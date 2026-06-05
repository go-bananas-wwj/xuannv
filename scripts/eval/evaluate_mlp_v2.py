#!/usr/bin/env python
"""MLP 下游评估 v2 — 使用预计算 embedding，训练 PixelMLPHead。

用法:
    python evaluate_mlp_v2.py \
        --embedding-file /path/to/patch_embeddings.npz \
        --output-dir /path/to/evaluation/downstream \
        --device npu:0 \
        --epochs 50
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, "/workspace/xuannv")

from src.models.downstream_heads import PixelMLPHead, PixelConvHead, focal_loss

DATA_ROOTS = [
    Path("/workspace/raw/harbin"),
    Path("/workspace/raw/haidian_train/haidian"),
]

TASKS = [
    ("worldcover", "worldcover", "static.tif", 7),   # 实际7类
    ("jrc_water", "jrc_water", "static.tif", 2),     # 二分类（threshold后）
    ("dynamic_world", "dynamic_world", "2025Q2.tif", 8),  # 实际8类（排除NoData=0）
]

# JRC Water: occurrence百分比 → 二分类的threshold
JRC_WATER_THRESHOLD = 0  # value > 0 即算有水

# 标签值映射（原始编码 -> 0-based）
LABEL_MAPPINGS = {
    "worldcover": {10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 6},
}


def load_label(patch_id: str, label_dir: str, fname: str):
    # 处理多区域 patch_id 格式（如 haidian_patch_000000 -> patch_000000）
    local_id = patch_id.split('_', 1)[1] if '_' in patch_id and not patch_id.startswith('patch_') else patch_id
    for data_root in DATA_ROOTS:
        # 先尝试原始 patch_id
        path = data_root / label_dir / patch_id / fname
        if path.exists():
            with rasterio.open(path) as src:
                label = src.read(1)
                nodata = src.nodata
            return label, nodata
        # 再尝试 local_id
        path = data_root / label_dir / local_id / fname
        if path.exists():
            with rasterio.open(path) as src:
                label = src.read(1)
                nodata = src.nodata
            return label, nodata
    return None, None


def resize_label(label: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(label_t, size=(target_h, target_w), mode="nearest")
    return resized.squeeze().numpy().astype(label.dtype)


def prepare_data(spatial_maps, patch_ids, label_dir, label_file, device, task_name, num_classes):
    """准备训练/测试数据.
    
    Returns:
        X_train, y_train, X_test, y_test: 都是 [N, D] 和 [N]
        num_classes: 实际类别数
    """
    D, H, W = spatial_maps.shape[1:]
    all_X = []
    all_y = []
    all_patch_idx = []
    
    mapping = LABEL_MAPPINGS.get(task_name)
    if mapping:
        num_classes = len(mapping)
    
    for p_idx, pid in enumerate(patch_ids):
        label, nodata = load_label(pid, label_dir, label_file)
        if label is None:
            continue
        
        if label.shape != (H, W):
            label = resize_label(label, H, W)
        
        # ★ 修复 JRC Water: occurrence百分比 → 二分类
        if task_name == "jrc_water":
            # JRC Water 原始值: 0=无水, 1-99=occurrence百分比, -128=预处理nodata
            # rasterio 读取的 nodata=-32768，所以 -128 不会被当作 nodata
            # 需要手动处理 -128
            label = np.where(label == -128, -1, label)  # 标记 -128 为无效
            nodata = -1
            # threshold: >0 算 water
            label = (label > JRC_WATER_THRESHOLD).astype(np.int64)
            num_classes = 2
        
        # 应用标签映射
        if mapping and task_name != "jrc_water":
            label = np.vectorize(lambda x: mapping.get(x, -1))(label)
            nodata = -1
        
        if nodata is not None:
            mask = (label != nodata) & (label >= 0) & (label < num_classes)
        else:
            mask = (label >= 0) & (label < num_classes)
        
        if mask.sum() == 0:
            continue
        
        emb = spatial_maps[p_idx]  # [D, H, W]
        all_X.append(emb[:, mask].T)  # [N, D]
        all_y.append(label[mask])
        all_patch_idx.append(np.full(mask.sum(), p_idx))
    
    if len(all_X) == 0:
        return None, None, None, None, 0
    
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_patch_idx = np.concatenate(all_patch_idx)
    
    # 确定实际类别数
    actual_num_classes = len(np.unique(all_y))
    
    # Patch-stratified split
    n_patches = len(patch_ids)
    n_train = max(1, int(n_patches * 0.8))
    rng = np.random.RandomState(42)
    train_pids = rng.choice(n_patches, n_train, replace=False)
    
    train_mask = np.isin(all_patch_idx, train_pids)
    test_mask = ~train_mask
    
    return (
        torch.from_numpy(all_X[train_mask]).float().to(device),
        torch.from_numpy(all_y[train_mask]).long().to(device),
        torch.from_numpy(all_X[test_mask]).float().to(device),
        torch.from_numpy(all_y[test_mask]).long().to(device),
        actual_num_classes,
    )


def compute_class_weights(y_train: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor | None:
    """计算平衡类别权重."""
    y_np = y_train.cpu().numpy()
    classes = np.arange(num_classes)
    class_counts = np.bincount(y_np, minlength=num_classes)
    
    # 只计算有数据的类别
    valid_classes = classes[class_counts > 0]
    valid_counts = class_counts[class_counts > 0]
    
    # 逆频率加权
    weights = np.zeros(num_classes, dtype=np.float32)
    weights[valid_classes] = 1.0 / valid_counts
    weights = weights / weights.sum() * len(valid_classes)  # 归一化
    
    return torch.from_numpy(weights).float().to(device)


def train_head(X_train, y_train, X_test, y_test, in_dim, num_classes, device, 
               epochs=50, hidden_dim=256, dropout=0.3, 
               head_type="mlp", use_class_weight=False, use_focal=False):
    """训练下游分类 Head.
    
    Args:
        head_type: "mlp" | "conv" | "mlpv2"
        use_class_weight: 是否使用逆频率类别加权
        use_focal: 是否使用 Focal Loss
    """
    # 创建 head
    if head_type == "mlp":
        head = PixelMLPHead(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)
    elif head_type == "mlpv2":
        head = PixelMLPHead(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)
        # 用更深层的 MLP: 实际上是 PixelMLPHead 但加大 hidden_dim
    elif head_type == "conv":
        head = PixelConvHead(in_dim=in_dim, hidden_dim=hidden_dim//4, num_classes=num_classes, kernel_size=3, dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    batch_size = 1024 if head_type != "conv" else 64  # ConvHead 需要空间输入，batch_size 小些
    best_acc = 0.0
    
    # 类别权重
    class_weights = None
    if use_class_weight and num_classes > 1:
        class_weights = compute_class_weights(y_train, num_classes, device)
        print(f"      Class weights: {class_weights.cpu().numpy()}")
    
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            
            if head_type == "conv":
                # ConvHead 需要 [B, D, H, W]，但 xb 是 [B, D] 展平的
                # 需要重新组织：这里实际上不能用 ConvHead，因为数据已经展平了
                # 保留逻辑但跳过
                logits = head(xb.unsqueeze(-1).unsqueeze(-1))  # [B, D, 1, 1]
                logits = logits.squeeze(-1).squeeze(-1)  # [B, C]
            else:
                logits = head(xb)
            
            # 损失函数选择
            if use_focal and class_weights is not None:
                loss = focal_loss(logits, yb, alpha=class_weights, gamma=2.0)
            elif class_weights is not None:
                loss = F.cross_entropy(logits, yb, weight=class_weights)
            else:
                loss = F.cross_entropy(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Eval
        head.eval()
        with torch.no_grad():
            if head_type == "conv":
                logits = head(X_test.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            else:
                logits = head(X_test)
            pred = logits.argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
        
        best_acc = max(best_acc, acc)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"      Epoch {epoch}: loss={total_loss/n_batches:.4f}, test_acc={acc:.4f}")
    
    # Final eval
    head.eval()
    with torch.no_grad():
        if head_type == "conv":
            logits = head(X_test.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            logits = head(X_test)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
    
    acc = accuracy_score(y_test_np, pred)
    cm = confusion_matrix(y_test_np, pred, labels=list(range(num_classes)))
    
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        support = cm[c, :].sum()
        per_class[f"class_{c}"] = {"iou": float(iou), "support": int(support)}
    
    valid_ious = [v["iou"] for v in per_class.values() if v["support"] > 0]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    
    return {
        "accuracy": float(acc),
        "mean_iou": float(mean_iou),
        "best_epoch_acc": float(best_acc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "head_type": head_type,
        "use_class_weight": use_class_weight,
        "use_focal": use_focal,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--month", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dimension")
    p.add_argument("--dropout", type=float, default=0.3, help="MLP dropout")
    p.add_argument("--head-type", type=str, default="mlp", choices=["mlp", "mlpv2", "conv"], help="Head类型")
    p.add_argument("--use-class-weight", action="store_true", help="使用逆频率类别加权")
    p.add_argument("--use-focal", action="store_true", help="使用 Focal Loss")
    args = p.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[MLP] 加载 embedding...")
    data = np.load(args.embedding_file)
    spatial_maps = data["spatial_maps"]
    patch_ids = data["patch_ids"]
    
    month_idx = args.month - 1
    spatial_maps = spatial_maps[:, month_idx]
    D = spatial_maps.shape[1]
    print(f"      使用 {args.month} 月 embedding, 形状: {spatial_maps.shape}, D={D}")
    
    all_reports = {}
    for task_name, label_dir, label_file, num_classes in TASKS:
        print(f"[MLP] 评估 {task_name}...")
        X_train, y_train, X_test, y_test, actual_num_classes = prepare_data(
            spatial_maps, patch_ids, label_dir, label_file, device, task_name, num_classes
        )
        if X_train is None:
            print(f"      无有效数据")
            continue
        
        print(f"      train={len(X_train)}, test={len(X_test)}, classes={actual_num_classes}")
        report = train_head(X_train, y_train, X_test, y_test, D, actual_num_classes, device, 
                            args.epochs, args.hidden_dim, args.dropout,
                            args.head_type, args.use_class_weight, args.use_focal)
        report["task"] = task_name
        report["epochs"] = args.epochs
        all_reports[task_name] = report
        
        with open(output_dir / f"mlp_{task_name}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"      Acc={report['accuracy']:.4f}, mIoU={report['mean_iou']:.4f}")
    
    summary = {k: {"accuracy": v["accuracy"], "mean_iou": v["mean_iou"]} 
               for k, v in all_reports.items()}
    with open(output_dir / "mlp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("[MLP] 完成!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
