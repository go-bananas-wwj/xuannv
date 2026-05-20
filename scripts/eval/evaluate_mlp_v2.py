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

from src.models.downstream_heads import PixelMLPHead

DATA_ROOT = Path("/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered")

TASKS = [
    ("worldcover", "worldcover", "static.tif", 10),
    ("jrc_water", "jrc_water", "static.tif", 2),
    ("dynamic_world", "dynamic_world", "2025Q2.tif", 9),
]

# 标签值映射（原始编码 -> 0-based）
LABEL_MAPPINGS = {
    "worldcover": {10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 6},
}


def load_label(patch_id: str, label_dir: str, fname: str):
    path = DATA_ROOT / label_dir / patch_id / fname
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        label = src.read(1)
        nodata = src.nodata
    return label, nodata


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
        
        # 应用标签映射
        if mapping:
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
    actual_num_classes = num_classes
    
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


def train_mlp_head(X_train, y_train, X_test, y_test, in_dim, num_classes, device, epochs=50, hidden_dim=256, dropout=0.3):
    head = PixelMLPHead(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    batch_size = 1024
    best_acc = 0.0
    
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            
            logits = head(xb)
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
            logits = head(X_test)
            pred = logits.argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
        
        best_acc = max(best_acc, acc)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"      Epoch {epoch}: loss={total_loss/n_batches:.4f}, test_acc={acc:.4f}")
    
    # Final eval
    head.eval()
    with torch.no_grad():
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
        report = train_mlp_head(X_train, y_train, X_test, y_test, D, actual_num_classes, device, args.epochs, args.hidden_dim, args.dropout)
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
