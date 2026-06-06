#!/usr/bin/env python3
"""OSM标签像素级线性探测评估 — 验证embedding对建筑/道路/水体的区分能力.

用法:
    python osm_linear_probe_eval.py \
        --config configs/config_v17_fix_collapse.yaml \
        --checkpoint /workspace/xuannv/outputs/exp_v17_fix_collapse_0606/epoch_5.pt \
        --osm-root /workspace/xuannv/data_raw/osm_labels \
        --device npu:0 --epochs 30 --max-patches 100
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
import rasterio
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import TensorDataset, DataLoader

from src.inference.engine import load_backbone


def parse_args():
    p = argparse.ArgumentParser(description="OSM标签像素级线性探测评估")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--osm-root", default="/workspace/xuannv/data_raw/osm_labels")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-patches", type=int, default=0, help="0表示全部")
    p.add_argument("--output", default="")
    return p.parse_args()


OSM_CLASSES = ["building", "road", "water"]


def load_osm_labels(osm_root: str, patch_id: str) -> dict[str, np.ndarray]:
    """加载一个patch的所有OSM标签."""
    labels = {}
    pid_dir = Path(osm_root) / patch_id
    if not pid_dir.exists():
        return labels
    for cls in OSM_CLASSES:
        tif_path = pid_dir / f"{cls}.tif"
        if tif_path.exists():
            with rasterio.open(tif_path) as src:
                labels[cls] = src.read(1)
    return labels


def extract_spatial_embeddings(model, dataset, cfg, device, patch_ids: list[str],
                                max_samples_per_patch: int = 3):
    """提取指定patches的空间embedding map.
    
    返回: {patch_id: [(month_emb_map, month), ...]}
    """
    model.eval()
    # 构建patch-month索引
    from src.data.dataset import HarbinPatchDataset
    patch_month_index: dict[tuple, int] = {}
    for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
        patch_month_index[(pid, year, month)] = idx
    
    results: dict[str, list] = {}
    for pid in patch_ids:
        # 找到该patch的所有月份
        months = [(y, m) for (p, y, m) in dataset.monthly_samples if p == pid]
        if not months:
            continue
        # 限制月份数量
        if len(months) > max_samples_per_patch:
            import random
            random.seed(42)
            months = random.sample(months, max_samples_per_patch)
        
        emb_maps = []
        for year, month in months:
            key = (pid, year, month)
            if key not in patch_month_index:
                continue
            item = dataset[patch_month_index[key]]
            with torch.no_grad():
                out = model(
                    source_frames=item["source_frames"].unsqueeze(0).to(device),
                    source_timestamps_ms=item["source_timestamps_ms"].unsqueeze(0).to(device),
                    source_frame_mask=item["source_frame_mask"].unsqueeze(0).to(device),
                    source_input_mask=item["source_input_mask"].unsqueeze(0).to(device),
                    source_type_ids=item["source_type_ids"].unsqueeze(0).to(device),
                    valid_start_ms=torch.tensor([item["valid_start_ms"]], device=device),
                    valid_end_ms=torch.tensor([item["valid_end_ms"]], device=device),
                    target_relative_time=torch.zeros(1, cfg.data.num_target_sources, device=device),
                    target_metadata=torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=device),
                    skip_decoder=True,
                )
                emb = out.embedding_map.squeeze(0).cpu().numpy()  # [D, H, W]
            emb_maps.append(emb)
        if emb_maps:
            results[pid] = emb_maps
    return results


def prepare_pixel_data(embeddings: dict[str, list], osm_root: str) -> tuple[np.ndarray, np.ndarray]:
    """将embedding maps和OSM标签展平为像素级训练数据.
    
    返回 X: [N_pixels, D], y: [N_pixels, n_classes]
    """
    X_list, y_list = [], []
    for pid, emb_maps in embeddings.items():
        labels = load_osm_labels(osm_root, pid)
        if not labels:
            continue
        # 合并所有OSM标签为单张多通道图 [C, H, W]
        H, W = next(iter(labels.values())).shape
        label_stack = np.zeros((len(OSM_CLASSES), H, W), dtype=np.float32)
        for i, cls in enumerate(OSM_CLASSES):
            if cls in labels:
                label_stack[i] = labels[cls].astype(np.float32)
        
        for emb_map in emb_maps:
            D, He, We = emb_map.shape
            # 将embedding map插值到标签分辨率
            if He != H or We != W:
                emb_tensor = torch.from_numpy(emb_map).unsqueeze(0)
                emb_resized = F.interpolate(emb_tensor, size=(H, W), mode="bilinear", align_corners=False)
                emb_map = emb_resized.squeeze(0).numpy()
            
            # 展平: [D, H, W] -> [H*W, D]
            emb_flat = emb_map.transpose(1, 2, 0).reshape(-1, D)
            label_flat = label_stack.reshape(len(OSM_CLASSES), -1).T  # [H*W, C]
            
            X_list.append(emb_flat)
            y_list.append(label_flat)
    
    if not X_list:
        return np.array([]), np.array([])
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def train_binary_linear_probe(X_train, y_train, X_val, y_val, epochs, lr, batch_size, device):
    """训练每个类别的二分类线性探测头."""
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    
    D = X_train.shape[1]
    n_classes = y_train.shape[1]
    
    # 共享的线性层 + sigmoid
    classifier = nn.Linear(D, n_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_f1 = 0.0
    for epoch in range(epochs):
        classifier.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        classifier.eval()
        with torch.no_grad():
            logits = classifier(X_val_t)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        
        # 平均F1
        f1s = []
        for c in range(n_classes):
            if y_val[:, c].sum() > 0:
                f1s.append(f1_score(y_val[:, c], preds[:, c], zero_division=0))
        avg_f1 = np.mean(f1s) if f1s else 0.0
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
    
    # 最终评估
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(np.float32)
    
    results = {}
    for c, cls_name in enumerate(OSM_CLASSES):
        y_true_c = y_val[:, c]
        y_pred_c = preds[:, c]
        if y_true_c.sum() == 0:
            continue
        results[cls_name] = {
            "accuracy": float(accuracy_score(y_true_c, y_pred_c)),
            "f1": float(f1_score(y_true_c, y_pred_c, zero_division=0)),
            "iou": float(jaccard_score(y_true_c, y_pred_c, zero_division=0)),
            "pos_rate": float(y_true_c.mean()),
        }
    
    return results


def main():
    args = parse_args()
    device = args.device
    
    print("=" * 60)
    print("  OSM标签像素级线性探测评估")
    print("=" * 60)
    
    print("\n加载模型...")
    model, dataset, cfg = load_backbone(args.config, args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # 获取所有有OSM标签的patch
    osm_root = Path(args.osm_root)
    available_pids = [d.name for d in osm_root.iterdir() if d.is_dir()]
    if args.max_patches > 0:
        available_pids = available_pids[:args.max_patches]
    print(f"\n找到 {len(available_pids)} 个有OSM标签的patch")
    if len(available_pids) == 0:
        print("[ERROR] 无OSM标签可用，请先运行 download_osm_labels.py")
        sys.exit(1)
    
    print("提取 spatial embedding maps...")
    embeddings = extract_spatial_embeddings(model, dataset, cfg, device, available_pids,
                                             max_samples_per_patch=2)
    print(f"  成功提取 {len(embeddings)} 个patch的embedding")
    
    print("准备像素级训练数据...")
    X, y = prepare_pixel_data(embeddings, args.osm_root)
    if len(X) == 0:
        print("[ERROR] 无有效训练数据")
        sys.exit(1)
    print(f"  总像素数: {len(X)}, embedding_dim: {X.shape[1]}")
    
    # 分层采样划分train/val（按像素采样，注意空间相关性）
    # 简单随机划分（实际应用应考虑空间划分）
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n训练集: {len(X_train)} 像素, 验证集: {len(X_val)} 像素")
    for i, cls in enumerate(OSM_CLASSES):
        print(f"  {cls}: train_pos={y_train[:, i].sum():.0f} ({y_train[:, i].mean()*100:.2f}%) "
              f"val_pos={y_val[:, i].sum():.0f} ({y_val[:, i].mean()*100:.2f}%)")
    
    print(f"\n训练线性探测头 ({args.epochs} epochs, lr={args.lr})...")
    results = train_binary_linear_probe(X_train, y_train, X_val, y_val,
                                        args.epochs, args.lr, args.batch_size, device)
    
    print("\n" + "=" * 60)
    print("  评估结果")
    print("=" * 60)
    for cls, metrics in results.items():
        print(f"  {cls:10s}: Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"IoU={metrics['iou']:.4f}  pos_rate={metrics['pos_rate']*100:.2f}%")
    
    out_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "osm_linear_probe_results.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
