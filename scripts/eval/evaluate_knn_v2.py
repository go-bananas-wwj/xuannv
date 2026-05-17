#!/usr/bin/env python
"""KNN 下游评估 v2 — 使用预计算 embedding。

用法:
    python evaluate_knn_v2.py \
        --embedding-file /path/to/patch_embeddings.npz \
        --output-dir /path/to/evaluation/downstream \
        --device npu:0 \
        --k 5
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch_npu
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, "/workspace/xuannv")

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
    """Resize label to target size using nearest neighbor."""
    import torch.nn.functional as F
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(label_t, size=(target_h, target_w), mode="nearest")
    return resized.squeeze().numpy().astype(label.dtype)


def knn_predict_npu(X_train, y_train, X_test, k, device):
    from scipy.stats import mode
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    batch_size = 256
    preds = []
    for i in range(0, len(X_test), batch_size):
        batch = torch.from_numpy(X_test[i:i+batch_size]).to(device)
        dist = torch.cdist(batch, X_train_t)
        _, idx = dist.topk(min(k, len(X_train)), largest=False, dim=1)
        neighbor_labels = y_train_t[idx].cpu().numpy()  # [B, k]
        batch_preds = mode(neighbor_labels, axis=1).mode.flatten()
        preds.append(batch_preds)
    return np.concatenate(preds)


def evaluate_task(task_name, label_dir, label_file, num_classes, spatial_maps, patch_ids, device, k, seed=42):
    """评估单个下游任务.
    
    spatial_maps: [num_patches, D, H, W]
    patch_ids: [num_patches] str
    """
    D, H, W = spatial_maps.shape[1:]
    rng = np.random.RandomState(seed)
    n_patches = len(patch_ids)

    # 收集所有有效像素
    all_X = []
    all_y = []
    all_patch_idx = []

    # 标签映射
    mapping = LABEL_MAPPINGS.get(task_name)
    if mapping:
        num_classes = len(mapping)
    
    for p_idx, pid in enumerate(patch_ids):
        label, nodata = load_label(pid, label_dir, label_file)
        if label is None:
            continue
        
        # resize label 到 embedding 空间尺寸
        if label.shape != (H, W):
            label = resize_label(label, H, W)
        
        # 应用标签映射
        if mapping:
            label = np.vectorize(lambda x: mapping.get(x, -1))(label)
            nodata = -1
        
        # 过滤 nodata 和无效类别
        if nodata is not None:
            mask = (label != nodata) & (label >= 0) & (label < num_classes)
        else:
            mask = (label >= 0) & (label < num_classes)
        
        if mask.sum() == 0:
            continue
        
        emb = spatial_maps[p_idx]  # [D, H, W]
        all_X.append(emb[:, mask].T)  # [N_pixels, D]
        all_y.append(label[mask])
        all_patch_idx.append(np.full(mask.sum(), p_idx))
    
    if len(all_X) == 0:
        print(f"  [{task_name}] 无有效数据")
        return None
    
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_patch_idx = np.concatenate(all_patch_idx)
    
    # Patch-stratified 80/20 split
    n_train = max(1, int(n_patches * 0.8))
    train_pids = rng.choice(n_patches, n_train, replace=False)
    test_pids = [i for i in range(n_patches) if i not in train_pids]
    
    train_mask = np.isin(all_patch_idx, train_pids)
    test_mask = np.isin(all_patch_idx, test_pids)
    
    X_train = all_X[train_mask]
    y_train = all_y[train_mask]
    X_test = all_X[test_mask]
    y_test = all_y[test_mask]
    
    # Subsample train if too large
    if len(X_train) > 100000:
        idx = rng.choice(len(X_train), 100000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    # KNN
    y_pred = knn_predict_npu(X_train, y_train, X_test, k, device)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    
    # Per-class accuracy / IoU
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        support = cm[c, :].sum()
        per_class[f"class_{c}"] = {"iou": float(iou), "support": int(support)}
    
    # Mean IoU (only classes with support > 0)
    valid_ious = [v["iou"] for v in per_class.values() if v["support"] > 0]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    
    report = {
        "task": task_name,
        "k": k,
        "accuracy": float(acc),
        "mean_iou": float(mean_iou),
        "num_train_pixels": int(len(X_train)),
        "num_test_pixels": int(len(X_test)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    
    print(f"  [{task_name}] Acc={acc:.4f}, mIoU={mean_iou:.4f}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--month", type=int, default=6, help="使用哪个月的 embedding")
    args = p.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 embedding
    print("[KNN] 加载 embedding...")
    data = np.load(args.embedding_file)
    spatial_maps = data["spatial_maps"]  # [424, 12, D, H, W]
    patch_ids = data["patch_ids"]  # [424]
    
    # 取指定月份
    month_idx = args.month - 1
    spatial_maps = spatial_maps[:, month_idx]  # [424, D, H, W]
    print(f"      使用 {args.month} 月 embedding, 形状: {spatial_maps.shape}")
    
    # 评估每个任务
    all_reports = {}
    for task_name, label_dir, label_file, num_classes in TASKS:
        print(f"[KNN] 评估 {task_name}...")
        report = evaluate_task(
            task_name, label_dir, label_file, num_classes,
            spatial_maps, patch_ids, device, args.k
        )
        if report:
            all_reports[task_name] = report
            with open(output_dir / f"knn_{task_name}.json", "w") as f:
                json.dump(report, f, indent=2)
    
    # 汇总
    summary = {k: {"accuracy": v["accuracy"], "mean_iou": v["mean_iou"]} 
               for k, v in all_reports.items()}
    with open(output_dir / "knn_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("[KNN] 完成!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
