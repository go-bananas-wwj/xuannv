#!/usr/bin/env python
"""训练 ChangeDetectionHead — 使用预计算 embedding。

用法:
    python train_cd_head.py \
        --embedding-file /path/to/patch_embeddings.npz \
        --output-dir /path/to/evaluation/change_detection \
        --device npu:0 \
        --epochs 30
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
import geopandas as gpd
from shapely.geometry import box
from sklearn.metrics import roc_auc_score
from rasterio import features

sys.path.insert(0, "/workspace/xuannv")

from src.models.heads import ChangeDetectionHead

ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")

PERIODS = {
    "apr_jun": {"before": 4, "after": 6, "shp": "june.shp"},
    "jun_aug": {"before": 6, "after": 8, "shp": "aug.shp"},
    "aug_sep": {"before": 8, "after": 9, "shp": "September.shp"},
    "sep_oct": {"before": 9, "after": 10, "shp": "October.shp"},
}


def load_patch_bounds():
    gdf = gpd.read_file(GRID_PATH)
    patch_bounds = {}
    for _, row in gdf.iterrows():
        pid = row.get("sample_id") or row.get("patch_id") or row.get("id")
        if pid is None:
            continue
        coords = list(row.geometry.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return patch_bounds


def rasterize_shapefile(shp_path, bounds, h, w):
    """将 shapefile 栅格化为 [h, w] 二值 mask."""
    if not shp_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    
    minx, miny, maxx, maxy = bounds
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale((maxx - minx) / w, (miny - maxy) / h)
    
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if len(shapes) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    mask = features.rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
    return mask


def build_dataset(spatial_maps, patch_ids, period_info, patch_bounds):
    """构建某个时期的训练数据集.
    
    Returns:
        emb_before_list, emb_after_list, mask_list
    """
    before_idx = period_info["before"] - 1
    after_idx = period_info["after"] - 1
    shp_path = ANNOT_DIR / period_info["shp"]
    
    emb_before_list = []
    emb_after_list = []
    mask_list = []
    
    for p_idx, pid in enumerate(patch_ids):
        pid_str = str(pid)
        if pid_str not in patch_bounds:
            continue
        
        emb_before = spatial_maps[p_idx, before_idx]  # [D, H, W]
        emb_after = spatial_maps[p_idx, after_idx]    # [D, H, W]
        
        # 栅格化标注
        mask = rasterize_shapefile(shp_path, patch_bounds[pid_str], emb_before.shape[1], emb_before.shape[2])
        
        emb_before_list.append(emb_before)
        emb_after_list.append(emb_after)
        mask_list.append(mask)
    
    return (
        np.stack(emb_before_list, axis=0),  # [N, D, H, W]
        np.stack(emb_after_list, axis=0),   # [N, D, H, W]
        np.stack(mask_list, axis=0),        # [N, H, W]
    )


def train_cd_head(emb_before, emb_after, mask, device, epochs=30):
    """训练 CD Head.
    
    Args:
        emb_before: [N, D, H, W] numpy
        emb_after: [N, D, H, W] numpy
        mask: [N, H, W] numpy
    """
    D = emb_before.shape[1]
    
    # 转换为 tensor
    emb_b = torch.from_numpy(emb_before).float().to(device)
    emb_a = torch.from_numpy(emb_after).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)
    
    # 划分 train/test (按 sample 划分)
    n = len(emb_b)
    n_train = max(1, int(n * 0.8))
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    head = ChangeDetectionHead(embedding_dim=D, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        head.train()
        pred = head(emb_b[train_idx], emb_a[train_idx])  # [N_train, 1, H, W]
        loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(1), mask_t[train_idx]
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Eval
        if epoch % 5 == 0 or epoch == epochs - 1:
            head.eval()
            with torch.no_grad():
                pred_test = head(emb_b[test_idx], emb_a[test_idx]).squeeze(1)  # [N_test, H, W]
                proba = torch.sigmoid(pred_test).cpu().numpy().flatten()
                y_true = mask_t[test_idx].cpu().numpy().flatten()
            
            if y_true.sum() > 0 and y_true.sum() < len(y_true):
                auc = roc_auc_score(y_true, proba)
                best_auc = max(best_auc, auc)
                print(f"      Epoch {epoch}: loss={loss.item():.4f}, test_auc={auc:.4f}")
            else:
                print(f"      Epoch {epoch}: loss={loss.item():.4f}, test_auc=N/A")
    
    return head, best_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 embedding
    print("[CD Head] 加载 embedding...")
    data = np.load(args.embedding_file)
    spatial_maps = data["spatial_maps"]  # [424, 12, D, H, W]
    patch_ids = data["patch_ids"]
    D = spatial_maps.shape[2]
    print(f"          形状: {spatial_maps.shape}, D={D}")
    
    # 加载 patch 边界
    patch_bounds = load_patch_bounds()
    
    # 训练每个时期的 CD Head
    results = {}
    for period_name, period_info in PERIODS.items():
        print(f"[CD Head] 训练时期: {period_name}...")
        
        emb_before, emb_after, mask = build_dataset(
            spatial_maps, patch_ids, period_info, patch_bounds
        )
        print(f"          样本数: {len(emb_before)}")
        
        head, auc = train_cd_head(emb_before, emb_after, mask, device, args.epochs)
        
        results[period_name] = {"auc": float(auc), "n_samples": len(emb_before)}
        
        # 保存模型
        torch.save(head.state_dict(), output_dir / f"cd_head_{period_name}.pt")
        print(f"          AUC={auc:.4f}")
    
    with open(output_dir / "cd_head_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("[CD Head] 完成!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
