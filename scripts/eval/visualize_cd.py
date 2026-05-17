#!/usr/bin/env python
"""变化检测可视化 — 生成 before/after embedding PCA-RGB + 变化概率图。

输出:
    每个标注 patch 每个时期 3 张图:
    - before_embedding_pca.png
    - after_embedding_pca.png
    - change_probability.png

用法:
    python visualize_cd.py \
        --embedding-file /path/to/patch_embeddings.npz \
        --output-dir /path/to/evaluation/visualizations/cd_before_after \
        --device npu:0
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import box
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio import features

sys.path.insert(0, "/workspace/xuannv")

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
    bounds = {}
    for _, row in gdf.iterrows():
        pid = row.get("sample_id") or row.get("patch_id") or row.get("id")
        if pid is None:
            continue
        coords = list(row.geometry.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return bounds


def rasterize_shapefile(shp_path, bounds, h, w):
    if not shp_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    minx, miny, maxx, maxy = bounds
    import rasterio
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale((maxx - minx) / w, (miny - maxy) / h)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if len(shapes) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    return features.rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)


def emb_to_rgb(emb_map):
    """将 [D, H, W] embedding map 用 PCA 降维到 [H, W, 3] RGB."""
    D, H, W = emb_map.shape
    flat = emb_map.reshape(D, -1).T  # [H*W, D]
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat)  # [H*W, 3]
    rgb = rgb.reshape(H, W, 3)
    # 归一化到 [0, 1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.clip(rgb, 0, 1)


def visualize_patch(args_tuple):
    """可视化单个 patch 的某个时期.
    
    args_tuple: (pid, period_name, period_info, emb_before, emb_after, 
                  patch_bounds, output_dir)
    """
    pid, period_name, period_info, emb_before, emb_after, patch_bounds, output_dir = args_tuple
    
    pid_str = str(pid)
    if pid_str not in patch_bounds:
        return None
    
    bounds = patch_bounds[pid_str]
    shp_path = ANNOT_DIR / period_info["shp"]
    
    # 栅格化标注
    H, W = emb_before.shape[1:]
    change_mask = rasterize_shapefile(shp_path, bounds, H, W)
    
    # PCA-RGB
    rgb_before = emb_to_rgb(emb_before)
    rgb_after = emb_to_rgb(emb_after)
    
    # 变化概率（简单用 cosine distance 作为概率）
    cos_map = np.sum(emb_before * emb_after, axis=0)  # [H, W]
    change_prob = 1.0 - cos_map
    change_prob = (change_prob - change_prob.min()) / (change_prob.max() - change_prob.min() + 1e-8)
    
    # 创建输出目录
    period_dir = output_dir / period_name
    period_dir.mkdir(parents=True, exist_ok=True)
    
    # 图 1: before embedding
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb_before, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    # 叠加变化边界
    if change_mask.any():
        contours = plt.contour(change_mask, levels=[0.5], colors='red', linewidths=2,
                               extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    ax.set_title(f"{pid} Before ({period_info['before']}月) Embedding PCA-RGB")
    ax.axis('off')
    fig.savefig(period_dir / f"{pid}_before_embedding_pca.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 图 2: after embedding
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb_after, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    if change_mask.any():
        plt.contour(change_mask, levels=[0.5], colors='red', linewidths=2,
                    extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    ax.set_title(f"{pid} After ({period_info['after']}月) Embedding PCA-RGB")
    ax.axis('off')
    fig.savefig(period_dir / f"{pid}_after_embedding_pca.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 图 3: 变化概率热力图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(change_prob, cmap='jet', extent=[bounds[0], bounds[2], bounds[1], bounds[3]], vmin=0, vmax=1)
    if change_mask.any():
        plt.contour(change_mask, levels=[0.5], colors='lime', linewidths=2,
                    extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    ax.set_title(f"{pid} Change Probability")
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Change Probability')
    fig.savefig(period_dir / f"{pid}_change_probability.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return pid_str


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-jobs", type=int, default=8, help="并行进程数")
    args = p.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 embedding
    print("[Viz] 加载 embedding...")
    data = np.load(args.embedding_file)
    spatial_maps = data["spatial_maps"]  # [424, 12, D, H, W]
    patch_ids = data["patch_ids"]
    print(f"      形状: {spatial_maps.shape}")
    
    # 加载 patch 边界
    patch_bounds = load_patch_bounds()
    
    # 收集所有可视化任务
    tasks = []
    for period_name, period_info in PERIODS.items():
        before_idx = period_info["before"] - 1
        after_idx = period_info["after"] - 1
        
        for p_idx, pid in enumerate(patch_ids):
            pid_str = str(pid)
            if pid_str not in patch_bounds:
                continue
            
            # 检查该 patch 是否有变化标注（简化：全部输出）
            emb_before = spatial_maps[p_idx, before_idx]  # [D, H, W]
            emb_after = spatial_maps[p_idx, after_idx]    # [D, H, W]
            
            tasks.append((pid, period_name, period_info, emb_before, emb_after, patch_bounds, output_dir))
    
    print(f"[Viz] 共 {len(tasks)} 个可视化任务")
    
    # 并行生成
    with Pool(args.n_jobs) as pool:
        results = pool.map(visualize_patch, tasks)
    
    completed = sum(1 for r in results if r is not None)
    print(f"[Viz] 完成! {completed}/{len(tasks)} 张图已生成")
    print(f"      输出目录: {output_dir}")


if __name__ == "__main__":
    main()
