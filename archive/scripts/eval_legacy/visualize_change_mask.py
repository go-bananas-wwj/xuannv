#!/usr/bin/env python3
"""可视化变化检测 mask 叠加到 S2 遥感图像上."""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
DATA_ROOT = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered"
OUTPUT_DIR = Path("/workspace/outputs/xuannv_round1/change_mask_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 时期定义: (名称, shapefile)
PERIODS = [
    ("apr→jun", "june.shp"),
    ("jun→aug", "aug.shp"),
    ("aug→sept", "September.shp"),
    ("sept→oct", "October.shp"),
]

# 加载 grid
with open(GRID_PATH) as f:
    grid_data = json.load(f)

patch_bounds = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

# 读取 S2 图像（取第一个可用帧）
def read_s2_for_patch(patch_id, image_size=256):
    s2_dir = Path(DATA_ROOT) / "s2" / patch_id
    if not s2_dir.exists():
        return None
    tif_files = sorted(s2_dir.glob("*.tif"))
    if not tif_files:
        return None
    
    with rasterio.open(tif_files[0]) as src:
        data = src.read()
        # data shape: [C, H, W]
        # S2 有多个波段，取 RGB: B4(红), B3(绿), B2(蓝)
        # 但我们的数据可能是 6 通道: [B2, B3, B4, B5, B6, B7] 或类似
        # 取前3个通道作为 RGB
        rgb = data[:3]
        # 归一化到 0-1
        rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb

# 生成 changed_mask
def generate_mask(patch_id, gdf, bounds, H, W):
    minx, miny, maxx, maxy = bounds
    mask = np.zeros((H, W), dtype=bool)
    
    patch_box = box(minx, miny, maxx, maxy)
    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        if not patch_box.intersects(row.geometry):
            continue
        
        for y in range(H):
            for x in range(W):
                px = minx + (x + 0.5) / W * (maxx - minx)
                py = maxy - (y + 0.5) / H * (maxy - miny)
                pt = box(px, py, px, py)
                if row.geometry.contains(pt) or row.geometry.intersects(pt):
                    mask[y, x] = True
    return mask

# 可视化
for period_name, shp_name in PERIODS:
    print(f"\n{'='*60}")
    print(f"  Period: {period_name}")
    print(f"  Shapefile: {shp_name}")
    print(f"{'='*60}")
    
    # 加载 shapefile
    gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    
    print(f"  总标注数: {len(gdf)}")
    
    # 找出有标注的 patch
    annotated_pids = []
    for pid, bounds in patch_bounds.items():
        patch_box = box(*bounds)
        for _, row in gdf.iterrows():
            if row.geometry is not None and patch_box.intersects(row.geometry):
                annotated_pids.append(pid)
                break
    
    print(f"  带标注 patch: {len(annotated_pids)} 个")
    
    # 选择前6个有标注的 patch 可视化
    viz_pids = annotated_pids[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, pid in enumerate(viz_pids):
        rgb = read_s2_for_patch(pid)
        if rgb is None:
            axes[idx].text(0.5, 0.5, f'{pid}\nNo S2 data', ha='center', va='center')
            axes[idx].axis('off')
            continue
        
        H, W = rgb.shape[:2]
        bounds = patch_bounds[pid]
        mask = generate_mask(pid, gdf, bounds, H, W)
        
        n_changed = mask.sum()
        
        # 显示原图
        axes[idx].imshow(rgb)
        
        # 叠加 mask（红色半透明）
        mask_overlay = np.zeros((H, W, 4))
        mask_overlay[mask] = [1, 0, 0, 0.5]
        axes[idx].imshow(mask_overlay)
        
        axes[idx].set_title(f'{pid}\n{n_changed} changed pixels ({n_changed/(H*W)*100:.1f}%)')
        axes[idx].axis('off')
    
    plt.suptitle(f'Change Detection Mask Visualization - {period_name}', fontsize=14)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'{period_name.replace("→", "_")}_mask_viz.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

print(f"\n{'='*60}")
print(f"  All visualizations saved to {OUTPUT_DIR}")
print(f"{'='*60}")
