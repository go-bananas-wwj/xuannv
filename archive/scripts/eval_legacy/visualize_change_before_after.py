#!/usr/bin/env python3
"""可视化变化前/后对比 + mask 叠加."""
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

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
DATA_ROOT = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered"
OUTPUT_DIR = Path("/workspace/outputs/xuannv_round1/change_mask_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 时期: (名称, before窗口ms, after窗口ms, shapefile, 描述)
PERIODS = [
    ("apr→jun", 1743436800000, 1746057599000, 1748736000000, 1751327999000, "june.shp", "2025 Apr-May → Jun-Jul"),
    ("jun→aug", 1748736000000, 1751327999000, 1754006400000, 1756655999000, "aug.shp", "2025 Jun-Jul → Aug-Sep"),
    ("aug→sept", 1754006400000, 1756655999000, 1756771200000, 1759247999000, "September.shp", "2025 Aug-Sep → Sep-Oct"),
    ("sept→oct", 1756771200000, 1759247999000, 1759449600000, 1761926399000, "October.shp", "2025 Sep-Oct → Oct-Nov"),
]

with open(GRID_PATH) as f:
    grid_data = json.load(f)

patch_bounds = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

def ms_to_date(ms):
    from datetime import datetime
    return datetime.utcfromtimestamp(ms // 1000).strftime("%Y%m%d")

def find_best_frame(patch_id, start_ms, end_ms):
    """找到窗口内最接近中间日期的帧."""
    s2_dir = Path(DATA_ROOT) / "s2" / patch_id
    if not s2_dir.exists():
        return None
    tif_files = sorted(s2_dir.glob("*.tif"))
    
    start_date = start_ms // 1000
    end_date = end_ms // 1000
    mid_date = (start_date + end_date) // 2
    
    best = None
    best_diff = float('inf')
    for tf in tif_files:
        try:
            date = int(tf.stem)
            ts = int(datetime.strptime(str(date), "%Y%m%d").timestamp())
            if start_date <= ts <= end_date:
                diff = abs(ts - mid_date)
                if diff < best_diff:
                    best_diff = diff
                    best = tf
        except:
            continue
    return best

def read_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()
        rgb = data[:3]
        rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb

def generate_mask(pid, gdf, bounds, H, W):
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

from datetime import datetime

for period_name, b_start, b_end, a_start, a_end, shp_name, desc in PERIODS:
    print(f"\n{'='*60}")
    print(f"  Period: {period_name} ({desc})")
    print(f"{'='*60}")
    
    gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    
    annotated_pids = []
    for pid, bounds in patch_bounds.items():
        patch_box = box(*bounds)
        for _, row in gdf.iterrows():
            if row.geometry is not None and patch_box.intersects(row.geometry):
                annotated_pids.append(pid)
                break
    
    # 选择变化像素最多的前6个 patch
    patch_info = []
    for pid in annotated_pids:
        bounds = patch_bounds[pid]
        b_frame = find_best_frame(pid, b_start, b_end)
        a_frame = find_best_frame(pid, a_start, a_end)
        if b_frame is None or a_frame is None:
            continue
        
        rgb = read_rgb(b_frame)
        H, W = rgb.shape[:2]
        mask = generate_mask(pid, gdf, bounds, H, W)
        n_changed = mask.sum()
        patch_info.append((pid, n_changed, b_frame, a_frame, mask, H, W))
    
    patch_info.sort(key=lambda x: x[1], reverse=True)
    top_patches = patch_info[:6]
    
    fig, axes = plt.subplots(len(top_patches), 3, figsize=(15, 5*len(top_patches)))
    if len(top_patches) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (pid, n_changed, b_frame, a_frame, mask, H, W) in enumerate(top_patches):
        rgb_before = read_rgb(b_frame)
        rgb_after = read_rgb(a_frame)
        
        mask_overlay = np.zeros((H, W, 4))
        mask_overlay[mask] = [1, 0, 0, 0.5]
        
        # Before
        axes[idx, 0].imshow(rgb_before)
        axes[idx, 0].imshow(mask_overlay)
        axes[idx, 0].set_title(f'{pid} BEFORE\n{n_changed} changed ({n_changed/(H*W)*100:.1f}%)')
        axes[idx, 0].axis('off')
        
        # After
        axes[idx, 1].imshow(rgb_after)
        axes[idx, 1].imshow(mask_overlay)
        axes[idx, 1].set_title(f'{pid} AFTER')
        axes[idx, 1].axis('off')
        
        # Difference
        diff = np.abs(rgb_after.astype(float) - rgb_before.astype(float))
        diff = np.clip(diff * 3, 0, 1)
        axes[idx, 2].imshow(diff)
        axes[idx, 2].set_title(f'{pid} DIFF (x3)')
        axes[idx, 2].axis('off')
    
    plt.suptitle(f'Change Detection: {period_name} ({desc})', fontsize=14)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'{period_name.replace("→", "_")}_before_after.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path} ({len(top_patches)} patches)")

print(f"\n{'='*60}")
print(f"  All visualizations saved to {OUTPUT_DIR}")
print(f"{'='*60}")
