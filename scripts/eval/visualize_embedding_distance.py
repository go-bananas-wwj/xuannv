#!/usr/bin/env python3
"""可视化: BEFORE / AFTER / Embedding Distance (3列)."""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
DATA_ROOT = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered"

PERIODS = [
    ("apr_jun", 1743436800000, 1746057599000, 1748736000000, 1751327999000, "june.shp"),
    ("jun_aug", 1748736000000, 1751327999000, 1754006400000, 1756655999000, "aug.shp"),
    ("aug_sept", 1754006400000, 1756655999000, 1756771200000, 1759247999000, "September.shp"),
    ("sept_oct", 1756771200000, 1759247999000, 1759449600000, 1761926399000, "October.shp"),
]

EXPERIMENTS = [
    "aef_baseline", "aef_high_consist", "aef_no_static", "aef_skip_l2",
    "aef_diff_recon", "aef_high_kappa", "aef_cyclic_unif", "aef_no_uniform",
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

def find_best_frame(patch_id, start_ms, end_ms):
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
        rgb = data[:3].astype(np.float32)
        for i in range(3):
            p99 = np.percentile(rgb[i], 99)
            if p99 > 0:
                rgb[i] = np.clip(rgb[i] / p99, 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb

def generate_mask_fast(gdf, bounds, shape):
    """Fast rasterize using rasterio."""
    minx, miny, maxx, maxy = bounds
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
        (maxx - minx) / shape[1], -(maxy - miny) / shape[0]
    )
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros(shape, dtype=bool)
    mask = rasterize(shapes, out_shape=shape, transform=transform, fill=0, default_value=1, dtype=np.uint8)
    return mask.astype(bool)

def compute_distance(before_emb, after_emb):
    b_norm = before_emb / (np.linalg.norm(before_emb, axis=0, keepdims=True) + 1e-8)
    a_norm = after_emb / (np.linalg.norm(after_emb, axis=0, keepdims=True) + 1e-8)
    sim = np.sum(b_norm * a_norm, axis=0)
    dist = 1.0 - sim
    return dist

def get_mask_outline(mask):
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask, iterations=1)
    outline = dilated & ~mask
    return outline

out_base = Path("/workspace/outputs/xuannv_round1/embedding_distance_viz")
out_base.mkdir(parents=True, exist_ok=True)

for exp_name in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*60}")
    
    for period_name, b_start, b_end, a_start, a_end, shp_name in PERIODS:
        emb_dir = Path(f"/workspace/outputs/xuannv_round1/{exp_name}/eval/embeddings/{period_name}")
        if not emb_dir.exists():
            print(f"  Skip {period_name}: no embeddings")
            continue

        gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)

        annotated_pids = set()
        for pid, bounds in patch_bounds.items():
            patch_box = box(*bounds)
            for geom in gdf.geometry:
                if geom is not None and patch_box.intersects(geom):
                    annotated_pids.add(pid)
                    break

        patch_info = []
        for pid in annotated_pids:
            bounds = patch_bounds[pid]
            b_frame = find_best_frame(pid, b_start, b_end)
            a_frame = find_best_frame(pid, a_start, a_end)
            b_emb_path = emb_dir / f"{pid}_before.npy"
            a_emb_path = emb_dir / f"{pid}_after.npy"

            if b_frame is None or a_frame is None or not b_emb_path.exists() or not a_emb_path.exists():
                continue

            rgb = read_rgb(b_frame)
            H, W = rgb.shape[:2]
            mask = generate_mask_fast(gdf, bounds, (H, W))
            n_changed = mask.sum()
            if n_changed < 10:
                continue
            patch_info.append((pid, n_changed, b_frame, a_frame, mask, H, W, b_emb_path, a_emb_path))

        if len(patch_info) == 0:
            print(f"  Skip {period_name}: no valid patches")
            continue

        patch_info.sort(key=lambda x: x[1], reverse=True)
        top_patches = patch_info[:6]

        fig, axes = plt.subplots(len(top_patches), 3, figsize=(15, 5*len(top_patches)))
        if len(top_patches) == 1:
            axes = axes.reshape(1, -1)

        for idx, (pid, n_changed, b_frame, a_frame, mask, H, W, b_emb_path, a_emb_path) in enumerate(top_patches):
            rgb_before = read_rgb(b_frame)
            rgb_after = read_rgb(a_frame)

            emb_b = np.load(b_emb_path)
            emb_a = np.load(a_emb_path)
            dist_map = compute_distance(emb_b, emb_a)

            # Resize mask to embedding size if needed
            if mask.shape != dist_map.shape:
                from scipy.ndimage import zoom
                zoom_y = dist_map.shape[0] / mask.shape[0]
                zoom_x = dist_map.shape[1] / mask.shape[1]
                mask_emb = zoom(mask.astype(float), (zoom_y, zoom_x), order=0) > 0.5
            else:
                mask_emb = mask

            outline = get_mask_outline(mask)
            outline_emb = get_mask_outline(mask_emb)

            axes[idx, 0].imshow(rgb_before)
            mask_overlay = np.zeros((H, W, 4))
            mask_overlay[mask] = [1, 0, 0, 0.18]
            mask_overlay[outline] = [1, 0, 0, 0.9]
            axes[idx, 0].imshow(mask_overlay)
            axes[idx, 0].set_title(f'{pid} BEFORE\n{n_changed} changed ({n_changed/(H*W)*100:.1f}%)')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(rgb_after)
            mask_overlay = np.zeros((H, W, 4))
            mask_overlay[mask] = [1, 0, 0, 0.18]
            mask_overlay[outline] = [1, 0, 0, 0.9]
            axes[idx, 1].imshow(mask_overlay)
            axes[idx, 1].set_title(f'{pid} AFTER')
            axes[idx, 1].axis('off')

            eh, ew = dist_map.shape
            im = axes[idx, 2].imshow(dist_map, cmap='hot', vmin=0, vmax=1.0)
            outline_overlay = np.zeros((eh, ew, 4))
            outline_overlay[outline_emb] = [0, 1, 0, 0.9]
            axes[idx, 2].imshow(outline_overlay)
            axes[idx, 2].set_title(f'{pid} EMB DIST\nmean={dist_map.mean():.3f} max={dist_map.max():.3f}')
            axes[idx, 2].axis('off')
            plt.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)

        fig.suptitle(f'{exp_name} — {period_name} — Embedding Cosine Distance', fontsize=14)
        plt.tight_layout()
        out_path = out_base / f'{exp_name}_{period_name}_emb_dist.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {period_name}: saved {out_path} ({len(top_patches)} patches)")

print(f"\n{'='*60}")
print(f"  All visualizations saved to {out_base}")
print(f"{'='*60}")
