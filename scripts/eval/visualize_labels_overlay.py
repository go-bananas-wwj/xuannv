#!/usr/bin/env python3
"""将变化检测标签叠加到真实 S2 图像上，验证坐标正确性."""
import sys, json
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from pathlib import Path

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
RAW_DIR = "/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered"


def rasterize_annotations(changes, bounds, H=64, W=64):
    resolution = (bounds[2] - bounds[0]) / W
    resolution_y = (bounds[3] - bounds[1]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


def rasterize_annotations_wrong(changes, bounds, H=64, W=64):
    resolution = (bounds[2] - bounds[0]) / W
    resolution_y = (bounds[3] - bounds[1]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for px in range(H):
            for py in range(W):
                wx = bounds[0] + (px + 0.5) * resolution
                wy = bounds[3] - (py + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[px, py] = 1.0
    return mask


def load_s2_rgb(patch_id):
    s2_dir = Path(RAW_DIR) / "s2" / patch_id
    if not s2_dir.exists():
        return None
    tif_files = sorted(s2_dir.glob("*.tif"))
    if not tif_files:
        return None
    mid_idx = len(tif_files) // 2
    tif_path = tif_files[mid_idx]
    try:
        with rasterio.open(tif_path) as src:
            img = src.read([1, 2, 3])
            img = np.transpose(img, (1, 2, 0))
            img = img.astype(np.float32)
            p2 = np.percentile(img, 2)
            p98 = np.percentile(img, 98)
            img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
            return img
    except Exception as e:
        print(f"  读取 {tif_path} 失败: {e}")
        return None


def main():
    print("=" * 60)
    print("  标签叠加可视化 — 真实 S2 图像 + 变化 mask")
    print("=" * 60)

    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    all_changes = []
    for shp_name in ["june.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None: continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                all_changes.append({"geometry": geom, "patch_id": row.get("patch_id", None)})
        except: pass

    patch_changes = {}
    for change in all_changes:
        if change["patch_id"]:
            patch_changes.setdefault(change["patch_id"], []).append(change["geometry"])
        else:
            pt = change["geometry"].centroid
            for pid, bounds in patch_bounds.items():
                if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                    patch_changes.setdefault(pid, []).append(change["geometry"])
                    break

    patch_areas = {pid: sum(g.area for g in geoms) for pid, geoms in patch_changes.items()}
    top_patches = sorted(patch_areas.items(), key=lambda x: x[1], reverse=True)[:6]
    n_show = min(len(top_patches), 6)

    # === 图1: 原始图像 + 正确 mask ===
    fig1, axes1 = plt.subplots(n_show, 2, figsize=(14, 7 * n_show))
    if n_show == 1:
        axes1 = axes1.reshape(1, -1)
    for idx in range(n_show):
        pid, _ = top_patches[idx]
        row = idx
        bounds = patch_bounds[pid]
        changes = patch_changes[pid]

        rgb = load_s2_rgb(pid)
        if rgb is None:
            axes1[row, 0].text(0.5, 0.5, f"{pid}\n无 S2 图像", ha='center', va='center')
            axes1[row, 0].axis('off')
            axes1[row, 1].axis('off')
            continue

        if rgb.shape[0] != 64 or rgb.shape[1] != 64:
            from PIL import Image
            rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb_pil = rgb_pil.resize((64, 64), Image.BILINEAR)
            rgb = np.array(rgb_pil) / 255.0

        mask_correct = rasterize_annotations(changes, bounds)

        # 原始图像
        ax = axes1[row, 0]
        ax.imshow(rgb)
        ax.set_title(f"{pid} — 原始 S2 图像")
        ax.axis('off')

        # 正确 mask 叠加
        ax = axes1[row, 1]
        ax.imshow(rgb)
        red_overlay = np.zeros((*mask_correct.shape, 4))
        red_overlay[..., 0] = 1.0
        red_overlay[..., 3] = mask_correct * 0.6
        ax.imshow(red_overlay)
        ax.set_title(f"{pid} — 正确 mask (ratio={mask_correct.mean():.3f})")
        ax.axis('off')

    plt.tight_layout()
    path1 = "/workspace/outputs/xuannv_backbone_v8_clean/label_overlay_correct.png"
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\n[1/2] 正确版本已保存: {path1}")

    # === 图2: 正确 vs 错误 对比 ===
    fig2, axes2 = plt.subplots(n_show, 2, figsize=(14, 7 * n_show))
    if n_show == 1:
        axes2 = axes2.reshape(1, -1)
    for idx in range(n_show):
        pid, _ = top_patches[idx]
        row = idx
        bounds = patch_bounds[pid]
        changes = patch_changes[pid]

        rgb = load_s2_rgb(pid)
        if rgb is None:
            axes2[row, 0].axis('off')
            axes2[row, 1].axis('off')
            continue

        if rgb.shape[0] != 64 or rgb.shape[1] != 64:
            from PIL import Image
            rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb_pil = rgb_pil.resize((64, 64), Image.BILINEAR)
            rgb = np.array(rgb_pil) / 255.0

        mask_correct = rasterize_annotations(changes, bounds)
        mask_wrong = rasterize_annotations_wrong(changes, bounds)

        # 正确 mask
        ax = axes2[row, 0]
        ax.imshow(rgb)
        red = np.zeros((*mask_correct.shape, 4))
        red[..., 0] = 1.0
        red[..., 3] = mask_correct * 0.6
        ax.imshow(red)
        ax.set_title(f"{pid} — 正确 mask")
        ax.axis('off')

        # 错误 mask
        ax = axes2[row, 1]
        ax.imshow(rgb)
        blue = np.zeros((*mask_wrong.shape, 4))
        blue[..., 2] = 1.0  # B
        blue[..., 3] = mask_wrong * 0.6
        ax.imshow(blue)
        ax.set_title(f"{pid} — 错误 mask (diff={np.abs(mask_correct-mask_wrong).sum():.0f})")
        ax.axis('off')

    plt.tight_layout()
    path2 = "/workspace/outputs/xuannv_backbone_v8_clean/label_overlay_compare.png"
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"[2/2] 对比版本已保存: {path2}")

    print(f"\n  共 {n_show} 个 patch")
    print(f"  左列 = 正确坐标 (row->y, col->x)")
    print(f"  右列 = 错误坐标 (px->x, py->y)")


if __name__ == "__main__":
    main()
