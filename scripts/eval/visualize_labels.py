#!/usr/bin/env python3
"""可视化变化检测标签 mask，检查坐标是否正确."""
import sys, json
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box, Point

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"


def rasterize_annotations(changes, bounds, H=64, W=64):
    """光栅化标注."""
    resolution = (bounds[2] - bounds[0]) / W  # x方向分辨率
    resolution_y = (bounds[3] - bounds[1]) / H  # y方向分辨率
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):  # row -> y
            for col in range(W):  # col -> x
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


def rasterize_annotations_wrong(changes, bounds, H=64, W=64):
    """错误版本（横纵坐标搞反）."""
    resolution = (bounds[2] - bounds[0]) / W
    resolution_y = (bounds[3] - bounds[1]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for px in range(H):  # 错误：px当作x
            for py in range(W):  # 错误：py当作y
                wx = bounds[0] + (px + 0.5) * resolution  # 错误
                wy = bounds[3] - (py + 0.5) * resolution_y  # 错误
                if geom.contains(Point(wx, wy)):
                    mask[px, py] = 1.0
    return mask


def main():
    print("=" * 60)
    print("  标签可视化检查")
    print("=" * 60)

    # 加载 Grid
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    # 加载标注
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

    # 按 patch 聚合
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

    # 选择变化面积最大的几个 patch 可视化
    patch_areas = {pid: sum(g.area for g in geoms) for pid, geoms in patch_changes.items()}
    top_patches = sorted(patch_areas.items(), key=lambda x: x[1], reverse=True)[:6]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (pid, _) in enumerate(top_patches):
        bounds = patch_bounds[pid]
        changes = patch_changes[pid]

        mask_correct = rasterize_annotations(changes, bounds)
        mask_wrong = rasterize_annotations_wrong(changes, bounds)

        # 检查是否有差异
        diff = np.abs(mask_correct - mask_wrong).sum()

        # 正确版本
        ax = axes[idx * 2]
        ax.imshow(mask_correct, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f"{pid}\n正确 (mask_ratio={mask_correct.mean():.3f})")
        ax.axis('off')

        # 错误版本
        ax = axes[idx * 2 + 1]
        ax.imshow(mask_wrong, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f"{pid}\n错误 (mask_ratio={mask_wrong.mean():.3f}, diff={diff:.0f})")
        ax.axis('off')

    plt.tight_layout()
    output_path = "/workspace/outputs/xuannv_backbone_v8_clean/label_visualization.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n可视化已保存到: {output_path}")

    # 统计所有 patch 的正确 vs 错误差异
    total_diff = 0
    total_pixels_correct = 0
    total_pixels_wrong = 0
    for pid, changes in patch_changes.items():
        bounds = patch_bounds[pid]
        mc = rasterize_annotations(changes, bounds)
        mw = rasterize_annotations_wrong(changes, bounds)
        total_diff += np.abs(mc - mw).sum()
        total_pixels_correct += mc.sum()
        total_pixels_wrong += mw.sum()

    print(f"\n统计:")
    print(f"  总 patch 数: {len(patch_changes)}")
    print(f"  正确版本总变化像素: {total_pixels_correct:.0f}")
    print(f"  错误版本总变化像素: {total_pixels_wrong:.0f}")
    print(f"  两版本差异像素数: {total_diff:.0f}")
    print(f"  差异比例: {total_diff / max(total_pixels_correct, 1) * 100:.1f}%")

    if total_diff > 0:
        print(f"\n  ⚠️  发现坐标映射错误！差异像素数: {total_diff:.0f}")
        print(f"     正确版本: px(row) -> y, py(col) -> x")
        print(f"     错误版本: px -> x, py -> y")
    else:
        print(f"\n  ✅ 坐标映射正确")


if __name__ == "__main__":
    main()
