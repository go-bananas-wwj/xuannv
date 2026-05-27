#!/usr/bin/env python3
"""
从变化检测shapefile生成每个patch的像素级变化mask。

处理4个时间段的变化标注：
- june.shp    → 4-6月变化（对应before=4月, after=6月）
- aug.shp     → 6-8月变化（对应before=6月, after=8月）
- September.shp → 8-9月变化（对应before=8月, after=9月）
- October.shp → 9-10月变化（对应before=9月, after=10月）

输出: /workspace/xuannv/data/change_masks/{period}/{patch_id}.npy
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer
from rasterio import features
from shapely.ops import transform
import shapely.affinity


def parse_shapefile_directly(shp_path: str) -> gpd.GeoDataFrame:
    """直接解析.shp文件，绕过损坏的.shx"""
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_path)
    return gdf


def load_patch_bounds(grid_path: str) -> dict[str, dict]:
    """从grid geojson加载每个patch的边界信息"""
    gdf = gpd.read_file(grid_path)
    bounds = {}
    for _, row in gdf.iterrows():
        pid = row["patch_id"]
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        bounds[pid] = {
            "geometry": geom,
            "bounds": (minx, miny, maxx, maxy),
            "crs": str(gdf.crs),
        }
    return bounds


def generate_masks_for_period(
    shapefile_path: str,
    patch_bounds: dict,
    period_name: str,
    output_dir: Path,
    target_crs: str = "EPSG:32652",
    image_size: int = 64,  # 与embedding_map尺寸对齐
) -> dict:
    """为某个时间段生成所有patch的变化mask"""
    print(f"\n=== 处理 {period_name} ===")
    print(f"  Shapefile: {shapefile_path}")

    gdf = parse_shapefile_directly(shapefile_path)
    print(f"  总变化图斑数: {len(gdf)}")

    # 确定shapefile的CRS
    src_crs = gdf.crs
    if src_crs is None:
        print("  Warning: CRS缺失，假设为EPSG:4490 (CGCS2000)")
        gdf = gdf.set_crs("EPSG:4490")
        src_crs = gdf.crs
    print(f"  Source CRS: {src_crs}")

    # 转换到目标CRS
    if str(src_crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
        print(f"  已转换到 {target_crs}")

    # 创建输出目录
    period_dir = output_dir / period_name
    period_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total_patches": 0, "patches_with_change": 0, "total_change_pixels": 0}

    for pid, pb in patch_bounds.items():
        stats["total_patches"] += 1
        patch_geom = pb["geometry"]
        minx, miny, maxx, maxy = patch_geom.bounds

        # 计算与该patch相交的变化polygon
        intersecting = gdf[gdf.geometry.intersects(patch_geom)]

        if len(intersecting) == 0:
            # 无变化，保存全零mask
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            np.save(period_dir / f"{pid}.npy", mask)
            continue

        # 有变化，计算交集并栅格化
        stats["patches_with_change"] += 1

        # 计算affine transform (像素坐标: 左上为(0,0), 右下为(128,128))
        # 注意: rasterio的transform是左上原点，y向下递增
        # 但我们的patch坐标是UTM，y向上递增
        # image_size=128, 分辨率=(maxx-minx)/128
        res_x = (maxx - minx) / image_size
        res_y = (maxy - miny) / image_size

        # 创建rasterio风格的transform
        # 注意: 图像坐标系中y轴向下，所以需要翻转
        transform = (
            res_x, 0.0, minx,
            0.0, -res_y, maxy,
            0.0, 0.0, 1.0,
        )

        # 栅格化
        shapes = [(geom, 1) for geom in intersecting.geometry]
        mask = features.rasterize(
            shapes=shapes,
            out_shape=(image_size, image_size),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        stats["total_change_pixels"] += mask.sum()
        np.save(period_dir / f"{pid}.npy", mask)

    print(f"  处理完成: {stats['total_patches']} patches, "
          f"{stats['patches_with_change']} 有变化, "
          f"{stats['total_change_pixels']} 变化像素")
    return stats


def main():
    output_dir = Path("/workspace/xuannv/data/change_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_path = "/workspace/index/harbin/grid/harbin_grid.geojson"
    patch_bounds = load_patch_bounds(grid_path)
    print(f"加载了 {len(patch_bounds)} 个patch边界 (CRS: EPSG:32652)")

    periods = [
        ("june", "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件/june.shp", "4-6月"),
        ("aug", "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件/aug.shp", "6-8月"),
        ("september", "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件/September.shp", "8-9月"),
        ("october", "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件/October.shp", "9-10月"),
    ]

    all_stats = {}
    for period_key, shp_path, period_label in periods:
        stats = generate_masks_for_period(
            shp_path,
            patch_bounds,
            period_key,
            output_dir,
        )
        all_stats[period_key] = stats

    # 保存统计信息
    with open(output_dir / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    print("\n=== 全部完成 ===")
    print(f"输出目录: {output_dir}")
    for pk, st in all_stats.items():
        print(f"  {pk}: {st['patches_with_change']}/{st['total_patches']} patches 有变化")


if __name__ == "__main__":
    main()
