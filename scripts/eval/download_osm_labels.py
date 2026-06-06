#!/usr/bin/env python3
"""下载OSM标签并栅格化为像素级标签 — 用于下游embedding质量验证.

用法:
    python download_osm_labels.py --grid /workspace/index/harbin/grid/harbin_grid.geojson \
        --output-dir /workspace/xuannv/data_raw/osm_labels --classes building,road,water
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


def parse_args():
    p = argparse.ArgumentParser(description="下载OSM标签")
    p.add_argument("--grid", required=True, help="Patch格网GeoJSON路径")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--classes", default="building,road,water",
                   help="要下载的OSM地物类别，逗号分隔")
    p.add_argument("--max-patches", type=int, default=0,
                   help="最大处理patch数，0表示全部")
    return p.parse_args()


OSM_QUERIES = {
    "building": {"tags": {"building": True}},
    "road":     {"tags": {"highway": True}},
    "water":    {"tags": {"natural": "water", "waterway": True}},
    "forest":   {"tags": {"landuse": "forest", "natural": "wood"}},
    "farmland": {"tags": {"landuse": "farmland", "landuse": "meadow"}},
}


def load_grid(grid_path: str) -> dict:
    with open(grid_path) as f:
        data = json.load(f)
    patches = {}
    for feat in data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        patches[pid] = (min(xs), min(ys), max(xs), max(ys))
    return patches


def download_osm_for_bbox(bbox: tuple, tags: dict) -> gpd.GeoDataFrame | None:
    """从OSM下载指定标签的矢量数据.
    
    bbox: (minx, miny, maxx, maxy) in EPSG:32652 (UTM meters)
    需要转换为EPSG:4326 (lat/lon) 供osmnx使用.
    """
    if not HAS_OSMNX:
        return None
    minx, miny, maxx, maxy = bbox
    try:
        # 扩大一点边界确保边缘完整
        buf = 50  # 50米缓冲区
        # 转换为EPSG:4326
        from shapely.geometry import box as shapely_box
        b = shapely_box(minx-buf, miny-buf, maxx+buf, maxy+buf)
        gdf_bbox = gpd.GeoDataFrame({"geometry": [b]}, crs="EPSG:32652")
        gdf_bbox = gdf_bbox.to_crs(epsg=4326)
        bounds_4326 = gdf_bbox.total_bounds  # (minx, miny, maxx, maxy)
        
        # 使用国内镜像加速
        ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        ox.settings.timeout = 60
        
        gdf = ox.features_from_bbox(
            bbox=(bounds_4326[0], bounds_4326[1], bounds_4326[2], bounds_4326[3]),
            tags=tags
        )
        if gdf is None or gdf.empty:
            return None
        # 统一CRS到patch的EPSG:32652
        if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        return gdf
    except Exception as e:
        print(f"    OSM下载失败: {e}")
        return None


def rasterize_gdf(gdf: gpd.GeoDataFrame, bounds: tuple, H: int = 128, W: int = 128) -> np.ndarray:
    """将GeoDataFrame栅格化为二值mask."""
    minx, miny, maxx, maxy = bounds
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
        (maxx - minx) / W, (miny - maxy) / H
    )
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros((H, W), dtype=np.uint8)
    mask = rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8, all_touched=True)
    return mask


def main():
    args = parse_args()
    if not HAS_OSMNX:
        print("[ERROR] osmnx未安装，请先运行: pip install osmnx")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",")]
    patches = load_grid(args.grid)
    print(f"加载 {len(patches)} 个patch边界")
    print(f"目标类别: {classes}")

    pids = sorted(patches.keys())
    if args.max_patches > 0:
        pids = pids[:args.max_patches]

    for pid in pids:
        bounds = patches[pid]
        print(f"\n处理 {pid}: {bounds}")

        for cls in classes:
            if cls not in OSM_QUERIES:
                print(f"  [跳过] 未知类别: {cls}")
                continue

            query = OSM_QUERIES[cls]
            out_tif = output_dir / pid / f"{cls}.tif"
            out_tif.parent.mkdir(parents=True, exist_ok=True)

            if out_tif.exists():
                print(f"  {cls}: 已存在，跳过")
                continue

            gdf = download_osm_for_bbox(bounds, query["tags"])
            if gdf is None or gdf.empty:
                print(f"  {cls}: 无数据")
                # 写入全0图像
                mask = np.zeros((128, 128), dtype=np.uint8)
            else:
                mask = rasterize_gdf(gdf, bounds, 128, 128)
                print(f"  {cls}: {mask.sum()}px ({mask.sum()/16384*100:.1f}%)")

            # 保存为GeoTIFF
            minx, miny, maxx, maxy = bounds
            transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
                (maxx - minx) / 128, (miny - maxy) / 128
            )
            with rasterio.open(
                out_tif, "w",
                driver="GTiff",
                height=128,
                width=128,
                count=1,
                dtype=mask.dtype,
                crs="EPSG:32652",
                transform=transform,
                compress="lzw",
            ) as dst:
                dst.write(mask, 1)

    print(f"\n完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
