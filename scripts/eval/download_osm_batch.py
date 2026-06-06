#!/usr/bin/env python3
"""批量OSM标签下载 — 一次性下载大区域OSM数据，本地裁剪到各patch.

比逐patch查询快10倍以上（减少Overpass API往返次数）.

用法:
    python download_osm_batch.py --grid /workspace/index/harbin/grid/harbin_grid.geojson \
        --output-dir /workspace/xuannv/data_raw/osm_labels --classes building,road,water
"""
from __future__ import annotations

import sys
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
    p = argparse.ArgumentParser(description="批量OSM标签下载")
    p.add_argument("--grid", required=True, help="Patch格网GeoJSON路径")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--classes", default="building,road,water",
                   help="要下载的OSM地物类别，逗号分隔")
    return p.parse_args()


OSM_QUERIES = {
    "building": {"building": True},
    "road":     {"highway": True},
    "water":    {"natural": "water", "waterway": True},
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


def download_osm_for_region(bounds_32652: tuple, tags: dict) -> gpd.GeoDataFrame | None:
    """一次性下载整个区域的OSM数据."""
    if not HAS_OSMNX:
        return None
    minx, miny, maxx, maxy = bounds_32652
    # 扩大5%边界
    buf_x = (maxx - minx) * 0.05
    buf_y = (maxy - miny) * 0.05
    b = box(minx-buf_x, miny-buf_y, maxx+buf_x, maxy+buf_y)
    gdf_bbox = gpd.GeoDataFrame({"geometry": [b]}, crs="EPSG:32652")
    gdf_bbox = gdf_bbox.to_crs(epsg=4326)
    bounds_4326 = gdf_bbox.total_bounds
    
    ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
    ox.settings.timeout = 300
    
    try:
        gdf = ox.features_from_bbox(
            bbox=(bounds_4326[0], bounds_4326[1], bounds_4326[2], bounds_4326[3]),
            tags=tags
        )
        if gdf is None or gdf.empty:
            return None
        if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        return gdf
    except Exception as e:
        print(f"  OSM下载失败: {e}")
        return None


def rasterize_for_patch(gdf: gpd.GeoDataFrame, bounds: tuple, H: int = 128, W: int = 128) -> np.ndarray:
    minx, miny, maxx, maxy = bounds
    patch_box = box(minx, miny, maxx, maxy)
    # 筛选与patch相交的几何体
    intersects = gdf[gdf.geometry.intersects(patch_box)]
    if intersects.empty:
        return np.zeros((H, W), dtype=np.uint8)
    
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
        (maxx - minx) / W, (miny - maxy) / H
    )
    shapes = [(geom, 1) for geom in intersects.geometry if geom is not None]
    mask = rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8, all_touched=True)
    return mask


def main():
    args = parse_args()
    if not HAS_OSMNX:
        print("[ERROR] osmnx未安装")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",")]
    patches = load_grid(args.grid)
    print(f"加载 {len(patches)} 个patch边界")
    print(f"目标类别: {classes}")
    
    # 计算整个区域的边界
    all_bounds = np.array(list(patches.values()))
    region_bounds = (
        all_bounds[:, 0].min(), all_bounds[:, 1].min(),
        all_bounds[:, 2].max(), all_bounds[:, 3].max(),
    )
    print(f"区域总边界: {region_bounds}")
    
    for cls in classes:
        if cls not in OSM_QUERIES:
            print(f"\n[跳过] 未知类别: {cls}")
            continue
        
        print(f"\n=== 下载类别: {cls} ===")
        tags = OSM_QUERIES[cls]
        gdf = download_osm_for_region(region_bounds, tags)
        
        if gdf is None or gdf.empty:
            print(f"  未获取到任何数据，所有patch将输出全0")
            for pid in sorted(patches.keys()):
                out_tif = output_dir / pid / f"{cls}.tif"
                out_tif.parent.mkdir(parents=True, exist_ok=True)
                if out_tif.exists():
                    continue
                mask = np.zeros((128, 128), dtype=np.uint8)
                minx, miny, maxx, maxy = patches[pid]
                transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
                    (maxx - minx) / 128, (miny - maxy) / 128
                )
                with rasterio.open(out_tif, "w", driver="GTiff", height=128, width=128,
                                   count=1, dtype=mask.dtype, crs="EPSG:32652",
                                   transform=transform, compress="lzw") as dst:
                    dst.write(mask, 1)
            continue
        
        print(f"  获取到 {len(gdf)} 个OSM要素，开始栅格化到 {len(patches)} 个patch...")
        for pid in sorted(patches.keys()):
            out_tif = output_dir / pid / f"{cls}.tif"
            out_tif.parent.mkdir(parents=True, exist_ok=True)
            if out_tif.exists():
                continue
            
            mask = rasterize_for_patch(gdf, patches[pid], 128, 128)
            minx, miny, maxx, maxy = patches[pid]
            transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
                (maxx - minx) / 128, (miny - maxy) / 128
            )
            with rasterio.open(out_tif, "w", driver="GTiff", height=128, width=128,
                               count=1, dtype=mask.dtype, crs="EPSG:32652",
                               transform=transform, compress="lzw") as dst:
                dst.write(mask, 1)
        print(f"  {cls} 完成")
    
    print(f"\n完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
