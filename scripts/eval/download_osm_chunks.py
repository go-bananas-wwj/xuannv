#!/usr/bin/env python3
"""分块OSM标签下载 — 将大区域分成小块，分别下载后合并.

用法:
    python download_osm_chunks.py --grid /workspace/index/harbin/grid/harbin_grid.geojson \
        --output-dir /workspace/raw/osm_labels_harbin --classes building,road,water --chunks 4
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
    p = argparse.ArgumentParser(description="分块OSM标签下载")
    p.add_argument("--grid", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--classes", default="building,road,water")
    p.add_argument("--chunks", type=int, default=4, help="分块数量")
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


def split_patches_into_chunks(patches: dict, n_chunks: int):
    """按x坐标排序后分块."""
    sorted_pids = sorted(patches.keys(), key=lambda p: patches[p][0])
    chunk_size = len(sorted_pids) // n_chunks + 1
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(sorted_pids))
        chunk_pids = sorted_pids[start:end]
        if not chunk_pids:
            continue
        bounds = np.array([patches[pid] for pid in chunk_pids])
        chunk_bounds = (
            bounds[:, 0].min(), bounds[:, 1].min(),
            bounds[:, 2].max(), bounds[:, 3].max(),
        )
        chunks.append((chunk_pids, chunk_bounds))
    return chunks


def download_osm_for_region(bounds_32652: tuple, tags: dict):
    if not HAS_OSMNX:
        return None
    minx, miny, maxx, maxy = bounds_32652
    buf = 100
    b = box(minx-buf, miny-buf, maxx+buf, maxy+buf)
    gdf_bbox = gpd.GeoDataFrame({"geometry": [b]}, crs="EPSG:32652")
    gdf_bbox = gdf_bbox.to_crs(epsg=4326)
    bounds_4326 = gdf_bbox.total_bounds
    
    ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
    ox.settings.timeout = 120
    
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
        print(f"    OSM下载失败: {e}")
        return None


def rasterize_for_patch(gdf, bounds, H=128, W=128):
    minx, miny, maxx, maxy = bounds
    patch_box = box(minx, miny, maxx, maxy)
    intersects = gdf[gdf.geometry.intersects(patch_box)]
    if intersects.empty:
        return np.zeros((H, W), dtype=np.uint8)
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
        (maxx - minx) / W, (miny - maxy) / H
    )
    shapes = [(geom, 1) for geom in intersects.geometry if geom is not None]
    return rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8, all_touched=True)


def save_tif(mask, bounds, out_tif):
    minx, miny, maxx, maxy = bounds
    transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(
        (maxx - minx) / 128, (miny - maxy) / 128
    )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", height=128, width=128,
                       count=1, dtype=mask.dtype, crs="EPSG:32652",
                       transform=transform, compress="lzw") as dst:
        dst.write(mask, 1)


def main():
    args = parse_args()
    if not HAS_OSMNX:
        print("[ERROR] osmnx未安装")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",")]
    patches = load_grid(args.grid)
    chunks = split_patches_into_chunks(patches, args.chunks)
    print(f"加载 {len(patches)} 个patch，分成 {len(chunks)} 块下载")

    for cls in classes:
        if cls not in OSM_QUERIES:
            continue
        tags = OSM_QUERIES[cls]
        print(f"\n=== 类别: {cls} ===")
        
        for chunk_idx, (chunk_pids, chunk_bounds) in enumerate(chunks):
            print(f"  块 {chunk_idx+1}/{len(chunks)}: {len(chunk_pids)} 个patches")
            gdf = download_osm_for_region(chunk_bounds, tags)
            
            for pid in chunk_pids:
                out_tif = output_dir / pid / f"{cls}.tif"
                if out_tif.exists():
                    continue
                if gdf is not None and not gdf.empty:
                    mask = rasterize_for_patch(gdf, patches[pid], 128, 128)
                else:
                    mask = np.zeros((128, 128), dtype=np.uint8)
                save_tif(mask, patches[pid], out_tif)
    
    print(f"\n完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
