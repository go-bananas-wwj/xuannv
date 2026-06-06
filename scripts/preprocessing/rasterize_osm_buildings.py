#!/usr/bin/env python3
"""将 OSM 建筑矢量数据栅格化为 per-patch 二值标签 TIF.

与 WorldCover 标签对齐: EPSG:32652, 128x128, 10m 分辨率.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RAW_DIR = Path("/workspace/xuannv/data_raw/harbin_scenes_scenes")
OSM_PATH = RAW_DIR / "osm_buildings_raw.geojson"
OUT_DIR = RAW_DIR / "osm_buildings"
WORLDCOVER_DIR = RAW_DIR / "worldcover"

# 哈尔滨 grid 的 patch IDs
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")


def load_patch_ids() -> list[str]:
    with open(GRID_PATH) as f:
        data = json.load(f)
    return sorted([feat["properties"]["patch_id"] for feat in data["features"]])


def rasterize_patch(buildings_utm: gpd.GeoDataFrame, pid: str) -> np.ndarray | None:
    """栅格化单个 patch 的建筑数据."""
    wc_path = WORLDCOVER_DIR / pid / "static.tif"
    if not wc_path.exists():
        return None

    with rasterio.open(wc_path) as src:
        shape = src.shape
        transform = src.transform
        bounds = src.bounds

    # 筛选与 patch bounds 相交的建筑
    minx, miny, maxx, maxy = bounds
    intersect = buildings_utm[
        (buildings_utm["bounds_minx"] <= maxx)
        & (buildings_utm["bounds_maxx"] >= minx)
        & (buildings_utm["bounds_miny"] <= maxy)
        & (buildings_utm["bounds_maxy"] >= miny)
    ]

    if len(intersect) == 0:
        return np.zeros(shape, dtype=np.int32)

    mask = features.rasterize(
        [(geom, 1) for geom in intersect.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    return mask


def main():
    print("[OSM] Loading building data...")
    buildings = gpd.read_file(OSM_PATH)
    print(f"[OSM] Loaded {len(buildings)} buildings")

    print("[OSM] Reprojecting to EPSG:32652...")
    buildings_utm = buildings.to_crs(epsg=32652)
    bounds_df = buildings_utm.geometry.bounds
    buildings_utm["bounds_minx"] = bounds_df.minx.values
    buildings_utm["bounds_maxx"] = bounds_df.maxx.values
    buildings_utm["bounds_miny"] = bounds_df.miny.values
    buildings_utm["bounds_maxy"] = bounds_df.maxy.values

    patch_ids = load_patch_ids()
    print(f"[OSM] Processing {len(patch_ids)} patches...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"with_buildings": 0, "empty": 0, "no_worldcover": 0, "total_pixels": 0, "building_pixels": 0}

    for pid in tqdm(patch_ids, desc="Rasterizing"):
        wc_path = WORLDCOVER_DIR / pid / "static.tif"
        if not wc_path.exists():
            stats["no_worldcover"] += 1
            continue

        mask = rasterize_patch(buildings_utm, pid)
        if mask is None:
            stats["no_worldcover"] += 1
            continue

        patch_dir = OUT_DIR / pid
        patch_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(wc_path) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.int32, nodata=-1, count=1)

        out_path = patch_dir / "static.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask.astype(np.int32), 1)

        building_count = int(mask.sum())
        stats["total_pixels"] += mask.size
        stats["building_pixels"] += building_count
        if building_count > 0:
            stats["with_buildings"] += 1
        else:
            stats["empty"] += 1

    print(f"\n[OSM] Done.")
    print(f"  Patches with buildings: {stats['with_buildings']}/{len(patch_ids)}")
    print(f"  Empty patches: {stats['empty']}")
    print(f"  No WorldCover ref: {stats['no_worldcover']}")
    if stats["total_pixels"] > 0:
        print(f"  Building pixel ratio: {stats['building_pixels']/stats['total_pixels']*100:.2f}%")

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
