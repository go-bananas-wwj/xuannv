#!/usr/bin/env python3
"""Resume AlphaEarth 2024 download for Harbin."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.merge import merge

import ee

SERVICE_ACCOUNT = "weijiewu0306@ee-weijiewu0306-5dfc67102021.iam.gserviceaccount.com"
CREDENTIALS_PATH = "/workspace/ee-weijiewu0306-5dfc67102021.json"
OUTPUT_DIR = Path("/workspace/outputs/alphaearth_harbin")

UTM_MINX = 290398.0
UTM_MINY = 5067541.2
UTM_MAXX = 325585.8
UTM_MAXY = 5100332.6
TARGET_CRS = "EPSG:32652"
TARGET_SCALE = 10
TILE_SIZE = 256
BANDS = [f"A{i:02d}" for i in range(64)]


def _init_ee() -> None:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, CREDENTIALS_PATH)
    ee.Initialize(credentials)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    _init_ee()

    year = 2024
    print(f"Resuming AlphaEarth embedding for {year}...", flush=True)

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    aoi = ee.Geometry.Rectangle([126.3057, 45.7392, 126.7465, 46.0248])
    collection = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
    )
    image = collection.mosaic().select(BANDS)

    width_px = int(np.ceil((UTM_MAXX - UTM_MINX) / TARGET_SCALE))
    height_px = int(np.ceil((UTM_MAXY - UTM_MINY) / TARGET_SCALE))
    n_tiles_x = int(np.ceil(width_px / TILE_SIZE))
    n_tiles_y = int(np.ceil(height_px / TILE_SIZE))
    total_tiles = n_tiles_x * n_tiles_y

    # Find existing tiles
    existing = set()
    for p in OUTPUT_DIR.glob(f"{year}_tile_*.tif"):
        parts = p.stem.split("_")
        tx = int(parts[2])
        ty = int(parts[3])
        existing.add((tx, ty))

    print(f"Found {len(existing)} existing tiles. Total needed: {total_tiles}", flush=True)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_idx = ty * n_tiles_x + tx + 1
            if (tx, ty) in existing:
                print(f"  Tile {tile_idx}/{total_tiles} already exists, skipping", flush=True)
                continue

            x_off = tx * TILE_SIZE
            y_off = ty * TILE_SIZE
            tw = min(TILE_SIZE, width_px - x_off)
            th = min(TILE_SIZE, height_px - y_off)

            t_minx = UTM_MINX + x_off * TARGET_SCALE
            t_maxy = UTM_MAXY - y_off * TARGET_SCALE
            transform = [TARGET_SCALE, 0, t_minx, 0, -TARGET_SCALE, t_maxy]

            url = image.getDownloadURL({
                "crs": TARGET_CRS,
                "crs_transform": transform,
                "dimensions": [tw, th],
                "format": "GEO_TIFF",
            })

            for attempt in range(3):
                try:
                    data = urlopen(url, timeout=120).read()
                    break
                except Exception as e:
                    print(f"  Tile {tile_idx}/{total_tiles} attempt {attempt+1} failed: {e}", flush=True)
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to download tile {tile_idx}")

            tile_path = OUTPUT_DIR / f"{year}_tile_{tx:03d}_{ty:03d}.tif"
            with open(tile_path, "wb") as f:
                f.write(data)
            print(f"  Tile {tile_idx}/{total_tiles} saved ({tw}x{th})", flush=True)
            time.sleep(0.3)

    # Merge all tiles
    tile_files = sorted(OUTPUT_DIR.glob(f"{year}_tile_*.tif"))
    print(f"Merging {len(tile_files)} tiles...", flush=True)
    datasets = [rasterio.open(p) for p in tile_files]
    mosaic, out_transform = merge(datasets)
    out_profile = datasets[0].profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
        "crs": TARGET_CRS,
        "transform": out_transform,
        "dtype": mosaic.dtype,
        "compress": "lzw",
    })
    out_path = OUTPUT_DIR / f"alphaearth_harbin_{year}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(mosaic)
    for ds in datasets:
        ds.close()

    for p in tile_files:
        p.unlink()

    print(f"Saved merged GeoTIFF: {out_path}", flush=True)
    print("✅ 2024 download complete.", flush=True)


if __name__ == "__main__":
    main()
