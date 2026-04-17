#!/usr/bin/env python3
"""Download AlphaEarth official embeddings from GEE for Harbin AOI.

Outputs two GeoTIFFs:
    /workspace/outputs/alphaearth_harbin/alphaearth_harbin_2023.tif
    /workspace/outputs/alphaearth_harbin/alphaearth_harbin_2024.tif
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds

import ee

# ── Configuration ──
SERVICE_ACCOUNT = "weijiewu0306@ee-weijiewu0306-5dfc67102021.iam.gserviceaccount.com"
CREDENTIALS_PATH = "/workspace/ee-weijiewu0306-5dfc67102021.json"
OUTPUT_DIR = Path("/workspace/outputs/alphaearth_harbin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Harbin AOI in UTM 52N (from grid)
UTM_MINX = 290398.0
UTM_MINY = 5067541.2
UTM_MAXX = 325585.8
UTM_MAXY = 5100332.6
TARGET_CRS = "EPSG:32652"
TARGET_SCALE = 10  # meters

# Tile size for GEE download (pixels)
# 256x256x64x4 = ~16.7 MB raw, well under the 48 MB GEE limit.
TILE_SIZE = 256

# Bands to download (all 64)
BANDS = [f"A{i:02d}" for i in range(64)]

# Years to download
YEARS = [2023, 2024]


def _init_ee() -> None:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, CREDENTIALS_PATH)
    ee.Initialize(credentials)


def _download_year(year: int) -> Path:
    """Download AlphaEarth embedding for a single year as GeoTIFF."""
    print(f"\n{'='*60}", flush=True)
    print(f"Downloading AlphaEarth embedding for {year}...", flush=True)
    print(f"{'='*60}", flush=True)

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
    print(f"AOI in {TARGET_CRS}: {UTM_MINX:.1f}, {UTM_MINY:.1f}, {UTM_MAXX:.1f}, {UTM_MAXY:.1f}", flush=True)
    print(f"Pixel dimensions: {width_px} x {height_px}", flush=True)

    n_tiles_x = int(np.ceil(width_px / TILE_SIZE))
    n_tiles_y = int(np.ceil(height_px / TILE_SIZE))
    total_tiles = n_tiles_x * n_tiles_y
    print(f"Downloading in {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles (size {TILE_SIZE}x{TILE_SIZE})", flush=True)

    tile_files = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_idx = ty * n_tiles_x + tx + 1
            x_off = tx * TILE_SIZE
            y_off = ty * TILE_SIZE
            tw = min(TILE_SIZE, width_px - x_off)
            th = min(TILE_SIZE, height_px - y_off)

            t_minx = UTM_MINX + x_off * TARGET_SCALE
            t_maxy = UTM_MAXY - y_off * TARGET_SCALE

            # crs_transform for this tile: [scaleX, shearX, translateX, shearY, scaleY, translateY]
            transform = [TARGET_SCALE, 0, t_minx, 0, -TARGET_SCALE, t_maxy]

            url = image.getDownloadURL(
                {
                    "crs": TARGET_CRS,
                    "crs_transform": transform,
                    "dimensions": [tw, th],
                    "format": "GEO_TIFF",
                }
            )

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
            tile_files.append(tile_path)
            print(f"  Tile {tile_idx}/{total_tiles} saved ({tw}x{th})", flush=True)
            time.sleep(0.3)  # be polite to GEE

    # Merge tiles
    print(f"Merging {len(tile_files)} tiles...", flush=True)
    datasets = [rasterio.open(p) for p in tile_files]
    mosaic, out_transform = merge(datasets)
    out_profile = datasets[0].profile.copy()
    out_profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": mosaic.shape[0],
            "crs": TARGET_CRS,
            "transform": out_transform,
            "dtype": mosaic.dtype,
            "compress": "lzw",
        }
    )
    out_path = OUTPUT_DIR / f"alphaearth_harbin_{year}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(mosaic)
    for ds in datasets:
        ds.close()

    # Clean up tile files
    for p in tile_files:
        p.unlink()

    print(f"Saved merged GeoTIFF: {out_path}", flush=True)
    return out_path


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    _init_ee()
    print("GEE initialized. Starting downloads...", flush=True)
    for year in YEARS:
        _download_year(year)
    print("\n✅ All downloads complete.", flush=True)


if __name__ == "__main__":
    main()
