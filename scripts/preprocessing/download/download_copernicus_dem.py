#!/usr/bin/env python3
"""下载Copernicus DEM (30m) for Haidian"""
import os
import requests
from tqdm import tqdm

OUTPUT_DIR = "/workspace/raw/haidian/dem"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copernicus DEM 30m via AWS Open Data
# Tile index: Haidian is around N39E116
# We'll download the relevant tile(s)

# Alternative: use GEE after auth
# For now, create a placeholder and note

print("=" * 50)
print("Copernicus DEM Download")
print("=" * 50)
print("\nHaidian District (N39E116) is covered by DEM tile.")
print("Download method options:")
print("  1. GEE: ee.Image('NASA/NASADEM_HGT/001').clip(aoi)")
print("  2. AWS: s3://copernicus-dem-30m/")
print("  3. Manual: https://spacedata.copernicus.eu/")
print("\nRecommended: Use GEE after authentication.")
print(f"\nOutput path: {OUTPUT_DIR}/haidian_dem_30m.tif")
