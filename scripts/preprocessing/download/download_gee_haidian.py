#!/usr/bin/env python3
"""
用Google Earth Engine下载海淀区数据
=====================================
使用说明:
1. 先运行: earthengine authenticate
2. 填入你的GEE项目ID
3. 运行: python download_gee_haidian.py
"""

import ee
import os

# ==================== 配置区 ====================
# GEE项目ID（可选，如果你有的话）
GEE_PROJECT = None  # 或填入 "your-project-id"
# =================================================

ee.Initialize(project=GEE_PROJECT)

# 海淀区AOI
aoi = ee.Geometry.Rectangle([116.05, 39.88, 116.38, 40.15])

print("=" * 50)
print("GEE Data Download for Haidian")
print("=" * 50)

# Sentinel-2 (云量<70%)
print("\n[1/4] Sentinel-2 L2A...")
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2025-01-01", "2026-05-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
)
print(f"  Found {s2.size().getInfo()} S2 images")

# Sentinel-1
print("\n[2/4] Sentinel-1 GRD...")
s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate("2025-01-01", "2026-05-31")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
)
print(f"  Found {s1.size().getInfo()} S1 images")

# Landsat-8/9
print("\n[3/4] Landsat-8/9 L2...")
landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(aoi)
    .filterDate("2025-01-01", "2026-05-31")
    .filter(ee.Filter.lt("CLOUD_COVER", 70))
)
print(f"  Found {landsat.size().getInfo()} Landsat images")

# DEM
print("\n[4/4] Copernicus DEM...")
dem = ee.Image("NASA/NASADEM_HGT/001").clip(aoi)
print(f"  DEM ready")

print("\n" + "=" * 50)
print("To export images, use ee.Export.image.toDrive() or toAsset()")
print("=" * 50)
