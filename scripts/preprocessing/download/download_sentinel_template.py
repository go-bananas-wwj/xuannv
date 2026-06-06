#!/usr/bin/env python3
"""
Sentinel数据下载脚本模板
=========================
使用说明:
1. 填入你的Copernicus账号密码
2. 设置下载区域和时间范围
3. 运行: python download_sentinel_template.py
"""

import os
from datetime import date
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

# ==================== 配置区 ====================
# 1. Copernicus 账号（必填）
# 注册地址: https://scihub.copernicus.eu/
COPERNICUS_USER = "YOUR_USERNAME"
COPERNICUS_PASS = "YOUR_PASSWORD"

# 2. AOI 文件路径
AOI_PATH = "/workspace/xuannv/data_raw/haidian/aoi/haidian_aoi.geojson"

# 3. 输出目录
OUTPUT_DIR = "/workspace/xuannv/data_raw/haidian/s1"  # 或 s2

# 4. 时间范围
START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 5, 31)

# 5. 产品类型: 'GRD'(S1) 或 'S2MSI2A'(S2)
PRODUCT_TYPE = "GRD"  # S1: GRD, S2: S2MSI2A
PLATFORM = "Sentinel-1"  # 或 "Sentinel-2"
# =================================================

def download_s1():
    """下载Sentinel-1 GRD"""
    api = SentinelAPI(COPERNICUS_USER, COPERNICUS_PASS, 
                      "https://scihub.copernicus.eu/dhus")
    footprint = geojson_to_wkt(read_geojson(AOI_PATH))
    
    products = api.query(
        footprint,
        date=(START_DATE, END_DATE),
        platformname="Sentinel-1",
        producttype="GRD",
        sensoroperationalmode="IW",
        polarisationmode="VV VH"
    )
    print(f"Found {len(products)} S1 products")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api.download_all(products, directory_path=OUTPUT_DIR)
    print("S1 download complete!")

def download_s2():
    """下载Sentinel-2 L2A"""
    api = SentinelAPI(COPERNICUS_USER, COPERNICUS_PASS,
                      "https://scihub.copernicus.eu/dhus")
    footprint = geojson_to_wkt(read_geojson(AOI_PATH))
    
    products = api.query(
        footprint,
        date=(START_DATE, END_DATE),
        platformname="Sentinel-2",
        producttype="S2MSI2A",
        cloudcoverpercentage=(0, 70)  # 云量 < 70%
    )
    print(f"Found {len(products)} S2 products")
    out_dir = "/workspace/xuannv/data_raw/haidian/s2"
    os.makedirs(out_dir, exist_ok=True)
    api.download_all(products, directory_path=out_dir)
    print("S2 download complete!")

if __name__ == "__main__":
    if COPERNICUS_USER == "YOUR_USERNAME":
        print("ERROR: 请先填写Copernicus账号密码!")
        print("注册地址: https://scihub.copernicus.eu/")
        exit(1)
    
    # 默认下载S1，可改为 download_s2()
    download_s1()
