#!/usr/bin/env python3
"""
Landsat数据下载脚本模板
======================
使用说明:
1. 填入你的USGS账号密码
2. 运行: python download_landsat_template.py
"""

from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import os

# ==================== 配置区 ====================
# 1. USGS 账号（必填）
# 注册地址: https://earthexplorer.usgs.gov/
USGS_USER = "YOUR_USERNAME"
USGS_PASS = "YOUR_PASSWORD"

# 2. AOI (lat, lon 中心点 + 缓冲区)
LAT, LON = 39.96, 116.20  # 海淀区中心
BUFFER = 0.15  # 度

# 3. 时间范围
START_DATE = "2025-01-01"
END_DATE = "2026-05-31"

# 4. 输出目录
OUTPUT_DIR = "/workspace/raw/haidian/landsat"
# =================================================

def download_landsat():
    api = API(USGS_USER, USGS_PASS)
    
    # 搜索
    scenes = api.search(
        dataset="landsat_ot_c2_l2",  # Landsat-8/9 Collection 2 Level-2
        latitude=LAT,
        longitude=LON,
        start_date=START_DATE,
        end_date=END_DATE,
        max_cloud_cover=70,
        max_results=500
    )
    print(f"Found {len(scenes)} Landsat scenes")
    
    # 下载
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ee = EarthExplorer(USGS_USER, USGS_PASS)
    for scene in scenes:
        entity_id = scene["entity_id"]
        print(f"Downloading {entity_id}...")
        ee.download(entity_id, OUTPUT_DIR)
    
    ee.logout()
    api.logout()
    print("Landsat download complete!")

if __name__ == "__main__":
    if USGS_USER == "YOUR_USERNAME":
        print("ERROR: 请先填写USGS账号密码!")
        print("注册地址: https://earthexplorer.usgs.gov/")
        exit(1)
    
    download_landsat()
