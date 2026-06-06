#!/usr/bin/env python3
"""
V11 数据扩展脚本：下载黑龙江省其他城市（齐齐哈尔、大庆等）的遥感数据

使用方法:
    export HTTP_PROXY="http://127.0.0.1:7890"
    export HTTPS_PROXY="http://127.0.0.1:7890"
    python scripts/preprocessing/download_heilongjiang_v11.py --city qiqihar --max-patches 300

依赖:
    earthengine-api, rasterio, numpy

数据格式对齐哈尔滨:
    - Patch: 128x128 像素 @ 10m 分辨率
    - CRS: EPSG:32652 (UTM Zone 52N)
    - S2: 6 波段 [B2, B3, B4, B5, B6, B7] (10m)
    - S1: 2 波段 [VV, VH] (10m)
    - Landsat: 6 波段 [B2, B3, B4, B5, B6, B7] (30m, 重采样到 10m)
    - DEM: 1 波段 (30m, 重采样到 10m)
    - WorldCover: 1 波段 (10m)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# 代理配置（必须在初始化前设置）
# ---------------------------------------------------------------------------
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")

# ---------------------------------------------------------------------------
# GEE 初始化
# ---------------------------------------------------------------------------
CREDENTIALS_PATH = "/workspace/ee-weijiewu0306-5dfc67102021.json"
SERVICE_ACCOUNT = "xuannv-gee@ee-weijiewu0306-5dfc67102021.iam.gserviceaccount.com"

if not Path(CREDENTIALS_PATH).exists():
    raise FileNotFoundError(f"GEE credentials not found: {CREDENTIALS_PATH}")

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, CREDENTIALS_PATH)
ee.Initialize(credentials)
print("[GEE] Initialized successfully.")

# ---------------------------------------------------------------------------
# 城市配置
# ---------------------------------------------------------------------------
def get_utm_crs(lon: float, lat: float) -> str:
    """根据经纬度返回合适的 UTM CRS。"""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


CITIES = {
    "qiqihar": {
        "name": "齐齐哈尔",
        "center": [123.918, 47.354],
        "bbox_deg": [123.5, 47.0, 124.5, 47.7],
        "patch_size_m": 1280,
    },
    "daqing": {
        "name": "大庆",
        "center": [125.0, 46.6],
        "bbox_deg": [124.5, 46.2, 125.6, 47.0],
        "patch_size_m": 1280,
    },
    "mudanjiang": {
        "name": "牡丹江",
        "center": [129.618, 44.582],
        "bbox_deg": [129.0, 44.0, 130.2, 45.0],
        "patch_size_m": 1280,
    },
}

# 时间范围（对齐哈尔滨）
DATE_START = "2023-01-01"
DATE_END = "2025-10-31"

# S2 云筛选阈值
S2_CLOUD_PROB_MAX = 20  # %

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def degree_bbox_to_utm_patches(
    bbox_deg: list[float],
    patch_size_m: float,
) -> list[dict]:
    """
    将经纬度 bbox 转换为 UTM 网格，返回 patch 列表。
    每个 patch 包含 utm_bounds [left, bottom, right, top] 和 center [lon, lat]。
    """
    from pyproj import Transformer

    w, s, e, n = bbox_deg
    center_lon = (w + e) / 2
    center_lat = (s + n) / 2
    crs = get_utm_crs(center_lon, center_lat)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    w, s, e, n = bbox_deg

    # 转换 corners 到 UTM
    (left, bottom), (right, top) = transformer.transform(
        [w, e], [s, n]
    )

    patches = []
    x = left
    patch_id = 0
    while x + patch_size_m <= right + 1e-6:
        y = bottom
        while y + patch_size_m <= top + 1e-6:
            patch_bounds = [x, y, x + patch_size_m, y + patch_size_m]
            # 中心点经纬度
            cx = (x + x + patch_size_m) / 2
            cy = (y + y + patch_size_m) / 2
            lon, lat = transformer.transform(cx, cy, direction="INVERSE")
            patches.append({
                "id": patch_id,
                "utm_bounds": patch_bounds,
                "center_lonlat": [lon, lat],
            })
            patch_id += 1
            y += patch_size_m
        x += patch_size_m

    return patches


def s2_mask_clouds(image: ee.Image) -> ee.Image:
    """使用 SCL 波段掩膜云层。"""
    scl = image.select("SCL")
    # 保留: 4=植被, 5=裸土, 6=水体, 11=雪/冰（可选）
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    return image.updateMask(mask)


def s2_cloud_score(image: ee.Image) -> ee.Image:
    """计算每张图像的云覆盖比例（0-100）。"""
    scl = image.select("SCL")
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    # 云 = 3 (云阴影), 8 (云低概率), 9 (云中概率), 10 (云高概率)
    cloudy = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    # 计算 cloudy 像素比例
    cloud_frac = cloudy.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=60,
        maxPixels=1e9,
    ).get("SCL")
    return image.set("cloud_fraction", cloud_frac)


def export_image_to_tiff(
    image: ee.Image,
    region: ee.Geometry,
    output_path: str,
    bands: list[str],
    crs: str,
    scale: float,
    dtype: str = "float32",
) -> bool:
    """
    从 GEE 下载图像为本地 GeoTIFF。
    使用 getDownloadURL + 本地写入，避免 GCS 依赖。
    """
    try:
        # 选择波段并重命名
        img = image.select(bands)

        # 获取下载 URL
        url = img.getDownloadURL({
            "region": region,
            "crs": crs,
            "scale": scale,
            "format": "GEO_TIFF",
        })

        # 使用代理下载
        import urllib.request
        proxy_handler = urllib.request.ProxyHandler({
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890",
        })
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

        # 下载
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as response:
            data = response.read()

        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)

        # 验证
        with rasterio.open(output_path) as src:
            if src.count != len(bands):
                print(f"  Warning: band count mismatch {src.count} vs {len(bands)}")
                return False
        return True

    except Exception as e:
        print(f"  Error exporting {output_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# 各数据源的下载函数
# ---------------------------------------------------------------------------

def download_s2_for_patch(
    patch: dict,
    city_name: str,
    output_root: str,
    crs: str,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> dict:
    """
    下载 Sentinel-2 数据到一个 patch。
    返回 {"patch_id": int, "n_frames": int, "status": str}
    """
    patch_id = patch["id"]
    bounds = patch["utm_bounds"]  # [left, bottom, right, top]
    # 使用中心点 + buffer 定义区域（GEE 兼容性最佳方式）
    lon, lat = patch["center_lonlat"]
    region = ee.Geometry.Point([lon, lat]).buffer(640).bounds()

    patch_dir = Path(output_root) / city_name / "s2" / f"patch_{patch_id:06d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    # 构建 S2 集合
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_PROB_MAX))
    )

    n_images = collection.size().getInfo()
    if n_images == 0:
        return {"patch_id": patch_id, "n_frames": 0, "status": "no_data"}

    # 获取图像列表
    img_list = collection.toList(n_images)
    downloaded = 0
    skipped = 0

    for i in range(min(n_images, 200)):  # 上限 200 帧/patch
        try:
            img = ee.Image(img_list.get(i))
            img_id = img.get("system:index").getInfo()
            date_str = img_id[:8]  # YYYYMMDD

            # 检查是否已存在
            out_path = patch_dir / f"{date_str}.tif"
            if out_path.exists():
                skipped += 1
                continue

            # 云掩膜 + 选择波段
            img_masked = s2_mask_clouds(img)
            # 缩放因子：SR 波段需要除以 10000
            img_scaled = img_masked.divide(10000.0).clamp(0, 1)

            bands = ["B2", "B3", "B4", "B5", "B6", "B7"]
            success = export_image_to_tiff(
                img_scaled, region, str(out_path), bands,
                crs=crs, scale=10, dtype="float32",
            )
            if success:
                downloaded += 1
            else:
                skipped += 1

            # 速率限制
            if (i + 1) % 10 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"  S2 patch {patch_id} frame {i}: {e}")
            continue

    return {
        "patch_id": patch_id,
        "n_frames": downloaded + skipped,
        "downloaded": downloaded,
        "status": "ok" if downloaded > 0 else "failed",
    }


def download_s1_for_patch(
    patch: dict,
    city_name: str,
    output_root: str,
    crs: str,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> dict:
    """下载 Sentinel-1 数据。"""
    patch_id = patch["id"]
    bounds = patch["utm_bounds"]
    region = ee.Geometry.Point(patch["center_lonlat"]).buffer(640).bounds()

    patch_dir = Path(output_root) / city_name / "s1" / f"patch_{patch_id:06d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    n_images = collection.size().getInfo()
    if n_images == 0:
        return {"patch_id": patch_id, "n_frames": 0, "status": "no_data"}

    img_list = collection.toList(min(n_images, 150))
    downloaded = 0

    for i in range(min(n_images, 150)):
        try:
            img = ee.Image(img_list.get(i))
            img_id = img.get("system:index").getInfo()
            date_str = img_id[:8]

            out_path = patch_dir / f"{date_str}.tif"
            if out_path.exists():
                continue

            # 转换为 dB 尺度
            img_db = img.select(["VV", "VH"]).log10().multiply(10)

            success = export_image_to_tiff(
                img_db, region, str(out_path), ["VV", "VH"],
                crs=crs, scale=10, dtype="float32",
            )
            if success:
                downloaded += 1

            if (i + 1) % 10 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"  S1 patch {patch_id} frame {i}: {e}")
            continue

    return {
        "patch_id": patch_id,
        "n_frames": downloaded,
        "status": "ok" if downloaded > 0 else "failed",
    }


def download_landsat_for_patch(
    patch: dict,
    city_name: str,
    output_root: str,
    crs: str,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> dict:
    """下载 Landsat 8/9 数据。"""
    patch_id = patch["id"]
    bounds = patch["utm_bounds"]
    region = ee.Geometry.Point(patch["center_lonlat"]).buffer(640).bounds()

    patch_dir = Path(output_root) / city_name / "landsat" / f"patch_{patch_id:06d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Landsat 8 + 9
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(region).filterDate(date_start, date_end)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(region).filterDate(date_start, date_end)
    collection = l8.merge(l9)

    n_images = collection.size().getInfo()
    if n_images == 0:
        return {"patch_id": patch_id, "n_frames": 0, "status": "no_data"}

    img_list = collection.toList(min(n_images, 150))
    downloaded = 0

    for i in range(min(n_images, 150)):
        try:
            img = ee.Image(img_list.get(i))
            img_id = img.get("system:index").getInfo()
            date_str = img_id[:8]

            out_path = patch_dir / f"{date_str}.tif"
            if out_path.exists():
                continue

            # 缩放因子 (L2 SR: multiply 0.0000275, add -0.2)
            bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
            img_scaled = img.select(bands).multiply(0.0000275).add(-0.2).clamp(0, 1)

            success = export_image_to_tiff(
                img_scaled, region, str(out_path),
                bands, crs=crs, scale=30, dtype="float32",
            )
            if success:
                downloaded += 1

            if (i + 1) % 10 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"  Landsat patch {patch_id} frame {i}: {e}")
            continue

    return {
        "patch_id": patch_id,
        "n_frames": downloaded,
        "status": "ok" if downloaded > 0 else "failed",
    }


def download_dem_for_patch(
    patch: dict,
    city_name: str,
    output_root: str,
    crs: str,
) -> dict:
    """下载 DEM 数据（静态，每 patch 一帧）。"""
    patch_id = patch["id"]
    bounds = patch["utm_bounds"]
    region = ee.Geometry.Point(patch["center_lonlat"]).buffer(640).bounds()

    patch_dir = Path(output_root) / city_name / "dem" / f"patch_{patch_id:06d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    out_path = patch_dir / "dem.tif"
    if out_path.exists():
        return {"patch_id": patch_id, "status": "skipped"}

    try:
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
        success = export_image_to_tiff(
            dem, region, str(out_path), ["elevation"],
            crs=crs, scale=30, dtype="float32",
        )
        return {"patch_id": patch_id, "status": "ok" if success else "failed"}
    except Exception as e:
        print(f"  DEM patch {patch_id}: {e}")
        return {"patch_id": patch_id, "status": "failed"}


def download_worldcover_for_patch(
    patch: dict,
    city_name: str,
    output_root: str,
    crs: str,
) -> dict:
    """下载 WorldCover 数据（静态，每 patch 一帧）。"""
    patch_id = patch["id"]
    bounds = patch["utm_bounds"]
    region = ee.Geometry.Point(patch["center_lonlat"]).buffer(640).bounds()

    patch_dir = Path(output_root) / city_name / "worldcover" / f"patch_{patch_id:06d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    out_path = patch_dir / "worldcover.tif"
    if out_path.exists():
        return {"patch_id": patch_id, "status": "skipped"}

    try:
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        success = export_image_to_tiff(
            wc, region, str(out_path), ["Map"],
            crs=crs, scale=10, dtype="uint8",
        )
        return {"patch_id": patch_id, "status": "ok" if success else "failed"}
    except Exception as e:
        print(f"  WorldCover patch {patch_id}: {e}")
        return {"patch_id": patch_id, "status": "failed"}


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="下载黑龙江省城市遥感数据")
    parser.add_argument("--city", required=True, choices=list(CITIES.keys()), help="目标城市")
    parser.add_argument("--output-root", default="/workspace/xuannv/data_raw/heilongjiang_new", help="输出根目录")
    parser.add_argument("--max-patches", type=int, default=0, help="最大 patch 数量（0=全部）")
    parser.add_argument("--workers", type=int, default=4, help="并行下载数")
    parser.add_argument("--sources", nargs="+", default=["s2", "s1", "landsat", "dem", "worldcover"],
                        help="要下载的数据源")
    parser.add_argument("--resume", action="store_true", help="断点续传（跳过已存在的文件）")
    args = parser.parse_args()

    city_cfg = CITIES[args.city]
    print(f"\n{'='*60}")
    print(f"[V11 Data Download] {city_cfg['name']}")
    print(f"{'='*60}")
    print(f"  Center: {city_cfg['center']}")
    print(f"  BBox (deg): {city_cfg['bbox_deg']}")
    print(f"  Patch size: {city_cfg['patch_size_m']}m")
    print(f"  CRS: {get_utm_crs(*city_cfg['center'])}")
    print(f"  Output: {args.output_root}/{args.city}")
    print(f"  Sources: {args.sources}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*60}\n")

    # 生成网格
    patches = degree_bbox_to_utm_patches(
        city_cfg["bbox_deg"], city_cfg["patch_size_m"]
    )
    if args.max_patches > 0:
        patches = patches[:args.max_patches]

    print(f"Total patches to download: {len(patches)}")

    # 保存 patch 元数据
    meta_path = Path(args.output_root) / args.city / "patches_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({
            "city": args.city,
            "city_name": city_cfg["name"],
            "n_patches": len(patches),
            "patch_size_m": city_cfg["patch_size_m"],
            "crs": get_utm_crs(*city_cfg["center"]),
            "date_start": DATE_START,
            "date_end": DATE_END,
            "patches": patches,
        }, f, indent=2)
    print(f"Patch metadata saved to {meta_path}")

    # 按源依次下载
    results_summary = {}

    crs = get_utm_crs(*city_cfg["center"])

    for source in args.sources:
        print(f"\n{'-'*50}")
        print(f"[Downloading {source.upper()}]")
        print(f"{'-'*50}")

        if source == "s2":
            download_fn = lambda p: download_s2_for_patch(p, args.city, args.output_root, crs)
        elif source == "s1":
            download_fn = lambda p: download_s1_for_patch(p, args.city, args.output_root, crs)
        elif source == "landsat":
            download_fn = lambda p: download_landsat_for_patch(p, args.city, args.output_root, crs)
        elif source == "dem":
            download_fn = lambda p: download_dem_for_patch(p, args.city, args.output_root, crs)
        elif source == "worldcover":
            download_fn = lambda p: download_worldcover_for_patch(p, args.city, args.output_root, crs)
        else:
            print(f"Unknown source: {source}")
            continue

        # 串行下载（GEE 有速率限制，并行效果有限）
        results = []
        for i, patch in enumerate(patches):
            print(f"  [{i+1}/{len(patches)}] Patch {patch['id']:06d} ...", end=" ", flush=True)
            try:
                res = download_fn(patch)
                results.append(res)
                if res.get("n_frames") is not None:
                    print(f"frames={res['n_frames']}, status={res['status']}")
                else:
                    print(f"status={res['status']}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"patch_id": patch["id"], "status": "error"})

            # 每 10 个 patch 暂停一下
            if (i + 1) % 10 == 0:
                time.sleep(2)

        # 汇总
        ok_count = sum(1 for r in results if r["status"] in ("ok", "skipped"))
        fail_count = len(results) - ok_count
        total_frames = sum(r.get("n_frames", 0) for r in results)

        results_summary[source] = {
            "ok": ok_count,
            "failed": fail_count,
            "total_frames": total_frames,
        }
        print(f"\n  Summary: {ok_count} ok, {fail_count} failed, {total_frames} total frames")

    # 最终汇总
    print(f"\n{'='*60}")
    print("[FINAL SUMMARY]")
    print(f"{'='*60}")
    for src, summ in results_summary.items():
        print(f"  {src}: {summ['ok']} ok / {summ['failed']} failed / {summ['total_frames']} frames")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
