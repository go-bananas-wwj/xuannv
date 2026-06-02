"""
使用 Google Earth Engine 下载黑龙江省其他城市数据。

不依赖 terragon，直接使用 GEE Python API。

目标城市:
- 齐齐哈尔 (Qiqihar): ~47.0N, ~124.0E
- 大庆 (Daqing): ~46.6N, ~125.0E
- 牡丹江 (Mudanjiang): ~44.6N, ~129.6E

每个城市: 150-200 patches, 128x128 pixels @ 10m = 1.28km x 1.28km
时间范围: 2023-01-01 至 2025-12-31
数据源: Sentinel-2 L1C, Sentinel-1 GRD, Landsat-8/9

使用前必须完成 GEE 认证:
  earthengine authenticate
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, "/workspace/xuannv")

import ee
import numpy as np


def initialize_gee():
    """初始化 GEE，如果未认证则提示用户。"""
    try:
        ee.Initialize()
        print("[GEE] 初始化成功")
        return True
    except Exception as e:
        print(f"[GEE] 初始化失败: {e}")
        print("请运行: earthengine authenticate")
        return False


# ────────────────────────────────────────────
# 城市定义
# ────────────────────────────────────────────

CITIES = {
    "qiqihar": {
        "name": "齐齐哈尔",
        "center": [47.0, 124.0],
        "description": "西部干旱平原，农业区",
    },
    "daqing": {
        "name": "大庆",
        "center": [46.6, 125.0],
        "description": "石油城市，盐碱地",
    },
    "mudanjiang": {
        "name": "牡丹江",
        "center": [44.6, 129.6],
        "description": "东南部山地森林",
    },
}


def create_patch_grid(
    center_lat: float,
    center_lon: float,
    n_patches_x: int = 10,
    n_patches_y: int = 10,
    patch_size_km: float = 1.28,
    spacing_km: float = 1.5,
) -> list[dict]:
    """以中心点创建 patch 网格。

    Args:
        center_lat, center_lon: 中心坐标
        n_patches_x, n_patches_y: 网格维度
        patch_size_km: 每个 patch 边长（km）
        spacing_km: patch 间距（km）

    Returns:
        List of patch dicts with {id, lat, lon, bounds}
    """
    patches = []
    # 粗略换算：1度纬度 ≈ 111km，1度经度 ≈ 111*cos(lat) km
    lat_spacing = spacing_km / 111.0
    lon_spacing = spacing_km / (111.0 * np.cos(np.radians(center_lat)))
    size_deg = patch_size_km / 111.0

    start_lat = center_lat - (n_patches_y - 1) * lat_spacing / 2
    start_lon = center_lon - (n_patches_x - 1) * lon_spacing / 2

    idx = 0
    for iy in range(n_patches_y):
        for ix in range(n_patches_x):
            lat = start_lat + iy * lat_spacing
            lon = start_lon + ix * lon_spacing
            patches.append({
                "id": f"patch_{idx:06d}",
                "lat": lat,
                "lon": lon,
                "bounds": {
                    "north": lat + size_deg / 2,
                    "south": lat - size_deg / 2,
                    "east": lon + size_deg / 2,
                    "west": lon - size_deg / 2,
                },
            })
            idx += 1

    return patches


def download_sentinel2_patch(
    patch: dict,
    start_date: str,
    end_date: str,
    out_dir: Path,
    max_per_month: int = 4,
) -> dict:
    """下载单个 patch 的 Sentinel-2 数据。

    策略:
    1. 查询该 patch 范围内的 S2 L1C 数据
    2. 按月分组，每月保留 cloud cover 最低的 N 帧
    3. 导出为 GeoTIFF

    Returns:
        {"patch_id": str, "n_frames": int, "status": str}
    """
    patch_id = patch["id"]
    bounds = patch["bounds"]
    patch_out = out_dir / patch_id
    patch_out.mkdir(parents=True, exist_ok=True)

    # 创建 AOI
    aoi = ee.Geometry.Rectangle([
        bounds["west"], bounds["south"],
        bounds["east"], bounds["north"],
    ])

    # 查询 S2 数据
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )

    # 获取图像列表
    image_list = collection.toList(collection.size())
    n_images = collection.size().getInfo()

    if n_images == 0:
        return {"patch_id": patch_id, "n_frames": 0, "status": "no_data"}

    # 按月分组，选择 cloud cover 最低的帧
    monthly_frames = defaultdict(list)
    for i in range(n_images):
        img = ee.Image(image_list.get(i))
        date = img.date().format("YYYYMM").getInfo()
        cloud = img.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        monthly_frames[date].append((i, cloud, img))

    selected = []
    for month, frames in sorted(monthly_frames.items()):
        frames.sort(key=lambda x: x[1])  # 按 cloud cover 排序
        selected.extend(frames[:max_per_month])

    # 导出选中的帧
    n_exported = 0
    for i, cloud, img in selected:
        date_str = img.date().format("YYYYMMdd").getInfo()
        out_path = patch_out / f"{date_str}.tif"

        if out_path.exists():
            n_exported += 1
            continue

        try:
            # 选择波段并裁剪
            img_cropped = img.select(["B2", "B3", "B4", "B8", "B11", "B12"]).clip(aoi)

            # 导出到本地（使用 getDownloadURL）
            url = img_cropped.getDownloadURL({
                "scale": 10,
                "crs": "EPSG:4326",
                "region": aoi,
                "format": "GEO_TIFF",
            })

            # 使用 urllib 下载
            import urllib.request
            urllib.request.urlretrieve(url, str(out_path))
            n_exported += 1

            # 避免 GEE 速率限制
            time.sleep(0.5)

        except Exception as e:
            print(f"  [WARN] {patch_id} {date_str} 导出失败: {e}")
            continue

    return {"patch_id": patch_id, "n_frames": n_exported, "status": "ok"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=["qiqihar", "daqing", "mudanjiang"], required=True)
    parser.add_argument("--n-patches", type=int, default=150, help="每个城市的 patch 数量")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--max-per-month", type=int, default=4)
    parser.add_argument("--out-dir", default="/workspace/raw/heilongjiang_new")
    parser.add_argument("--workers", type=int, default=1, help="GEE 建议单线程避免限流")
    args = parser.parse_args()

    if not initialize_gee():
        sys.exit(1)

    city_info = CITIES[args.city]
    print(f"[Download] 开始下载 {city_info['name']} ({args.city})")
    print(f"  描述: {city_info['description']}")
    print(f"  中心: {city_info['center']}")
    print(f"  Patch 数量: {args.n_patches}")
    print(f"  时间范围: {args.start_date} ~ {args.end_date}")

    # 计算网格维度（尽量接近正方形）
    n_xy = int(np.sqrt(args.n_patches))
    n_patches_x = n_xy
    n_patches_y = n_xy
    while n_patches_x * n_patches_y < args.n_patches:
        n_patches_x += 1

    patches = create_patch_grid(
        city_info["center"][0],
        city_info["center"][1],
        n_patches_x=n_patches_x,
        n_patches_y=n_patches_y,
    )

    # 保存 patch 元数据
    out_dir = Path(args.out_dir) / args.city / "s2"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = out_dir / "../patches_meta.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump({
            "city": args.city,
            "city_name": city_info["name"],
            "center": city_info["center"],
            "n_patches": len(patches),
            "patches": patches,
            "date_range": [args.start_date, args.end_date],
        }, f, indent=2)

    print(f"[Download] 已生成 {len(patches)} 个 patches")

    # 逐 patch 下载
    results = []
    for i, patch in enumerate(patches):
        print(f"[{i+1}/{len(patches)}] 下载 {patch['id']}...")
        result = download_sentinel2_patch(
            patch,
            args.start_date,
            args.end_date,
            out_dir,
            args.max_per_month,
        )
        results.append(result)
        print(f"  → {result['n_frames']} frames, status={result['status']}")

    # 保存结果统计
    stats = {
        "city": args.city,
        "n_patches": len(patches),
        "successful": sum(1 for r in results if r["status"] == "ok"),
        "total_frames": sum(r["n_frames"] for r in results),
        "results": results,
    }
    with open(out_dir / "../download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[Download] {city_info['name']} 下载完成")
    print(f"  成功 patches: {stats['successful']}/{stats['n_patches']}")
    print(f"  总帧数: {stats['total_frames']}")


if __name__ == "__main__":
    main()
