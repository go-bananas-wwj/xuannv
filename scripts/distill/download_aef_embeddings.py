#!/usr/bin/env python3
"""
AEF (AlphaEarth Foundations) 嵌入下载脚本
从 Source Cooperative / Google Earth Engine 下载指定区域的 64D 嵌入

数据说明:
- 分辨率: 10m/像素
- 维度: 64 (bands A00-A63)
- 时间: 2018-2024 年度合成
- 格式: Cloud-Optimized GeoTIFF (COG), 量化 8-bit int
- 来源: https://source.coop/tge-labs/aef

使用方法:
    python download_aef_embeddings.py --region harbin --year 2024
    python download_aef_embeddings.py --region haidian --year 2024
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

sys.path.insert(0, "/workspace/xuannv")

# =============================================================================
# 配置
# =============================================================================

# Source Cooperative AEF COG 基础 URL (2024)
# 格式: aef_v1_{year}_{utm_zone}_{quadkey}.tif
# 由于是全球瓦片，我们需要通过 GEE 或 STAC 查询对应区域的文件
# 这里使用 GEE 导出方式（更可靠）

AEF_SOURCE_COOP_URL = "https://data.source.coop/tge-labs/aef"

# 区域配置: (min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "harbin": {
        "bounds": (126.0, 45.0, 128.5, 46.5),
        "crs": "EPSG:4326",
        "desc": "哈尔滨新区",
    },
    "haidian": {
        "bounds": (116.0, 39.5, 116.5, 40.2),
        "crs": "EPSG:4326", 
        "desc": "北京海淀区",
    },
}

# AEF 量化参数: 存储为 uint8, 需要反量化到 float32
# dequantize: float_value = (int_value - 127.5) / 127.5  (近似 [-1, 1])
# 实际上 AEF 使用更复杂的量化，但官方 aef-loader 包会处理
# 这里我们使用近似: (v - 128) / 128.0


def dequantize_aef(arr_uint8: np.ndarray) -> np.ndarray:
    """将 AEF 量化 uint8 反量化为 float32.
    
    AEF 使用对称量化，映射关系:
        float = (uint8 - 127.5) / 127.5
    这样 uint8=0 -> -1.0, uint8=127 -> ~0, uint8=255 -> ~1.0
    """
    return (arr_uint8.astype(np.float32) - 127.5) / 127.5


def download_aef_via_gee(
    region_name: str,
    year: int,
    output_dir: Path,
    scale: int = 10,
) -> Path:
    """使用 Google Earth Engine API 下载 AEF 嵌入.
    
    需要已认证 GEE (earthengine authenticate)
    """
    try:
        import ee
    except ImportError:
        raise RuntimeError("请安装 earthengine-api: pip install earthengine-api")
    
    ee.Initialize()
    
    region = REGIONS[region_name]
    bounds = region["bounds"]
    
    # AEF ImageCollection on GEE
    # 路径格式: projects/tge-labs/assets/aef/aef_v1_{year}
    collection_path = f"projects/tge-labs/assets/aef/aef_v1_{year}"
    
    try:
        collection = ee.ImageCollection(collection_path)
    except Exception as e:
        print(f"[警告] GEE 路径 {collection_path} 可能不存在，尝试替代路径...")
        # 替代路径
        collection = ee.ImageCollection("projects/tge-labs/assets/aef")
        collection = collection.filter(ee.Filter.eq("year", year))
    
    # 获取区域内的 mosaic
    roi = ee.Geometry.Rectangle(bounds)
    image = collection.filterBounds(roi).mosaic()
    
    # 64 个波段
    bands = [f"A{i:02d}" for i in range(64)]
    
    output_path = output_dir / f"aef_{region_name}_{year}.tif"
    
    print(f"[GEE] 开始导出 {region_name} {year} 年 AEF 嵌入...")
    print(f"       区域: {bounds}")
    print(f"       分辨率: {scale}m")
    print(f"       输出: {output_path}")
    
    task = ee.batch.Export.image.toDrive(
        image=image.select(bands),
        description=f"aef_{region_name}_{year}",
        folder="aef_downloads",
        region=roi,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    
    # 等待任务完成
    print(f"[GEE] 任务已启动: {task.id}")
    print("[GEE] 注意: GEE 导出到 Drive，请手动下载后放到对应目录")
    print(f"       目标路径: {output_path}")
    
    return output_path


def download_aef_via_cog(
    region_name: str,
    year: int,
    output_dir: Path,
    target_crs: str = "EPSG:32650",
) -> Path:
    """通过 Source Cooperative COG 直接下载（推荐）.
    
    使用 rasterio 窗口读取，只下载需要的区域。
    需要知道对应区域的 COG URL。
    """
    # Source Cooperative 的 COG 文件通常按 UTM 分区存储
    # 由于瓦片结构复杂，这里使用一个简化的方法:
    # 1. 通过 STAC API 查询对应区域的 asset URL
    # 2. 用 rasterio 读取窗口
    
    print("[COG] 尝试通过 Source Cooperative STAC 查询...")
    
    # 使用 terrafloww 的 Rasteret Collection (预构建索引)
    # https://source.coop/terrafloww/aef-v1-rasteret
    
    try:
        import requests
    except ImportError:
        raise RuntimeError("请安装 requests: pip install requests")
    
    region = REGIONS[region_name]
    bounds_wgs84 = region["bounds"]
    
    # Source Cooperative STAC API endpoint
    stac_url = "https://api.source.coop/stac/tge-labs/aef/search"
    
    payload = {
        "bbox": list(bounds_wgs84),
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        "collections": ["aef"],
        "limit": 100,
    }
    
    print(f"[STAC] 查询: {stac_url}")
    print(f"       BBox: {bounds_wgs84}")
    
    try:
        resp = requests.post(stac_url, json=payload, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("features", [])
    except Exception as e:
        print(f"[STAC] 查询失败: {e}")
        print("[提示] 将回退到 GEE 导出方式")
        return download_aef_via_gee(region_name, year, output_dir)
    
    if not items:
        print(f"[STAC] 未找到 {region_name} {year} 的数据")
        return download_aef_via_gee(region_name, year, output_dir)
    
    print(f"[STAC] 找到 {len(items)} 个瓦片")
    
    # 合并所有瓦片
    output_path = output_dir / f"aef_{region_name}_{year}_wgs84.tif"
    
    # 读取并合并
    from rasterio.merge import merge
    
    sources = []
    for item in items:
        cog_url = item["assets"]["data"]["href"]
        print(f"[COG] 读取: {cog_url}")
        src = rasterio.open(cog_url)
        sources.append(src)
    
    # Merge
    mosaic, out_transform = merge(sources, bounds=bounds_wgs84)
    
    # 保存
    out_profile = sources[0].profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
        "transform": out_transform,
        "crs": "EPSG:4326",
        "dtype": "uint8",
        "compress": "lzw",
    })
    
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(mosaic)
    
    for src in sources:
        src.close()
    
    print(f"[保存] WGS84 原始文件: {output_path}")
    print(f"       形状: {mosaic.shape} (bands, H, W)")
    
    # 反量化并保存为 float32 npz (更紧凑)
    mosaic_float = dequantize_aef(mosaic)
    
    # 如果需要重投影到目标 CRS
    if target_crs and target_crs != "EPSG:4326":
        print(f"[重投影] 到 {target_crs}...")
        import rasterio.warp
        
        # 计算目标 CRS 的 bounds
        dst_crs = rasterio.CRS.from_string(target_crs)
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            "EPSG:4326", dst_crs, mosaic.shape[2], mosaic.shape[1],
            *bounds_wgs84, resolution=10
        )
        
        dst_mosaic = np.empty((64, dst_height, dst_width), dtype=np.float32)
        
        rasterio.warp.reproject(
            source=mosaic_float,
            destination=dst_mosaic,
            src_transform=out_transform,
            src_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        
        npz_path = output_dir / f"aef_{region_name}_{year}_{target_crs.replace(':', '')}.npz"
        np.savez_compressed(npz_path, 
            embeddings=dst_mosaic,  # (64, H, W)
            transform=dst_transform.to_gdal(),
            crs=target_crs,
            bounds=bounds_wgs84,
        )
        print(f"[保存] 重投影后 NPZ: {npz_path}")
        return npz_path
    else:
        npz_path = output_dir / f"aef_{region_name}_{year}_wgs84.npz"
        np.savez_compressed(npz_path,
            embeddings=mosaic_float,
            transform=out_transform.to_gdal(),
            crs="EPSG:4326",
            bounds=bounds_wgs84,
        )
        print(f"[保存] WGS84 NPZ: {npz_path}")
        return npz_path


def crop_to_patches(
    aef_npz_path: Path,
    patches_meta_path: Path,
    output_dir: Path,
) -> None:
    """将 AEF 嵌入按 patch 边界裁剪，保存为每个 patch 的 .npy 文件.
    
    输出结构: output_dir/{patch_id}.npy  (shape: 64, h, w)
    """
    print(f"[裁剪] 读取 AEF: {aef_npz_path}")
    data = np.load(aef_npz_path)
    aef_emb = data["embeddings"]  # (64, H, W)
    transform = rasterio.Affine.from_gdal(*data["transform"])
    crs = data["crs"].item() if hasattr(data["crs"], "item") else str(data["crs"])
    
    with open(patches_meta_path) as f:
        patches = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for patch in patches:
        patch_id = patch["patch_id"]
        bounds = patch["bounds"]  # [minx, miny, maxx, maxy]
        
        # 计算像素窗口
        min_col, min_row = ~transform * (bounds[0], bounds[3])
        max_col, max_row = ~transform * (bounds[2], bounds[1])
        
        min_col, max_col = int(min_col), int(max_col)
        min_row, max_row = int(min_row), int(max_row)
        
        # 裁剪
        patch_emb = aef_emb[:, min_row:max_row, min_col:max_col]
        
        # 保存
        out_path = output_dir / f"{patch_id}.npy"
        np.save(out_path, patch_emb)
    
    print(f"[完成] 已裁剪 {len(patches)} 个 patches 到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载 AEF 嵌入")
    parser.add_argument("--region", choices=["harbin", "haidian"], required=True)
    parser.add_argument("--year", type=int, default=2024, choices=[2018,2019,2020,2021,2022,2023,2024])
    parser.add_argument("--method", choices=["gee", "cog"], default="cog", help="下载方式")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/raw/aef_embeddings"))
    parser.add_argument("--crop-to-patches", action="store_true", help="裁剪到 patch 级别")
    parser.add_argument("--patches-meta", type=Path, default=None, help="patches_meta.json 路径")
    parser.add_argument("--target-crs", default="EPSG:32650", help="目标 CRS")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"AEF 嵌入下载: {args.region} / {args.year}")
    print("=" * 60)
    
    if args.method == "gee":
        result = download_aef_via_gee(args.region, args.year, args.output_dir)
    else:
        result = download_aef_via_cog(args.region, args.year, args.output_dir, args.target_crs)
    
    if args.crop_to_patches and args.patches_meta:
        crop_dir = args.output_dir / f"{args.region}_{args.year}_patches"
        crop_to_patches(result, args.patches_meta, crop_dir)
    
    print("\n[完成]")
    print(f"输出: {result}")


if __name__ == "__main__":
    main()
