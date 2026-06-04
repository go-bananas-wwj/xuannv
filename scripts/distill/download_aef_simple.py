#!/usr/bin/env python3
"""
AEF 嵌入下载简化版 — 使用 Google Earth Engine API

前置条件:
    1. 安装 earthengine-api: pip install earthengine-api
    2. 认证 GEE: earthengine authenticate

用法:
    python download_aef_simple.py --region harbin --year 2024
    python download_aef_simple.py --region haidian --year 2024
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, "/workspace/xuannv")

# 区域配置
REGIONS = {
    "harbin": {
        "bounds": [126.0, 45.0, 128.5, 46.5],
        "patches_meta": "/workspace/raw/harbin_newarea_olmoearth/patches_meta.json",
        "output_dir": "/workspace/raw/aef_embeddings/harbin_2024_patches",
        "crs": "EPSG:32650",
    },
    "haidian": {
        "bounds": [116.0, 39.5, 116.5, 40.2],
        "patches_meta": "/workspace/raw/haidian_olmoearth/patches_meta.json",
        "output_dir": "/workspace/raw/aef_embeddings/haidian_2024_patches",
        "crs": "EPSG:32650",
    },
}


def download_aef_gee(region_name: str, year: int, output_tif: Path):
    """通过 GEE 下载 AEF 嵌入为 GeoTIFF."""
    import ee
    ee.Initialize()
    
    region = REGIONS[region_name]
    bounds = region["bounds"]
    roi = ee.Geometry.Rectangle(bounds)
    
    # AEF Collection
    collection = ee.ImageCollection("projects/tge-labs/assets/aef")
    image = collection.filter(ee.Filter.eq("year", year)).mosaic()
    
    bands = [f"A{i:02d}" for i in range(64)]
    
    print(f"[GEE] 导出 {region_name} {year} 年 AEF 嵌入...")
    print(f"       区域: {bounds}")
    
    task = ee.batch.Export.image.toDrive(
        image=image.select(bands),
        description=f"aef_{region_name}_{year}",
        folder="aef_downloads",
        region=roi,
        scale=10,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"[GEE] 任务 ID: {task.id}")
    print("[提示] 数据将导出到 Google Drive 的 aef_downloads 文件夹")
    print("[提示] 下载后请手动将文件复制到:")
    print(f"       {output_tif}")
    return task.id


def crop_aef_to_patches(aef_tif: Path, patches_meta: Path, output_dir: Path):
    """将 AEF GeoTIFF 按 patch 边界裁剪为 .npy 文件.
    
    AEF 是量化 uint8 存储，需要反量化为 float32.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(patches_meta) as f:
        patches = json.load(f)
    
    with rasterio.open(aef_tif) as src:
        aef_data = src.read()  # (64, H, W) uint8
        aef_transform = src.transform
        aef_crs = src.crs
        
        print(f"[读取] AEF: {aef_tif}")
        print(f"       形状: {aef_data.shape}, dtype: {aef_data.dtype}")
        print(f"       CRS: {aef_crs}")
        
        # 反量化: (uint8 - 127.5) / 127.5
        aef_float = (aef_data.astype(np.float32) - 127.5) / 127.5
        
        for patch in patches:
            patch_id = patch["patch_id"]
            pbounds = patch["bounds"]  # [minx, miny, maxx, maxy]
            
            # 计算像素坐标
            min_col, min_row = ~aef_transform * (pbounds[0], pbounds[3])
            max_col, max_row = ~aef_transform * (pbounds[2], pbounds[1])
            
            min_col, max_col = int(min_col), int(max_col)
            min_row, max_row = int(min_row), int(max_row)
            
            # 边界检查
            min_col = max(0, min_col)
            min_row = max(0, min_row)
            max_col = min(aef_data.shape[2], max_col)
            max_row = min(aef_data.shape[1], max_row)
            
            if max_col <= min_col or max_row <= min_row:
                print(f"[跳过] {patch_id}: 超出范围")
                continue
            
            patch_emb = aef_float[:, min_row:max_row, min_col:max_col]
            out_path = output_dir / f"{patch_id}.npy"
            np.save(out_path, patch_emb)
        
        print(f"[完成] 已裁剪 {len(patches)} 个 patches 到 {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["harbin", "haidian"], required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--action", choices=["download", "crop", "both"], default="both")
    parser.add_argument("--aef-tif", type=Path, default=None, help="已有的 AEF GeoTIFF 路径")
    args = parser.parse_args()
    
    region = REGIONS[args.region]
    
    if args.action in ("download", "both"):
        output_tif = Path(f"/workspace/raw/aef_embeddings/aef_{args.region}_{args.year}.tif")
        output_tif.parent.mkdir(parents=True, exist_ok=True)
        download_aef_gee(args.region, args.year, output_tif)
        print("\n[下一步] 请从 Google Drive 下载文件后运行:")
        print(f"   python download_aef_simple.py --region {args.region} --action crop --aef-tif <下载的文件路径>")
    
    if args.action in ("crop", "both"):
        if args.aef_tif is None:
            args.aef_tif = Path(f"/workspace/raw/aef_embeddings/aef_{args.region}_{args.year}.tif")
        if not args.aef_tif.exists():
            print(f"[错误] 找不到 AEF 文件: {args.aef_tif}")
            print("[提示] 先运行 --action download 获取文件")
            return
        crop_aef_to_patches(args.aef_tif, region["patches_meta"], region["output_dir"])


if __name__ == "__main__":
    main()
