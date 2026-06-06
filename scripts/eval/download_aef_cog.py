#!/usr/bin/env python3
"""
下载AEF官方嵌入（COG格式），覆盖海淀区和哈尔滨新区。
使用aef_index.parquet索引定位文件，用rasterio读取COG。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.session import AWSSession
from rasterio.windows import Window
from pyproj import Transformer
from scipy.ndimage import zoom


def dequantize_aef(data_int8: np.ndarray) -> np.ndarray:
    """AEF int8 -> float32 反量化，结果在S^63球面上。"""
    data = data_int8.astype(np.float32)
    return ((data / 127.5) ** 2) * np.sign(data)


def find_covering_files(
    index_df: pd.DataFrame,
    year: int,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.DataFrame:
    """从索引中查找覆盖指定区域的COG文件。"""
    mask = (
        (index_df["year"] == year) &
        (index_df["wgs84_west"] < lon_max) &
        (index_df["wgs84_east"] > lon_min) &
        (index_df["wgs84_south"] < lat_max) &
        (index_df["wgs84_north"] > lat_min)
    )
    return index_df[mask].copy()


def read_cog_region(
    filepath: str,
    session: AWSSession,
    utm_bounds: tuple[float, float, float, float] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, rasterio.Affine]:
    """读取COG文件，可选裁剪到UTM边界。"""
    with rasterio.Env(session=session):
        with rasterio.open(filepath) as src:
            if utm_bounds is not None:
                # 计算窗口
                left, bottom, right, top = utm_bounds
                row_start, col_start = rasterio.transform.rowcol(src.transform, left, top)
                row_end, col_end = rasterio.transform.rowcol(src.transform, right, bottom)
                row_start = max(0, row_start)
                col_start = max(0, col_start)
                row_end = min(src.height, row_end)
                col_end = min(src.width, col_end)
                
                if row_start >= row_end or col_start >= col_end:
                    return None, None
                
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                data = src.read(window=window)
                # 更新transform
                new_transform = src.transform * rasterio.Affine.translation(col_start, row_start)
            else:
                data = src.read()
                new_transform = src.transform
            
            return data, new_transform


def merge_tiles(
    tiles: list[tuple[np.ndarray, rasterio.Affine]],
    target_bounds: tuple[float, float, float, float],
    target_res: float = 10.0,
    target_shape: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """将多个COG tile合并并下采样到目标尺寸。
    
    Args:
        tiles: [(data, transform), ...]
        target_bounds: (left, bottom, right, top) in UTM
        target_res: 目标分辨率（米）
        target_shape: (H, W)
    """
    if len(tiles) == 1:
        data, transform = tiles[0]
    else:
        # 简单合并：找到所有数据的bounds，创建统一数组
        # 这里简化处理，假设tiles有重叠或相邻
        # 实际应该用rasterio.merge，但为了简单，先手动处理
        all_data = []
        for data, transform in tiles:
            if data is not None:
                all_data.append((data, transform))
        
        if len(all_data) == 0:
            return None
        if len(all_data) == 1:
            data, transform = all_data[0]
        else:
            # 计算合并后的bounds
            lefts = [t[2] for _, t in all_data]
            tops = [t[5] for _, t in all_data]
            rights = [t[2] + t[0] * t[1] for _, t in all_data]
            bottoms = [t[5] + t[4] * t[0] for _, t in all_data]
            
            merged_left = min(lefts)
            merged_top = max(tops)
            merged_right = max(rights)
            merged_bottom = min(bottoms)
            
            merged_w = int((merged_right - merged_left) / target_res)
            merged_h = int((merged_top - merged_bottom) / target_res)
            merged_data = np.full((64, merged_h, merged_w), -128, dtype=np.int8)
            
            for data, transform in all_data:
                tile_left = transform[2]
                tile_top = transform[5]
                col_off = int((tile_left - merged_left) / target_res)
                row_off = int((merged_top - tile_top) / target_res)
                
                h, w = data.shape[1], data.shape[2]
                merged_data[:, row_off:row_off+h, col_off:col_off+w] = data
            
            data = merged_data
            transform = rasterio.Affine.translation(merged_left, merged_top) * rasterio.Affine.scale(target_res, -target_res)
    
    # 下采样到目标尺寸
    if data.shape[1] != target_shape[0] or data.shape[2] != target_shape[1]:
        zoom_y = target_shape[0] / data.shape[1]
        zoom_x = target_shape[1] / data.shape[2]
        data = zoom(data, (1, zoom_y, zoom_x), order=1)
    
    return data


def extract_patches_from_cogs(
    index_df: pd.DataFrame,
    patches: list[dict],
    year: int,
    session: AWSSession,
    target_size: int = 128,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """从COG文件中提取每个patch的嵌入。"""
    results = {}
    
    # 找到所有覆盖这些patches的COG文件
    lons = [p["center_lonlat"][0] for p in patches]
    lats = [p["center_lonlat"][1] for p in patches]
    margin = 0.02
    files_df = find_covering_files(
        index_df, year,
        min(lons) - margin, max(lons) + margin,
        min(lats) - margin, max(lats) + margin,
    )
    
    if verbose:
        print(f"  Found {len(files_df)} COG files for year {year}")
        for _, row in files_df.iterrows():
            print(f"    {row['path'].split('/')[-1]}: WGS84 [{row['wgs84_west']:.3f}, {row['wgs84_south']:.3f}, {row['wgs84_east']:.3f}, {row['wgs84_north']:.3f}]")
    
    # 预读取所有COG文件（合并为一个大数组）
    # 但为了简化，我们逐个patch处理
    for i, patch in enumerate(patches):
        patch_id = patch["id"]
        utm_bounds = patch["utm_bounds"]  # [left, bottom, right, top]
        crs = patch.get("crs", "EPSG:32650")
        
        # 找到覆盖这个patch的COG文件
        # 先将UTM bounds转为WGS84来匹配索引
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        left, bottom = transformer.transform(utm_bounds[0], utm_bounds[1])
        right, top = transformer.transform(utm_bounds[2], utm_bounds[3])
        
        patch_files = find_covering_files(index_df, year, left, right, bottom, top)
        
        if len(patch_files) == 0:
            if verbose:
                print(f"  Warning: patch {patch_id} not covered by any COG file")
            continue
        
        # 读取并合并覆盖这个patch的COG tile
        tiles = []
        for _, row in patch_files.iterrows():
            # 扩大边界以包含完整patch
            data, transform = read_cog_region(row['path'], session, utm_bounds, verbose=False)
            if data is not None:
                tiles.append((data, transform))
        
        if len(tiles) == 0:
            continue
        
        patch_data = merge_tiles(tiles, utm_bounds, target_shape=(target_size, target_size))
        
        if patch_data is None:
            continue
        
        # 反量化
        patch_data = dequantize_aef(patch_data)
        
        # 重新归一化到单位球面
        norms = np.linalg.norm(patch_data, axis=0, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        patch_data = patch_data / norms
        
        results[str(patch_id)] = patch_data.astype(np.float32)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(patches)} patches")
    
    if verbose:
        print(f"  Total extracted: {len(results)}/{len(patches)} patches")
    
    return results


def process_region(
    index_df: pd.DataFrame,
    patches_meta_path: str,
    output_path: str,
    year: int = 2025,
    verbose: bool = True,
):
    """处理一个区域的AEF嵌入下载和patch裁剪。"""
    patches_meta_path = Path(patches_meta_path)
    with open(patches_meta_path) as f:
        meta = json.load(f)
    
    patches = meta["patches"]
    patch_size_m = meta.get("patch_size_m", 1280)
    
    region_name = patches_meta_path.stem.replace("_patches", "").replace("patches_meta", "region")
    print(f"\n{'='*60}")
    print(f"Processing: {region_name}")
    print(f"  Patches: {len(patches)}")
    print(f"  Year: {year}")
    
    session = AWSSession(aws_unsigned=True, region_name='us-west-2')
    
    print("  Step 1: Extracting patches from COG files...")
    t0 = time.time()
    patch_embeddings = extract_patches_from_cogs(
        index_df, patches, year, session, target_size=128, verbose=verbose
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")
    
    # 保存
    print(f"  Step 2: Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        **{k: v for k, v in patch_embeddings.items()},
        _meta=json.dumps({
            "source": "AEF_official_COG",
            "year": year,
            "n_patches": len(patch_embeddings),
            "patch_size_m": patch_size_m,
            "embedding_dim": 64,
            "region": region_name,
        })
    )
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {file_size:.1f} MB")
    
    return patch_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haidian-meta", default="/workspace/xuannv/patches_meta.json")
    parser.add_argument("--harbin-meta", default="/workspace/xuannv/data_raw/harbin/scenes/patches_meta.json")
    parser.add_argument("--index", default="/tmp/aef_index.parquet")
    parser.add_argument("--output-dir", default="/workspace/xuannv/outputs/aef_official_embeddings")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--haidian-only", action="store_true")
    parser.add_argument("--harbin-only", action="store_true")
    args = parser.parse_args()
    
    print("Loading AEF index...")
    index_df = pd.read_parquet(args.index)
    print(f"Loaded {len(index_df)} records.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.harbin_only:
        process_region(
            index_df,
            args.haidian_meta,
            output_dir / f"aef_haidian_{args.year}.npz",
            year=args.year,
        )
    
    if not args.haidian_only:
        process_region(
            index_df,
            args.harbin_meta,
            output_dir / f"aef_harbin_{args.year}.npz",
            year=args.year,
        )
    
    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
