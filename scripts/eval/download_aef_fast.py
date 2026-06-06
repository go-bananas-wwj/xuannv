#!/usr/bin/env python3
"""
高效下载AEF官方嵌入：只读取覆盖patches的最小窗口，本地裁剪。
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
from pyproj import Transformer
from scipy.ndimage import zoom


def dequantize_aef(data_int8: np.ndarray) -> np.ndarray:
    data = data_int8.astype(np.float32)
    return ((data / 127.5) ** 2) * np.sign(data)


def read_cog_window(
    filepath: str,
    session: AWSSession,
    utm_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, rasterio.Affine] | tuple[None, None]:
    """读取COG文件中与UTM边界相交的窗口。"""
    with rasterio.Env(session=session):
        with rasterio.open(filepath) as src:
            left, bottom, right, top = utm_bounds
            row_start, col_start = rasterio.transform.rowcol(src.transform, left, top)
            row_end, col_end = rasterio.transform.rowcol(src.transform, right, bottom)
            
            row_start = max(0, row_start)
            col_start = max(0, col_start)
            row_end = min(src.height, row_end)
            col_end = min(src.width, col_end)
            
            if row_start >= row_end or col_start >= col_end:
                return None, None
            
            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            data = src.read(window=window)
            new_transform = src.transform * rasterio.Affine.translation(col_start, row_start)
            return data, new_transform


def process_region(
    index_df: pd.DataFrame,
    patches_meta_path: str,
    output_path: str,
    year: int = 2025,
):
    patches_meta_path = Path(patches_meta_path)
    with open(patches_meta_path) as f:
        meta = json.load(f)
    
    patches = meta["patches"]
    patch_size_m = meta.get("patch_size_m", 1280)
    crs = patches[0].get("crs", "EPSG:32650")
    
    region_name = patches_meta_path.stem.replace("_patches", "").replace("patches_meta", "region")
    print(f"\n{'='*60}")
    print(f"Processing: {region_name}")
    print(f"  Patches: {len(patches)}, Year: {year}, CRS: {crs}")
    
    # 找到覆盖区域的COG文件（限制为实际覆盖patches的）
    lons = [p["center_lonlat"][0] for p in patches]
    lats = [p["center_lonlat"][1] for p in patches]
    margin = 0.02
    
    files_df = index_df[
        (index_df["year"] == year) &
        (index_df["wgs84_west"] < max(lons) + margin) &
        (index_df["wgs84_east"] > min(lons) - margin) &
        (index_df["wgs84_south"] < max(lats) + margin) &
        (index_df["wgs84_north"] > min(lats) - margin)
    ].copy()
    
    # 进一步筛选：只保留与patches实际UTM范围相交的文件
    # 先将所有patches的UTM bounds合并为一个总bounds
    utm_lefts = [p["utm_bounds"][0] for p in patches]
    utm_bottoms = [p["utm_bounds"][1] for p in patches]
    utm_rights = [p["utm_bounds"][2] for p in patches]
    utm_tops = [p["utm_bounds"][3] for p in patches]
    total_utm = (min(utm_lefts), min(utm_bottoms), max(utm_rights), max(utm_tops))
    
    print(f"  Total UTM bounds: {total_utm}")
    print(f"  Found {len(files_df)} candidate COG files")
    for _, row in files_df.iterrows():
        print(f"    {row['path'].split('/')[-1]}: WGS84 [{row['wgs84_west']:.3f}, {row['wgs84_south']:.3f}, {row['wgs84_east']:.3f}, {row['wgs84_north']:.3f}]")
    
    session = AWSSession(aws_unsigned=True, region_name='us-west-2')
    
    # 对每个COG文件，计算需要读取的窗口 = 所有patches与该文件UTM范围的交集
    # 先获取每个文件的UTM bounds
    tiles_data = {}
    for _, row in files_df.iterrows():
        vrt_path = row['path'].replace('.tiff', '.vrt')
        fname = vrt_path.split('/')[-1]
        print(f"\n  Loading {fname}...")
        t0 = time.time()
        
        # 计算该文件与所有patches的合并窗口
        with rasterio.Env(session=session):
            with rasterio.open(vrt_path) as src:
                tile_left = src.bounds.left
                tile_bottom = src.bounds.bottom
                tile_right = src.bounds.right
                tile_top = src.bounds.top
                
                # 交集窗口
                win_left = max(tile_left, total_utm[0])
                win_bottom = max(tile_bottom, total_utm[1])
                win_right = min(tile_right, total_utm[2])
                win_top = min(tile_top, total_utm[3])
                
                if win_left >= win_right or win_bottom >= win_top:
                    print(f"    No overlap, skipping")
                    continue
                
                row_start, col_start = rasterio.transform.rowcol(src.transform, win_left, win_top)
                row_end, col_end = rasterio.transform.rowcol(src.transform, win_right, win_bottom)
                row_start = max(0, row_start)
                col_start = max(0, col_start)
                row_end = min(src.height, row_end)
                col_end = min(src.width, col_end)
                
                window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
                print(f"    Reading window: {window}")
                data = src.read(window=window)
                transform = src.transform * rasterio.Affine.translation(col_start, row_start)
                tiles_data[fname] = (data, transform)
                print(f"    Done in {time.time()-t0:.1f}s, shape={data.shape}")
    
    if len(tiles_data) == 0:
        print("  No tiles loaded!")
        return {}
    
    # 提取patches
    print(f"\n  Extracting {len(patches)} patches...")
    t0 = time.time()
    results = {}
    
    for i, patch in enumerate(patches):
        patch_id = patch["id"]
        utm_bounds = tuple(patch["utm_bounds"])  # [left, bottom, right, top]
        patch_crs = patch.get("crs", crs)
        
        # 找到覆盖这个patch的tile
        for fname, (tile_data, tile_transform) in tiles_data.items():
            # 计算窗口
            t_left, t_bottom, t_right, t_top = utm_bounds
            
            # 如果CRS不同，转换
            if patch_crs != crs:
                transformer = Transformer.from_crs(patch_crs, crs, always_xy=True)
                t_left, t_bottom = transformer.transform(t_left, t_bottom)
                t_right, t_top = transformer.transform(t_right, t_top)
            
            row_start, col_start = rasterio.transform.rowcol(tile_transform, t_left, t_top)
            row_end, col_end = rasterio.transform.rowcol(tile_transform, t_right, t_bottom)
            
            row_start = max(0, row_start)
            col_start = max(0, col_start)
            row_end = min(tile_data.shape[1], row_end)
            col_end = min(tile_data.shape[2], col_end)
            
            if row_start >= row_end or col_start >= col_end:
                continue
            
            patch_data = tile_data[:, row_start:row_end, col_start:col_end]
            
            # 下采样到128x128
            if patch_data.shape[1] != 128 or patch_data.shape[2] != 128:
                zoom_y = 128 / patch_data.shape[1]
                zoom_x = 128 / patch_data.shape[2]
                patch_data = zoom(patch_data, (1, zoom_y, zoom_x), order=1)
            
            # 反量化 + 归一化
            patch_data = dequantize_aef(patch_data)
            norms = np.linalg.norm(patch_data, axis=0, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            patch_data = patch_data / norms
            
            results[str(patch_id)] = patch_data.astype(np.float32)
            break
        
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(patches)} done")
    
    print(f"  Extracted {len(results)}/{len(patches)} patches in {time.time()-t0:.1f}s")
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{k: v for k, v in results.items()},
        _meta=json.dumps({
            "source": "AEF_official_COG",
            "year": year,
            "n_patches": len(results),
            "patch_size_m": patch_size_m,
            "embedding_dim": 64,
            "region": region_name,
        })
    )
    print(f"  Saved: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return results


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
        process_region(index_df, args.haidian_meta, output_dir / f"aef_haidian_{args.year}.npz", year=args.year)
    
    if not args.haidian_only:
        process_region(index_df, args.harbin_meta, output_dir / f"aef_harbin_{args.year}.npz", year=args.year)
    
    print(f"\n{'='*60}\nAll done!")


if __name__ == "__main__":
    main()
