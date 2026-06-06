#!/usr/bin/env python3
"""
下载AEF官方嵌入（Source Cooperative Zarr mosaic），覆盖海淀区和哈尔滨新区。
反量化后按patch裁剪并重采样到128x128。

AEF数据:
- S3: s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/
- 格式: Zarr v3, EPSG:4326
- 时间: 2017-2025 (年度)
- 通道: 64 (int8量化)
- 反量化: float = ((int8 / 127.5) ** 2) * sign(int8)
- 反量化后已在 S^63 单位球面上
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr
import fsspec
from pyproj import Transformer
from scipy.ndimage import zoom


def dequantize_aef(data_int8: np.ndarray) -> np.ndarray:
    """AEF int8 -> float32 反量化，结果在S^63球面上。"""
    data = data_int8.astype(np.float32)
    return ((data / 127.5) ** 2) * np.sign(data)


def read_aef_region(
    root: zarr.Group,
    year: int,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从AEF Zarr读取指定经纬度范围的嵌入。
    
    Returns:
        data: (64, H, W) int8 数组
        x_coords: (W,) 经度坐标
        y_coords: (H,) 纬度坐标（降序）
    """
    x_coords = np.array(root["x"])
    y_coords = np.array(root["y"])
    times = np.array(root["time"])
    
    t_idx = int(np.where(times == year)[0][0])
    
    # x是升序
    x0 = int(np.searchsorted(x_coords, lon_min, side="left"))
    x1 = int(np.searchsorted(x_coords, lon_max, side="right"))
    # y是降序 (83.68 -> -83.36)
    y0 = int(np.searchsorted(-y_coords, -lat_max, side="left"))
    y1 = int(np.searchsorted(-y_coords, -lat_min, side="right"))
    
    if verbose:
        print(f"  Zarr slice: t={t_idx}({year}), y={y0}:{y1}({y1-y0}px), x={x0}:{x1}({x1-x0}px)")
    
    embeddings = root["embeddings"]
    
    # 分块读取，使用较大的块减少HTTP请求次数
    chunk_h, chunk_w = 1024, 1024
    total_h = y1 - y0
    total_w = x1 - x0
    data = np.empty((64, total_h, total_w), dtype=np.int8)
    
    n_chunks = ((total_h + chunk_h - 1) // chunk_h) * ((total_w + chunk_w - 1) // chunk_w)
    chunk_idx = 0
    t_start = time.time()
    
    for y_start in range(y0, y1, chunk_h):
        y_end = min(y_start + chunk_h, y1)
        for x_start in range(x0, x1, chunk_w):
            x_end = min(x_start + chunk_w, x1)
            chunk_idx += 1
            
            for attempt in range(3):
                try:
                    chunk = embeddings[t_idx, :, y_start:y_end, x_start:x_end]
                    data[:, y_start - y0 : y_end - y0, x_start - x0 : x_end - x0] = chunk
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(f"Failed to read chunk y={y_start}:{y_end}, x={x_start}:{x_end}: {e}")
            
            if verbose and chunk_idx % 5 == 0:
                elapsed = time.time() - t_start
                mb_done = chunk_idx * chunk_h * chunk_w * 64 / 1024 / 1024
                mb_total = n_chunks * chunk_h * chunk_w * 64 / 1024 / 1024
                print(f"  Progress: {chunk_idx}/{n_chunks} chunks ({mb_done:.0f}/{mb_total:.0f}MB, {elapsed:.0f}s)")
    
    return data, x_coords[x0:x1], y_coords[y0:y1]


def extract_patches_from_aef(
    aef_data: np.ndarray,
    aef_x: np.ndarray,
    aef_y: np.ndarray,
    patches: list[dict],
    patch_size_m: int = 1280,
    target_size: int = 128,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """从AEF区域数据中裁剪每个patch并下采样到target_size。
    
    Args:
        aef_data: (64, H, W) float32, 已反量化
        aef_x: (W,) 经度
        aef_y: (H,) 纬度（降序）
        patches: patches_meta.json中的patches列表
        
    Returns:
        dict: patch_id -> (64, target_size, target_size) embedding
    """
    results = {}
    
    for i, patch in enumerate(patches):
        patch_id = patch["id"]
        utm_bounds = patch["utm_bounds"]  # [left, bottom, right, top]
        crs = patch.get("crs", "EPSG:32650")
        
        # UTM -> WGS84
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        left, bottom = transformer.transform(utm_bounds[0], utm_bounds[1])
        right, top = transformer.transform(utm_bounds[2], utm_bounds[3])
        
        # 在AEF中找到对应像素范围
        x0 = int(np.searchsorted(aef_x, left, side="left"))
        x1 = int(np.searchsorted(aef_x, right, side="right"))
        y0 = int(np.searchsorted(-aef_y, -top, side="left"))
        y1 = int(np.searchsorted(-aef_y, -bottom, side="right"))
        
        if x0 >= x1 or y0 >= y1:
            print(f"  Warning: patch {patch_id} out of AEF bounds, skipping")
            continue
        
        patch_data = aef_data[:, y0:y1, x0:x1]
        
        # 下采样到 target_size x target_size
        if patch_data.shape[1] != target_size or patch_data.shape[2] != target_size:
            zoom_y = target_size / patch_data.shape[1]
            zoom_x = target_size / patch_data.shape[2]
            patch_data = zoom(patch_data, (1, zoom_y, zoom_x), order=1)
        
        # 重新归一化到单位球面（双线性插值可能破坏了归一化）
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
    root: zarr.Group,
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
    
    # 计算WGS84边界
    lons = [p["center_lonlat"][0] for p in patches]
    lats = [p["center_lonlat"][1] for p in patches]
    
    # 扩大边界以包含完整patch
    margin = 0.02  # 约2km的margin
    lon_min, lon_max = min(lons) - margin, max(lons) + margin
    lat_min, lat_max = min(lats) - margin, max(lats) + margin
    
    region_name = patches_meta_path.stem.replace("_patches", "").replace("patches_meta", "region")
    print(f"\n{'='*60}")
    print(f"Processing: {region_name}")
    print(f"  Patches: {len(patches)}")
    print(f"  Year: {year}")
    print(f"  WGS84 bbox: [{lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}]")
    
    # 1. 下载AEF区域数据
    print("  Step 1: Downloading AEF region data...")
    t0 = time.time()
    aef_int8, aef_x, aef_y = read_aef_region(
        root, year, lon_min, lat_min, lon_max, lat_max, verbose=verbose
    )
    print(f"  Downloaded in {time.time()-t0:.1f}s, shape={aef_int8.shape}")
    
    # 2. 反量化
    print("  Step 2: Dequantizing...")
    aef_float = dequantize_aef(aef_int8)
    del aef_int8
    
    # 3. 提取patch
    print("  Step 3: Extracting patches...")
    t0 = time.time()
    patch_embeddings = extract_patches_from_aef(
        aef_float, aef_x, aef_y, patches, patch_size_m=patch_size_m, verbose=verbose
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")
    
    # 4. 保存
    print(f"  Step 4: Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        **{k: v for k, v in patch_embeddings.items()},
        _meta=json.dumps({
            "source": "AEF_official",
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
    parser.add_argument("--output-dir", default="/workspace/xuannv/outputs/aef_official_embeddings")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--haidian-only", action="store_true")
    parser.add_argument("--harbin-only", action="store_true")
    args = parser.parse_args()
    
    print("Connecting to AEF Zarr mosaic on Source Cooperative...")
    fs = fsspec.filesystem("s3", anon=True)
    store = zarr.storage.FsspecStore(fs, path="us-west-2.opendata.source.coop/tge-labs/aef-mosaic/")
    root = zarr.open_group(store, mode="r")
    print("Connected.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.harbin_only:
        process_region(
            root,
            args.haidian_meta,
            output_dir / f"aef_haidian_{args.year}.npz",
            year=args.year,
        )
    
    if not args.haidian_only:
        process_region(
            root,
            args.harbin_meta,
            output_dir / f"aef_harbin_{args.year}.npz",
            year=args.year,
        )
    
    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
