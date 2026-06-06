#!/usr/bin/env python3
"""
下载AEF官方嵌入（Source Cooperative Zarr mosaic），覆盖海淀区和哈尔滨新区。
使用xarray直接读取，避免zarr.open_group枚举metadata。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import zoom


def dequantize_aef(data_int8: np.ndarray) -> np.ndarray:
    """AEF int8 -> float32 反量化，结果在S^63球面上。"""
    data = data_int8.astype(np.float32)
    return ((data / 127.5) ** 2) * np.sign(data)


def read_aef_region_xarray(
    ds: xr.Dataset,
    year: int,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用xarray读取AEF区域数据。"""
    if verbose:
        print(f"  Reading year={year}, bbox=[{lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}]")
    
    t0 = time.time()
    # y是降序，slice需要从大到小
    subset = ds.sel(
        time=year,
        x=slice(lon_min, lon_max),
        y=slice(lat_max, lat_min),
    )
    
    data = subset["embeddings"].values  # (64, H, W) int8
    x_coords = subset["x"].values
    y_coords = subset["y"].values
    
    if verbose:
        print(f"  Read done in {time.time()-t0:.1f}s, shape={data.shape}")
    
    return data, x_coords, y_coords


def extract_patches_from_aef(
    aef_data: np.ndarray,
    aef_x: np.ndarray,
    aef_y: np.ndarray,
    patches: list[dict],
    patch_size_m: int = 1280,
    target_size: int = 128,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """从AEF区域数据中裁剪每个patch并下采样到target_size。"""
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
    ds: xr.Dataset,
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
    
    lons = [p["center_lonlat"][0] for p in patches]
    lats = [p["center_lonlat"][1] for p in patches]
    
    margin = 0.02
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
    aef_int8, aef_x, aef_y = read_aef_region_xarray(
        ds, year, lon_min, lat_min, lon_max, lat_max, verbose=verbose
    )
    
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
    
    print("Opening AEF Zarr mosaic with xarray...")
    ds = xr.open_zarr(
        "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/",
        storage_options={"anon": True},
        consolidated=False,
    )
    print(f"Opened. Dims: {dict(ds.dims)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.harbin_only:
        process_region(
            ds,
            args.haidian_meta,
            output_dir / f"aef_haidian_{args.year}.npz",
            year=args.year,
        )
    
    if not args.haidian_only:
        process_region(
            ds,
            args.harbin_meta,
            output_dir / f"aef_harbin_{args.year}.npz",
            year=args.year,
        )
    
    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
