#!/usr/bin/env python3
"""
高分辨率光学数据预处理 — 将 2m 镶嵌大图切分为 patch 级 128x128 TIFF
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import geopandas as gpd

# ── 配置 ─────────────────────────────────────────────────────────
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
OPT_DIR = Path("/workspace/01哈尔滨新区（全覆盖）（2025年4月、6月、8月、9月、10月）/光学-哈尔滨新区/光学-哈尔滨新区2025年4_9月2米高分影像-镶嵌")
OUT_DIR = Path("/workspace/raw/harbin_scenes/s2_hr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 月份 → 文件映射
OPTICAL_FILES = {
    "20250409": OPT_DIR / "哈尔滨新区20250409.tif",
    "20250622": OPT_DIR / "哈尔滨新区20250622.tif",
    "20250804": OPT_DIR / "哈尔滨新区20250804.tif",
    "20250901": OPT_DIR / "哈尔滨新区202509.tif",
    "20251001": OPT_DIR / "哈尔滨10月.tif",
}

TARGET_SIZE = 128
INPUT_DIM = 6  # 模型 input_dim


def load_grid():
    gdf = gpd.read_file(str(GRID_PATH))
    return gdf


def process_patch_for_month(args):
    """处理单个 patch 的某个月份."""
    patch_id, bounds, month, filepath = args
    minx, miny, maxx, maxy = bounds

    out_patch_dir = OUT_DIR / patch_id
    out_patch_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_patch_dir / f"{month}.tif"

    if out_path.exists():
        return {"patch_id": patch_id, "month": month, "status": "exists"}

    try:
        with rasterio.open(str(filepath)) as src:
            # 将 patch bounds (32652) 转换到影像 CRS
            src_bounds = transform_bounds("EPSG:32652", src.crs, minx, miny, maxx, maxy)
            # 读取窗口
            window = from_bounds(*src_bounds, src.transform)
            data = src.read(window=window, out_shape=(src.count, TARGET_SIZE, TARGET_SIZE), resampling=Resampling.bilinear)

            # 检查是否全是 NoData
            if src.nodata is not None:
                valid_ratio = np.mean(data != src.nodata)
                if valid_ratio < 0.5:
                    return {"patch_id": patch_id, "month": month, "status": "low_coverage", "valid_ratio": float(valid_ratio)}

            # Pad 到 6 通道
            if data.shape[0] < INPUT_DIM:
                padded = np.zeros((INPUT_DIM, TARGET_SIZE, TARGET_SIZE), dtype=data.dtype)
                padded[:data.shape[0]] = data
                data = padded
            elif data.shape[0] > INPUT_DIM:
                data = data[:INPUT_DIM]

            # 写入
            profile = {
                "driver": "GTiff",
                "height": TARGET_SIZE,
                "width": TARGET_SIZE,
                "count": INPUT_DIM,
                "dtype": data.dtype,
                "crs": "EPSG:32652",
                "transform": rasterio.Affine.identity() * rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale((maxx - minx) / TARGET_SIZE, -(maxy - miny) / TARGET_SIZE),
            }
            with rasterio.open(str(out_path), "w", **profile) as dst:
                dst.write(data)

        return {"patch_id": patch_id, "month": month, "status": "ok"}
    except Exception as e:
        return {"patch_id": patch_id, "month": month, "status": "error", "error": str(e)}


def main():
    print("Loading grid...")
    grid = load_grid()
    print(f"Total patches: {len(grid)}")

    tasks = []
    for _, row in grid.iterrows():
        pid = row["patch_id"]
        bounds = row.geometry.bounds
        for month, filepath in OPTICAL_FILES.items():
            if filepath.exists():
                tasks.append((pid, bounds, month, filepath))
            else:
                print(f"[!] Missing file: {filepath}")

    print(f"Total tasks: {len(tasks)}")

    results = []
    ok_count = 0
    error_count = 0
    low_cov_count = 0
    exist_count = 0

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_patch_for_month, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            results.append(res)
            status = res["status"]
            if status == "ok":
                ok_count += 1
            elif status == "error":
                error_count += 1
                print(f"  ERROR {res['patch_id']} {res['month']}: {res.get('error')}")
            elif status == "low_coverage":
                low_cov_count += 1
            elif status == "exists":
                exist_count += 1

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{len(tasks)} | ok={ok_count} exists={exist_count} low_cov={low_cov_count} err={error_count}")

    print(f"\nDone. ok={ok_count} exists={exist_count} low_cov={low_cov_count} err={error_count}")

    # Save report
    report_path = OUT_DIR / "preprocess_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_tasks": len(tasks),
            "ok": ok_count,
            "exists": exist_count,
            "low_coverage": low_cov_count,
            "error": error_count,
            "details": results,
        }, f, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
