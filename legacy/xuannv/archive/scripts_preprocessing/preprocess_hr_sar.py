#!/usr/bin/env python3
"""
高分辨率 SAR 数据预处理 — 解压、重投影、切分为 patch 级 128x128 TIFF
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as rio_from_bounds
import geopandas as gpd

# ── 配置 ─────────────────────────────────────────────────────────
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
SAR_DIR = Path("/workspace/哈尔滨SAR")
OUT_DIR = Path("/workspace/raw/harbin_scenes/s1_hr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 128
DST_CRS = "EPSG:32652"

# 月份 → ZIP 文件映射
SAR_ZIPS = {
    "20250627": SAR_DIR / "6月/BC4-015141/BC4-SM-ORG-2SVV-20250627T021632-015141-000010-003B25.zip",
    "20250810": SAR_DIR / "8月/BC4-017850/BC4-SM-ORG-2SVV-20250810T021545-017850-000010-0045BA.zip",
    "20250912": SAR_DIR / "9月/BC4-019684/BC4-SM-ORG-2SVV-20250912T021509-019684-000009-004CE4.zip",
    "20251005": SAR_DIR / "10月/BC5-007134/BC5-SM-ORG-2SVV-20251005T021616-007134-000130-001BDE.zip",
}


def load_grid():
    return gpd.read_file(str(GRID_PATH))


def extract_and_reproject_sar(zip_path: Path, tmpdir: str):
    """解压 ZIP 并将 SAR 重投影到 EPSG:32652，返回输出路径."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        tiff_names = [n for n in z.namelist() if n.endswith('.tiff')]
        if not tiff_names:
            return None
        z.extract(tiff_names[0], tmpdir)
        src_path = Path(tmpdir) / tiff_names[0]

    reprojected_path = Path(tmpdir) / "reprojected.tif"

    with rasterio.open(str(src_path)) as src:
        # 读取全部数据
        data = src.read(1)

        # dB 转换: 10 * log10(max(x, 1e-10))
        data_db = 10.0 * np.log10(np.maximum(data, 1e-10))
        data_db = data_db.astype(np.float32)

        # 重投影到 32652
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': DST_CRS,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999,
        })

        dst_data = np.empty((height, width), dtype=np.float32)
        reproject(
            source=data_db,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=DST_CRS,
            resampling=WarpResampling.bilinear,
        )

        with rasterio.open(str(reprojected_path), 'w', **kwargs) as dst:
            dst.write(dst_data, 1)

    return reprojected_path


def process_sar_for_month(grid_rows, month, zip_path):
    """处理某个月份的所有 patch."""
    results = []
    tmpdir = tempfile.mkdtemp(prefix=f"sar_{month}_")

    try:
        reprojected_path = extract_and_reproject_sar(zip_path, tmpdir)
        if reprojected_path is None:
            return [{"month": month, "status": "no_tiff"}]

        with rasterio.open(str(reprojected_path)) as src:
            for _, row in grid_rows.iterrows():
                pid = row["patch_id"]
                minx, miny, maxx, maxy = row.geometry.bounds

                out_patch_dir = OUT_DIR / pid
                out_patch_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_patch_dir / f"{month}.tif"

                if out_path.exists():
                    results.append({"patch_id": pid, "month": month, "status": "exists"})
                    continue

                try:
                    window = from_bounds(minx, miny, maxx, maxy, src.transform)
                    data = src.read(1, window=window, out_shape=(TARGET_SIZE, TARGET_SIZE), resampling=Resampling.bilinear)

                    # 检查覆盖度
                    valid_mask = data != src.nodata
                    valid_ratio = np.mean(valid_mask)
                    if valid_ratio < 0.3:
                        results.append({"patch_id": pid, "month": month, "status": "low_coverage", "valid_ratio": float(valid_ratio)})
                        continue

                    # 标准化 (z-score on valid pixels)
                    valid = data[valid_mask]
                    if len(valid) > 0:
                        mean_val = float(np.mean(valid))
                        std_val = float(np.std(valid))
                        if std_val > 1e-8:
                            data = (data - mean_val) / std_val
                        else:
                            data = data - mean_val

                    profile = {
                        "driver": "GTiff",
                        "height": TARGET_SIZE,
                        "width": TARGET_SIZE,
                        "count": 1,
                        "dtype": "float32",
                        "crs": DST_CRS,
                        "transform": rio_from_bounds(minx, miny, maxx, maxy, TARGET_SIZE, TARGET_SIZE),
                    }
                    with rasterio.open(str(out_path), "w", **profile) as dst:
                        dst.write(data, 1)

                    results.append({"patch_id": pid, "month": month, "status": "ok"})
                except Exception as e:
                    results.append({"patch_id": pid, "month": month, "status": "error", "error": str(e)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def main():
    print("Loading grid...")
    grid = load_grid()
    print(f"Total patches: {len(grid)}")

    all_results = []
    for month, zip_path in SAR_ZIPS.items():
        if not zip_path.exists():
            print(f"[!] Missing SAR zip: {zip_path}")
            continue
        print(f"\nProcessing SAR {month} from {zip_path}...")
        res = process_sar_for_month(grid, month, zip_path)
        ok = sum(1 for r in res if r.get("status") == "ok")
        exist = sum(1 for r in res if r.get("status") == "exists")
        err = sum(1 for r in res if r.get("status") == "error")
        low = sum(1 for r in res if r.get("status") == "low_coverage")
        print(f"  Done. ok={ok} exists={exist} low_cov={low} err={err}")
        all_results.extend(res)

    # Save report
    ok_total = sum(1 for r in all_results if r.get("status") == "ok")
    exist_total = sum(1 for r in all_results if r.get("status") == "exists")
    err_total = sum(1 for r in all_results if r.get("status") == "error")
    low_total = sum(1 for r in all_results if r.get("status") == "low_coverage")

    print(f"\n=== SAR Preprocessing Summary ===")
    print(f"ok={ok_total} exists={exist_total} low_cov={low_total} err={err_total}")

    report_path = OUT_DIR / "preprocess_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_tasks": len(all_results),
            "ok": ok_total,
            "exists": exist_total,
            "low_coverage": low_total,
            "error": err_total,
            "details": all_results,
        }, f, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
