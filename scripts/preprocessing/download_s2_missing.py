#!/usr/bin/env python3
"""补充下载 S2 缺失的 2026 年数据.

特点:
- 只下载缺失的文件（已存在则跳过）
- 单线程逐个下载，稳定避免 503
- 每 patch 下载后记录日志
- 支持断点续传（重新运行会自动跳过已完成的）
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# 限制线程数
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dask

dask.config.set(scheduler="synchronous")

import numpy as np
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, "/workspace/xuannv")
from scripts.preprocessing.download_from_planetary_computer import (
    get_catalog, compute_bbox, determine_utm_epsg, search_items,
    download_source_stackstac, save_geotiff, COLLECTION_MAP, ASSET_MAP, RESOLUTION_MAP,
)

DATA_ROOT = Path("/workspace/raw/harbin_scenes")
PATCHES_META = DATA_ROOT / "patches_meta.json"

MONTH_RANGES = [
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
    ("2026-04-01", "2026-04-30"),
    ("2026-05-01", "2026-05-31"),
]


def find_missing_s2() -> list[tuple[str, str, str]]:
    """找出所有缺失 S2 2026 数据的 (patch_id, month, date_range).
    
    返回: [(patch_id, "2026-01", "2026-01-01_2026-01-31"), ...]
    """
    with open(PATCHES_META) as f:
        meta = json.load(f)
    
    patches = [p["id"] for p in meta["patches"]]
    missing = []
    
    for pid in patches:
        s2_dir = DATA_ROOT / "s2" / f"patch_{pid:06d}"
        for (start, end), month_label in zip(MONTH_RANGES, ["01", "02", "03", "04", "05"]):
            month_prefix = f"2026{month_label}"
            if s2_dir.exists():
                files = [f for f in s2_dir.glob("*.tif") if f.stem.startswith(month_prefix)]
                if files:
                    continue  # 已存在
            missing.append((f"patch_{pid:06d}", f"2026-{month_label}", f"{start}_{end}"))
    
    return missing


def download_single_patch_month(patch_id: str, month_label: str, date_start: str, date_end: str) -> dict:
    """下载单个 patch 的单个月份的 S2 数据."""
    # 解析 patch_id
    pid_num = int(patch_id.split("_")[1])
    
    with open(PATCHES_META) as f:
        meta = json.load(f)
    patch = meta["patches"][pid_num]
    lon, lat = patch["center_lonlat"]
    
    bbox = compute_bbox(lon, lat)
    epsg = determine_utm_epsg(lon, lat)
    
    catalog = get_catalog()
    source = "s2"
    
    try:
        items = search_items(catalog, source, bbox, date_start, date_end)
    except Exception as e:
        return {"status": "search_error", "error": str(e)}
    
    if not items:
        return {"status": "no_data", "n_items": 0}
    
    patch_dir = DATA_ROOT / source / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for item in items:
        try:
            dt = item.datetime
            date_str = dt.strftime("%Y%m%d") if dt else item.id.split("_")[-1][:8]
            out_path = patch_dir / f"{date_str}.tif"
            
            if out_path.exists():
                # 验证格式
                try:
                    with rasterio.open(out_path) as src:
                        if src.dtypes[0] == "float32":
                            skipped += 1
                            continue
                        else:
                            out_path.unlink()
                except Exception:
                    out_path.unlink()
            
            data_np = download_source_stackstac(item, source, bbox, epsg, divide_10000=True)
            save_geotiff(out_path, data_np, bbox, epsg)
            downloaded += 1
            
        except Exception as e:
            failed += 1
            print(f"    [WARN] {patch_id} {date_str} failed: {e}", file=sys.stderr)
            continue
    
    return {
        "status": "ok" if downloaded > 0 else ("failed" if failed > 0 else "no_data"),
        "n_items": len(items),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


def main():
    missing = find_missing_s2()
    print(f"Found {len(missing)} missing S2 files to download")
    
    if not missing:
        print("S2 data is already complete!")
        return
    
    # 按 patch 分组
    patch_months: dict[str, list[tuple[str, str]]] = {}
    for pid, month_label, date_range in missing:
        patch_months.setdefault(pid, []).append((month_label, date_range))
    
    print(f"Missing patches: {len(patch_months)}")
    
    start_time = time.time()
    success = 0
    fail = 0
    
    for idx, (pid, months) in enumerate(sorted(patch_months.items()), 1):
        print(f"\n[{idx}/{len(patch_months)}] {pid} — missing months: {[m[0] for m in months]}")
        
        for month_label, date_range in months:
            start, end = date_range.split("_")
            print(f"  Downloading {month_label} ({start} ~ {end})...")
            
            try:
                result = download_single_patch_month(pid, month_label, start, end)
                print(f"    Result: {result}")
                
                if result["status"] in ("ok", "no_data"):
                    success += 1
                else:
                    fail += 1
                    
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                fail += 1
            
            # 短暂延迟，避免对 PC 服务器的压力
            time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done: success={success}, fail={fail}, total={len(missing)}")
    print(f"Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
