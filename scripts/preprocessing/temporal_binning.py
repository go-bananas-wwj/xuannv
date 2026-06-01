#!/usr/bin/env python3
"""
时序数据 11 天分箱脚本
======================
将每个 patch 的原始时序影像按 11 天间隔分箱，
同一 bin 内多帧取 median，重命名为 bin 中心日期。

分箱规则：
    Bin 0: 2025-01-01  → 窗口 [2024-12-27, 2025-01-06]
    Bin 1: 2025-01-12  → 窗口 [2025-01-07, 2025-01-17]
    Bin 2: 2025-01-23  → 窗口 [2025-01-18, 2025-01-28]
    ...

用法:
    python temporal_binning.py \
        --input-dir /workspace/raw/national_china/national_china/s2 \
        --output-dir /workspace/raw/national_china/national_china/s2_binned \
        --bin-days 11 \
        --start-date 2025-01-01 \
        --end-date 2026-06-01
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge


def generate_bins(start_date: str, end_date: str, bin_days: int) -> list[datetime]:
    """生成所有 bin 中心日期"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    bins = []
    current = start
    while current <= end:
        bins.append(current)
        current += timedelta(days=bin_days)
    return bins


def parse_date_from_filename(filename: str) -> datetime | None:
    """从文件名解析日期，如 '20250120.tif'"""
    m = re.match(r"(\d{8})\.tif", filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return None


def find_best_bin(frame_date: datetime, bins: list[datetime]) -> datetime | None:
    """找到帧最近的 bin，且距离不超过 bin_days/2"""
    if not bins:
        return None
    half_window = timedelta(days=5)  # 11天窗口的半宽（向下取整）

    best_bin = None
    best_diff = None
    for b in bins:
        diff = abs((frame_date - b).days)
        if diff <= 5:
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_bin = b
    return best_bin


def bin_patch(
    patch_dir: Path,
    output_patch_dir: Path,
    bins: list[datetime],
    method: str = "closest",
):
    """对单个 patch 的所有帧进行时间分箱"""
    output_patch_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有帧
    frames = []
    for tif_path in sorted(patch_dir.glob("*.tif")):
        date = parse_date_from_filename(tif_path.name)
        if date:
            frames.append({"date": date, "path": tif_path})

    if not frames:
        return {"n_frames": 0, "n_bins": 0}

    # 按 bin 分组
    bin_groups: dict[str, list] = {}
    for f in frames:
        best_bin = find_best_bin(f["date"], bins)
        if best_bin:
            bin_str = best_bin.strftime("%Y%m%d")
            bin_groups.setdefault(bin_str, []).append(f)

    n_bins_created = 0
    for bin_str, group in bin_groups.items():
        out_path = output_patch_dir / f"{bin_str}.tif"

        if len(group) == 1:
            # 只有一帧，直接复制
            shutil.copy2(group[0]["path"], out_path)
            n_bins_created += 1
        else:
            # 多帧：取 median
            datasets = []
            for g in group:
                src = rasterio.open(g["path"])
                datasets.append(src)

            # 读取所有数据并取 median
            arrays = [np.asarray(ds.read()) for ds in datasets]
            stacked = np.stack(arrays, axis=0)  # [n_frames, band, h, w]
            median = np.median(stacked, axis=0).astype(arrays[0].dtype)

            # 保存
            ref = datasets[0]
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=ref.height,
                width=ref.width,
                count=ref.count,
                dtype=median.dtype,
                crs=ref.crs,
                transform=ref.transform,
                compress="lzw",
            ) as dst:
                dst.write(median)

            for ds in datasets:
                ds.close()
            n_bins_created += 1

    return {"n_frames": len(frames), "n_bins": n_bins_created}


def main():
    parser = argparse.ArgumentParser(description="时序数据 11 天分箱")
    parser.add_argument("--input-dir", required=True, help="原始时序数据目录")
    parser.add_argument("--output-dir", required=True, help="分箱后输出目录")
    parser.add_argument("--bin-days", type=int, default=11, help="分箱间隔（天）")
    parser.add_argument("--start-date", default="2025-01-01", help="起始日期")
    parser.add_argument("--end-date", default="2026-06-01", help="结束日期")
    parser.add_argument("--method", default="closest", choices=["closest", "median"],
                        help="多帧时的处理方法")
    parser.add_argument("--workers", type=int, default=8, help="并行 worker 数")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 生成 bins
    bins = generate_bins(args.start_date, args.end_date, args.bin_days)
    print(f"生成分箱窗口: {len(bins)} 个 ({bins[0].strftime('%Y-%m-%d')} ~ {bins[-1].strftime('%Y-%m-%d')})")

    # 找到所有 patch 目录
    patch_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("patch_")])
    print(f"处理 patches: {len(patch_dirs)}")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for patch_dir in patch_dirs:
            output_patch_dir = output_dir / patch_dir.name
            future = executor.submit(bin_patch, patch_dir, output_patch_dir, bins, args.method)
            futures[future] = patch_dir.name

        for i, future in enumerate(as_completed(futures)):
            patch_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 100 == 0:
                    total_frames = sum(r["n_frames"] for r in results)
                    total_bins = sum(r["n_bins"] for r in results)
                    print(f"[{i+1}/{len(patch_dirs)}] {patch_name} | 累计: {total_frames} frames → {total_bins} bins")
            except Exception as e:
                print(f"[ERROR] {patch_name}: {e}")

    total_frames = sum(r["n_frames"] for r in results)
    total_bins = sum(r["n_bins"] for r in results)
    print(f"\n分箱完成: {total_frames} 帧 → {total_bins} bins")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
