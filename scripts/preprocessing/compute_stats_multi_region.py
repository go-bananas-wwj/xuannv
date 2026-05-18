#!/usr/bin/env python3
"""为多区域数据计算统计量 (mean/std).

用法:
    python scripts/preprocessing/compute_stats_multi_region.py \
        --data_root /workspace/raw/phase2_heilongjiang/daqing \
        --s2_dir s2_cloud_filtered \
        --output_dir /workspace/statistics/daqing \
        --workers 16
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "/workspace/xuannv")

from src.data.transforms import read_tif

SOURCES = ["s2", "s1", "landsat", "dem", "worldcover"]
MAX_PATCHES = 50


def compute_source_stats(data_root: Path, source_name: str, s2_dir: str, max_patches: int = 50) -> dict | None:
    """计算单个数据源的各通道 mean/std."""
    src_dir = data_root / (s2_dir if source_name == "s2" else source_name)
    if not src_dir.exists():
        print(f"  [{source_name}] 目录不存在，跳过")
        return None

    patches = sorted([p.name for p in src_dir.iterdir() if p.is_dir()])
    if not patches:
        print(f"  [{source_name}] 无patch，跳过")
        return None

    # 采样
    if len(patches) > max_patches:
        np.random.seed(42)
        patches = np.random.choice(patches, max_patches, replace=False).tolist()

    print(f"  [{source_name}] 采样 {len(patches)} patches...")

    all_samples = []
    for patch_id in patches:
        patch_dir = src_dir / patch_id
        tif_files = sorted(patch_dir.glob("*.tif"))
        if not tif_files:
            continue
        # 只取第一张计算通道数
        try:
            data = read_tif(tif_files[0], image_size=-1)  # 原始尺寸
            if data is not None:
                all_samples.append(data)
        except Exception as e:
            pass

    if not all_samples:
        print(f"  [{source_name}] 无有效样本")
        return None

    n_channels = all_samples[0].shape[0]
    stats = {"n_channels": n_channels}

    for c in range(n_channels):
        channel_vals = np.concatenate([s[c].flatten() for s in all_samples])
        channel_vals = channel_vals[np.isfinite(channel_vals)]
        if len(channel_vals) == 0:
            continue
        mean = float(np.mean(channel_vals))
        std = float(np.std(channel_vals))
        stats[f"band_{c}"] = {"mean": mean, "std": std if std > 1e-8 else 1.0}

    print(f"    Done: {len(all_samples)} samples, {n_channels} channels")
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--s2_dir", default="s2_cloud_filtered", help="S2子目录名")
    parser.add_argument("--output_dir", required=True, help="统计量输出目录")
    parser.add_argument("--max_patches", type=int, default=50)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"S2 source: {args.s2_dir}")
    print(f"Computing statistics (sampling {args.max_patches} patches per source)...\n")

    for source_name in SOURCES:
        stats = compute_source_stats(data_root, source_name, args.s2_dir, args.max_patches)
        if stats:
            out_file = output_dir / f"{source_name}_stats.json"
            with open(out_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"    Saved -> {out_file}")

    print("\nAll statistics computed.")


if __name__ == "__main__":
    main()
