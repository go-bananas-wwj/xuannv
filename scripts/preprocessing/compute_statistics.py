#!/usr/bin/env python
"""计算各数据源的通道统计量 (mean/std) 并保存为 JSON。

支持多区域通用，默认路径对应哈尔滨数据。

用法:
    # 哈尔滨（默认路径）
    python scripts/preprocessing/compute_statistics.py

    # 自定义数据目录
    python scripts/preprocessing/compute_statistics.py \
        --data-root /workspace/raw/phase2_heilongjiang/daqing \
        --s2-dir s2_cloud_filtered \
        --output-dir /workspace/statistics/daqing \
        --workers 16
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.data.transforms import read_tif

SOURCES = ["s2", "s1", "landsat", "dem", "worldcover", "dynamic_world", "jrc_water"]


def compute_source_stats(data_root: Path, source_name: str, s2_dir: str,
                         max_patches: int = 50) -> dict | None:
    """计算单个数据源的 mean/std 统计量。"""
    src_dir = data_root / (s2_dir if source_name == "s2" else source_name)
    if not src_dir.exists():
        print(f"  [{source_name}] 目录不存在，跳过")
        return None

    patches = sorted([p.name for p in src_dir.iterdir() if p.is_dir()])
    if not patches:
        print(f"  [{source_name}] 无 patch，跳过")
        return None

    # 随机采样
    if len(patches) > max_patches:
        np.random.seed(42)
        patches = np.random.choice(patches, max_patches, replace=False).tolist()

    print(f"  [{source_name}] 采样 {len(patches)} 个 patch...")

    all_samples: list[np.ndarray] = []
    for patch_id in patches:
        patch_dir = src_dir / patch_id
        for tif_path in sorted(patch_dir.glob("*.tif")):
            try:
                data = read_tif(str(tif_path), image_size=-1)
                if data is None:
                    continue
                if all_samples and data.shape != all_samples[0].shape:
                    continue

                # jrc_water: -128 是 nodata，替换为 nan
                if source_name == "jrc_water":
                    data = data.astype(np.float32)
                    data[data == -128] = np.nan

                # 光学源：log(x+1)/10 变换
                if source_name in {"s2", "s2_hr", "landsat"}:
                    if data.max() < 2.0:
                        data = data * 10000.0
                    data = np.log(np.clip(data, 0, None) + 1) / 10.0

                # SAR 源：DN → dB
                if source_name in {"s1", "s1_hr"} and data.max() > 100:
                    data = np.log10(np.clip(data / 10000.0, 1e-10, None)) * 10.0

                all_samples.append(data)
            except Exception:
                pass

    if not all_samples:
        print(f"  [{source_name}] 无有效样本")
        return None

    stacked = np.stack(all_samples, axis=0)  # [N, C, H, W]
    n_channels = stacked.shape[1]
    stats: dict = {"n_channels": n_channels}

    for c in range(n_channels):
        channel_data = stacked[:, c].ravel()
        channel_data = channel_data[np.isfinite(channel_data)]
        mean = float(np.mean(channel_data))
        std  = float(np.std(channel_data))
        stats[f"channel_{c}"] = {"mean": mean, "std": std}

    return stats


def main():
    pa = argparse.ArgumentParser(description="计算数据源通道统计量")
    pa.add_argument("--data-root",    default="/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered",
                    help="数据根目录")
    pa.add_argument("--s2-dir",       default="s2",
                    help="S2 子目录名（云筛选后可能是 s2_cloud_filtered）")
    pa.add_argument("--output-dir",   default="/workspace/statistics/harbin_scenes",
                    help="统计量 JSON 输出目录")
    pa.add_argument("--max-patches",  type=int, default=50,
                    help="每源最多采样多少个 patch（统计量对样本数不敏感）")
    pa.add_argument("--workers",      type=int, default=1,
                    help="并发进程数（多源并行）")
    args = pa.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据根目录: {data_root}")
    print(f"输出目录:   {output_dir}")
    print(f"S2 目录名:  {args.s2_dir}")
    print(f"采样上限:   {args.max_patches} patches/源\n")

    for src in SOURCES:
        stats = compute_source_stats(data_root, src, args.s2_dir, args.max_patches)
        if stats is None:
            continue
        out_fp = output_dir / f"{src}_stats.json"
        out_fp.write_text(json.dumps(stats, indent=2))
        mean0 = stats.get("channel_0", {}).get("mean", "?")
        std0  = stats.get("channel_0", {}).get("std",  "?")
        print(f"  [{src}] 已保存 (ch0 mean={mean0:.4f} std={std0:.4f}) → {out_fp}")

    print("\n统计量计算完成。")


if __name__ == "__main__":
    main()
