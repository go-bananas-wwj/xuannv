#!/usr/bin/env python3
"""计算各数据源的通道统计量 (mean/std) 并保存为 JSON.

用法:
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv
    python scripts/preprocessing/compute_statistics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.data.dataset import HarbinPatchDataset
from src.data.transforms import read_tif, normalize_data

# 需要统计的源
SOURCES = ["s2", "s1", "landsat", "dem", "worldcover", "dynamic_world", "jrc_water"]

# 为加速，只采样部分 patch（统计量对样本数不敏感）
MAX_PATCHES = 50


def compute_source_stats(dataset: HarbinPatchDataset, source_name: str, max_patches: int = 50) -> dict:
    """计算单个数据源的各通道 mean/std."""
    print(f"  Computing stats for {source_name}...")
    
    # 收集所有帧的原始数据（未归一化）
    all_samples: list[np.ndarray] = []
    
    patch_ids = dataset.patches[:max_patches]
    for patch_id in patch_ids:
        source_dir = dataset._resolve_source_dir(source_name, patch_id)
        if source_dir is None:
            continue
        
        tif_files = sorted(source_dir.glob("*.tif"))
        if not tif_files:
            continue
        
        # 对每个 patch 最多采样 5 个时间步
        sample_files = tif_files[::max(1, len(tif_files) // 5)]
        for tif_path in sample_files:
            data = read_tif(tif_path, dataset.image_size)
            if data is None:
                continue
            # 修复：光学源先log变换，再算统计量
            if source_name in {"s2", "s2_hr", "landsat"}:
                data = np.log(np.clip(data, 0, None) + 1) / 10.0
            all_samples.append(data)
    
    if not all_samples:
        print(f"    WARNING: No data found for {source_name}")
        return {}
    
    # 按通道计算
    n_channels = all_samples[0].shape[0]
    stats: dict[str, dict[str, float]] = {}
    
    for c in range(n_channels):
        channel_vals = np.concatenate([s[c].flatten() for s in all_samples])
        # 排除 nodata / nan
        channel_vals = channel_vals[np.isfinite(channel_vals)]
        if len(channel_vals) == 0:
            continue
        
        mean = float(np.mean(channel_vals))
        std = float(np.std(channel_vals))
        stats[f"band_{c}"] = {"mean": mean, "std": std if std > 1e-8 else 1.0}
    
    print(f"    Done: {len(all_samples)} samples, {n_channels} channels")
    return stats


def main():
    cfg = load_config("configs/qwen_v1_scenes.yaml")
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    
    print(f"Dataset: {len(dataset.patches)} patches")
    print(f"Computing statistics (sampling {MAX_PATCHES} patches per source)...\n")
    
    stats_dir = Path("/workspace/statistics/harbin_scenes")
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    for source_name in SOURCES:
        stats = compute_source_stats(dataset, source_name, MAX_PATCHES)
        if stats:
            out_file = stats_dir / f"{source_name}_stats.json"
            with open(out_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"    Saved -> {out_file}")
    
    print("\nAll statistics computed.")


if __name__ == "__main__":
    main()
