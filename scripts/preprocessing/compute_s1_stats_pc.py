#!/usr/bin/env python3
"""
计算 PC S1 RTC (dB) 的统计量，用于更新 haidian 的 s1_stats.json。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import rasterio

S1_DIR = "/workspace/xuannv/data_raw/haidian/scenes"
OUT_FILE = "/workspace/xuannv/statistics/haidian/s1_stats.json"


def main():
    files = glob.glob(f"{S1_DIR}/*/s1/*.tif")
    print(f"找到 {len(files)} 个 S1 文件")

    vv_vals = []
    vh_vals = []

    for i, f in enumerate(files):
        ds = rasterio.open(f)
        vv = ds.read(1).astype(np.float32)
        vh = ds.read(2).astype(np.float32)

        # 排除 nodata (0 或极值)
        vv = vv[(vv > -29.99) & (vv < 9.99)]
        vh = vh[(vh > -29.99) & (vh < 9.99)]

        if len(vv) > 0:
            vv_vals.extend(vv.tolist())
        if len(vh) > 0:
            vh_vals.extend(vh.tolist())

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(files)}")

    vv_vals = np.array(vv_vals)
    vh_vals = np.array(vh_vals)

    stats = {
        "n_channels": 2,
        "channel_0": {
            "mean": float(vv_vals.mean()),
            "std": float(vv_vals.std()),
        },
        "channel_1": {
            "mean": float(vh_vals.mean()),
            "std": float(vh_vals.std()),
        },
    }

    print(f"\nVV: mean={stats['channel_0']['mean']:.4f}, std={stats['channel_0']['std']:.4f}")
    print(f"VH: mean={stats['channel_1']['mean']:.4f}, std={stats['channel_1']['std']:.4f}")

    with open(OUT_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n统计量已保存到 {OUT_FILE}")


if __name__ == "__main__":
    main()
