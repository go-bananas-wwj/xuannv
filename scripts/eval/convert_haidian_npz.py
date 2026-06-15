#!/usr/bin/env python3
"""将 extract_haidian_embeddings.py 输出的 npz 转换为 evaluate_haidian2026_labels.py 所需格式.

输入: patch_ids, emb_dec, emb_apr
输出: spatial_maps [N, M, D, H, W], month_labels, patch_ids
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 npz (含 patch_ids/emb_dec/emb_apr)")
    parser.add_argument("--output", required=True, help="输出 npz 路径")
    parser.add_argument("--dec-month", default="2025-12")
    parser.add_argument("--apr-month", default="2026-04")
    args = parser.parse_args()

    data = np.load(args.input)
    patch_ids = data["patch_ids"]
    emb_dec = data["emb_dec"]  # [N, D, H, W]
    emb_apr = data["emb_apr"]

    spatial_maps = np.stack([emb_dec, emb_apr], axis=1).astype(np.float32)  # [N, 2, D, H, W]
    month_labels = np.array([args.dec_month, args.apr_month])

    np.savez_compressed(args.output, spatial_maps=spatial_maps, month_labels=month_labels, patch_ids=patch_ids)
    print(f"[Convert] 保存到 {args.output}, shape={spatial_maps.shape}, months={list(month_labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
