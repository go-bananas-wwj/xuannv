#!/usr/bin/env python3
"""
预提取 V2 模型在 2025 年各月份的 embedding，处理所有 424 个 patch。
支持按 GPU 分段并行。

用法:
  CUDA_VISIBLE_DEVICES=6 python scripts/inference/extract_monthly_embeddings_all_patches.py --gpu_idx 0 --total_gpus 2
  CUDA_VISIBLE_DEVICES=7 python scripts/inference/extract_monthly_embeddings_all_patches.py --gpu_idx 1 --total_gpus 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
from tqdm import tqdm

from src.inference.engine import extract_embedding_map, load_backbone
from src.utils.device import get_device

CONFIG_PATH = "configs/xuannv_v2_expE_pure_recon.yaml"
CKPT_PATH = "/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/epoch_best_epoch43.pt"
OUTPUT_DIR = Path("/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/monthly_embeddings_2025")

MONTHLY_WINDOWS = {
    "2025-04": (1743465600000.0, 1746057600000.0),
    "2025-05": (1746057600000.0, 1748736000000.0),
    "2025-06": (1748736000000.0, 1751328000000.0),
    "2025-07": (1751328000000.0, 1754006400000.0),
    "2025-08": (1754006400000.0, 1756684800000.0),
    "2025-09": (1756684800000.0, 1759267200000.0),
    "2025-10": (1759267200000.0, 1761945600000.0),
}

NEEDED_MONTHS = {"2025-04", "2025-06", "2025-08", "2025-09", "2025-10"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=int, default=0, help="当前 GPU 分段索引")
    parser.add_argument("--total_gpus", type=int, default=2, help="总 GPU 分段数")
    args = parser.parse_args()

    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, dataset, _ = load_backbone(CONFIG_PATH, CKPT_PATH, device=device)

    all_patches = dataset.patches
    total = len(all_patches)

    chunk_size = (total + args.total_gpus - 1) // args.total_gpus
    start_idx = args.gpu_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    my_patches = all_patches[start_idx:end_idx]

    print(f"GPU segment {args.gpu_idx}/{args.total_gpus} → patches {start_idx}:{end_idx} ({len(my_patches)} patches)")

    meta_records = []
    skipped = 0
    extracted = 0
    errors = 0

    for pid in tqdm(my_patches, desc=f"GPU{args.gpu_idx}"):
        pidx = dataset.patches.index(pid)
        for month, (ws, we) in MONTHLY_WINDOWS.items():
            if month not in NEEDED_MONTHS:
                continue
            out_path = OUTPUT_DIR / f"{pid}_{month}.npy"
            if out_path.exists():
                skipped += 1
                continue
            try:
                emb = extract_embedding_map(model, dataset, pidx, ws, we, device)
                np.save(out_path, emb)
                meta_records.append({"patch_id": pid, "month": month, "shape": list(emb.shape)})
                extracted += 1
            except Exception as e:
                errors += 1
                tqdm.write(f"  Skip {pid} {month}: {e}")

    meta_path = OUTPUT_DIR / f"meta_gpu{args.gpu_idx}.json"
    with open(meta_path, "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"\nGPU{args.gpu_idx} Done. skipped={skipped} extracted={extracted} errors={errors}")


if __name__ == "__main__":
    main()
