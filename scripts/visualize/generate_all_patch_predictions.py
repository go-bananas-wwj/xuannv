#!/usr/bin/env python3
"""
Run the post-bugfix CD head on all 424 patches for the 4 month pairs.
Saves 64×64 probability maps as .npy files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np

from src.inference.engine import load_cd_head, run_change_detection
from src.utils.device import get_device

EMBEDDING_DIR = Path("/workspace/xuannv/outputs/aef_qwen_v2/monthly_embeddings_2025")
HEAD_PATH = Path("/workspace/xuannv/outputs/aef_qwen_v2/monthly_cd_head/best_cv_fold0_v3_ohem_head.pt")
OUTPUT_DIR = Path("/workspace/xuannv/outputs/aef_qwen_v2/patch_predictions")

PERIODS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}


def get_all_patch_ids() -> list[str]:
    """Infer patch IDs from embedding filenames."""
    patch_ids = set()
    for p in EMBEDDING_DIR.glob("patch_*.npy"):
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].startswith("2025-"):
            patch_ids.add(parts[0])
    return sorted(patch_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="torch device string, e.g. npu:0")
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    args = parser.parse_args()

    device = get_device(device_str=args.device)
    head = load_cd_head(HEAD_PATH, device=device)
    print(f"Head loaded on {device}")

    patch_ids = get_all_patch_ids()
    print(f"Found {len(patch_ids)} patches with embeddings")

    segment = len(patch_ids) // args.total_gpus
    start = args.gpu_idx * segment
    end = start + segment if args.gpu_idx < args.total_gpus - 1 else len(patch_ids)
    my_patches = patch_ids[start:end]
    print(f"GPU {args.gpu_idx}/{args.total_gpus} processing patches {start}:{end} ({len(my_patches)} patches)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    extracted = 0
    errors = 0

    for pid in my_patches:
        for period, (bm, am) in PERIODS.items():
            bpath = EMBEDDING_DIR / f"{pid}_{bm}.npy"
            apath = EMBEDDING_DIR / f"{pid}_{am}.npy"
            out_path = OUTPUT_DIR / f"{pid}_{period.replace('~', '_')}.npy"

            if not bpath.exists() or not apath.exists():
                skipped += 1
                continue

            try:
                emb_b = np.load(bpath)
                emb_a = np.load(apath)
                probs = run_change_detection(head, emb_b, emb_a, device)
                np.save(out_path, probs.astype(np.float32))
                extracted += 1
            except Exception as e:
                print(f"Error on {pid} {period}: {e}")
                errors += 1

    print(f"Done. skipped={skipped} extracted={extracted} errors={errors}")


if __name__ == "__main__":
    main()
