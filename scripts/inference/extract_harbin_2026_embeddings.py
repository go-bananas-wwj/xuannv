#!/usr/bin/env python3
"""预提取 V5 Mixed Scale 模型在哈尔滨 2026 年1-5月的 embedding，用于变化检测.

月份窗口:
  2026-01: Jan 1  ~ Jan 31
  2026-02: Feb 1  ~ Feb 28
  2026-03: Mar 1  ~ Mar 31
  2026-04: Apr 1  ~ Apr 30
  2026-05: May 1  ~ May 31

输出:
  /workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026/{patch_id}_{month}.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
from tqdm import tqdm

from src.inference.engine import extract_embedding_map, load_backbone

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v5_harbin_inference.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")

# 2026年1-5月时间窗口 (毫秒时间戳)
MONTHLY_WINDOWS = {
    "2026-01": (1735689600000.0, 1738368000000.0),
    "2026-02": (1738368000000.0, 1740787200000.0),
    "2026-03": (1740787200000.0, 1743465600000.0),
    "2026-04": (1743465600000.0, 1746057600000.0),
    "2026-05": (1746057600000.0, 1748736000000.0),
}

NEEDED_MONTHS = {"2026-01", "2026-02", "2026-03", "2026-04", "2026-05"}


def main():
    # 当前环境为 CUDA GPU，显式指定 cuda:0
    device = "cuda:0"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, dataset, cfg = load_backbone(CONFIG_PATH, CKPT_PATH, device=device)

    all_patches = dataset.patches
    print(f"Loaded {len(all_patches)} patches")

    meta_records = []
    for pid in tqdm(all_patches, desc="Patches"):
        pidx = dataset.patches.index(pid)
        for month, (ws, we) in MONTHLY_WINDOWS.items():
            if month not in NEEDED_MONTHS:
                continue
            out_path = OUTPUT_DIR / f"{pid}_{month}.npy"
            if out_path.exists():
                continue
            try:
                emb = extract_embedding_map(model, dataset, pidx, ws, we, device, normalize=True)
                np.save(out_path, emb)
                meta_records.append({"patch_id": pid, "month": month, "shape": emb.shape})
            except Exception as e:
                print(f"\n  Skip {pid} {month}: {e}")

    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"\nDone. Saved {len(meta_records)} embeddings to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
