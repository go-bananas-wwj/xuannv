#!/usr/bin/env python3
"""
预提取 V2 模型在 2025 年各月份的 embedding，用于月度变化检测 Head 训练.

月份窗口:
  2025-04: Apr 1  ~ Apr 30
  2025-05: May 1  ~ May 31 (备用)
  2025-06: Jun 1  ~ Jun 30
  2025-07: Jul 1  ~ Jul 31 (备用)
  2025-08: Aug 1  ~ Aug 31
  2025-09: Sep 1  ~ Sep 30
  2025-10: Oct 1  ~ Oct 31

输出:
  /workspace/outputs/aef_qwen_v2/monthly_embeddings_2025/{patch_id}_{month}.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
from tqdm import tqdm

from demo_v2.utils.harbin_annotations_v2 import get_annotated_patches
from src.inference.engine import extract_embedding_map, load_backbone
from src.utils.device import get_device

CONFIG_PATH = "configs/qwen_v1_scenes.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/epoch_499.pt"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2/monthly_embeddings_2025")

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
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, dataset, _ = load_backbone(CONFIG_PATH, CKPT_PATH, device=device)

    annotated_patches = set(get_annotated_patches())
    needed_patches = [pid for pid in dataset.patches if pid in annotated_patches]
    print(f"Loaded {len(dataset.patches)} patches, {len(needed_patches)} annotated")

    meta_records = []
    for pid in tqdm(needed_patches, desc="Patches"):
        pidx = dataset.patches.index(pid)
        for month, (ws, we) in MONTHLY_WINDOWS.items():
            if month not in NEEDED_MONTHS:
                continue
            out_path = OUTPUT_DIR / f"{pid}_{month}.npy"
            if out_path.exists():
                continue
            try:
                emb = extract_embedding_map(model, dataset, pidx, ws, we, device)
                np.save(out_path, emb)
                meta_records.append({"patch_id": pid, "month": month, "shape": emb.shape})
            except Exception as e:
                print(f"\n  Skip {pid} {month}: {e}")

    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"\nDone. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
