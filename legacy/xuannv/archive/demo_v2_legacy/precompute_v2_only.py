#!/usr/bin/env python3
"""预计算 V2 常用时间窗口组合的全域变化检测数据。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_v2.cache_manager import cache
from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.utils.constants import TIME_WINDOWS

PRECOMPUTE_DIR = Path(__file__).resolve().parent / "precomputed_cd"
PRECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)

COMMON_PAIRS = [
    ("2023 Full Year vs 2024 Full Year", "2023 全年", "2024 全年"),
    ("2024 Full Year vs 2025 Full Year", "2024 全年", "2025 全年"),
    ("2023 Q3-Q4 vs 2024 Q3-Q4", "2023 Q3-Q4", "2024 Q3-Q4"),
    ("2024 Q3-Q4 vs 2025 Q3-Q4", "2024 Q3-Q4", "2025 Q3-Q4"),
    ("2023-10 vs 2024-10", "2023-10", "2024-10"),
    ("2024-10 vs 2025-10", "2024-10", "2025-10"),
]


def _safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def main():
    cache.load()
    version = "v2"
    version_dir = PRECOMPUTE_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Precompute] Version: {version}")

    engine = ChangeDetectionEngine(version)

    for pair_name, before_key, after_key in COMMON_PAIRS:
        before = TIME_WINDOWS.get(before_key)
        after = TIME_WINDOWS.get(after_key)
        if before is None or after is None:
            continue

        out_path = version_dir / f"{_safe_filename(pair_name)}.npz"
        if out_path.exists():
            print(f"  Skip existing: {pair_name}")
            continue

        print(f"  Computing: {pair_name} ...")
        canvas, msg = engine.compute_global_change_map(
            before, after, max_patches=None, use_precomputed=False
        )
        if canvas is None:
            print(f"    Failed: {msg}")
            continue

        valid_mask = canvas > 0
        mean_score = float(canvas.mean())
        max_score = float(canvas.max())
        computed_patches = int(valid_mask.sum() > 0)
        m = re.search(r"计算 patch 数\s*\|\s*(\d+)", msg)
        if m:
            computed_patches = int(m.group(1))

        np.savez_compressed(
            out_path,
            change_scores=canvas.astype(np.float32),
            computed_patches=computed_patches,
            mean_score=mean_score,
            max_score=max_score,
            pair_name=pair_name,
            before_key=before_key,
            after_key=after_key,
        )
        print(f"    Saved: {out_path} shape={canvas.shape} mean={mean_score:.4f} max={max_score:.4f}")

    print("\n[Precompute] V2 done.")


if __name__ == "__main__":
    main()
