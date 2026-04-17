#!/usr/bin/env python3
"""快速预计算 V3 的核心组合，用于 Demo 即时体验."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_v2.cache_manager import cache
from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.utils.constants import TIME_WINDOWS

PRECOMPUTE_BASE = Path(__file__).resolve().parent / "precomputed_cd"
PRECOMPUTE_BASE.mkdir(parents=True, exist_ok=True)


def main():
    cache.load()
    version = "v3"
    pair_name = "2023 Q3-Q4 vs 2024 Q3-Q4"
    before_key = "2023 Q3-Q4"
    after_key = "2024 Q3-Q4"

    version_dir = PRECOMPUTE_BASE / version
    version_dir.mkdir(parents=True, exist_ok=True)

    safe_name = pair_name.replace(" ", "_").replace("-", "_")
    out_path = version_dir / f"{safe_name}.npz"

    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    print(f"Computing {pair_name} for {version} ...")
    engine = ChangeDetectionEngine(version)
    before = TIME_WINDOWS[before_key]
    after = TIME_WINDOWS[after_key]

    canvas, msg = engine.compute_global_change_map(
        before, after, max_patches=None, use_precomputed=False
    )
    if canvas is None:
        print(f"Failed: {msg}")
        return

    import re
    m = re.search(r"计算 patch 数\s*\|\s*(\d+)", msg)
    computed_patches = int(m.group(1)) if m else 0

    import numpy as np
    np.savez_compressed(
        out_path,
        change_scores=canvas.astype(np.float32),
        computed_patches=computed_patches,
        mean_score=float(canvas.mean()),
        max_score=float(canvas.max()),
        pair_name=pair_name,
        before_key=before_key,
        after_key=after_key,
    )
    print(f"Saved: {out_path} shape={canvas.shape}")


if __name__ == "__main__":
    main()
