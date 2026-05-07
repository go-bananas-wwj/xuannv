#!/usr/bin/env python3
"""测试 TaskHead 与 ChangeDetectionEngine 集成效果."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/xuannv")

from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.utils.harbin_annotations_v2 import get_annotated_patches, get_period_for_patch, PERIODS
from demo_v2.utils.visualization import change_heatmap_fig, fig_to_pil

VERSION = "v2"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2_taskheads/test_visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_comparison(patch_id: str, before_window, after_window, gt_mask, engine: ChangeDetectionEngine):
    """对比原始 cosine distance vs task head 输出."""
    # 原始方法
    score_raw = engine.compute_change_score(
        patch_id, before_window, after_window, use_precomputed=True, use_task_head=False
    )
    # Task head 方法
    score_head = engine.compute_change_score(
        patch_id, before_window, after_window, use_precomputed=True, use_task_head=True
    )
    if score_raw is None or score_head is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = max(score_head.max(), score_raw.max(), 0.001)

    ax = axes[0]
    im = ax.imshow(score_raw, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title("Raw Cosine Distance", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im = ax.imshow(score_head, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title("TaskHead Output", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    rgb[gt_mask == 1] = [255, 0, 0]
    rgb[gt_mask == 0] = [20, 20, 20]
    ax.imshow(rgb)
    ax.set_title("GT Mask", fontsize=12)
    ax.axis("off")

    fig.suptitle(f"{patch_id} — {get_period_for_patch(patch_id)}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{patch_id}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    engine = ChangeDetectionEngine(VERSION, device="npu:0")
    patches = get_annotated_patches()[:3]  # 测试前 3 个

    for pid in patches:
        period = get_period_for_patch(pid)
        if period is None or period not in PERIODS:
            continue
        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0

        from demo_v2.utils.harbin_annotations_v2 import rasterize_patch_changes
        gt_mask, _ = rasterize_patch_changes(pid, grid_size=64)
        save_comparison(pid, (bs, mid), (mid, be), gt_mask, engine)

    print(f"\nAll visuals saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
