#!/usr/bin/env python3
"""全量 annotated patch 基准测试：Raw Cosine vs TaskHead CD.

用法:
    CUDA_VISIBLE_DEVICES=2 python scripts/benchmark_all_annotated.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, "/workspace/xuannv")

from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.engines.task_head_engine import TaskHeadEngine
from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
    PERIODS,
)

VERSION = "v2"
DEVICE = "cuda:0"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2_taskheads/benchmark_all")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_patch(patch_id: str, engine: ChangeDetectionEngine, task_engine: TaskHeadEngine):
    """评估单个 patch 的 Raw vs TaskHead."""
    period = get_period_for_patch(patch_id)
    if period is None or period not in PERIODS:
        return None
    bs, be = PERIODS[period]
    mid = (bs + be) / 2.0

    gt_mask, _ = rasterize_patch_changes(patch_id, grid_size=64)
    labels = gt_mask.flatten()

    score_raw = engine.compute_change_score(patch_id, (bs, mid), (mid, be), use_precomputed=True, use_task_head=False)
    score_head = engine.compute_change_score(patch_id, (bs, mid), (mid, be), use_precomputed=True, use_task_head=True)

    if score_raw is None or score_head is None:
        return None

    probs_raw = score_raw.flatten()
    probs_head = score_head.flatten()

    def _metrics(probs, labels):
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5
        preds = (probs > 0.5).astype(int)
        ba = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        return {"auc": auc, "ba": ba, "f1": f1}

    return {
        "patch_id": patch_id,
        "period": period,
        "raw": _metrics(probs_raw, labels),
        "head": _metrics(probs_head, labels),
    }


def save_comparison_figure(patch_id: str, period: str, score_raw, score_head, gt_mask, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = max(score_head.max(), score_raw.max(), 0.001)

    ax = axes[0]
    im = ax.imshow(score_raw, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title("Raw Cosine Distance", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im = ax.imshow(score_head, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title("TaskHead CD", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    rgb[gt_mask == 1] = [255, 0, 0]
    rgb[gt_mask == 0] = [20, 20, 20]
    ax.imshow(rgb)
    ax.set_title("GT Mask", fontsize=12)
    ax.axis("off")

    fig.suptitle(f"{patch_id} — {period}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"{patch_id}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    engine = ChangeDetectionEngine(VERSION, device=DEVICE)
    task_engine = TaskHeadEngine.get_instance(DEVICE)
    patches = get_annotated_patches()
    print(f"[Benchmark] Evaluating {len(patches)} annotated patches...")

    records = []
    visuals_dir = OUTPUT_DIR / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    for pid in tqdm(patches, desc="Benchmark"):
        res = evaluate_patch(pid, engine, task_engine)
        if res is None:
            continue
        records.append(res)

        # save figure for every patch
        period = res["period"]
        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0
        gt_mask, _ = rasterize_patch_changes(pid, grid_size=64)
        score_raw = engine.compute_change_score(pid, (bs, mid), (mid, be), use_precomputed=True, use_task_head=False)
        score_head = engine.compute_change_score(pid, (bs, mid), (mid, be), use_precomputed=True, use_task_head=True)
        if score_raw is not None and score_head is not None:
            save_comparison_figure(pid, period, score_raw, score_head, gt_mask, visuals_dir)

    # aggregate stats
    raw_aucs = [r["raw"]["auc"] for r in records]
    head_aucs = [r["head"]["auc"] for r in records]
    raw_bas = [r["raw"]["ba"] for r in records]
    head_bas = [r["head"]["ba"] for r in records]

    summary = {
        "n_patches": len(records),
        "raw": {
            "auc_mean": float(np.mean(raw_aucs)),
            "auc_std": float(np.std(raw_aucs)),
            "auc_median": float(np.median(raw_aucs)),
            "auc_min": float(np.min(raw_aucs)),
            "auc_max": float(np.max(raw_aucs)),
            "ba_mean": float(np.mean(raw_bas)),
        },
        "head": {
            "auc_mean": float(np.mean(head_aucs)),
            "auc_std": float(np.std(head_aucs)),
            "auc_median": float(np.median(head_aucs)),
            "auc_min": float(np.min(head_aucs)),
            "auc_max": float(np.max(head_aucs)),
            "ba_mean": float(np.mean(head_bas)),
        },
        "improved_patches": int(sum(h > r for h, r in zip(head_aucs, raw_aucs))),
        "records": records,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Benchmark] Summary:")
    print(f"  Evaluated patches: {summary['n_patches']}")
    print(f"  Raw  AUC: mean={summary['raw']['auc_mean']:.4f} std={summary['raw']['auc_std']:.4f} median={summary['raw']['auc_median']:.4f}")
    print(f"  Head AUC: mean={summary['head']['auc_mean']:.4f} std={summary['head']['auc_std']:.4f} median={summary['head']['auc_median']:.4f}")
    print(f"  Improved patches: {summary['improved_patches']} / {summary['n_patches']}")
    print(f"  Visuals saved to: {visuals_dir}")


if __name__ == "__main__":
    main()
