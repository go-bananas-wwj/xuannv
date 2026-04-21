#!/usr/bin/env python3
"""
评估 V5 月度 Embedding + ChangeDetectionHead 的 patch-level AUC.
与 V4 benchmark 对齐评估协议.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
)
from src.models.heads import ChangeDetectionHeadV3

EMBEDDING_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025")
HEAD_PATH = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/monthly_cd_head_v3.pt")

PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
    "2025-all": ("2025-04", "2025-10"),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_head():
    head = ChangeDetectionHeadV3(embedding_dim=128, hidden_dim=64).to(DEVICE)
    ckpt = torch.load(HEAD_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["cd_head"])
    head.eval()
    return head


def evaluate_patch_level(head: torch.nn.Module) -> dict:
    annotated = get_annotated_patches()
    results = []
    raw_results = []

    with torch.no_grad():
        for pid in annotated:
            period = get_period_for_patch(pid)
            if period is None or period not in PERIOD_TO_MONTHS:
                continue
            bm, am = PERIOD_TO_MONTHS[period]
            bpath = EMBEDDING_DIR / f"{pid}_{bm}.npy"
            apath = EMBEDDING_DIR / f"{pid}_{am}.npy"
            if not bpath.exists() or not apath.exists():
                continue

            emb_b = np.load(bpath)  # [D, H, W]
            emb_a = np.load(apath)
            mask, _ = rasterize_patch_changes(pid, grid_size=64)
            if mask.sum() == 0:
                continue

            # Head prediction
            eb = torch.from_numpy(emb_b).unsqueeze(0).float().to(DEVICE)
            ea = torch.from_numpy(emb_a).unsqueeze(0).float().to(DEVICE)
            logits = head(eb, ea).squeeze().cpu().numpy()  # [H, W]
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Raw cosine distance
            D, H, W = emb_b.shape
            fb = emb_b.reshape(D, -1)
            fa = emb_a.reshape(D, -1)
            fb = fb / np.linalg.norm(fb, axis=0, keepdims=True)
            fa = fa / np.linalg.norm(fa, axis=0, keepdims=True)
            cos_sim = np.sum(fb * fa, axis=0)
            raw_dist = ((1.0 - cos_sim) / 2.0).reshape(H, W)

            flat_mask = mask.flatten()
            flat_probs = probs.flatten()
            flat_raw = raw_dist.flatten()

            if len(np.unique(flat_mask)) < 2:
                continue

            auc_head = roc_auc_score(flat_mask, flat_probs)
            auc_raw = roc_auc_score(flat_mask, flat_raw)
            ba_head = balanced_accuracy_score(flat_mask, (flat_probs > 0.5).astype(int))
            f1_head = f1_score(flat_mask, (flat_probs > 0.5).astype(int), zero_division=0)

            results.append({
                "patch_id": pid,
                "period": period,
                "auc": auc_head,
                "ba": ba_head,
                "f1": f1_head,
            })
            raw_results.append({
                "patch_id": pid,
                "period": period,
                "auc": auc_raw,
                "ba": 0.5,
                "f1": 0.0,
            })

    def summarize(records):
        aucs = [r["auc"] for r in records]
        return {
            "auc_mean": float(np.mean(aucs)),
            "auc_median": float(np.median(aucs)),
            "auc_std": float(np.std(aucs)),
            "n": len(aucs),
        }

    return {
        "head": summarize(results),
        "raw": summarize(raw_results),
        "records": results,
        "raw_records": raw_results,
    }


def main():
    print("=" * 60)
    print("  V5 月度 CD Head Patch-Level Benchmark")
    print("=" * 60)

    if not HEAD_PATH.exists():
        print(f"❌ Head not found: {HEAD_PATH}")
        print("Please run train_v5_monthly_cd_head.py first.")
        return

    head = load_head()
    result = evaluate_patch_level(head)

    head_sum = result["head"]
    raw_sum = result["raw"]

    print(f"\nEvaluated patches: {head_sum['n']}")
    print(f"Raw  cosine | AUC mean={raw_sum['auc_mean']:.4f} median={raw_sum['auc_median']:.4f} std={raw_sum['auc_std']:.4f}")
    print(f"Head predict | AUC mean={head_sum['auc_mean']:.4f} median={head_sum['auc_median']:.4f} std={head_sum['auc_std']:.4f}")

    improved = sum(1 for h, r in zip(result["records"], result["raw_records"]) if h["auc"] > r["auc"])
    print(f"Improved patches: {improved}/{head_sum['n']} ({improved/head_sum['n']*100:.1f}%)")

    out_path = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval/benchmark_full69_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
