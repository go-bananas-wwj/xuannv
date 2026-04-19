#!/usr/bin/env python3
"""
V4 vs V2 变化检测对比可视化.
对同一批 patch 并排显示 V2 和 V4 的 Head 预测结果，直观展示改进.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom

sys.path.insert(0, "/workspace/xuannv")

from demo_v2.engines.patch_image_loader import load_patch_source_rgb
from demo_v2.utils.constants import TIME_WINDOWS
from demo_v2.utils.harbin_annotations_v2 import (
    rasterize_patch_changes,
    rasterize_patch_categories,
    get_annotated_patches,
    get_period_for_patch,
)

CATEGORY_NAMES = {
    0: "unchanged",
    1: "construction",
    2: "demolition",
    3: "road",
    4: "water_change",
    5: "farmland",
}
from src.models.heads import ChangeDetectionHeadV3

# V2 paths
V2_EMB_DIR = Path("/workspace/outputs/aef_qwen_v2/monthly_embeddings_2025")
V2_HEAD_PATH = Path("/workspace/outputs/aef_qwen_v2/monthly_cd_head/best_cv_fold0_v3_ohem_head.pt")

# V4 paths
V4_EMB_DIR = Path("/workspace/outputs/aef_qwen_v4_official/monthly_embeddings_2025")
V4_HEAD_PATH = Path("/workspace/outputs/aef_qwen_v4_official/monthly_cd_head/monthly_cd_head_v3_ohem.pt")

OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v4_official/visualization/05_v4_vs_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}


def load_head(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    head = ChangeDetectionHeadV3(
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.3),
    )
    head.load_state_dict(state["cd_head"])
    head.eval()
    return head


def compute_prediction(head, emb_b: np.ndarray, emb_a: np.ndarray):
    with torch.no_grad():
        eb = torch.from_numpy(emb_b).unsqueeze(0).float()
        ea = torch.from_numpy(emb_a).unsqueeze(0).float()
        logits = head(eb, ea).squeeze(1)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()
    return pred


def compute_raw_score(emb_b: np.ndarray, emb_a: np.ndarray):
    D, H, W = emb_b.shape
    fb = emb_b.reshape(D, -1)
    fa = emb_a.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    return ((1.0 - cos_sim) / 2.0).reshape(H, W)


def load_s2_rgb(patch_id: str, month: str):
    window = TIME_WINDOWS.get(month)
    if window is None:
        return None
    return load_patch_source_rgb(patch_id, "s2", window)


def plot_v4_vs_v2_patch(pid: str, v2_head, v4_head, out_dir: Path):
    """为单个 patch 生成 V4 vs V2 对比图 (1x6 子图)."""
    period = get_period_for_patch(pid)
    if period is None or period not in PERIOD_TO_MONTHS:
        return False
    bm, am = PERIOD_TO_MONTHS[period]

    # Load embeddings
    v2_b = np.load(V2_EMB_DIR / f"{pid}_{bm}.npy") if (V2_EMB_DIR / f"{pid}_{bm}.npy").exists() else None
    v2_a = np.load(V2_EMB_DIR / f"{pid}_{am}.npy") if (V2_EMB_DIR / f"{pid}_{am}.npy").exists() else None
    v4_b = np.load(V4_EMB_DIR / f"{pid}_{bm}.npy") if (V4_EMB_DIR / f"{pid}_{bm}.npy").exists() else None
    v4_a = np.load(V4_EMB_DIR / f"{pid}_{am}.npy") if (V4_EMB_DIR / f"{pid}_{am}.npy").exists() else None

    if v2_b is None or v2_a is None or v4_b is None or v4_a is None:
        return False

    # Compute predictions
    v2_pred = compute_prediction(v2_head, v2_b, v2_a)
    v4_pred = compute_prediction(v4_head, v4_b, v4_a)
    v2_raw = compute_raw_score(v2_b, v2_a)
    v4_raw = compute_raw_score(v4_b, v4_a)

    # Load GT mask
    mask_bin, _ = rasterize_patch_changes(pid, grid_size=64)

    # Load S2 RGB
    s2_b = load_s2_rgb(pid, bm)
    s2_a = load_s2_rgb(pid, am)
    if s2_b is None:
        s2_b = np.zeros((512, 512, 3), dtype=np.float32)
    if s2_a is None:
        s2_a = np.zeros((512, 512, 3), dtype=np.float32)

    # AUCs
    from sklearn.metrics import roc_auc_score
    flat_mask = mask_bin.flatten()
    v2_auc = roc_auc_score(flat_mask, v2_pred.flatten()) if len(np.unique(flat_mask)) > 1 else 0.5
    v4_auc = roc_auc_score(flat_mask, v4_pred.flatten()) if len(np.unique(flat_mask)) > 1 else 0.5
    v2_raw_auc = roc_auc_score(flat_mask, v2_raw.flatten()) if len(np.unique(flat_mask)) > 1 else 0.5
    v4_raw_auc = roc_auc_score(flat_mask, v4_raw.flatten()) if len(np.unique(flat_mask)) > 1 else 0.5

    # Plot
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))

    titles = [
        "Before S2", "After S2",
        f"V2 Raw\nAUC={v2_raw_auc:.3f}", f"V4 Raw\nAUC={v4_raw_auc:.3f}",
        f"V2 Head\nAUC={v2_auc:.3f}", f"V4 Head\nAUC={v4_auc:.3f}",
    ]
    imgs = [s2_b, s2_a, v2_raw, v4_raw, v2_pred, v4_pred]

    for j, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        if j < 2:
            ax.imshow(img, interpolation="bilinear")
        elif j == 2 or j == 3:
            pmin, pmax = img.min(), img.max()
            norm_img = (img - pmin) / (pmax - pmin + 1e-8) if pmax > pmin else img
            ax.imshow(norm_img, cmap="coolwarm", vmin=0, vmax=1, interpolation="bilinear")
        else:
            pmin, pmax = img.min(), img.max()
            norm_img = (img - pmin) / (pmax - pmin + 1e-8) if pmax > pmin else img
            im = ax.imshow(norm_img, cmap="hot", vmin=0, vmax=1, interpolation="bilinear")
            if j == 5:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")

    # Add GT mask contour on Head pred panels
    for j in [4, 5]:
        h, w = imgs[j].shape
        mask_hr = zoom(mask_bin.astype(float), (h / 64.0, w / 64.0), order=1)
        axes[j].contour(mask_hr, levels=[0.5], colors='lime', linewidths=1.5)

    delta = v4_auc - v2_auc
    delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    fig.suptitle(f"{pid} | V4 vs V2 | Delta={delta_str}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / f"{pid}_v4_vs_v2.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return True


def main():
    print("=" * 60)
    print("  V4 vs V2 Change Detection Comparison")
    print("=" * 60)

    print("\nLoading V2 head...")
    v2_head = load_head(V2_HEAD_PATH)
    print("Loading V4 head...")
    v4_head = load_head(V4_HEAD_PATH)

    # Select representative patches: best, median, worst improvements
    annotated = get_annotated_patches()
    results = []
    for pid in annotated:
        period = get_period_for_patch(pid)
        if period is None or period not in PERIOD_TO_MONTHS:
            continue
        bm, am = PERIOD_TO_MONTHS[period]
        v2_b_path = V2_EMB_DIR / f"{pid}_{bm}.npy"
        v2_a_path = V2_EMB_DIR / f"{pid}_{am}.npy"
        v4_b_path = V4_EMB_DIR / f"{pid}_{bm}.npy"
        v4_a_path = V4_EMB_DIR / f"{pid}_{am}.npy"
        if not all(p.exists() for p in [v2_b_path, v2_a_path, v4_b_path, v4_a_path]):
            continue
        results.append(pid)

    print(f"\nGenerating comparison figures for {len(results)} patches...")
    generated = 0
    for pid in results:
        ok = plot_v4_vs_v2_patch(pid, v2_head, v4_head, OUTPUT_DIR)
        if ok:
            generated += 1

    print(f"\nDone. Generated {generated} comparison figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
