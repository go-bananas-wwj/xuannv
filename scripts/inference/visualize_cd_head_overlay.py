#!/usr/bin/env python3
"""CD Head 变化检测可视化 — Mask 叠加到 RGB 上，标注月份，生成大量例子.

输出:
  per_patch_overlay/ — 每个 patch 一张图，4 行(月度对) × 3 列(Before+Mask / After+Mask / CD Prob)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
import rasterio

from src.models.heads import ChangeDetectionHeadV3
from src.utils.device import get_device

EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")
DATA_ROOT = Path("/workspace/raw/harbin_scenes")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/cd_head_visualization_2026/per_patch_overlay")

CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/monthly_cd_head_v5_final.pt"

MONTH_PAIRS = [
    ("2026-01", "2026-02", "Jan → Feb"),
    ("2026-02", "2026-03", "Feb → Mar"),
    ("2026-03", "2026-04", "Mar → Apr"),
    ("2026-04", "2026-05", "Apr → May"),
]

CD_THRESHOLD = 0.5
TOP_N_PATCHES = 40

CHANGE_CMAP = LinearSegmentedColormap.from_list("change", ["#2166ac", "#f7f7f7", "#b2182b"])


def load_cd_head(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    head = ChangeDetectionHeadV3(
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.3),
    )
    head.load_state_dict(ckpt["cd_head"])
    head.to(device)
    head.eval()
    return head


def compute_cd_probs(head, emb_before, emb_after, device):
    with torch.no_grad():
        eb = torch.from_numpy(emb_before).unsqueeze(0).float().to(device)
        ea = torch.from_numpy(emb_after).unsqueeze(0).float().to(device)
        logits = head(eb, ea)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def enhance_rgb(rgb):
    rgb = rgb.copy()
    for c in range(3):
        band = rgb[..., c]
        p2, p98 = np.percentile(band, [2, 98])
        rgb[..., c] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    rgb = np.power(rgb, 0.8)
    return np.clip(rgb, 0, 1)


def read_rgb_for_patch_month(patch_id, month, target_size=(64, 64)):
    month_prefix = month.replace("-", "")
    
    s2_dir = DATA_ROOT / "s2" / patch_id
    if s2_dir.exists():
        files = sorted([f for f in s2_dir.glob("*.tif") if f.stem.startswith(month_prefix)])
        if files:
            with rasterio.open(files[0]) as src:
                data = src.read()
                if data.shape[0] >= 3:
                    rgb = np.stack([data[2], data[1], data[0]], axis=0)
                    rgb = _resize_rgb(rgb, target_size)
                    return enhance_rgb(rgb)
    
    landsat_dir = DATA_ROOT / "landsat" / patch_id
    if landsat_dir.exists():
        files = sorted([f for f in landsat_dir.glob("*.tif") if f.stem.startswith(month_prefix)])
        if files:
            with rasterio.open(files[0]) as src:
                data = src.read()
                if data.shape[0] >= 3:
                    rgb = np.stack([data[0], data[1], data[2]], axis=0)
                    rgb = _resize_rgb(rgb, target_size)
                    return enhance_rgb(rgb)
    
    s1_dir = DATA_ROOT / "s1" / patch_id
    if s1_dir.exists():
        files = sorted([f for f in s1_dir.glob("*.tif") if f.stem.startswith(month_prefix)])
        if files:
            with rasterio.open(files[0]) as src:
                data = src.read()
                if data.shape[0] >= 1:
                    gray = data[0:1]
                    vmin, vmax = np.percentile(gray, [2, 98])
                    gray = np.clip((gray - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    rgb = np.repeat(gray, 3, axis=0)
                    rgb = _resize_rgb(rgb, target_size)
                    return rgb
    return None


def _resize_rgb(rgb, target_size):
    tensor = torch.from_numpy(rgb).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).numpy()
    resized = np.transpose(resized, (1, 2, 0))
    return np.clip(resized, 0, 1)


def overlay_mask_on_rgb(rgb, binary_mask, alpha=0.4, color=(0.7, 0.1, 0.1)):
    """将二值 mask 以半透明红色叠加到 RGB 图像上."""
    overlay = rgb.copy()
    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    color_arr = np.array(color).reshape(1, 1, 3)
    overlay = np.where(mask_3ch > 0, rgb * (1 - alpha) + color_arr * alpha, rgb)
    return np.clip(overlay, 0, 1)


def select_top_patches(head, device, n=40):
    """选择 top N 个变化最显著的 patches（基于所有月份对的平均 cd_max）."""
    all_patches = sorted(set(p.name.split("_2026")[0] for p in EMB_DIR.glob("patch_*.npy")))
    
    patch_scores = {}
    for pid in all_patches:
        max_scores = []
        for m1, m2, _ in MONTH_PAIRS:
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            if not emb1_path.exists() or not emb2_path.exists():
                continue
            emb1 = np.load(emb1_path)
            emb2 = np.load(emb2_path)
            cd_probs = compute_cd_probs(head, emb1, emb2, device)
            max_scores.append(float(cd_probs.max()))
        if max_scores:
            patch_scores[pid] = np.mean(max_scores)
    
    sorted_patches = sorted(patch_scores.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_patches[:n]]


def plot_patch_overlay(head, pid, device, out_path):
    """为单个 patch 生成叠加 Mask 的对比图."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 18))
    
    col_titles = ["Before + Change Mask", "After + Change Mask", "CD Probability"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold")
    
    for row_idx, (m1, m2, label) in enumerate(MONTH_PAIRS):
        emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
        emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
        
        rgb_before = read_rgb_for_patch_month(pid, m1)
        rgb_after = read_rgb_for_patch_month(pid, m2)
        
        if emb1_path.exists() and emb2_path.exists():
            emb1 = np.load(emb1_path)
            emb2 = np.load(emb2_path)
            cd_probs = compute_cd_probs(head, emb1, emb2, device)
            binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
        else:
            cd_probs = None
            binary = None
        
        # 1. Before + Mask Overlay
        ax = axes[row_idx, 0]
        if rgb_before is not None and binary is not None:
            overlay = overlay_mask_on_rgb(rgb_before, binary)
            ax.imshow(overlay)
        ax.set_ylabel(f"{label}", fontsize=11, fontweight="bold", rotation=0, ha="right", va="center")
        ax.tick_params(labelleft=False)
        ax.axis("off")
        
        # 2. After + Mask Overlay
        ax = axes[row_idx, 1]
        if rgb_after is not None and binary is not None:
            overlay = overlay_mask_on_rgb(rgb_after, binary)
            ax.imshow(overlay)
        ax.axis("off")
        
        # 3. CD Prob
        ax = axes[row_idx, 2]
        if cd_probs is not None:
            im = ax.imshow(cd_probs, cmap=CHANGE_CMAP, vmin=0, vmax=1)
            ax.set_title(f"max={cd_probs.max():.2f}, mean={cd_probs.mean():.3f}, rate={binary.mean():.3f}", fontsize=9)
        ax.axis("off")
    
    fig.suptitle(f"{pid} — Change Detection with Mask Overlay (threshold={CD_THRESHOLD})", 
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device(device_str="cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    head = load_cd_head(CKPT_PATH, device)
    
    top_patches = select_top_patches(head, device, TOP_N_PATCHES)
    print(f"Selected top {len(top_patches)} patches: {top_patches}")
    
    for pid in top_patches:
        out_path = OUTPUT_DIR / f"{pid}_overlay.png"
        plot_patch_overlay(head, pid, device, out_path)
        print(f"  Saved: {out_path.name}")
    
    print(f"\nAll done. {len(top_patches)} patches saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
