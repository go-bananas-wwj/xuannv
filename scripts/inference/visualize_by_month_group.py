#!/usr/bin/env python3
"""按月份对分组生成变化检测可视化大图，每月份对独立一张图.

输出:
  - month_group_01_to_02.png  (Jan→Feb)
  - month_group_02_to_03.png  (Feb→Mar)
  - month_group_03_to_04.png  (Mar→Apr)
  - month_group_04_to_05.png  (Apr→May)
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
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/cd_head_visualization_2026")

CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/monthly_cd_head_v5_final.pt"

MONTH_PAIRS = [
    ("2026-01", "2026-02", "Jan → Feb"),
    ("2026-02", "2026-03", "Feb → Mar"),
    ("2026-03", "2026-04", "Mar → Apr"),
    ("2026-04", "2026-05", "Apr → May"),
]

CD_THRESHOLD = 0.5
TOP_N = 16  # 每图展示 top 16 个 patch

CHANGE_CMAP = LinearSegmentedColormap.from_list("change", ["#2166ac", "#f7f7f7", "#b2182b"])
BINARY_CMAP = LinearSegmentedColormap.from_list("binary", ["#ffffff", "#b2182b"])


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


def plot_month_group(head, m1, m2, label, device, out_path, top_n=16):
    """为单个月份对生成大图，展示 top N 个 patch."""
    all_patches = sorted(set(p.name.split("_2026")[0] for p in EMB_DIR.glob("patch_*.npy")))
    
    # 计算所有 patches 的分数
    results = []
    for pid in all_patches:
        emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
        emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
        if not emb1_path.exists() or not emb2_path.exists():
            continue
        emb1 = np.load(emb1_path)
        emb2 = np.load(emb2_path)
        cd_probs = compute_cd_probs(head, emb1, emb2, device)
        binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
        results.append({
            "pid": pid,
            "cd_max": float(cd_probs.max()),
            "cd_mean": float(cd_probs.mean()),
            "binary_rate": float(binary.mean()),
        })
    
    results.sort(key=lambda r: r["cd_max"], reverse=True)
    top = results[:top_n]
    
    # 布局: 4 列 × N/4 行
    n_cols = 4
    n_rows = (len(top) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 4, n_rows * 7))
    if n_rows == 1:
        axes = axes.reshape(2, n_cols)
    
    fig.suptitle(f"Harbin 2026 — {label} Change Detection (Top {len(top)} patches)\n"
                 f"CD Head: V5 Final (AUC=0.916) | Threshold={CD_THRESHOLD}",
                 fontsize=16, fontweight="bold", y=1.01)
    
    for idx, combo in enumerate(top):
        row = idx // n_cols
        col = idx % n_cols
        pid = combo["pid"]
        
        emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
        emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
        
        rgb_before = read_rgb_for_patch_month(pid, m1)
        rgb_after = read_rgb_for_patch_month(pid, m2)
        
        emb1 = np.load(emb1_path)
        emb2 = np.load(emb2_path)
        cd_probs = compute_cd_probs(head, emb1, emb2, device)
        binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
        
        ax_before = axes[row * 2, col]
        ax_after = axes[row * 2 + 1, col]
        
        # 合并显示 Before + After + CD Prob + Binary 在一个 subplot 内
        ax_before.axis("off")
        ax_after.axis("off")
        
        # 使用 inset axes 在单个大单元格内放 4 个小图
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Before RGB (左上)
        ax_b = inset_axes(ax_before, width="48%", height="48%", loc="upper left")
        if rgb_before is not None:
            ax_b.imshow(rgb_before)
        ax_b.set_title(f"{pid}\n{label}", fontsize=9, fontweight="bold", pad=2)
        ax_b.axis("off")
        
        # After RGB (右上)
        ax_a = inset_axes(ax_before, width="48%", height="48%", loc="upper right")
        if rgb_after is not None:
            ax_a.imshow(rgb_after)
        ax_a.set_title(f"After {m2}", fontsize=9, pad=2)
        ax_a.axis("off")
        
        # CD Prob (左下)
        ax_p = inset_axes(ax_after, width="48%", height="48%", loc="upper left")
        im = ax_p.imshow(cd_probs, cmap=CHANGE_CMAP, vmin=0, vmax=1)
        ax_p.set_title(f"CD max={combo['cd_max']:.2f}\nmean={combo['cd_mean']:.3f}", fontsize=8, pad=2)
        ax_p.axis("off")
        
        # Binary (右下)
        ax_bin = inset_axes(ax_after, width="48%", height="48%", loc="upper right")
        ax_bin.imshow(binary, cmap=BINARY_CMAP, vmin=0, vmax=1)
        ax_bin.set_title(f"Binary rate={combo['binary_rate']:.3f}", fontsize=8, pad=2)
        ax_bin.axis("off")
    
    # 隐藏多余的 axes
    for idx in range(len(top), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row * 2, col].axis("off")
        axes[row * 2 + 1, col].axis("off")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {label}: {out_path}")


def main():
    device = get_device(device_str="cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    head = load_cd_head(CKPT_PATH, device)
    
    for m1, m2, label in MONTH_PAIRS:
        print(f"\nGenerating {label}...")
        plot_month_group(
            head, m1, m2, label, device,
            OUTPUT_DIR / f"month_group_{m1}_to_{m2}.png",
            top_n=TOP_N
        )
    
    print(f"\nAll done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
