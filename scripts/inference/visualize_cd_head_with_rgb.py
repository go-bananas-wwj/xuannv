#!/usr/bin/env python3
"""CD Head 变化检测可视化 — 带 RGB 真彩色合成图，方便人工审查.

输出:
  1. cd_head_rgb_comparison.png — 8 个 top patches 的最显著变化对比（每行 4 列）
  2. per_patch/ — 每个 patch 的详细月度对比图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
import rasterio

from src.models.heads import ChangeDetectionHeadV3
from src.utils.device import get_device

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")
DATA_ROOT = Path("/workspace/raw/harbin_scenes")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/cd_head_visualization_2026")

CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/monthly_cd_head_v5_final.pt"

MONTH_PAIRS = [
    ("2026-01", "2026-02"),
    ("2026-02", "2026-03"),
    ("2026-03", "2026-04"),
    ("2026-04", "2026-05"),
]

MONTH_PAIR_LABELS = ["Jan→Feb", "Feb→Mar", "Mar→Apr", "Apr→May"]

CD_THRESHOLD = 0.5
TOP_N_PATCHES = 8

CHANGE_CMAP = LinearSegmentedColormap.from_list("change", ["#2166ac", "#f7f7f7", "#b2182b"])
BINARY_CMAP = LinearSegmentedColormap.from_list("binary", ["#ffffff", "#b2182b"])

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def load_cd_head(ckpt_path: str, device: torch.device) -> ChangeDetectionHeadV3:
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


def compute_cosine_distance(emb1, emb2):
    cos_sim = np.sum(emb1 * emb2, axis=0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return (1.0 - cos_sim) / 2.0


def read_rgb_for_patch_month(patch_id: str, month: str, target_size: tuple[int, int] = (64, 64)) -> np.ndarray | None:
    """读取指定 patch 和月份的 RGB 图像，缩放到 target_size.
    
    优先顺序: S2 → Landsat → S1(灰度)
    返回: [H, W, 3] numpy array, 值域 [0, 1], 或 None
    """
    month_prefix = month.replace("-", "")
    
    # 1. 尝试 S2
    s2_dir = DATA_ROOT / "s2" / patch_id
    if s2_dir.exists():
        files = sorted([f for f in s2_dir.glob("*.tif") if f.stem.startswith(month_prefix)])
        if files:
            with rasterio.open(files[0]) as src:
                data = src.read()  # [C, H, W]
                if data.shape[0] >= 3:
                    # B04(红)=波段3, B03(绿)=波段2, B02(蓝)=波段1
                    rgb = np.stack([data[2], data[1], data[0]], axis=0)
                    rgb = _resize_rgb(rgb, target_size)
                    return rgb
    
    # 2. 尝试 Landsat
    landsat_dir = DATA_ROOT / "landsat" / patch_id
    if landsat_dir.exists():
        files = sorted([f for f in landsat_dir.glob("*.tif") if f.stem.startswith(month_prefix)])
        if files:
            with rasterio.open(files[0]) as src:
                data = src.read()
                if data.shape[0] >= 3:
                    # red, green, blue = 波段1, 2, 3
                    rgb = np.stack([data[0], data[1], data[2]], axis=0)
                    rgb = _resize_rgb(rgb, target_size)
                    return rgb
    
    # 3. 尝试 S1 (灰度，复制到 3 通道)
    s1_dir = DATA_ROOT / "s1" / patch_id
    if s1_dir.exists():
        files = sorted([f for f in s1_dir.glob("*.tif") if f.stem.startswith(month_prefix)])
        if files:
            with rasterio.open(files[0]) as src:
                data = src.read()
                if data.shape[0] >= 1:
                    # VV 波段
                    gray = data[0:1]
                    # 归一化到 0-1
                    vmin, vmax = np.percentile(gray, [2, 98])
                    gray = np.clip((gray - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    rgb = np.repeat(gray, 3, axis=0)
                    rgb = _resize_rgb(rgb, target_size)
                    return rgb
    
    return None


def _resize_rgb(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """将 [C, H, W] RGB 缩放到 target_size (H, W)，并转为 [H, W, C]."""
    import torch
    tensor = torch.from_numpy(rgb).unsqueeze(0).float()  # [1, C, H, W]
    resized = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).numpy()  # [C, H, W]
    resized = np.transpose(resized, (1, 2, 0))  # [H, W, C]
    return np.clip(resized, 0, 1)


def select_top_patches_and_pairs(n: int) -> list[tuple[str, str, str, float]]:
    """选择变化最显著的 N 个 (patch, month1, month2, score) 组合."""
    meta_path = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/change_scores_2026/meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sorted_records = sorted(meta, key=lambda r: r["mean_score"], reverse=True)
        # 去重：每个 patch 只取最显著的一个月份对
        seen_patches = set()
        result = []
        for r in sorted_records:
            pid = r["patch_id"]
            if pid not in seen_patches:
                mp = r["month_pair"]
                m1, m2 = mp.split("_to_")
                result.append((pid, m1, m2, r["mean_score"]))
                seen_patches.add(pid)
                if len(result) >= n:
                    break
        return result
    
    # fallback
    all_patches = sorted(set(p.name.split("_2026")[0] for p in EMB_DIR.glob("patch_*.npy")))
    return [(p, "2026-03", "2026-04", 0.0) for p in all_patches[:n]]


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_rgb_comparison_grid(head, top_combos, device, out_path):
    """生成大图: 每行一个 patch，展示最显著月份对的 Before/After/CD/Binary."""
    n_rows = len(top_combos)
    n_cols = 4  # Before RGB | After RGB | CD Prob | Binary
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
    
    col_titles = ["Before (RGB)", "After (RGB)", "CD Head Probability", f"Binary (>{CD_THRESHOLD})"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold")
    
    for row_idx, (pid, m1, m2, score) in enumerate(top_combos):
        emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
        emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
        
        # 读取 RGB
        rgb_before = read_rgb_for_patch_month(pid, m1)
        rgb_after = read_rgb_for_patch_month(pid, m2)
        
        # 读取 embedding 并计算 CD
        if emb1_path.exists() and emb2_path.exists():
            emb1 = np.load(emb1_path)
            emb2 = np.load(emb2_path)
            cd_probs = compute_cd_probs(head, emb1, emb2, device)
            binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
        else:
            cd_probs = None
            binary = None
        
        # 1. Before RGB
        ax = axes[row_idx, 0]
        if rgb_before is not None:
            ax.imshow(rgb_before)
        ax.set_ylabel(f"{pid}\n{m1}→{m2}\ncos={score:.3f}", fontsize=10, fontweight="bold", rotation=0, ha="right", va="center")
        ax.tick_params(labelleft=False)
        ax.axis("off")
        
        # 2. After RGB
        ax = axes[row_idx, 1]
        if rgb_after is not None:
            ax.imshow(rgb_after)
        ax.axis("off")
        
        # 3. CD Prob
        ax = axes[row_idx, 2]
        if cd_probs is not None:
            im = ax.imshow(cd_probs, cmap=CHANGE_CMAP, vmin=0, vmax=1)
            ax.set_title(f"mean={cd_probs.mean():.3f}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        
        # 4. Binary
        ax = axes[row_idx, 3]
        if binary is not None:
            im = ax.imshow(binary, cmap=BINARY_CMAP, vmin=0, vmax=1)
            ax.set_title(f"rate={binary.mean():.3f}", fontsize=9)
        ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RGB comparison grid: {out_path}")


def plot_per_patch_detail(head, top_combos, device, out_dir):
    """为每个 patch 生成详细的月度对比图（所有 4 个月度对）."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for pid, _, _, _ in top_combos:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
        # 列标题
        col_titles = ["Before RGB", "After RGB", "Cosine Dist", "CD Prob", f"Binary (>{CD_THRESHOLD})"]
        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold")
        
        for row_idx, (m1, m2) in enumerate(MONTH_PAIRS):
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            
            rgb_before = read_rgb_for_patch_month(pid, m1)
            rgb_after = read_rgb_for_patch_month(pid, m2)
            
            if emb1_path.exists() and emb2_path.exists():
                emb1 = np.load(emb1_path)
                emb2 = np.load(emb2_path)
                cd_probs = compute_cd_probs(head, emb1, emb2, device)
                cos_dist = compute_cosine_distance(emb1, emb2)
                binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
            else:
                cd_probs = None
                cos_dist = None
                binary = None
            
            # Before RGB
            ax = axes[row_idx, 0]
            if rgb_before is not None:
                ax.imshow(rgb_before)
            ax.set_ylabel(MONTH_PAIR_LABELS[row_idx], fontsize=10, fontweight="bold", rotation=0, ha="right", va="center")
            ax.tick_params(labelleft=False)
            ax.axis("off")
            
            # After RGB
            ax = axes[row_idx, 1]
            if rgb_after is not None:
                ax.imshow(rgb_after)
            ax.axis("off")
            
            # Cosine Dist
            ax = axes[row_idx, 2]
            if cos_dist is not None:
                im = ax.imshow(cos_dist, cmap=CHANGE_CMAP, vmin=0, vmax=1)
                ax.set_title(f"mean={cos_dist.mean():.3f}", fontsize=9)
            ax.axis("off")
            
            # CD Prob
            ax = axes[row_idx, 3]
            if cd_probs is not None:
                im = ax.imshow(cd_probs, cmap=CHANGE_CMAP, vmin=0, vmax=1)
                ax.set_title(f"mean={cd_probs.mean():.3f}", fontsize=9)
            ax.axis("off")
            
            # Binary
            ax = axes[row_idx, 4]
            if binary is not None:
                im = ax.imshow(binary, cmap=BINARY_CMAP, vmin=0, vmax=1)
                ax.set_title(f"rate={binary.mean():.3f}", fontsize=9)
            ax.axis("off")
        
        fig.suptitle(f"{pid} — Change Detection Detail", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(out_dir / f"{pid}_detail.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    print(f"Saved per-patch details to {out_dir}")


def main():
    device = get_device(device_str="cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    head = load_cd_head(CKPT_PATH, device)
    print(f"Loaded CD Head: AUC={head.training}")
    
    top_combos = select_top_patches_and_pairs(TOP_N_PATCHES)
    print(f"\nSelected top {len(top_combos)} patches:")
    for pid, m1, m2, score in top_combos:
        print(f"  {pid}: {m1}→{m2} (cos_score={score:.4f})")
    
    # 1. 大图
    print("\nGenerating RGB comparison grid...")
    plot_rgb_comparison_grid(head, top_combos, device, OUTPUT_DIR / "cd_head_rgb_comparison_grid.png")
    
    # 2. 每个 patch 的详细图
    print("\nGenerating per-patch detail images...")
    plot_per_patch_detail(head, top_combos, device, OUTPUT_DIR / "per_patch")
    
    print(f"\nAll done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
