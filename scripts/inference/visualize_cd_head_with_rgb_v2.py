#!/usr/bin/env python3
"""CD Head 变化检测可视化 V2 — 大量例子 + RGB 增强，筛选真正有变化的样本.

输出:
  - cd_head_rgb_top30_grid.png — Top 30 最显著变化例子
  - cd_head_rgb_all_significant.png — 所有显著变化的例子（>0.1 rate）
  - per_patch_enhanced/ — 每个 patch 增强版详细图
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


def enhance_rgb(rgb: np.ndarray) -> np.ndarray:
    """增强 RGB 对比度，让图像更适合审查."""
    rgb = rgb.copy()
    # 逐波段对比度拉伸 (2%-98% percentile)
    for c in range(3):
        band = rgb[..., c]
        p2, p98 = np.percentile(band, [2, 98])
        rgb[..., c] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    # Gamma 校正提亮暗部
    rgb = np.power(rgb, 0.8)
    return np.clip(rgb, 0, 1)


def read_rgb_for_patch_month(patch_id: str, month: str, target_size: tuple[int, int] = (64, 64)) -> np.ndarray | None:
    """读取指定 patch 和月份的 RGB 图像."""
    month_prefix = month.replace("-", "")
    
    # 1. 尝试 S2
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
    
    # 2. 尝试 Landsat
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
    
    # 3. 尝试 S1 (灰度)
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


def _resize_rgb(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    import torch
    tensor = torch.from_numpy(rgb).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).numpy()
    resized = np.transpose(resized, (1, 2, 0))
    return np.clip(resized, 0, 1)


# ---------------------------------------------------------------------------
# 核心：计算所有组合的 CD Head 分数，筛选显著变化
# ---------------------------------------------------------------------------

def compute_all_cd_scores(head, device) -> list[dict]:
    """计算所有 (patch, month_pair) 的 CD Head 分数，返回排序后的列表."""
    
    # 获取所有 patches
    all_patches = sorted(set(p.name.split("_2026")[0] for p in EMB_DIR.glob("patch_*.npy")))
    
    results = []
    for pid in all_patches:
        for (m1, m2), mp_label in zip(MONTH_PAIRS, MONTH_PAIR_LABELS):
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            
            if not emb1_path.exists() or not emb2_path.exists():
                continue
            
            emb1 = np.load(emb1_path)
            emb2 = np.load(emb2_path)
            
            cd_probs = compute_cd_probs(head, emb1, emb2, device)
            cos_dist = compute_cosine_distance(emb1, emb2)
            binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
            
            results.append({
                "patch_id": pid,
                "month_pair_label": mp_label,
                "m1": m1,
                "m2": m2,
                "cd_mean": float(cd_probs.mean()),
                "cd_max": float(cd_probs.max()),
                "cd_std": float(cd_probs.std()),
                "binary_rate": float(binary.mean()),
                "cos_mean": float(cos_dist.mean()),
                "cos_max": float(cos_dist.max()),
            })
    
    # 排序：优先 cd_max 高且 binary_rate > 0 的
    results.sort(key=lambda r: (r["cd_max"] > 0.3, r["binary_rate"] > 0, r["cd_max"], r["binary_rate"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_large_grid(head, combos, device, out_path, max_rows=50):
    """生成大图：每行一个 (patch, month_pair) 组合."""
    n_rows = min(len(combos), max_rows)
    n_cols = 4  # Before RGB | After RGB | CD Prob | Binary
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
    
    col_titles = ["Before (RGB)", "After (RGB)", f"CD Prob", f"Binary (>{CD_THRESHOLD})"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold")
    
    for row_idx in range(n_rows):
        combo = combos[row_idx]
        pid = combo["patch_id"]
        m1 = combo["m1"]
        m2 = combo["m2"]
        mp_label = combo["month_pair_label"]
        
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
        
        # 1. Before RGB
        ax = axes[row_idx, 0]
        if rgb_before is not None:
            ax.imshow(rgb_before)
        ax.set_ylabel(f"{pid}\n{mp_label}\ncd_max={combo['cd_max']:.2f}\nbin_rate={combo['binary_rate']:.3f}", 
                      fontsize=9, fontweight="bold", rotation=0, ha="right", va="center")
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
            ax.set_title(f"mean={cd_probs.mean():.3f}, max={cd_probs.max():.3f}", fontsize=9)
        ax.axis("off")
        
        # 4. Binary
        ax = axes[row_idx, 3]
        if binary is not None:
            im = ax.imshow(binary, cmap=BINARY_CMAP, vmin=0, vmax=1)
            ax.set_title(f"rate={binary.mean():.3f}", fontsize=9)
        ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved large grid ({n_rows} rows): {out_path}")


def plot_monthly_breakdown(head, top_patches, device, out_dir):
    """为每个 patch 生成所有月份对的详细对比图（增强版）."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for pid in top_patches:
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        
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
                ax.imshow(cos_dist, cmap=CHANGE_CMAP, vmin=0, vmax=1)
                ax.set_title(f"mean={cos_dist.mean():.3f}", fontsize=9)
            ax.axis("off")
            
            # CD Prob
            ax = axes[row_idx, 3]
            if cd_probs is not None:
                ax.imshow(cd_probs, cmap=CHANGE_CMAP, vmin=0, vmax=1)
                ax.set_title(f"mean={cd_probs.mean():.3f}, max={cd_probs.max():.3f}", fontsize=9)
            ax.axis("off")
            
            # Binary
            ax = axes[row_idx, 4]
            if binary is not None:
                ax.imshow(binary, cmap=BINARY_CMAP, vmin=0, vmax=1)
                ax.set_title(f"rate={binary.mean():.3f}", fontsize=9)
            ax.axis("off")
        
        fig.suptitle(f"{pid} — Change Detection Detail (Enhanced RGB)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(out_dir / f"{pid}_detail_enhanced.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    print(f"Saved enhanced per-patch details to {out_dir}")


def main():
    device = get_device(device_str="cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    head = load_cd_head(CKPT_PATH, device)
    
    # 1. 计算所有组合的分数
    print("Computing CD scores for all patches and month pairs...")
    all_scores = compute_all_cd_scores(head, device)
    print(f"Total valid combinations: {len(all_scores)}")
    
    # 打印统计
    cd_max_values = [r["cd_max"] for r in all_scores]
    binary_rates = [r["binary_rate"] for r in all_scores]
    print(f"CD max: min={min(cd_max_values):.3f}, max={max(cd_max_values):.3f}, mean={np.mean(cd_max_values):.3f}")
    print(f"Binary rate: min={min(binary_rates):.3f}, max={max(binary_rates):.3f}, mean={np.mean(binary_rates):.3f}")
    
    # 2. 筛选显著变化的组合 (cd_max > 0.3 或 binary_rate > 0.01)
    significant = [r for r in all_scores if r["cd_max"] > 0.3 or r["binary_rate"] > 0.01]
    print(f"Significant combinations: {len(significant)}")
    
    # 3. 生成 Top 50 大图
    print("\nGenerating top 50 grid...")
    plot_large_grid(head, all_scores[:50], device, OUTPUT_DIR / "cd_head_rgb_top50_grid.png", max_rows=50)
    
    # 4. 生成所有显著变化的大图
    if significant:
        print("\nGenerating all significant changes grid...")
        plot_large_grid(head, significant, device, OUTPUT_DIR / "cd_head_rgb_all_significant.png", max_rows=100)
    
    # 5. 为 top 15 patches 生成增强版详细图
    top_patches = list(dict.fromkeys(r["patch_id"] for r in all_scores))[:15]
    print(f"\nGenerating enhanced details for top 15 patches...")
    plot_monthly_breakdown(head, top_patches, device, OUTPUT_DIR / "per_patch_enhanced")
    
    print(f"\nAll done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
