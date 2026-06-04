#!/usr/bin/env python3
"""使用训练好的 CD Head 对 2026 年月度 embedding 运行变化检测，并生成可视化大图.

输出: /workspace/outputs/aef_qwen_v5_mixed_scale/cd_head_visualization_2026/
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

from src.models.heads import ChangeDetectionHeadV3
from src.utils.device import get_device

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/cd_head_visualization_2026")

# 选择最佳 CD Head
CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/monthly_cd_head_v5_final.pt"
# 备选:
# CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/monthly_cd_head_v3_hd128_ohem.pt"

MONTH_PAIRS = [
    ("2026-01", "2026-02"),
    ("2026-02", "2026-03"),
    ("2026-03", "2026-04"),
    ("2026-04", "2026-05"),
]

# 展示前 N 个变化最显著的 patch
TOP_N_PATCHES = 12

# CD Head 概率阈值（用于二值化显示）
CD_THRESHOLD = 0.5

# 自定义颜色映射
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "change", ["#2166ac", "#f7f7f7", "#b2182b"]
)
BINARY_CMAP = LinearSegmentedColormap.from_list(
    "binary", ["#ffffff", "#b2182b"]
)


def load_cd_head(ckpt_path: str, device: torch.device) -> ChangeDetectionHeadV3:
    """加载 CD Head."""
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
    print(f"Loaded CD Head from {ckpt_path}")
    print(f"  Config: {cfg}")
    print(f"  Metrics: {ckpt.get('metrics', 'N/A')}")
    return head


def compute_cd_probs(head: ChangeDetectionHeadV3, emb_before: np.ndarray, emb_after: np.ndarray, device: torch.device) -> np.ndarray:
    """使用 CD Head 计算变化概率图 [H, W]."""
    with torch.no_grad():
        eb = torch.from_numpy(emb_before).unsqueeze(0).float().to(device)
        ea = torch.from_numpy(emb_after).unsqueeze(0).float().to(device)
        logits = head(eb, ea)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """计算 cosine distance [H, W]."""
    cos_sim = np.sum(emb1 * emb2, axis=0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return (1.0 - cos_sim) / 2.0


def select_top_patches(n: int) -> list[str]:
    """选择变化最显著的 N 个 patches（基于 cosine distance 均值）."""
    meta_path = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/change_scores_2026/meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        # 计算每个 patch 的平均变化分数
        patch_scores: dict[str, list[float]] = {}
        for r in meta:
            pid = r["patch_id"]
            patch_scores.setdefault(pid, []).append(r["mean_score"])
        avg_scores = {pid: np.mean(scores) for pid, scores in patch_scores.items()}
        sorted_patches = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_patches[:n]]
    
    # fallback: 按文件名顺序
    all_patches = sorted(set(
        p.name.split("_2026")[0] 
        for p in EMB_DIR.glob("patch_*.npy")
    ))
    return all_patches[:n]


def plot_comparison_grid(head: ChangeDetectionHeadV3, patches: list[str], device: torch.device, out_path: Path):
    """生成大图: 每行一个 patch, 每列一个月份对.
    
    每行展示:
      - Cosine Distance (baseline)
      - CD Head Probability
      - CD Head Binary (thresholded)
    """
    n_rows = len(patches)
    n_cols = len(MONTH_PAIRS)
    
    # 创建大图: 3 行子图 × n_cols 列 × n_rows 个 patch
    # 布局: (n_rows * 3) 行, n_cols 列
    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 4, n_rows * 3 * 2.5))
    if n_rows == 1:
        axes = axes.reshape(3, n_cols)
    
    for row_idx, pid in enumerate(patches):
        for col_idx, (m1, m2) in enumerate(MONTH_PAIRS):
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            
            if not emb1_path.exists() or not emb2_path.exists():
                # 空白图
                for k in range(3):
                    ax = axes[row_idx * 3 + k, col_idx] if n_rows > 1 else axes[k, col_idx]
                    ax.axis("off")
                continue
            
            emb1 = np.load(emb1_path)
            emb2 = np.load(emb2_path)
            
            # 1. Cosine Distance
            cos_dist = compute_cosine_distance(emb1, emb2)
            ax_cd = axes[row_idx * 3 + 0, col_idx] if n_rows > 1 else axes[0, col_idx]
            im_cd = ax_cd.imshow(cos_dist, cmap=CHANGE_CMAP, vmin=0, vmax=1)
            ax_cd.set_title(f"{pid}  {m1}→{m2}\nCosine Distance", fontsize=9)
            ax_cd.axis("off")
            
            # 2. CD Head Probability
            cd_probs = compute_cd_probs(head, emb1, emb2, device)
            ax_prob = axes[row_idx * 3 + 1, col_idx] if n_rows > 1 else axes[1, col_idx]
            im_prob = ax_prob.imshow(cd_probs, cmap=CHANGE_CMAP, vmin=0, vmax=1)
            ax_prob.set_title(f"CD Head Prob  mean={cd_probs.mean():.3f}", fontsize=9)
            ax_prob.axis("off")
            
            # 3. CD Head Binary
            binary = (cd_probs > CD_THRESHOLD).astype(np.float32)
            ax_bin = axes[row_idx * 3 + 2, col_idx] if n_rows > 1 else axes[2, col_idx]
            im_bin = ax_bin.imshow(binary, cmap=BINARY_CMAP, vmin=0, vmax=1)
            ax_bin.set_title(f"Binary (>{CD_THRESHOLD})  rate={binary.mean():.3f}", fontsize=9)
            ax_bin.axis("off")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid: {out_path}")


def plot_trend_comparison(head: ChangeDetectionHeadV3, patches: list[str], device: torch.device, out_path: Path):
    """绘制 CD Head vs Cosine Distance 的月度趋势对比."""
    month_pair_labels = [f"{m1[5:]}→{m2[5:]}" for m1, m2 in MONTH_PAIRS]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cd_means = []
    cd_stds = []
    cos_means = []
    cos_stds = []
    
    for m1, m2 in MONTH_PAIRS:
        cd_scores = []
        cos_scores = []
        for pid in patches:
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            if emb1_path.exists() and emb2_path.exists():
                emb1 = np.load(emb1_path)
                emb2 = np.load(emb2_path)
                cd_probs = compute_cd_probs(head, emb1, emb2, device)
                cos_dist = compute_cosine_distance(emb1, emb2)
                cd_scores.append(cd_probs.mean())
                cos_scores.append(cos_dist.mean())
        
        cd_means.append(np.mean(cd_scores) if cd_scores else 0)
        cd_stds.append(np.std(cd_scores) if cd_scores else 0)
        cos_means.append(np.mean(cos_scores) if cos_scores else 0)
        cos_stds.append(np.std(cos_scores) if cos_scores else 0)
    
    # 左图: Mean Change Score
    axes[0].plot(month_pair_labels, cos_means, "o-", label="Cosine Distance", color="#2166ac", linewidth=2, markersize=8)
    axes[0].fill_between(month_pair_labels, 
                         [m - s for m, s in zip(cos_means, cos_stds)],
                         [m + s for m, s in zip(cos_means, cos_stds)],
                         alpha=0.2, color="#2166ac")
    axes[0].plot(month_pair_labels, cd_means, "s-", label="CD Head Prob", color="#b2182b", linewidth=2, markersize=8)
    axes[0].fill_between(month_pair_labels, 
                         [m - s for m, s in zip(cd_means, cd_stds)],
                         [m + s for m, s in zip(cd_means, cd_stds)],
                         alpha=0.2, color="#b2182b")
    axes[0].set_ylabel("Mean Change Score", fontsize=12)
    axes[0].set_title("Change Score Trend Comparison", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 右图: 散点对比 (所有 patch-month 对)
    all_cos = []
    all_cd = []
    for pid in patches:
        for m1, m2 in MONTH_PAIRS:
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            if emb1_path.exists() and emb2_path.exists():
                emb1 = np.load(emb1_path)
                emb2 = np.load(emb2_path)
                cd_probs = compute_cd_probs(head, emb1, emb2, device)
                cos_dist = compute_cosine_distance(emb1, emb2)
                all_cos.append(cos_dist.mean())
                all_cd.append(cd_probs.mean())
    
    axes[1].scatter(all_cos, all_cd, alpha=0.5, s=20, color="#444444")
    # 拟合线
    if len(all_cos) > 1:
        z = np.polyfit(all_cos, all_cd, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_cos), max(all_cos), 100)
        axes[1].plot(x_line, p(x_line), "r--", linewidth=2, label=f"Linear fit (slope={z[0]:.2f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    axes[1].set_xlabel("Cosine Distance (mean)", fontsize=12)
    axes[1].set_ylabel("CD Head Probability (mean)", fontsize=12)
    axes[1].set_title("Cosine Distance vs CD Head Prob", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trend comparison: {out_path}")


def main():
    device = get_device(device_str="cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载 CD Head
    head = load_cd_head(CKPT_PATH, device)
    
    # 选择 Top patches
    top_patches = select_top_patches(TOP_N_PATCHES)
    print(f"\nSelected top {len(top_patches)} patches: {top_patches}")
    
    # 1. 生成对比大图
    print("\nGenerating comparison grid...")
    plot_comparison_grid(head, top_patches, device, OUTPUT_DIR / "cd_head_comparison_grid.png")
    
    # 2. 生成趋势对比
    print("\nGenerating trend comparison...")
    plot_trend_comparison(head, top_patches, device, OUTPUT_DIR / "cd_head_trend_comparison.png")
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
