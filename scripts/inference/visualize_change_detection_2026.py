#!/usr/bin/env python3
"""可视化哈尔滨 2026 年变化检测结果.

输出:
  1. 典型 patch 的变化热力图 (top 20 变化最显著的 patch)
  2. 全区域月度变化趋势曲线
  3. 变化强度分布直方图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

CHANGE_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/change_scores_2026")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/visualizations_2026")

# 自定义变化热力图颜色：蓝色(无变化) -> 黄色 -> 红色(强变化)
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "change", ["#2166ac", "#f7f7f7", "#b2182b"]
)


def plot_patch_heatmap(change_score: np.ndarray, patch_id: str, month_pair: str, out_path: Path):
    """绘制单个 patch 的变化热力图."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(change_score, cmap=CHANGE_CMAP, vmin=0, vmax=1)
    ax.set_title(f"{patch_id}\n{month_pair} Change Score", fontsize=12)
    ax.axis("off")
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Change Score", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trend_curve(meta_records: list, out_path: Path):
    """绘制月度变化趋势曲线."""
    month_pairs = ["2026-01_to_2026-02", "2026-02_to_2026-03", 
                   "2026-03_to_2026-04", "2026-04_to_2026-05"]
    pair_labels = ["Jan→Feb", "Feb→Mar", "Mar→Apr", "Apr→May"]
    
    means = []
    stds = []
    for mp in month_pairs:
        scores = [r["mean_score"] for r in meta_records if r["month_pair"] == mp]
        means.append(np.mean(scores) if scores else 0)
        stds.append(np.std(scores) if scores else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pair_labels, means, "o-", linewidth=2, markersize=8, color="#d73027", label="Mean Change Score")
    ax.fill_between(pair_labels, 
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="#d73027")
    
    ax.set_xlabel("Month Pair", fontsize=12)
    ax.set_ylabel("Mean Change Score", fontsize=12)
    ax.set_title("Harbin 2026 — Monthly Change Trend (All 424 Patches)", fontsize=14)
    ax.set_ylim(0, max(means) * 1.3 if means else 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trend curve: {out_path}")


def plot_histogram(meta_records: list, out_path: Path):
    """绘制变化强度分布直方图."""
    all_scores = []
    for r in meta_records:
        # 不加载完整的 numpy 数组，只使用已记录的统计值
        # 为了绘制直方图，我们需要实际加载数据
        pass
    
    # 改为加载所有 change map 并展平
    for npy_path in sorted(CHANGE_DIR.glob("patch_*_to_*.npy")):
        score = np.load(npy_path)
        all_scores.extend(score.flatten().tolist())
    
    all_scores = np.array(all_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：全范围直方图
    axes[0].hist(all_scores, bins=100, color="#4575b4", edgecolor="white", alpha=0.8)
    axes[0].axvline(np.median(all_scores), color="red", linestyle="--", linewidth=2, label=f"Median={np.median(all_scores):.4f}")
    axes[0].set_xlabel("Change Score", fontsize=12)
    axes[0].set_ylabel("Pixel Count", fontsize=12)
    axes[0].set_title("Distribution of All Change Scores", fontsize=13)
    axes[0].legend()
    axes[0].set_yscale("log")
    
    # 右图：按月度对分组
    month_pairs = ["2026-01_to_2026-02", "2026-02_to_2026-03", 
                   "2026-03_to_2026-04", "2026-04_to_2026-05"]
    pair_labels = ["Jan→Feb", "Feb→Mar", "Mar→Apr", "Apr→May"]
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]
    
    for mp, label, color in zip(month_pairs, pair_labels, colors):
        scores = []
        for npy_path in sorted(CHANGE_DIR.glob(f"patch_*_{mp}.npy")):
            scores.extend(np.load(npy_path).flatten().tolist())
        if scores:
            axes[1].hist(scores, bins=50, alpha=0.5, label=label, color=color, density=True)
    
    axes[1].set_xlabel("Change Score", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Change Score Distribution by Month Pair", fontsize=13)
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载 meta
    meta_path = CHANGE_DIR / "meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run compute_change_detection_2026.py first.")
        return
    
    with open(meta_path) as f:
        meta_records = json.load(f)
    
    print(f"Loaded {len(meta_records)} change map records")
    
    # --- 1. 绘制 top 20 变化最显著的 patch 热力图 ---
    print("\nGenerating top-20 patch heatmaps...")
    
    # 按 mean_score 排序
    sorted_records = sorted(meta_records, key=lambda r: r["mean_score"], reverse=True)
    top_records = sorted_records[:20]
    
    heatmap_dir = OUTPUT_DIR / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    for r in tqdm(top_records, desc="Heatmaps"):
        pid = r["patch_id"]
        mp = r["month_pair"]
        npy_path = CHANGE_DIR / f"{pid}_{mp}.npy"
        if npy_path.exists():
            score = np.load(npy_path)
            out_path = heatmap_dir / f"{pid}_{mp}.png"
            plot_patch_heatmap(score, pid, mp, out_path)
    
    # --- 2. 月度变化趋势曲线 ---
    print("\nGenerating trend curve...")
    plot_trend_curve(meta_records, OUTPUT_DIR / "change_trend.png")
    
    # --- 3. 变化强度分布直方图 ---
    print("\nGenerating histogram...")
    plot_histogram(meta_records, OUTPUT_DIR / "change_histogram.png")
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
