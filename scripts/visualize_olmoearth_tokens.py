#!/usr/bin/env python3
"""可视化 OlmoEarth spatial tokens — 全局 t-SNE/UMAP 对比两个月份."""
import sys, os
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

REGIONS = {
    "haidian": "/workspace/outputs/olmoearth_haidian",
    "harbin": "/workspace/outputs/olmoearth_harbin",
}


def load_tokens(region: str, month: str):
    """加载指定区域和月份的 spatial tokens.
    
    month 格式:
      - "01" -> 2025-01
      - "2026/01" -> 2026-01
    """
    root = REGIONS[region]
    path = os.path.join(root, month, "spatial_tokens.npz")
    d = np.load(path)
    tokens = d["tokens"]  # (N, 32, 32, 768) fp16
    patch_ids = d["patch_ids"]
    return tokens.astype(np.float32), patch_ids


def visualize_two_months(region: str, month_a: str, month_b: str, out_path: str):
    """对比可视化两个月的 tokens (spatial mean -> 768D -> t-SNE 2D)."""
    tok_a, ids_a = load_tokens(region, month_a)
    tok_b, ids_b = load_tokens(region, month_b)

    # spatial mean: (N, 32, 32, 768) -> (N, 768)
    emb_a = tok_a.mean(axis=(1, 2))
    emb_b = tok_b.mean(axis=(1, 2))

    print(f"[{region}] {month_a}: {emb_a.shape}")
    print(f"[{region}] {month_b}: {emb_b.shape}")

    # 合并降维
    X = np.concatenate([emb_a, emb_b], axis=0)
    y = np.array([0] * len(emb_a) + [1] * len(emb_b))

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, verbose=1)
    Z = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: 合并散点
    ax = axes[0]
    colors = ["#e74c3c", "#3498db"]
    labels = [month_a, month_b]
    for idx in (0, 1):
        mask = y == idx
        ax.scatter(Z[mask, 0], Z[mask, 1], c=colors[idx], s=8, alpha=0.6, label=labels[idx])
    ax.set_title(f"{region}: {month_a} vs {month_b} (t-SNE)")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot 2: 仅 month_a
    ax = axes[1]
    mask = y == 0
    ax.scatter(Z[mask, 0], Z[mask, 1], c=colors[0], s=8, alpha=0.6)
    ax.set_title(f"{region}: {month_a}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot 3: 仅 month_b
    ax = axes[2]
    mask = y == 1
    ax.scatter(Z[mask, 0], Z[mask, 1], c=colors[1], s=8, alpha=0.6)
    ax.set_title(f"{region}: {month_b}")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    os.makedirs("/workspace/outputs/olmoearth_viz", exist_ok=True)

    # Haidian: 01 (冬季) vs 07 (夏季) — 应该有显著季节差异
    visualize_two_months(
        "haidian", "01", "07",
        "/workspace/outputs/olmoearth_viz/haidian_01_vs_07.png"
    )

    # Harbin: 01 (冬季) vs 07 (夏季) — 应该有显著季节差异
    visualize_two_months(
        "harbin", "01", "07",
        "/workspace/outputs/olmoearth_viz/harbin_01_vs_07.png"
    )

    # Haidian: 2025-04 vs 2026-04 — 跨年同月对比
    visualize_two_months(
        "haidian", "04", "2026/04",
        "/workspace/outputs/olmoearth_viz/haidian_2025_vs_2026_apr.png"
    )
