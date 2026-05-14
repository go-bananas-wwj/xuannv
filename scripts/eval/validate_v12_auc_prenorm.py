#!/usr/bin/env python3
"""V12 Pre-norm 空间 AUC 验证 — 基于时间 gap 的变化检测.

不使用哈尔滨标注，而是用"时间间隔"作为变化的代理标签：
  - 相邻月份 (gap=1): 视为"无变化" (label=0)
  - 相隔多月 (gap>=6): 视为"有变化" (label=1)

同时计算 global mean 和 per-pixel spatial map 的 AUC.
"""
import sys, os, argparse
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-means", required=True,
                        help="global_means.npz 路径")
    parser.add_argument("--spatial-maps", default=None,
                        help="spatial_maps_sample.npz 路径 (可选)")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def compute_auc_from_pairs(distances, labels, name):
    """计算 AUC 并返回统计信息."""
    distances = np.array(distances)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return None

    auc = roc_auc_score(labels, distances)

    # 分组的 mean distance
    d0 = distances[labels == 0]
    d1 = distances[labels == 1]

    return {
        "name": name,
        "auc": float(auc),
        "n_total": len(distances),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "dist_mean_neg": float(d0.mean()),
        "dist_mean_pos": float(d1.mean()),
        "dist_median_neg": float(np.median(d0)),
        "dist_median_pos": float(np.median(d1)),
        "dist_std_neg": float(d0.std()),
        "dist_std_pos": float(d1.std()),
        "separation": float(d1.mean() - d0.mean()),
    }


def build_pairs_from_global(pre_norm, patch_ids, year_months, gap_threshold=6):
    """从 global mean embeddings 构建 pairs."""
    from collections import defaultdict

    patch_series = defaultdict(list)
    for i, (pid, ym) in enumerate(zip(patch_ids, year_months)):
        patch_series[pid].append((ym[0], ym[1], pre_norm[i]))

    for pid in patch_series:
        patch_series[pid].sort(key=lambda x: (x[0], x[1]))

    distances = []
    labels = []
    gap_sizes = []

    for pid, series in patch_series.items():
        n = len(series)
        for i in range(n):
            for j in range(i + 1, n):
                y1, m1, emb1 = series[i]
                y2, m2, emb2 = series[j]
                month_gap = (y2 - y1) * 12 + (m2 - m1)
                if month_gap < 1:
                    continue

                dist = np.linalg.norm(emb1 - emb2)
                distances.append(dist)
                gap_sizes.append(month_gap)

                # label: 相邻=0, 远隔=1
                label = 1 if month_gap >= gap_threshold else 0
                labels.append(label)

    return distances, labels, gap_sizes


def build_pairs_from_spatial(pre_norm_maps, patch_ids, year_months, gap_threshold=6):
    """从 spatial maps 构建 per-pixel pairs."""
    from collections import defaultdict

    # 按 patch 组织
    patch_series = defaultdict(list)
    for i, (pid, ym) in enumerate(zip(patch_ids, year_months)):
        patch_series[pid].append((ym[0], ym[1], pre_norm_maps[i]))

    for pid in patch_series:
        patch_series[pid].sort(key=lambda x: (x[0], x[1]))

    all_pixel_dists = []
    all_pixel_labels = []

    for pid, series in patch_series.items():
        n = len(series)
        for i in range(n):
            for j in range(i + 1, n):
                y1, m1, map1 = series[i]
                y2, m2, map2 = series[j]
                month_gap = (y2 - y1) * 12 + (m2 - m1)
                if month_gap < 1:
                    continue

                # Per-pixel L2 distance
                # map1: [D, H, W], map2: [D, H, W]
                diff = map1 - map2  # [D, H, W]
                dist_map = np.linalg.norm(diff, axis=0)  # [H, W]

                label = 1 if month_gap >= gap_threshold else 0

                # Flatten
                all_pixel_dists.extend(dist_map.flatten().tolist())
                all_pixel_labels.extend([label] * dist_map.size)

    return all_pixel_dists, all_pixel_labels


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("V12 Pre-norm 空间 AUC 验证")
    print("=" * 60)

    # ── 加载 global means ──
    print(f"\n加载: {args.global_means}")
    g = np.load(args.global_means)
    pre_norm = g["pre_norm"]          # [N, D]
    patch_ids = g["patch_ids"]
    year_months = g["year_months"]    # [N, 2]
    print(f"  Samples: {pre_norm.shape[0]}, Dim: {pre_norm.shape[1]}")

    # ── Global Mean AUC ──
    print("\n--- Global Mean Embedding ---")
    for gap_th in [3, 6]:
        distances, labels, gap_sizes = build_pairs_from_global(
            pre_norm, patch_ids, year_months, gap_threshold=gap_th
        )
        result = compute_auc_from_pairs(distances, labels, f"global_gap{gap_th}")
        if result:
            print(f"\n  Gap threshold = {gap_th} months:")
            print(f"    AUC = {result['auc']:.4f}")
            print(f"    Pairs: total={result['n_total']}, pos={result['n_pos']}, neg={result['n_neg']}")
            print(f"    Distance: neg_mean={result['dist_mean_neg']:.4f}, pos_mean={result['dist_mean_pos']:.4f}")
            print(f"    Separation: {result['separation']:.4f}")

    # 按 gap 分组统计
    print("\n  Per-gap 距离统计:")
    unique_gaps = sorted(set(gap_sizes))
    for g_size in unique_gaps[:12]:  # 最多显示 12 个 gap
        d_g = [d for d, gs in zip(distances, gap_sizes) if gs == g_size]
        if d_g:
            print(f"    Gap={g_size:2d}m: n={len(d_g):4d}, mean={np.mean(d_g):.4f}, median={np.median(d_g):.4f}")

    # ── Spatial Map AUC ──
    spatial_result = None
    if args.spatial_maps and os.path.exists(args.spatial_maps):
        print(f"\n--- Spatial Map (Per-Pixel) ---")
        s = np.load(args.spatial_maps)
        pre_norm_maps = s["pre_norm_maps"]    # [M, D, H, W]
        sp_patch_ids = s["patch_ids"]
        sp_year_months = s["year_months"]
        print(f"  Spatial samples: {pre_norm_maps.shape[0]}, Map: {pre_norm_maps.shape[2]}x{pre_norm_maps.shape[3]}")

        for gap_th in [3, 6]:
            pixel_dists, pixel_labels = build_pairs_from_spatial(
                pre_norm_maps, sp_patch_ids, sp_year_months, gap_threshold=gap_th
            )
            spatial_result = compute_auc_from_pairs(pixel_dists, pixel_labels, f"spatial_gap{gap_th}")
            if spatial_result:
                print(f"\n  Gap threshold = {gap_th} months:")
                print(f"    AUC = {spatial_result['auc']:.4f}")
                print(f"    Pixels: total={spatial_result['n_total']}, pos={spatial_result['n_pos']}, neg={spatial_result['n_neg']}")
                print(f"    Distance: neg_mean={spatial_result['dist_mean_neg']:.4f}, pos_mean={spatial_result['dist_mean_pos']:.4f}")
                print(f"    Separation: {spatial_result['separation']:.4f}")

    # ── 可视化 ──
    print("\n生成可视化...")

    # Fig 1: Distance distribution by gap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    neg_dists = [d for d, l in zip(distances, labels) if l == 0]
    pos_dists = [d for d, l in zip(distances, labels) if l == 1]
    ax.hist(neg_dists, bins=50, alpha=0.6, label=f'Neg (gap<{gap_th}m)', density=True, color='steelblue')
    ax.hist(pos_dists, bins=50, alpha=0.6, label=f'Pos (gap>={gap_th}m)', density=True, color='coral')
    ax.set_xlabel('L2 Distance (Pre-norm)')
    ax.set_ylabel('Density')
    ax.set_title(f'Global Mean: Distance Distribution (AUC={result["auc"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # 按 gap 分组的 boxplot
    gap_groups = {}
    for d, gs in zip(distances, gap_sizes):
        gap_groups.setdefault(gs, []).append(d)
    gaps_to_plot = [g for g in sorted(gap_groups.keys()) if gap_groups[g]]
    data_to_plot = [gap_groups[g] for g in gaps_to_plot]
    bp = ax.boxplot(data_to_plot, labels=[f'{g}m' for g in gaps_to_plot], showfliers=False)
    ax.set_xlabel('Month Gap')
    ax.set_ylabel('L2 Distance (Pre-norm)')
    ax.set_title('Distance by Gap Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "auc_distance_distribution.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  保存: {fig_path}")
    plt.close()

    # Fig 2: ROC Curve
    if len(np.unique(labels)) == 2:
        fig, ax = plt.subplots(figsize=(7, 7))
        fpr, tpr, _ = roc_curve(labels, distances)
        ax.plot(fpr, tpr, linewidth=2, label=f'Global (AUC={result["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve: Pre-norm Temporal Change Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig_path = os.path.join(args.output_dir, "auc_roc_curve.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"  保存: {fig_path}")
        plt.close()

    # ── 保存报告 ──
    report_path = os.path.join(args.output_dir, "auc_report.md")
    with open(report_path, "w") as f:
        f.write("# V12 Pre-norm AUC 验证报告\n\n")
        f.write(f"**Global samples**: {pre_norm.shape[0]}\n\n")
        f.write(f"**Model**: v12_expB (inter=0.1), epoch 39\n\n")

        f.write("## Global Mean Embedding\n\n")
        f.write(f"- AUC (gap>=6m): **{result['auc']:.4f}**\n")
        f.write(f"- Pairs: total={result['n_total']}, pos={result['n_pos']}, neg={result['n_neg']}\n")
        f.write(f"- Distance: neg_mean={result['dist_mean_neg']:.4f}, pos_mean={result['dist_mean_pos']:.4f}\n")
        f.write(f"- Separation: {result['separation']:.4f}\n\n")

        f.write("### Per-Gap Statistics\n\n")
        f.write("| Gap | N | Mean Dist | Median Dist |\n")
        f.write("|-----|---|-----------|-------------|\n")
        for g_size in unique_gaps[:12]:
            d_g = [d for d, gs in zip(distances, gap_sizes) if gs == g_size]
            if d_g:
                f.write(f"| {g_size}m | {len(d_g)} | {np.mean(d_g):.4f} | {np.median(d_g):.4f} |\n")
        f.write("\n")

        if spatial_result:
            f.write("## Spatial Map (Per-Pixel)\n\n")
            f.write(f"- AUC (gap>=6m): **{spatial_result['auc']:.4f}**\n")
            f.write(f"- Pixels: total={spatial_result['n_total']}, pos={spatial_result['n_pos']}, neg={spatial_result['n_neg']}\n")
            f.write(f"- Distance: neg_mean={spatial_result['dist_mean_neg']:.4f}, pos_mean={spatial_result['dist_mean_pos']:.4f}\n")
            f.write(f"- Separation: {spatial_result['separation']:.4f}\n\n")

        if result['auc'] > 0.7:
            f.write("## 结论\n\n**时间敏感性良好** ✅ — Pre-norm 空间能有效区分相邻月份和相隔月份。\n")
        elif result['auc'] > 0.6:
            f.write("## 结论\n\n**时间敏感性中等** ⚠️ — 有一定区分能力，但不够强。\n")
        else:
            f.write("## 结论\n\n**时间敏感性弱** ❌ — 难以区分不同时间间隔。\n")

    print(f"\n报告保存: {report_path}")
    print("=" * 60)
    print("AUC 验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
