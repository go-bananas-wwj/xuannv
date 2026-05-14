#!/usr/bin/env python3
"""V12 Embedding 全面诊断分析.

输入: extract_v12_embedding_diagnostics.py 的输出目录
输出: 诊断报告 + 图表到 {input_dir}/report/
"""
import sys, os, argparse
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="包含 global_means.npz 和 spatial_maps_sample.npz 的目录")
    return parser.parse_args()


def compute_uniformity(z, t=2.0):
    """Hyperspherical uniformity: log(mean(exp(-t * ||zi - zj||^2))).
    越负越好 (分布越均匀).
    z: [N, D]
    """
    N = z.shape[0]
    if N > 5000:
        # 采样计算
        idx = np.random.choice(N, 5000, replace=False)
        z = z[idx]
        N = 5000
    sq_dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    # 排除对角线
    mask = ~np.eye(N, dtype=bool)
    vals = np.exp(-t * sq_dists[mask])
    return float(np.log(vals.mean()))


def compute_variance_stats(z):
    """Per-dimension variance statistics.
    z: [N, D]
    """
    stds = np.std(z, axis=0)  # [D]
    active = stds > 0.1
    very_active = stds > 0.5
    dead = stds < 0.01
    return {
        "stds": stds,
        "active_ratio": float(active.mean()),
        "very_active_ratio": float(very_active.mean()),
        "dead_ratio": float(dead.mean()),
        "std_mean": float(stds.mean()),
        "std_median": float(np.median(stds)),
        "std_min": float(stds.min()),
        "std_max": float(stds.max()),
    }


def compute_covariance_offdiag(z):
    """Covariance matrix off-diagonal energy.
    z: [N, D], 已经中心化
    """
    zc = z - z.mean(axis=0, keepdims=True)
    D = z.shape[1]
    cov = (zc.T @ zc) / (z.shape[0] - 1)  # [D, D]
    # 排除对角线
    mask = ~np.eye(D, dtype=bool)
    offdiag_vals = cov[mask]
    return {
        "offdiag_mean_sq": float(np.mean(offdiag_vals ** 2)),
        "offdiag_max_abs": float(np.max(np.abs(offdiag_vals))),
        "cov_matrix": cov,
    }


def analyze_temporal_sensitivity(patch_ids, year_months, pre_norm):
    """分析时间敏感性.
    
    对每个 patch，计算相邻月份、相隔3月、相隔6月、相隔12月的 embedding 变化.
    pre_norm: [N, D]
    """
    from collections import defaultdict

    # 按 patch 组织
    patch_series = defaultdict(list)
    for i, (pid, ym) in enumerate(zip(patch_ids, year_months)):
        patch_series[pid].append((ym[0], ym[1], pre_norm[i]))

    # 排序
    for pid in patch_series:
        patch_series[pid].sort(key=lambda x: (x[0], x[1]))

    gaps = [1, 3, 6, 12]
    gap_deltas = {g: [] for g in gaps}

    for pid, series in patch_series.items():
        n = len(series)
        for i in range(n):
            y1, m1, emb1 = series[i]
            for j in range(i + 1, n):
                y2, m2, emb2 = series[j]
                # 计算月份差
                month_diff = (y2 - y1) * 12 + (m2 - m1)
                if month_diff <= 0:
                    continue
                delta = np.linalg.norm(emb1 - emb2)
                for g in gaps:
                    if month_diff == g:
                        gap_deltas[g].append(delta)

    results = {}
    for g in gaps:
        vals = gap_deltas[g]
        results[f"gap_{g}m"] = {
            "n_pairs": len(vals),
            "mean": float(np.mean(vals)) if vals else 0.0,
            "median": float(np.median(vals)) if vals else 0.0,
            "std": float(np.std(vals)) if vals else 0.0,
            "values": np.array(vals),
        }
    return results, patch_series


def analyze_per_dimension_temporal(patch_series):
    """Per-dimension 时间敏感度排序.
    
    对每对相邻月份，计算每个维度的变化量，然后聚合.
    """
    D = None
    dim_changes = []

    for pid, series in patch_series.items():
        n = len(series)
        for i in range(n - 1):
            y1, m1, emb1 = series[i]
            y2, m2, emb2 = series[i + 1]
            month_diff = (y2 - y1) * 12 + (m2 - m1)
            if month_diff != 1:
                continue
            if D is None:
                D = len(emb1)
            dim_changes.append(np.abs(emb1 - emb2))

    if not dim_changes:
        return None

    dim_changes = np.stack(dim_changes, axis=0)  # [N_pairs, D]
    mean_change = dim_changes.mean(axis=0)  # [D]
    std_change = dim_changes.std(axis=0)  # [D]

    # 排序
    sorted_idx = np.argsort(mean_change)[::-1]
    return {
        "mean_change": mean_change,
        "std_change": std_change,
        "top_dims": sorted_idx[:20],
        "bottom_dims": sorted_idx[-20:],
    }


def main():
    args = parse_args()
    report_dir = os.path.join(args.input_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 60)
    print("V12 Embedding 全面诊断分析")
    print("=" * 60)

    # ── 加载数据 ──
    global_path = os.path.join(args.input_dir, "global_means.npz")
    spatial_path = os.path.join(args.input_dir, "spatial_maps_sample.npz")

    print(f"\n加载: {global_path}")
    g = np.load(global_path)
    pre_norm = g["pre_norm"]          # [N, D]
    embedding = g["embedding"]        # [N, D] (L2 normalized)
    patch_ids = g["patch_ids"]
    year_months = g["year_months"]    # [N, 2]
    N, D = pre_norm.shape
    print(f"  Global samples: {N}, Dim: {D}")

    has_spatial = os.path.exists(spatial_path)
    if has_spatial:
        print(f"加载: {spatial_path}")
        s = np.load(spatial_path)
        pre_norm_maps = s["pre_norm_maps"]    # [M, D, H, W]
        embedding_maps = s["embedding_maps"]  # [M, D, H, W]
        M = pre_norm_maps.shape[0]
        H, W = pre_norm_maps.shape[2], pre_norm_maps.shape[3]
        print(f"  Spatial samples: {M}, Map size: {H}x{W}")
    else:
        print("  无 spatial map 数据")

    # ═══════════════════════════════════════
    # 维度 1: Embedding 空间质量
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("维度 1: Embedding 空间质量 (Global Mean Space)")
    print("=" * 60)

    # 1.1 Per-Dimension Variance
    var_stats_pre = compute_variance_stats(pre_norm)
    var_stats_emb = compute_variance_stats(embedding)

    print(f"\n[Pre-norm space]")
    print(f"  Active dims (std>0.1): {var_stats_pre['active_ratio']*100:.1f}% ({int(var_stats_pre['active_ratio']*D)}/{D})")
    print(f"  Very active (std>0.5): {var_stats_pre['very_active_ratio']*100:.1f}%")
    print(f"  Dead dims (std<0.01):  {var_stats_pre['dead_ratio']*100:.1f}%")
    print(f"  Std mean/median: {var_stats_pre['std_mean']:.4f} / {var_stats_pre['std_median']:.4f}")
    print(f"  Std min/max: {var_stats_pre['std_min']:.4f} / {var_stats_pre['std_max']:.4f}")

    print(f"\n[L2-normalized space]")
    print(f"  Active dims (std>0.1): {var_stats_emb['active_ratio']*100:.1f}% ({int(var_stats_emb['active_ratio']*D)}/{D})")
    print(f"  Very active (std>0.5): {var_stats_emb['very_active_ratio']*100:.1f}%")
    print(f"  Dead dims (std<0.01):  {var_stats_emb['dead_ratio']*100:.1f}%")
    print(f"  Std mean/median: {var_stats_emb['std_mean']:.4f} / {var_stats_emb['std_median']:.4f}")

    # 1.2 Covariance Off-Diagonal
    cov_pre = compute_covariance_offdiag(pre_norm)
    cov_emb = compute_covariance_offdiag(embedding)
    print(f"\n[Covariance Off-Diagonal Energy]")
    print(f"  Pre-norm:  mean_sq={cov_pre['offdiag_mean_sq']:.6f}, max_abs={cov_pre['offdiag_max_abs']:.4f}")
    print(f"  L2-norm:   mean_sq={cov_emb['offdiag_mean_sq']:.6f}, max_abs={cov_emb['offdiag_max_abs']:.4f}")

    # 1.3 Uniformity
    print(f"\n[Uniformity] 计算中 (采样 5000)...")
    uni_pre = compute_uniformity(pre_norm, t=2.0)
    uni_emb = compute_uniformity(embedding, t=2.0)
    print(f"  Pre-norm:  {uni_pre:.4f}  (期望: -4.0 ~ -1.0)")
    print(f"  L2-norm:   {uni_emb:.4f}")

    # Spatial quality (if available)
    if has_spatial:
        print(f"\n[Spatial Map Statistics]")
        # Flatten all spatial pixels
        spatial_flat = pre_norm_maps.reshape(-1, D)  # [M*H*W, D]
        var_stats_spatial = compute_variance_stats(spatial_flat)
        cov_spatial = compute_covariance_offdiag(spatial_flat)
        print(f"  Pre-norm spatial pixels: {spatial_flat.shape[0]}")
        print(f"  Active dims (std>0.1): {var_stats_spatial['active_ratio']*100:.1f}%")
        print(f"  Dead dims (std<0.01):  {var_stats_spatial['dead_ratio']*100:.1f}%")
        print(f"  Std mean: {var_stats_spatial['std_mean']:.4f}")
        print(f"  Cov offdiag mean_sq: {cov_spatial['offdiag_mean_sq']:.6f}")

    # ═══════════════════════════════════════
    # 维度 2: 时间敏感性
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("维度 2: 时间敏感性")
    print("=" * 60)

    temporal_results, patch_series = analyze_temporal_sensitivity(
        patch_ids, year_months, pre_norm
    )

    print(f"\n[Pre-norm Embedding 变化幅度]")
    for g in [1, 3, 6, 12]:
        r = temporal_results[f"gap_{g}m"]
        print(f"  相隔{g:2d}月: n={r['n_pairs']:4d}, mean={r['mean']:.4f}, median={r['median']:.4f}, std={r['std']:.4f}")

    # 检查单调性
    means = [temporal_results[f"gap_{g}m"]["mean"] for g in [1, 3, 6, 12]]
    monotonic = all(means[i] < means[i+1] for i in range(len(means)-1))
    print(f"\n  单调递增? {monotonic}  (期望: True)")
    print(f"  1月→3月变化比: {means[1]/means[0]:.2f}x")
    print(f"  1月→12月变化比: {means[3]/means[0]:.2f}x")

    # Per-dimension temporal sensitivity
    dim_temporal = analyze_per_dimension_temporal(patch_series)
    if dim_temporal is not None:
        print(f"\n[Per-Dimension 时间敏感度 (相邻月份)]")
        print(f"  Top 10 敏感维度: {dim_temporal['top_dims'][:10].tolist()}")
        print(f"  Top 10 平均变化: {dim_temporal['mean_change'][dim_temporal['top_dims'][:10]].tolist()}")
        print(f"  Bottom 10 维度: {dim_temporal['bottom_dims'][-10:].tolist()}")
        print(f"  变化量范围: [{dim_temporal['mean_change'].min():.4f}, {dim_temporal['mean_change'].max():.4f}]")

    # ═══════════════════════════════════════
    # 可视化
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    # Fig 1: Per-dimension std
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(range(D), var_stats_pre["stds"], alpha=0.7, color='steelblue')
    ax.axhline(0.1, color='r', linestyle='--', label='active threshold=0.1')
    ax.axhline(0.5, color='g', linestyle='--', label='very active=0.5')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Std')
    ax.set_title(f'Pre-norm Per-Dimension Std (active={var_stats_pre["active_ratio"]*100:.1f}%)')
    ax.legend()

    ax = axes[0, 1]
    ax.bar(range(D), var_stats_emb["stds"], alpha=0.7, color='coral')
    ax.axhline(0.1, color='r', linestyle='--', label='active threshold=0.1')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Std')
    ax.set_title(f'L2-norm Per-Dimension Std (active={var_stats_emb["active_ratio"]*100:.1f}%)')
    ax.legend()

    # Fig 2: Covariance heatmap
    ax = axes[1, 0]
    im = ax.imshow(cov_pre["cov_matrix"], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Pre-norm Covariance Matrix')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(cov_emb["cov_matrix"], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('L2-norm Covariance Matrix')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig_path = os.path.join(report_dir, "fig1_variance_covariance.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  保存: {fig_path}")
    plt.close()

    # Fig 3: Temporal sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    gaps = [1, 3, 6, 12]
    means = [temporal_results[f"gap_{g}m"]["mean"] for g in gaps]
    stds = [temporal_results[f"gap_{g}m"]["std"] for g in gaps]
    ax.errorbar(gaps, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Gap (months)')
    ax.set_ylabel('L2 Distance')
    ax.set_title('Temporal Sensitivity: Mean Change vs Gap')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for g in gaps:
        vals = temporal_results[f"gap_{g}m"]["values"]
        if len(vals) > 0:
            ax.hist(vals, bins=30, alpha=0.5, label=f'{g}m (n={len(vals)})', density=True)
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Change by Gap')
    ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(report_dir, "fig2_temporal_sensitivity.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  保存: {fig_path}")
    plt.close()

    # Fig 4: Per-dimension temporal sensitivity
    if dim_temporal is not None:
        fig, ax = plt.subplots(figsize=(14, 5))
        sorted_idx = np.argsort(dim_temporal["mean_change"])[::-1]
        ax.bar(range(D), dim_temporal["mean_change"][sorted_idx], alpha=0.7, color='purple')
        ax.set_xlabel('Dimension (sorted by sensitivity)')
        ax.set_ylabel('Mean Abs Change')
        ax.set_title('Per-Dimension Temporal Sensitivity (Adjacent Months)')
        fig_path = os.path.join(report_dir, "fig3_dimension_sensitivity.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"  保存: {fig_path}")
        plt.close()

    # Fig 5: Spatial std map (if available)
    if has_spatial:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Compute per-pixel std across spatial samples
        per_pixel_std = pre_norm_maps.std(axis=0).mean(axis=0)  # [H, W]
        im = ax.imshow(per_pixel_std, cmap='hot')
        ax.set_title('Mean Per-Pixel Std Across Spatial Samples')
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(report_dir, "fig4_spatial_std.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"  保存: {fig_path}")
        plt.close()

    # ═══════════════════════════════════════
    # 保存报告
    # ═══════════════════════════════════════
    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# V12 Embedding 全面诊断报告\n\n")
        f.write(f"**样本数**: {N} global mean embeddings\n\n")
        f.write(f"**模型**: v12_expB (inter=0.1), epoch 39\n\n")

        f.write("## 1. Embedding 空间质量\n\n")
        f.write("### 1.1 Per-Dimension Variance\n\n")
        f.write(f"| 空间 | Active (std>0.1) | Very Active (std>0.5) | Dead (std<0.01) | Std Mean | Std Median |\n")
        f.write(f"|------|------------------|----------------------|-----------------|----------|------------|\n")
        f.write(f"| Pre-norm | {var_stats_pre['active_ratio']*100:.1f}% | {var_stats_pre['very_active_ratio']*100:.1f}% | {var_stats_pre['dead_ratio']*100:.1f}% | {var_stats_pre['std_mean']:.4f} | {var_stats_pre['std_median']:.4f} |\n")
        f.write(f"| L2-norm | {var_stats_emb['active_ratio']*100:.1f}% | {var_stats_emb['very_active_ratio']*100:.1f}% | {var_stats_emb['dead_ratio']*100:.1f}% | {var_stats_emb['std_mean']:.4f} | {var_stats_emb['std_median']:.4f} |\n")
        f.write("\n")

        f.write("### 1.2 Covariance Off-Diagonal Energy\n\n")
        f.write(f"- Pre-norm: mean_sq={cov_pre['offdiag_mean_sq']:.6f}, max_abs={cov_pre['offdiag_max_abs']:.4f}\n")
        f.write(f"- L2-norm: mean_sq={cov_emb['offdiag_mean_sq']:.6f}, max_abs={cov_emb['offdiag_max_abs']:.4f}\n")
        f.write(f"- 理想值: mean_sq < 0.05\n\n")

        f.write("### 1.3 Uniformity\n\n")
        f.write(f"- Pre-norm: {uni_pre:.4f} (期望: -4.0 ~ -1.0)\n")
        f.write(f"- L2-norm: {uni_emb:.4f}\n\n")

        if has_spatial:
            f.write("### 1.4 Spatial Map Statistics\n\n")
            f.write(f"- Active dims: {var_stats_spatial['active_ratio']*100:.1f}%\n")
            f.write(f"- Dead dims: {var_stats_spatial['dead_ratio']*100:.1f}%\n")
            f.write(f"- Std mean: {var_stats_spatial['std_mean']:.4f}\n\n")

        f.write("## 2. 时间敏感性\n\n")
        f.write("### 2.1 变化幅度 vs 时间间隔\n\n")
        f.write("| 间隔 | 样本对数 | Mean L2 | Median L2 | Std |\n")
        f.write("|------|---------|---------|-----------|-----|\n")
        for g in [1, 3, 6, 12]:
            r = temporal_results[f"gap_{g}m"]
            f.write(f"| {g}个月 | {r['n_pairs']} | {r['mean']:.4f} | {r['median']:.4f} | {r['std']:.4f} |\n")
        f.write("\n")
        f.write(f"**单调递增**: {monotonic}\n\n")
        f.write(f"- 1月→3月变化比: {means[1]/means[0]:.2f}x\n")
        f.write(f"- 1月→12月变化比: {means[3]/means[0]:.2f}x\n\n")

        if dim_temporal is not None:
            f.write("### 2.2 Per-Dimension 时间敏感度\n\n")
            f.write(f"- Top 10 敏感维度: {dim_temporal['top_dims'][:10].tolist()}\n")
            f.write(f"- 变化量范围: [{dim_temporal['mean_change'].min():.4f}, {dim_temporal['mean_change'].max():.4f}]\n\n")

        f.write("## 3. 诊断结论\n\n")
        # 自动生成诊断
        issues = []
        if var_stats_pre['active_ratio'] < 0.5:
            issues.append(f"大量维度 inactive (仅 {var_stats_pre['active_ratio']*100:.1f}% active)")
        if cov_pre['offdiag_mean_sq'] > 0.1:
            issues.append(f"维度间冗余度高 (cov offdiag={cov_pre['offdiag_mean_sq']:.4f})")
        if uni_pre > -0.5:
            issues.append(f"Uniformity 过高 ({uni_pre:.2f})，可能发生坍缩")
        if not monotonic:
            issues.append("时间敏感性不单调——模型时间盲")
        if means[3] / means[0] < 1.5 if means[0] > 0 else True:
            issues.append("12月 vs 1月变化比过低，时间敏感度极弱")

        if issues:
            f.write("**发现的问题**:\n\n")
            for issue in issues:
                f.write(f"- ❌ {issue}\n")
        else:
            f.write("**未发现明显问题** ✅\n\n")

        f.write("\n")

    print(f"\n报告保存: {report_path}")
    print("=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
