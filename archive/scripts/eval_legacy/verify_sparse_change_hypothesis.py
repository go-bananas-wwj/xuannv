#!/usr/bin/env python3
"""验证「变化信号被全局 cosine similarity 平均淹没」猜想.

实验 1-5：维度差异分析 / 多度量 AUC / 奇异值谱 / 变化像素维度分析 / 最优子集搜索
用法:
    cd /workspace/xuannv
    python scripts/eval/verify_sparse_change_hypothesis.py \
        --embeddings /workspace/outputs/xuannv_backbone_v8_clean/precomputed_embeddings.pt \
        --output-dir /workspace/outputs/xuannv_backbone_v8_clean/sparse_change_analysis
"""
from __future__ import annotations

import sys, json, argparse, warnings, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
DIMS = 64  # embedding 维度

# 设置 matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_embeddings(path: str) -> dict:
    """加载预提取的 embedding."""
    print(f"[Load] 加载 embedding: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    # data[patch_id] = {"eb": [D,H,W], "ea": [D,H,W]}
    return data


def load_annotations() -> tuple[dict, dict]:
    """加载 grid 和标注，返回 (patch_bounds, patch_changes)."""
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                all_changes.append({"geometry": geom, "patch_id": row.get("patch_id", None)})
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

    patch_changes = {}
    for change in all_changes:
        if change["patch_id"]:
            pid = change["patch_id"]
            patch_changes.setdefault(pid, []).append(change["geometry"])
        else:
            pt = change["geometry"].centroid
            for pid, bounds in patch_bounds.items():
                if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                    patch_changes.setdefault(pid, []).append(change["geometry"])
                    break

    return patch_bounds, patch_changes


def rasterize_annotations(changes, bounds, H=64, W=64):
    """将变化图斑光栅化为 HxW mask."""
    resolution = (bounds[2] - bounds[0]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


# ──────────────────────────────────────────
# 距离度量函数 (像素级)
# ──────────────────────────────────────────

def metric_cosine_distance(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    """eb, ea: [D, H, W] -> [H, W]"""
    D, H, W = eb.shape
    fb = eb.reshape(D, -1)
    fa = ea.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    return ((1.0 - cos_sim) / 2.0).reshape(H, W)


def metric_euclidean(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    D, H, W = eb.shape
    diff = eb - ea
    return np.linalg.norm(diff.reshape(D, -1), axis=0).reshape(H, W)


def metric_manhattan(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    D, H, W = eb.shape
    return np.sum(np.abs(eb - ea), axis=0)


def metric_max_diff(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    return np.max(np.abs(eb - ea), axis=0)


def metric_topk_mean_diff(eb: np.ndarray, ea: np.ndarray, k: int) -> np.ndarray:
    diff = np.abs(eb - ea)  # [D, H, W]
    D, H, W = diff.shape
    flat = diff.reshape(D, -1)
    topk = np.partition(flat, -k, axis=0)[-k:]
    return np.mean(topk, axis=0).reshape(H, W)


def metric_sam(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    """Spectral Angle Mapper: arccos(cos_sim)"""
    D, H, W = eb.shape
    fb = eb.reshape(D, -1)
    fa = ea.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    cos_sim = np.sum(fb * fa, axis=0) / np.maximum(nb * na, 1e-8)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim).reshape(H, W)


def metric_diem(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    """DIEM simplified: L2 / sqrt(D)"""
    D, H, W = eb.shape
    diff = eb - ea
    return np.linalg.norm(diff.reshape(D, -1), axis=0).reshape(H, W) / math.sqrt(D)


def metric_minkowski_p05(eb: np.ndarray, ea: np.ndarray) -> np.ndarray:
    """Minkowski with p=0.5"""
    D, H, W = eb.shape
    diff = np.abs(eb - ea)
    return (np.sum(np.sqrt(diff).reshape(D, -1), axis=0) ** 2).reshape(H, W)


def metric_count_threshold(eb: np.ndarray, ea: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """统计差异超过阈值的维度数"""
    diff = np.abs(eb - ea)  # [D, H, W]
    return np.sum(diff > threshold, axis=0).astype(np.float32)


def metric_variance_weighted_l2(eb: np.ndarray, ea: np.ndarray, dim_var: np.ndarray) -> np.ndarray:
    """按维度方差加权的 L2"""
    D, H, W = eb.shape
    diff = eb - ea
    weights = np.sqrt(dim_var).reshape(D, 1, 1)
    weighted = diff * weights
    return np.linalg.norm(weighted.reshape(D, -1), axis=0).reshape(H, W)


ALL_METRICS = {
    "cosine_distance": metric_cosine_distance,
    "euclidean_l2": metric_euclidean,
    "manhattan_l1": metric_manhattan,
    "max_diff": metric_max_diff,
    "top5_mean_diff": lambda eb, ea: metric_topk_mean_diff(eb, ea, 5),
    "top10_mean_diff": lambda eb, ea: metric_topk_mean_diff(eb, ea, 10),
    "top20_mean_diff": lambda eb, ea: metric_topk_mean_diff(eb, ea, 20),
    "sam": metric_sam,
    "diem": metric_diem,
    "minkowski_p05": metric_minkowski_p05,
    "count_threshold_0.1": lambda eb, ea: metric_count_threshold(eb, ea, 0.1),
    "count_threshold_0.2": lambda eb, ea: metric_count_threshold(eb, ea, 0.2),
}


# ──────────────────────────────────────────
# 实验 1: 维度级差异分布分析
# ──────────────────────────────────────────

def experiment_1_dim_diff_distribution(embeddings: dict, patch_changes: dict, patch_bounds: dict, output_dir: Path):
    print("\n" + "=" * 70)
    print("  实验 1: 维度级差异分布分析")
    print("=" * 70)

    # 收集所有有标注 patch 的维度差异
    all_diffs = []
    for pid, changes in patch_changes.items():
        if pid not in embeddings:
            continue
        eb = embeddings[pid]["eb"].numpy()
        ea = embeddings[pid]["ea"].numpy()
        diff = np.abs(eb - ea)  # [D, H, W]
        all_diffs.append(diff.reshape(DIMS, -1).mean(axis=1))  # [D]

    all_diffs = np.stack(all_diffs, axis=0)  # [N_patches, D]
    mean_diff_per_dim = all_diffs.mean(axis=0)  # [D]
    sorted_idx = np.argsort(mean_diff_per_dim)[::-1]
    sorted_diff = mean_diff_per_dim[sorted_idx]
    cumsum = np.cumsum(sorted_diff)
    cumsum_ratio = cumsum / cumsum[-1]

    # 统计
    stats = {}
    for k in [1, 3, 5, 10, 15, 20, 30]:
        stats[f"top{k}_contrib"] = float(cumsum_ratio[min(k - 1, DIMS - 1)])

    print(f"  有标注 patch 数: {len(all_diffs)}")
    print(f"  Top-1 维度贡献:  {stats['top1_contrib'] * 100:.1f}%")
    print(f"  Top-5 维度贡献:  {stats['top5_contrib'] * 100:.1f}%")
    print(f"  Top-10 维度贡献: {stats['top10_contrib'] * 100:.1f}%")
    print(f"  Top-20 维度贡献: {stats['top20_contrib'] * 100:.1f}%")
    print(f"  最大差异维度:    dim-{sorted_idx[0]} = {sorted_diff[0]:.4f}")
    print(f"  最小差异维度:    dim-{sorted_idx[-1]} = {sorted_diff[-1]:.4f}")
    print(f"  差异极差比:      {sorted_diff[0] / sorted_diff[-1]:.2f}x")

    # 保存 JSON
    with open(output_dir / "exp1_dim_diff_stats.json", "w") as f:
        json.dump({
            "stats": stats,
            "mean_diff_per_dim": mean_diff_per_dim.tolist(),
            "sorted_dims": sorted_idx.tolist(),
            "sorted_diff": sorted_diff.tolist(),
        }, f, indent=2)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(range(DIMS), sorted_diff, color="steelblue")
    ax.set_xlabel("Dimension (sorted by diff)")
    ax.set_ylabel("Mean Absolute Diff")
    ax.set_title("Exp1: Per-Dimension Change Magnitude")
    ax.axhline(sorted_diff.mean(), color="red", linestyle="--", label=f"mean={sorted_diff.mean():.4f}")
    ax.legend()

    ax = axes[1]
    ax.plot(range(1, DIMS + 1), cumsum_ratio * 100, color="steelblue", linewidth=2)
    ax.axhline(50, color="orange", linestyle="--", label="50%")
    ax.axhline(80, color="green", linestyle="--", label="80%")
    ax.axhline(90, color="red", linestyle="--", label="90%")
    ax.set_xlabel("Number of Top Dimensions")
    ax.set_ylabel("Cumulative Contribution (%)")
    ax.set_title("Exp1: Cumulative Contribution of Top-k Dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_dim_diff_distribution.png", dpi=150)
    plt.close()
    print(f"  图表已保存: {output_dir / 'exp1_dim_diff_distribution.png'}")

    return mean_diff_per_dim, sorted_idx


# ──────────────────────────────────────────
# 实验 2: 多种距离度量的 AUC 对比
# ──────────────────────────────────────────

def experiment_2_metric_auc_comparison(embeddings: dict, patch_changes: dict, patch_bounds: dict, output_dir: Path):
    print("\n" + "=" * 70)
    print("  实验 2: 多种距离度量的变化检测 AUC 对比")
    print("=" * 70)

    results = defaultdict(list)

    # 计算全局维度方差（用于 variance-weighted metric）
    all_embs = []
    for pid, emb_dict in embeddings.items():
        if pid not in patch_changes:
            continue
        all_embs.append(emb_dict["eb"].numpy().reshape(DIMS, -1))
        all_embs.append(emb_dict["ea"].numpy().reshape(DIMS, -1))
    all_embs = np.concatenate(all_embs, axis=1)  # [D, N_pixels]
    dim_var = np.var(all_embs, axis=1)  # [D]

    for pid, changes in patch_changes.items():
        if pid not in embeddings:
            continue
        eb = embeddings[pid]["eb"].numpy()
        ea = embeddings[pid]["ea"].numpy()
        bounds = patch_bounds[pid]
        mask = rasterize_annotations(changes, bounds)

        flat_mask = mask.flatten()
        if flat_mask.sum() <= 10 or (1 - flat_mask).sum() <= 10:
            continue

        for name, metric_fn in ALL_METRICS.items():
            if name == "variance_weighted_l2":
                cd_map = metric_variance_weighted_l2(eb, ea, dim_var)
            else:
                cd_map = metric_fn(eb, ea)
            flat_cd = cd_map.flatten()

            # 过滤 NaN
            valid = ~(np.isnan(flat_cd) | np.isinf(flat_cd))
            if valid.sum() < 10:
                continue
            try:
                auc = roc_auc_score(flat_mask[valid], flat_cd[valid])
                results[name].append(auc)
            except Exception:
                pass

    # 汇总
    summary = []
    for name, aucs in results.items():
        if len(aucs) > 0:
            summary.append({
                "metric": name,
                "n_patches": len(aucs),
                "mean_auc": float(np.mean(aucs)),
                "median_auc": float(np.median(aucs)),
                "std_auc": float(np.std(aucs)),
                "min_auc": float(np.min(aucs)),
                "max_auc": float(np.max(aucs)),
                "aucs": [float(a) for a in aucs],
            })

    summary.sort(key=lambda x: x["mean_auc"], reverse=True)

    print(f"\n  {'Metric':<25} {'N':>4} {'Mean AUC':>10} {'Median':>10} {'Std':>8} {'Max':>8}")
    print("  " + "-" * 75)
    for s in summary:
        print(f"  {s['metric']:<25} {s['n_patches']:>4} {s['mean_auc']:>10.4f} {s['median_auc']:>10.4f} {s['std_auc']:>8.4f} {s['max_auc']:>8.4f}")

    with open(output_dir / "exp2_metric_auc_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 柱状图: mean AUC
    ax = axes[0]
    names = [s["metric"] for s in summary]
    means = [s["mean_auc"] for s in summary]
    colors = ["red" if m < 0.52 else "orange" if m < 0.55 else "green" if m > 0.60 else "steelblue" for m in means]
    ax.barh(range(len(names)), means, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean AUC")
    ax.set_title("Exp2: Change Detection AUC by Distance Metric")
    ax.axvline(0.5, color="black", linestyle="--", label="random=0.5")
    ax.axvline(0.6, color="green", linestyle="--", alpha=0.5, label="good=0.6")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # 箱线图
    ax = axes[1]
    data_for_box = [s["aucs"] for s in summary]
    bp = ax.boxplot(data_for_box, vert=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names)
    ax.set_xlabel("AUC")
    ax.set_title("Exp2: AUC Distribution per Metric")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.axvline(0.6, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_metric_auc_comparison.png", dpi=150)
    plt.close()
    print(f"  图表已保存: {output_dir / 'exp2_metric_auc_comparison.png'}")

    return summary


# ──────────────────────────────────────────
# 实验 3: 奇异值谱与有效秩分析
# ──────────────────────────────────────────

def experiment_3_svd_spectrum(embeddings: dict, patch_changes: dict, output_dir: Path):
    print("\n" + "=" * 70)
    print("  实验 3: 奇异值谱与有效秩分析")
    print("=" * 70)

    # 收集所有 patch 的 before 和 after embedding
    all_before = []
    all_after = []
    for pid, emb_dict in embeddings.items():
        all_before.append(emb_dict["eb"].numpy().reshape(DIMS, -1))
        all_after.append(emb_dict["ea"].numpy().reshape(DIMS, -1))

    all_before = np.concatenate(all_before, axis=1)  # [D, N_pixels]
    all_after = np.concatenate(all_after, axis=1)

    def _analyze_svd(mat, label):
        # SVD
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)

        # Effective Rank
        p = S / S.sum()
        p = np.maximum(p, 1e-12)
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))

        # Information Abundance
        ia = float(S.sum() / S.max())

        # 方差累积
        cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)

        print(f"\n  [{label}]")
        print(f"    Effective Rank:  {eff_rank:.2f} / {DIMS}")
        print(f"    Information Abundance: {ia:.2f}")
        print(f"    Top-5 方差占比:  {cumvar[4] * 100:.1f}%")
        print(f"    Top-10 方差占比: {cumvar[9] * 100:.1f}%")
        print(f"    Top-20 方差占比: {cumvar[19] * 100:.1f}%")

        return S, cumvar, eff_rank, ia

    S_before, cumvar_b, er_b, ia_b = _analyze_svd(all_before, "Before")
    S_after, cumvar_a, er_a, ia_a = _analyze_svd(all_after, "After")

    stats = {
        "before": {"eff_rank": er_b, "ia": ia_b, "top5": float(cumvar_b[4]), "top10": float(cumvar_b[9]), "top20": float(cumvar_b[19])},
        "after": {"eff_rank": er_a, "ia": ia_a, "top5": float(cumvar_a[4]), "top10": float(cumvar_a[9]), "top20": float(cumvar_a[19])},
    }

    with open(output_dir / "exp3_svd_spectrum.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(range(1, DIMS + 1), S_before / S_before[0], "o-", label="Before", markersize=3)
    ax.plot(range(1, DIMS + 1), S_after / S_after[0], "s-", label="After", markersize=3)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value")
    ax.set_title("Exp3: Singular Value Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, DIMS + 1), cumvar_b * 100, "o-", label="Before", markersize=3)
    ax.plot(range(1, DIMS + 1), cumvar_a * 100, "s-", label="After", markersize=3)
    ax.axhline(80, color="green", linestyle="--", alpha=0.5)
    ax.axhline(90, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Exp3: Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    categories = ["Eff Rank", "IA", "Top5%", "Top10%", "Top20%"]
    before_vals = [er_b, ia_b, cumvar_b[4] * 100, cumvar_b[9] * 100, cumvar_b[19] * 100]
    after_vals = [er_a, ia_a, cumvar_a[4] * 100, cumvar_a[9] * 100, cumvar_a[19] * 100]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, before_vals, width, label="Before")
    ax.bar(x + width / 2, after_vals, width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Value")
    ax.set_title("Exp3: Spectrum Summary Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_svd_spectrum.png", dpi=150)
    plt.close()
    print(f"  图表已保存: {output_dir / 'exp3_svd_spectrum.png'}")


# ──────────────────────────────────────────
# 实验 4: 变化像素 vs 不变像素的维度差异分布
# ──────────────────────────────────────────

def experiment_4_change_vs_nochange_dims(embeddings: dict, patch_changes: dict, patch_bounds: dict, output_dir: Path):
    print("\n" + "=" * 70)
    print("  实验 4: 变化像素 vs 不变像素的维度差异分布")
    print("=" * 70)

    diff_change_sum = np.zeros(DIMS)
    diff_nochange_sum = np.zeros(DIMS)
    change_count = 0
    nochange_count = 0

    for pid, changes in patch_changes.items():
        if pid not in embeddings:
            continue
        eb = embeddings[pid]["eb"].numpy()
        ea = embeddings[pid]["ea"].numpy()
        bounds = patch_bounds[pid]
        mask = rasterize_annotations(changes, bounds)

        diff = np.abs(eb - ea)  # [D, H, W]
        change_pixels = mask > 0.5
        nochange_pixels = ~change_pixels

        if change_pixels.sum() == 0:
            continue

        diff_change_sum += diff[:, change_pixels].sum(axis=1)
        diff_nochange_sum += diff[:, nochange_pixels].sum(axis=1)
        change_count += change_pixels.sum()
        nochange_count += nochange_pixels.sum()

    diff_change = diff_change_sum / np.maximum(change_count, 1)
    diff_nochange = diff_nochange_sum / np.maximum(nochange_count, 1)
    ratio = diff_change / (diff_nochange + 1e-12)

    # 排序
    sorted_idx = np.argsort(ratio)[::-1]

    print(f"  变化像素数:   {change_count}")
    print(f"  不变像素数:   {nochange_count}")
    print(f"  变化/不变比 > 2.0 的维度: {np.sum(ratio > 2.0)} / {DIMS}")
    print(f"  变化/不变比 > 1.5 的维度: {np.sum(ratio > 1.5)} / {DIMS}")
    print(f"\n  Top-10 变化敏感维度 (ratio > 1 排序):")
    for i in range(10):
        d = sorted_idx[i]
        print(f"    dim-{d:02d}: change={diff_change[d]:.4f} nochange={diff_nochange[d]:.4f} ratio={ratio[d]:.2f}")

    with open(output_dir / "exp4_change_vs_nochange.json", "w") as f:
        json.dump({
            "diff_change": diff_change.tolist(),
            "diff_nochange": diff_nochange.tolist(),
            "ratio": ratio.tolist(),
            "sorted_dims": sorted_idx.tolist(),
            "n_ratio_gt_2": int(np.sum(ratio > 2.0)),
            "n_ratio_gt_1_5": int(np.sum(ratio > 1.5)),
        }, f, indent=2)

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    x = np.arange(DIMS)
    width = 0.35
    ax.bar(x - width / 2, diff_change, width, label="Change pixels", alpha=0.8)
    ax.bar(x + width / 2, diff_nochange, width, label="No-change pixels", alpha=0.8)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Absolute Diff")
    ax.set_title("Exp4: Per-Dim Diff: Change vs No-Change")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    colors = ["red" if r > 2 else "orange" if r > 1.5 else "steelblue" for r in ratio]
    ax.bar(range(DIMS), ratio, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", label="ratio=1 (no difference)")
    ax.axhline(1.5, color="orange", linestyle="--", alpha=0.5, label="ratio=1.5")
    ax.axhline(2.0, color="red", linestyle="--", alpha=0.5, label="ratio=2.0")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Ratio (Change / No-Change)")
    ax.set_title("Exp4: Change Sensitivity Ratio per Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    top_dims = sorted_idx[:20]
    ax.barh(range(20), ratio[top_dims], color="steelblue")
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"dim-{d}" for d in top_dims])
    ax.set_xlabel("Ratio")
    ax.set_title("Exp4: Top-20 Most Change-Sensitive Dimensions")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "exp4_change_vs_nochange.png", dpi=150)
    plt.close()
    print(f"  图表已保存: {output_dir / 'exp4_change_vs_nochange.png'}")

    return sorted_idx


# ──────────────────────────────────────────
# 实验 5: 最优子集搜索
# ──────────────────────────────────────────

def experiment_5_optimal_subset(embeddings: dict, patch_changes: dict, patch_bounds: dict, dim_ranking: np.ndarray, output_dir: Path):
    print("\n" + "=" * 70)
    print("  实验 5: 最优子集搜索")
    print("=" * 70)

    k_values = [1, 2, 3, 5, 10, 15, 20, 30, 40, 64]
    results_cosine = []
    results_maxdiff = []

    for k in k_values:
        selected_dims = dim_ranking[:k]

        aucs_cosine = []
        aucs_maxdiff = []

        for pid, changes in patch_changes.items():
            if pid not in embeddings:
                continue
            eb_full = embeddings[pid]["eb"].numpy()
            ea_full = embeddings[pid]["ea"].numpy()
            bounds = patch_bounds[pid]
            mask = rasterize_annotations(changes, bounds)
            flat_mask = mask.flatten()

            if flat_mask.sum() <= 10 or (1 - flat_mask).sum() <= 10:
                continue

            # 子空间 embedding
            eb = eb_full[selected_dims]
            ea = ea_full[selected_dims]

            # Cosine distance on subset
            cd_cos = metric_cosine_distance(eb, ea).flatten()
            valid = ~(np.isnan(cd_cos) | np.isinf(cd_cos))
            if valid.sum() >= 10:
                try:
                    aucs_cosine.append(roc_auc_score(flat_mask[valid], cd_cos[valid]))
                except Exception:
                    pass

            # Max diff on subset
            cd_max = metric_max_diff(eb, ea).flatten()
            valid = ~(np.isnan(cd_max) | np.isinf(cd_max))
            if valid.sum() >= 10:
                try:
                    aucs_maxdiff.append(roc_auc_score(flat_mask[valid], cd_max[valid]))
                except Exception:
                    pass

        results_cosine.append({
            "k": k,
            "mean_auc": float(np.mean(aucs_cosine)) if aucs_cosine else 0.0,
            "std_auc": float(np.std(aucs_cosine)) if aucs_cosine else 0.0,
        })
        results_maxdiff.append({
            "k": k,
            "mean_auc": float(np.mean(aucs_maxdiff)) if aucs_maxdiff else 0.0,
            "std_auc": float(np.std(aucs_maxdiff)) if aucs_maxdiff else 0.0,
        })

    print(f"\n  {'k':>4} {'Cosine AUC':>12} {'MaxDiff AUC':>12}")
    print("  " + "-" * 32)
    for rc, rm in zip(results_cosine, results_maxdiff):
        print(f"  {rc['k']:>4} {rc['mean_auc']:>12.4f} {rm['mean_auc']:>12.4f}")

    with open(output_dir / "exp5_optimal_subset.json", "w") as f:
        json.dump({"cosine": results_cosine, "maxdiff": results_maxdiff}, f, indent=2)

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ks = [r["k"] for r in results_cosine]
    cos_vals = [r["mean_auc"] for r in results_cosine]
    max_vals = [r["mean_auc"] for r in results_maxdiff]

    ax.plot(ks, cos_vals, "o-", label="Cosine Distance (subspace)", linewidth=2, markersize=6)
    ax.plot(ks, max_vals, "s-", label="Max Diff (subspace)", linewidth=2, markersize=6)
    ax.axhline(0.5, color="black", linestyle="--", label="Random")
    ax.axhline(0.6, color="green", linestyle="--", alpha=0.5, label="Good")
    ax.set_xlabel("Number of Top Dimensions (k)")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Exp5: AUC vs Subset Size (Top-k Dimensions)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp5_optimal_subset.png", dpi=150)
    plt.close()
    print(f"  图表已保存: {output_dir / 'exp5_optimal_subset.png'}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True, help="预提取 embedding .pt 文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  验证「变化信号被全局 cosine similarity 平均淹没」猜想")
    print("=" * 70)
    print(f"  Embedding:  {args.embeddings}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)

    # 加载数据
    embeddings = load_embeddings(args.embeddings)
    patch_bounds, patch_changes = load_annotations()

    # 过滤有 embedding 的 patch
    valid_pids = [pid for pid in patch_changes if pid in embeddings]
    print(f"\n  有标注且有 embedding 的 patch: {len(valid_pids)} / {len(patch_changes)}")

    # 执行 5 个实验
    mean_diff, sorted_dims_exp1 = experiment_1_dim_diff_distribution(embeddings, patch_changes, patch_bounds, output_dir)
    summary_exp2 = experiment_2_metric_auc_comparison(embeddings, patch_changes, patch_bounds, output_dir)
    experiment_3_svd_spectrum(embeddings, patch_changes, output_dir)
    dim_ranking = experiment_4_change_vs_nochange_dims(embeddings, patch_changes, patch_bounds, output_dir)
    experiment_5_optimal_subset(embeddings, patch_changes, patch_bounds, dim_ranking, output_dir)

    # 汇总结论
    print("\n" + "=" * 70)
    print("  汇总结论")
    print("=" * 70)

    # 统计成功指标
    successes = 0

    # H1: top-5 贡献 > 50%
    with open(output_dir / "exp1_dim_diff_stats.json") as f:
        exp1 = json.load(f)
    top5 = exp1["stats"]["top5_contrib"]
    if top5 > 0.5:
        print(f"  [PASS] H1 (维度坍缩): top-5 贡献 = {top5*100:.1f}% > 50%")
        successes += 1
    else:
        print(f"  [FAIL] H1 (维度坍缩): top-5 贡献 = {top5*100:.1f}% < 50%")

    # H2: 某度量 AUC > 0.55
    best_metric = max(summary_exp2, key=lambda x: x["mean_auc"])
    if best_metric["mean_auc"] > 0.55:
        print(f"  [PASS] H2 (度量敏感): 最佳度量 = {best_metric['metric']}, AUC = {best_metric['mean_auc']:.4f}")
        successes += 1
    else:
        print(f"  [FAIL] H2 (度量敏感): 最佳度量 AUC = {best_metric['mean_auc']:.4f} < 0.55")

    # H3: Effective Rank < 20
    with open(output_dir / "exp3_svd_spectrum.json") as f:
        exp3 = json.load(f)
    er = max(exp3["before"]["eff_rank"], exp3["after"]["eff_rank"])
    if er < 20:
        print(f"  [PASS] H3 (有效秩): Effective Rank = {er:.1f} < 20")
        successes += 1
    else:
        print(f"  [FAIL] H3 (有效秩): Effective Rank = {er:.1f} >= 20")

    # H4: 存在 ratio > 2.0 的维度
    with open(output_dir / "exp4_change_vs_nochange.json") as f:
        exp4 = json.load(f)
    if exp4["n_ratio_gt_2"] > 0:
        print(f"  [PASS] H4 (变化敏感维度): {exp4['n_ratio_gt_2']} 个维度 ratio > 2.0")
        successes += 1
    else:
        print(f"  [FAIL] H4 (变化敏感维度): 无维度 ratio > 2.0")

    # H5: k=5~10 达到峰值（简化判断：k=10 >= k=64 的 90%）
    with open(output_dir / "exp5_optimal_subset.json") as f:
        exp5 = json.load(f)
    cos_k10 = next(r for r in exp5["cosine"] if r["k"] == 10)["mean_auc"]
    cos_k64 = next(r for r in exp5["cosine"] if r["k"] == 64)["mean_auc"]
    if cos_k10 >= cos_k64 * 0.9:
        print(f"  [PASS] H5 (子空间有效): k=10 AUC = {cos_k10:.4f} >= 90% of k=64 AUC = {cos_k64:.4f}")
        successes += 1
    else:
        print(f"  [FAIL] H5 (子空间有效): k=10 AUC = {cos_k10:.4f} < 90% of k=64 AUC = {cos_k64:.4f}")

    print(f"\n  总体: {successes}/5 个假设被支持")
    if successes >= 3:
        print("  ★ 结论: 猜想被强烈支持！变化信号确实稀疏，全局 cosine similarity 平均淹没了信号。")
        print("  ★ 建议: 优先改进距离度量（使用 max-diff / top-k）或引入维度注意力机制。")
    elif successes <= 1:
        print("  ★ 结论: 猜想不成立。变化信号是全局分布的，问题在 backbone 训练。")
    else:
        print("  ★ 结论: 部分支持。需要进一步分析。")
    print("=" * 70)


if __name__ == "__main__":
    main()
