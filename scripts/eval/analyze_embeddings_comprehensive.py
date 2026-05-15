"""Round7 8实验 Embedding 全面分析脚本.

在 E50 训练完成后运行，对每个实验的 embedding 做深度分析:
1. Active Dimensions（多阈值）
2. Uniformity（raw / l2 / directional）
3. 维度 std 分布直方图
4. 维度间相关性矩阵
5. 样本间距离分布
6. 时序敏感性（模拟 before/after 窗口差异）
7. Pre-norm vs L2-norm 对比

用法:
    python scripts/eval/analyze_embeddings_comprehensive.py \
        --experiments round7_exp1,round7_exp2,... \
        --n-batches 100 \
        --device cpu \
        --output /workspace/outputs/round7_embedding_analysis.json

输出:
    - JSON 报告: 每个实验的完整指标
    - 文本表格: 8 实验对比
    - 可选: 各维度 std 分布 CSV
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import HarbinPatchDataset
from src.models.model import AEFModel
from src.utils.checkpoint import load_checkpoint
from src.training.losses import (
    raw_uniformity_loss,
    pre_norm_uniformity_loss,
    directional_uniformity_loss,
    variance_regularizer,
    covariance_loss,
    decorrelation_loss,
    bottleneck_orthogonality_loss,
)


# ────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────

def compute_active_dims(embeddings: torch.Tensor, thresholds: list[float]) -> dict[float, int]:
    """计算多个阈值下的 active dimensions.

    Args:
        embeddings: [N, D] 或 [N*H*W, D]
        thresholds: 阈值列表

    Returns:
        {threshold: active_count}
    """
    std_per_dim = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-6)
    return {t: (std_per_dim > t).sum().item() for t in thresholds}


def compute_std_histogram(embeddings: torch.Tensor, bins: list[float] = None) -> dict:
    """计算维度 std 的分布直方图.

    Args:
        embeddings: [N, D]
        bins: 分桶边界

    Returns:
        {bin_label: count, mean_std, min_std, max_std, median_std}
    """
    if bins is None:
        bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 10.0]

    std_per_dim = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-6)
    std_np = std_per_dim.cpu().numpy()

    hist, _ = np.histogram(std_np, bins=bins)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]

    return {
        "histogram": {label: int(count) for label, count in zip(labels, hist)},
        "mean_std": float(std_np.mean()),
        "min_std": float(std_np.min()),
        "max_std": float(std_np.max()),
        "median_std": float(np.median(std_np)),
        "p25_std": float(np.percentile(std_np, 25)),
        "p75_std": float(np.percentile(std_np, 75)),
        "std_of_std": float(std_np.std()),
    }


def compute_dim_correlation_matrix(embeddings: torch.Tensor) -> dict:
    """计算维度间相关性矩阵的统计量.

    Args:
        embeddings: [N, D]

    Returns:
        {mean_abs_corr, max_corr, off_diag_mean, off_diag_std}
    """
    z = embeddings - embeddings.mean(dim=0)
    std = z.std(dim=0) + 1e-4
    z = z / std

    N = z.shape[0]
    c = (z.T @ z) / N  # [D, D]

    abs_corr = c.abs()
    # 排除对角线
    mask = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
    off_diag = abs_corr[mask]

    return {
        "mean_abs_corr": float(off_diag.mean()),
        "max_abs_corr": float(off_diag.max()),
        "min_abs_corr": float(off_diag.min()),
        "off_diag_std": float(off_diag.std()),
        "fraction_above_0.5": float((off_diag > 0.5).sum()) / off_diag.numel(),
        "fraction_above_0.8": float((off_diag > 0.8).sum()) / off_diag.numel(),
    }


def compute_pairwise_distance_stats(embeddings: torch.Tensor, max_samples: int = 2000) -> dict:
    """计算样本间距离分布统计.

    Args:
        embeddings: [N, D]
        max_samples: 最大采样数（避免 OOM）

    Returns:
        {mean_dist, median_dist, min_dist, max_dist, std_dist}
    """
    N = embeddings.shape[0]
    if N > max_samples:
        indices = torch.randperm(N)[:max_samples]
        embeddings = embeddings[indices]
        N = max_samples

    # 计算成对距离 [N, N]
    dist = torch.cdist(embeddings, embeddings, p=2)

    # 排除对角线
    mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    dist_pairs = dist[mask]

    return {
        "mean_dist": float(dist_pairs.mean()),
        "median_dist": float(dist_pairs.median()),
        "min_dist": float(dist_pairs.min()),
        "max_dist": float(dist_pairs.max()),
        "std_dist": float(dist_pairs.std()),
        "p5_dist": float(torch.quantile(dist_pairs, 0.05)),
        "p95_dist": float(torch.quantile(dist_pairs, 0.95)),
        "fraction_below_0.1": float((dist_pairs < 0.1).sum()) / dist_pairs.numel(),
    }


def compute_temporal_sensitivity(model, loader, device, n_batches: int = 30) -> dict:
    """计算时序敏感性：模拟两个时间窗口的 embedding 差异.

    策略:
    1. 取两个不同的 batch 作为 window1 和 window2
    2. 计算每个 patch 在两批中的 embedding 差异
    3. 统计距离分布

    Args:
        model: AEFModel
        loader: DataLoader
        device: torch.device
        n_batches: 采样的 batch 数

    Returns:
        {mean_cosine_sim, mean_l2_dist, fraction_similar}
    """
    model.eval()
    all_pre_w1 = []
    all_pre_w2 = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches * 2:
                break

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            outputs = model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
            )

            pre_norm = outputs.pre_norm_embedding  # [B, D]

            if batch_idx % 2 == 0:
                all_pre_w1.append(pre_norm.cpu())
            else:
                all_pre_w2.append(pre_norm.cpu())

    if len(all_pre_w1) == 0 or len(all_pre_w2) == 0:
        return {"error": "not enough batches"}

    # 合并并截断到相同长度
    w1 = torch.cat(all_pre_w1, dim=0)  # [N1, D]
    w2 = torch.cat(all_pre_w2, dim=0)  # [N2, D]
    min_n = min(w1.shape[0], w2.shape[0])
    w1 = w1[:min_n]
    w2 = w2[:min_n]

    # L2 归一化后计算 cosine similarity
    w1_norm = F.normalize(w1, p=2, dim=1)
    w2_norm = F.normalize(w2, p=2, dim=1)
    cos_sim = (w1_norm * w2_norm).sum(dim=1)  # [N]

    # L2 距离（pre-norm）
    l2_dist = torch.norm(w1 - w2, p=2, dim=1)  # [N]

    return {
        "mean_cosine_sim": float(cos_sim.mean()),
        "median_cosine_sim": float(cos_sim.median()),
        "fraction_similar": float((cos_sim > 0.5).sum()) / cos_sim.numel(),
        "fraction_orthogonal": float((cos_sim < 0.2).sum()) / cos_sim.numel(),
        "mean_l2_dist": float(l2_dist.mean()),
        "median_l2_dist": float(l2_dist.median()),
    }


# ────────────────────────────────────────────
# 单个实验分析
# ────────────────────────────────────────────

def analyze_single_experiment(
    exp_id: int,
    exp_dir: Path,
    n_batches: int = 100,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """分析单个实验的 embedding."""
    print(f"\n{'='*70}")
    print(f"分析 exp{exp_id}: {exp_dir.name}")
    print(f"{'='*70}")

    t0 = time.time()
    device = torch.device(device_str)

    # 查找 config 和 checkpoint
    config_dir = Path("/workspace/xuannv/configs/round7_8gpu")
    config_candidates = list(config_dir.glob(f"exp{exp_id}_*.yaml"))
    if not config_candidates:
        return {"error": f"config not found for exp{exp_id}"}
    config_path = config_candidates[0]

    checkpoint_candidates = list(exp_dir.glob("epoch_*.pt")) + list(exp_dir.glob("epoch_best_*.pt"))
    if not checkpoint_candidates:
        return {"error": f"checkpoint not found for exp{exp_id}"}
    # 取最新的
    checkpoint_path = sorted(checkpoint_candidates, key=lambda p: p.stat().st_mtime)[-1]

    print(f"Config: {config_path.name}")
    print(f"Checkpoint: {checkpoint_path.name}")

    # 加载 config
    cfg = load_config(str(config_path))
    cfg.data.preload = False  # 避免长时间预加载

    # 构建模型
    model = AEFModel(cfg).to(device)
    state = load_checkpoint(str(checkpoint_path), device=device_str)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # 构建 dataloader
    dataset = HarbinPatchDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    # 收集 embedding
    all_pre_norm_emb = []      # [N, D] 全局 mean
    all_pre_norm_map = []      # [N, D, H, W] 空间 map
    all_l2_norm_emb = []       # [N, D] L2 归一化后的 mean
    all_l2_norm_map = []       # [N, D, H, W] L2 归一化后的 map

    print(f"收集 {n_batches} batches 的 embedding...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            outputs = model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
            )

            # Pre-norm
            pre_emb = outputs.pre_norm_embedding  # [B, D]
            pre_map = outputs.pre_norm_map        # [B, D, H, W]

            # L2-norm（手动计算，因为模型训练时 skip_l2）
            l2_map = F.normalize(pre_map, p=2, dim=1)
            l2_emb = l2_map.mean(dim=(-2, -1))
            l2_emb = F.normalize(l2_emb, p=2, dim=1)

            all_pre_norm_emb.append(pre_emb.cpu())
            all_pre_norm_map.append(pre_map.cpu())
            all_l2_norm_emb.append(l2_emb.cpu())
            all_l2_norm_map.append(l2_map.cpu())

    # 合并
    pre_emb = torch.cat(all_pre_norm_emb, dim=0)      # [N, D]
    pre_map = torch.cat(all_pre_norm_map, dim=0)      # [N, D, H, W]
    l2_emb = torch.cat(all_l2_norm_emb, dim=0)        # [N, D]
    l2_map = torch.cat(all_l2_norm_map, dim=0)        # [N, D, H, W]

    # 空间 flatten
    N, D, H, W = pre_map.shape
    pre_spatial_flat = pre_map.permute(0, 2, 3, 1).reshape(-1, D)  # [N*H*W, D]
    l2_spatial_flat = l2_map.permute(0, 2, 3, 1).reshape(-1, D)    # [N*H*W, D]

    print(f"总样本: {N}, 维度: {D}, 空间尺寸: {H}x{W}")

    # ── 1. Active Dimensions ──
    print("计算 Active Dimensions...")
    thresholds = [0.05, 0.10, 0.15, 0.20]
    active_emb = compute_active_dims(pre_emb, thresholds)
    active_spatial = compute_active_dims(pre_spatial_flat, thresholds)
    active_l2_emb = compute_active_dims(l2_emb, thresholds)
    active_l2_spatial = compute_active_dims(l2_spatial_flat, thresholds)

    # ── 2. Uniformity ──
    print("计算 Uniformity...")
    unif_pre_raw = raw_uniformity_loss(pre_emb).item()
    unif_pre_normed = pre_norm_uniformity_loss(pre_emb).item()
    unif_pre_directional = directional_uniformity_loss(pre_emb).item()
    unif_l2_raw = raw_uniformity_loss(l2_emb).item()
    unif_l2_normed = pre_norm_uniformity_loss(l2_emb).item()

    # ── 3. STD 分布 ──
    print("计算 STD 分布...")
    std_emb = compute_std_histogram(pre_emb)
    std_spatial = compute_std_histogram(pre_spatial_flat)
    std_l2_emb = compute_std_histogram(l2_emb)

    # ── 4. 维度相关性 ──
    print("计算维度相关性...")
    corr_emb = compute_dim_correlation_matrix(pre_emb)
    corr_spatial = compute_dim_correlation_matrix(pre_spatial_flat)

    # ── 5. 样本间距离 ──
    print("计算样本间距离...")
    dist_emb = compute_pairwise_distance_stats(pre_emb, max_samples=1000)
    dist_l2 = compute_pairwise_distance_stats(l2_emb, max_samples=1000)

    # ── 6. 时序敏感性 ──
    print("计算时序敏感性...")
    temporal = compute_temporal_sensitivity(model, loader, device, n_batches=min(30, n_batches))

    # ── 7. VICReg 指标 ──
    print("计算 VICReg 指标...")
    var_emb = variance_regularizer(pre_emb, min_std=1.0).item()
    cov_emb = covariance_loss(pre_emb).item()
    decorr_emb = decorrelation_loss(pre_emb).item()

    # 收集权重正交性（从模型）
    orth_loss = bottleneck_orthogonality_loss(
        model.bottleneck.to_embedding.weight
    ).item()

    elapsed = time.time() - t0
    print(f"分析完成，耗时 {elapsed:.1f}s")

    return {
        "exp_id": exp_id,
        "exp_name": exp_dir.name,
        "checkpoint": checkpoint_path.name,
        "n_samples": N,
        "embedding_dim": D,
        "spatial_size": f"{H}x{W}",
        "analysis_time_sec": elapsed,

        # 1. Active Dimensions
        "active_dims": {
            "pre_norm_emb": active_emb,
            "pre_norm_spatial": active_spatial,
            "l2_norm_emb": active_l2_emb,
            "l2_norm_spatial": active_l2_spatial,
        },

        # 2. Uniformity
        "uniformity": {
            "pre_norm_raw": unif_pre_raw,
            "pre_norm_directional": unif_pre_directional,
            "pre_norm_normed": unif_pre_normed,
            "l2_norm_raw": unif_l2_raw,
            "l2_norm_normed": unif_l2_normed,
        },

        # 3. STD 分布
        "std_distribution": {
            "pre_norm_emb": std_emb,
            "pre_norm_spatial": std_spatial,
            "l2_norm_emb": std_l2_emb,
        },

        # 4. 维度相关性
        "dim_correlation": {
            "pre_norm_emb": corr_emb,
            "pre_norm_spatial": corr_spatial,
        },

        # 5. 样本间距离
        "pairwise_distance": {
            "pre_norm_emb": dist_emb,
            "l2_norm_emb": dist_l2,
        },

        # 6. 时序敏感性
        "temporal_sensitivity": temporal,

        # 7. VICReg + Orth
        "vicreg_orth": {
            "variance": var_emb,
            "covariance": cov_emb,
            "decorrelation": decorr_emb,
            "bottleneck_orthogonality": orth_loss,
        },
    }


# ────────────────────────────────────────────
# 报告生成
# ────────────────────────────────────────────

def print_comparison_table(results: list[dict]):
    """打印 8 实验对比表格."""
    print("\n" + "=" * 120)
    print(" Round7 Embedding 全面分析 — 8 实验对比")
    print("=" * 120)

    # 表 1: Active Dimensions
    print("\n【表 1】Active Dimensions (多阈值对比)")
    print("-" * 120)
    print(f"{'Exp':<8} {'Name':<22} {'T=0.05':<10} {'T=0.10':<10} {'T=0.15':<10} {'T=0.20':<10} {'L2(0.05)':<10}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            continue
        name = r["exp_name"].replace("round7_", "")
        a = r["active_dims"]["pre_norm_emb"]
        l2 = r["active_dims"]["l2_norm_emb"]
        print(f"exp{r['exp_id']:<2} {name:<22} {a[0.05]:<10} {a[0.10]:<10} {a[0.15]:<10} {a[0.20]:<10} {l2[0.05]:<10}")

    # 表 2: Uniformity
    print("\n【表 2】Uniformity (越负越分散)")
    print("-" * 120)
    print(f"{'Exp':<8} {'Pre Raw':<12} {'Pre Dir':<12} {'Pre Normed':<12} {'L2 Raw':<12} {'L2 Normed':<12}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            continue
        u = r["uniformity"]
        print(f"exp{r['exp_id']:<2} {u['pre_norm_raw']:<12.3f} {u['pre_norm_directional']:<12.3f} "
              f"{u['pre_norm_normed']:<12.3f} {u['l2_norm_raw']:<12.3f} {u['l2_norm_normed']:<12.3f}")

    # 表 3: STD 分布
    print("\n【表 3】Pre-norm Embedding STD 分布")
    print("-" * 120)
    print(f"{'Exp':<8} {'Mean':<10} {'Min':<10} {'P25':<10} {'Median':<10} {'P75':<10} {'Max':<10}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            continue
        s = r["std_distribution"]["pre_norm_emb"]
        print(f"exp{r['exp_id']:<2} {s['mean_std']:<10.4f} {s['min_std']:<10.4f} {s['p25_std']:<10.4f} "
              f"{s['median_std']:<10.4f} {s['p75_std']:<10.4f} {s['max_std']:<10.4f}")

    # 表 4: 维度相关性
    print("\n【表 4】维度间相关性 (Pre-norm Embedding)")
    print("-" * 120)
    print(f"{'Exp':<8} {'MeanAbs':<10} {'MaxAbs':<10} {'>0.5%':<10} {'>0.8%':<10}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            continue
        c = r["dim_correlation"]["pre_norm_emb"]
        print(f"exp{r['exp_id']:<2} {c['mean_abs_corr']:<10.4f} {c['max_abs_corr']:<10.4f} "
              f"{c['fraction_above_0.5']*100:<9.1f}% {c['fraction_above_0.8']*100:<9.1f}%")

    # 表 5: 时序敏感性
    print("\n【表 5】时序敏感性 (不同 batch 的 embedding 差异)")
    print("-" * 120)
    print(f"{'Exp':<8} {'MeanCos':<12} {'MedCos':<12} {'Sim>0.5':<10} {'Orth<0.2':<10} {'MeanL2':<12}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            continue
        t = r["temporal_sensitivity"]
        if "error" in t:
            print(f"exp{r['exp_id']:<2} N/A")
            continue
        print(f"exp{r['exp_id']:<2} {t['mean_cosine_sim']:<12.4f} {t['median_cosine_sim']:<12.4f} "
              f"{t['fraction_similar']*100:<9.1f}% {t['fraction_orthogonal']*100:<9.1f}% {t['mean_l2_dist']:<12.4f}")

    # 表 6: VICReg + Orth
    print("\n【表 6】VICReg + Bottleneck 正交性")
    print("-" * 120)
    print(f"{'Exp':<8} {'Var':<10} {'Cov':<10} {'Decorr':<12} {'Orth':<10}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            continue
        v = r["vicreg_orth"]
        print(f"exp{r['exp_id']:<2} {v['variance']:<10.4f} {v['covariance']:<10.4f} "
              f"{v['decorrelation']:<12.4f} {v['bottleneck_orthogonality']:<10.4f}")

    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="1,2,3,4,5,6,7,8",
                        help="实验 ID 列表，逗号分隔，如 '1,2,3,4,5,6,7,8'")
    parser.add_argument("--n-batches", type=int, default=100,
                        help="每个实验采样的 batch 数")
    parser.add_argument("--device", default="cpu",
                        help="运行设备 (cpu / npu:0 / npu:1 / ...)")
    parser.add_argument("--output", default="/workspace/outputs/round7_embedding_analysis.json",
                        help="JSON 报告输出路径")
    args = parser.parse_args()

    exp_ids = [int(x.strip()) for x in args.experiments.split(",")]

    results = []
    for exp_id in exp_ids:
        dirs = list(Path("/workspace/outputs").glob(f"round7_exp{exp_id}_*"))
        if not dirs:
            print(f"⚠️ exp{exp_id}: 目录未找到，跳过")
            results.append({"exp_id": exp_id, "error": "directory not found"})
            continue
        exp_dir = dirs[0]

        try:
            result = analyze_single_experiment(
                exp_id, exp_dir,
                n_batches=args.n_batches,
                device_str=args.device,
            )
            results.append(result)
        except Exception as e:
            print(f"❌ exp{exp_id} 分析失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({"exp_id": exp_id, "error": str(e)})

    # 打印对比表
    print_comparison_table(results)

    # 保存 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 JSON 报告已保存: {output_path}")
    print(f"   大小: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
