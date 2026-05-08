#!/usr/bin/env python3
"""V7 Minimal embedding 质量分析 — 对比两个 checkpoint 的 embedding 空间分布."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import torch
import torch.nn.functional as F
import numpy as np

from src.config import load_config
from src.data.builder import build_dataloader
from src.models.model import AEFModel
from src.training.vicreg_loss import koleo_loss
from src.training.losses import pre_norm_uniformity_loss, directional_uniformity_loss


def analyze_embeddings(model, dataloader, device, max_batches: int = 50):
    """提取 embedding 并计算质量指标."""
    model.eval()
    all_pre_norm = []
    all_norm = []
    all_recon = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
                target_loss_type=batch.get("target_loss_type"),
                target_source_idx=batch.get("target_source_idx"),
            )

            all_pre_norm.append(out.pre_norm_embedding.cpu())
            all_norm.append(out.embedding.cpu())

    pre_norm = torch.cat(all_pre_norm, dim=0)  # [N, D]
    norm = torch.cat(all_norm, dim=0)  # [N, D]

    # 1. 基本统计
    N, D = pre_norm.shape
    pre_mean = pre_norm.mean(dim=0)
    pre_std = pre_norm.std(dim=0)
    norm_mean = norm.mean(dim=0)
    norm_std = norm.std(dim=0)

    # 2. VICReg variance (每维 std >= gamma)
    gamma = 1.0
    var_loss = torch.mean(F.relu(gamma - pre_std)).item()
    n_dim_low_std = (pre_std < gamma).sum().item()

    # 3. VICReg covariance
    z = pre_norm - pre_mean
    cov = (z.T @ z) / (N - 1)
    cov_loss = ((cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D).item()

    # 4. KoLeo
    koleo = koleo_loss(pre_norm.to(device)).item()

    # 5. Uniformity
    pre_unif = pre_norm_uniformity_loss(pre_norm.to(device)).item()
    enc_unif = directional_uniformity_loss(pre_norm.to(device)).item()

    # 6. 最近邻距离分布
    pre_norm_n = F.normalize(pre_norm, p=2, dim=-1)
    dists = torch.cdist(pre_norm_n, pre_norm_n, p=2)
    dists = dists + torch.eye(N, dtype=dists.dtype) * 1e6
    nn_dists = dists.min(dim=1)[0]
    nn_mean = nn_dists.mean().item()
    nn_std = nn_dists.std().item()
    nn_min = nn_dists.min().item()
    nn_max = nn_dists.max().item()

    # 7. L2 归一化后的统计
    norm_n = F.normalize(norm, p=2, dim=-1)
    norm_dists = torch.cdist(norm_n, norm_n, p=2)
    norm_dists = norm_dists + torch.eye(N, dtype=norm_dists.dtype) * 1e6
    norm_nn = norm_dists.min(dim=1)[0]

    return {
        "N": N, "D": D,
        "pre_norm_mean_abs": pre_norm.abs().mean().item(),
        "pre_norm_std_mean": pre_std.mean().item(),
        "pre_norm_std_min": pre_std.min().item(),
        "pre_norm_std_max": pre_std.max().item(),
        "n_dim_low_std": n_dim_low_std,
        "vicreg_var": var_loss,
        "vicreg_cov": cov_loss,
        "koleo": koleo,
        "pre_unif": pre_unif,
        "enc_unif": enc_unif,
        "nn_mean": nn_mean,
        "nn_std": nn_std,
        "nn_min": nn_min,
        "nn_max": nn_max,
        "norm_std_mean": norm_std.mean().item(),
        "norm_nn_mean": norm_nn.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--name", type=str, default="model", help="模型名称标签")
    parser.add_argument("--max-batches", type=int, default=50, help="分析 batch 数")
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    cfg = load_config("configs/qwen_v7_minimal.yaml")
    
    # 使用项目统一的 device 获取逻辑
    from src.utils.device import get_device
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Analyzing: {args.name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # 加载模型
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 加载数据（少量 batch，禁用预加载加速启动）
    cfg.data.preload = False
    dataloader = build_dataloader(cfg, training=False, distributed=False)

    # 分析
    results = analyze_embeddings(model, dataloader, device, args.max_batches)

    print(f"\n样本数: {results['N']} | 维度: {results['D']}")
    print(f"\n[Pre-Norm Embedding]")
    print(f"  平均绝对值: {results['pre_norm_mean_abs']:.4f}")
    print(f"  每维 std (均值): {results['pre_norm_std_mean']:.4f}")
    print(f"  每维 std (范围): [{results['pre_norm_std_min']:.4f}, {results['pre_norm_std_max']:.4f}]")
    print(f"  低 std 维度数 (<1.0): {results['n_dim_low_std']} / {results['D']}")
    print(f"  VICReg Variance: {results['vicreg_var']:.4f}")
    print(f"  VICReg Covariance: {results['vicreg_cov']:.4f}")
    print(f"  KoLeo: {results['koleo']:.4f}")
    print(f"  Pre-Unif: {results['pre_unif']:.4f}")
    print(f"  Enc-Unif: {results['enc_unif']:.4f}")
    print(f"\n[Nearest Neighbor Distances (pre-norm L2)]")
    print(f"  Mean: {results['nn_mean']:.4f} | Std: {results['nn_std']:.4f}")
    print(f"  Min:  {results['nn_min']:.4f} | Max: {results['nn_max']:.4f}")
    print(f"\n[L2-Normalized Embedding]")
    print(f"  每维 std (均值): {results['norm_std_mean']:.4f}")
    print(f"  NN distance (mean): {results['norm_nn_mean']:.4f}")
    print(f"{'='*60}\n")

    # 保存结果到 JSON
    output_path = Path(f"/tmp/v7_emb_{args.name.replace(' ', '_')}.json")
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
