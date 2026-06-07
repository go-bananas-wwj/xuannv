"""Embedding推散损失 — 防止64维坍缩."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def uniformity_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    RBF uniformity loss (Wang & Isola 2020)。
    手动计算距离矩阵，避免NPU上torch.cdist的兼容性问题。

    Args:
        embeddings: [B, D]

    Returns:
        loss: scalar tensor
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    z = F.normalize(embeddings, p=2, dim=-1)

    # 手动计算成对距离（NPU-safe）
    z_i = z.unsqueeze(1)  # [B, 1, D]
    z_j = z.unsqueeze(0)  # [1, B, D]
    sq_pdist = ((z_i - z_j) ** 2).sum(-1)  # [B, B]

    # 只算上三角（排除对角线）
    B = z.shape[0]
    mask = torch.triu(torch.ones(B, B, device=z.device, dtype=torch.bool), diagonal=1)
    sq_pdist_pairs = sq_pdist[mask]

    if sq_pdist_pairs.numel() == 0:
        return embeddings.new_tensor(0.0)

    loss = torch.logsumexp(-2.0 * sq_pdist_pairs, dim=0) - math.log(sq_pdist_pairs.shape[0])
    return loss


def spatial_uniformity_loss(embedding_map: torch.Tensor, n_samples: int = 256) -> torch.Tensor:
    """
    空间embedding的推散损失。
    从空间特征图中采样位置，计算uniformity。

    Args:
        embedding_map: [B, D, H, W]
        n_samples: 每batch随机采样的空间位置数

    Returns:
        loss: scalar tensor
    """
    B, D, H, W = embedding_map.shape
    if B == 0 or H == 0 or W == 0:
        return embedding_map.new_tensor(0.0)

    # 把所有空间位置flatten: [B, H*W, D]
    emb = embedding_map.permute(0, 2, 3, 1).reshape(B, H * W, D)

    total_positions = H * W
    if total_positions <= n_samples:
        # 位置不够，全部使用
        sampled = emb.reshape(-1, D)  # [B*H*W, D]
    else:
        # 随机采样位置
        idx = torch.randint(0, total_positions, (n_samples,), device=emb.device)
        sampled = emb[:, idx, :]  # [B, n_samples, D]
        sampled = sampled.reshape(-1, D)  # [B*n_samples, D]

    return uniformity_loss(sampled)
