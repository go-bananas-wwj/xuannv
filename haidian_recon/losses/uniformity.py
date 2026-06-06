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
