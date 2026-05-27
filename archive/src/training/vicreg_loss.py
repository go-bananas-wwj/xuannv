"""VICReg + KoLeo 损失函数 — 替代反坍缩四件套.

参考:
- VICReg: Bardes et al., 2021
- KoLeo: Sablayrolles et al., 2018 (used in DINOv2)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
) -> torch.Tensor:
    """VICReg: Variance-Invariance-Covariance Regularization.

    Args:
        z1, z2: [N, D] — 同一 batch 的两个视图（teacher vs student，或不同增强）
        lambda_var: variance 项权重
        lambda_cov: covariance 项权重（论文推荐 1/25 = 0.04）

    Returns:
        scalar loss
    """
    # 1. Invariance (对齐)
    inv = F.mse_loss(z1, z2)

    # 2. Variance (每维标准差 ≥ 1)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    # 3. Covariance (去相关)
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (z1_c.shape[0] - 1)
    cov_z2 = (z2_c.T @ z2_c) / (z2_c.shape[0] - 1)
    cov_loss = (cov_z1.pow(2).sum() - cov_z1.diagonal().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov_z2.pow(2).sum() - cov_z2.diagonal().pow(2).sum()) / z2.shape[1]

    return inv + lambda_var * var + lambda_cov * cov_loss


def vicreg_loss_components(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg with component breakdown for logging."""
    inv = F.mse_loss(z1, z2)

    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (z1_c.shape[0] - 1)
    cov_z2 = (z2_c.T @ z2_c) / (z2_c.shape[0] - 1)
    cov_loss = (cov_z1.pow(2).sum() - cov_z1.diagonal().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov_z2.pow(2).sum() - cov_z2.diagonal().pow(2).sum()) / z2.shape[1]

    total = inv + lambda_var * var + lambda_cov * cov_loss
    return total, inv, var, cov_loss


def koleo_loss(x: torch.Tensor) -> torch.Tensor:
    """KoLeo: Kozachenko-Leonenko 熵估计正则化.

    强制 batch 内 embedding 的最近邻距离最大化。
    比 uniformity loss 更直接，对小 batch size 更稳定。

    Args:
        x: [N, D] — pre-norm embedding

    Returns:
        scalar loss
    """
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    # 排除自身
    dists = dists + torch.eye(dists.shape[0], device=dists.device, dtype=dists.dtype) * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
