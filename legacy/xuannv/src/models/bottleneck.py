"""VMF Bottleneck — 训练时跳过 L2 归一化，推理时做 L2 归一化.

核心设计:
- 训练时: Conv1x1 → 原始幅度空间 → uniformity/decorrelation/variance 在此空间计算
- 推理时: Conv1x1 → L2 Norm → VMF noise → 球面 embedding

这彻底消除了 L2 Normalization 的 Jacobian 梯度屏障问题。
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def sample_vmf(mean_direction: torch.Tensor, kappa: float) -> torch.Tensor:
    """Von Mises-Fisher 采样近似.

    大 kappa 时: v = mu + N(0, 1/kappa) 投影到球面
    """
    if kappa <= 0:
        return mean_direction
    noise = torch.randn_like(mean_direction) * math.sqrt(1.0 / kappa)
    sample = mean_direction + noise
    return F.normalize(sample, p=2, dim=1)


class VMFBottleneck(nn.Module):
    """改进版 VMF 瓶颈.

    Args:
        channels: 输入通道数 (来自 summary_map)
        embedding_dim: 输出 embedding 维度 (论文标准: 64)
        kappa: VMF 浓度参数 (越大噪声越小)
        skip_l2_training: 训练时是否跳过 L2 归一化
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        kappa: float = 2000.0,
        skip_l2_training: bool = True,
    ) -> None:
        super().__init__()
        self.to_embedding = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.skip_l2_training = skip_l2_training

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W]  summary_map from STP encoder

        Returns:
            embedding_map: [B, D, H, W]  空间密集 embedding (推理时 L2 normalized)
            embedding_vector: [B, D]  全局 embedding (spatial mean)
            pre_norm_embedding: [B, D]  L2 norm 前的全局 embedding
            pre_norm_map: [B, D, H, W]  L2 norm 前的空间 embedding (用于可视化)
        """
        # Conv1x1 压缩维度
        pre_norm_map = self.to_embedding(features)  # [B, D, H, W]

        # 决定 L2 归一化模式
        if self.training and self.skip_l2_training:
            # 训练模式: 跳过 L2, 保留原始幅度
            if self.kappa > 0:
                noise = torch.randn_like(pre_norm_map) * math.sqrt(1.0 / self.kappa)
                embedding_map = pre_norm_map + noise
            else:
                embedding_map = pre_norm_map
        else:
            # 推理模式: 标准 L2 + VMF
            direction = F.normalize(pre_norm_map, p=2, dim=1)
            if self.kappa > 0:
                embedding_map = sample_vmf(direction, self.kappa)
            else:
                embedding_map = direction

        # 全局 embedding = spatial mean
        pre_norm_vector = pre_norm_map.mean(dim=(-2, -1))  # [B, D]
        embedding_vector = embedding_map.mean(dim=(-2, -1))  # [B, D]

        # 推理时对全局 embedding 做 L2 norm
        if not (self.training and self.skip_l2_training):
            embedding_vector = F.normalize(embedding_vector, p=2, dim=1)

        return embedding_map, embedding_vector, pre_norm_vector, pre_norm_map
