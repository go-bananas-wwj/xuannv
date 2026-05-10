"""VMF Bottleneck — V10: 加入 Difference Module.

核心设计:
- 训练时: Conv1x1 → 原始幅度空间 → uniformity/decorrelation/variance 在此空间计算
- 推理时: Conv1x1 → L2 Norm → VMF noise → 球面 embedding
- V10 新增: forward_dual_window — 显式编码双窗口差分

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
    """改进版 VMF 瓶颈 — V10 加入 Difference Module."""

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

        # ── V10: Difference Module ──
        # 差分编码: 将两个窗口的 pre-norm embedding 拼接后编码差分特征
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim // 2, kernel_size=1),
            nn.GroupNorm(8, embedding_dim // 2),
            nn.GELU(),
        )
        # 变化门控: 预测每个像素位置的变化概率
        self.change_gate = nn.Conv2d(embedding_dim // 2, 1, kernel_size=1)
        # 融合: 将原始 embedding 与差分特征融合回 embedding_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim + embedding_dim // 2, embedding_dim, kernel_size=1),
            nn.GroupNorm(8, embedding_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """标准单窗口前向.

        Args:
            features: [B, C, H, W]  summary_map from STP encoder

        Returns:
            embedding_map: [B, D, H, W]
            embedding_vector: [B, D]
            pre_norm_embedding: [B, D]
            pre_norm_map: [B, D, H, W]
        """
        pre_norm_map = self.to_embedding(features)  # [B, D, H, W]
        embedding_map = self._apply_norm(pre_norm_map)
        pre_norm_vector = pre_norm_map.mean(dim=(-2, -1))
        embedding_vector = embedding_map.mean(dim=(-2, -1))
        if not (self.training and self.skip_l2_training):
            embedding_vector = F.normalize(embedding_vector, p=2, dim=1)
        return embedding_map, embedding_vector, pre_norm_vector, pre_norm_map

    def forward_dual_window(
        self,
        feat_w1: torch.Tensor,
        feat_w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """V10: 双窗口前向 — 显式编码差分.

        Args:
            feat_w1: [B, C, H, W] summary_map (window 1)
            feat_w2: [B, C, H, W] summary_map (window 2)

        Returns:
            emb_w1: [B, D, H, W] 窗口1 embedding (L2 normalized if inference)
            emb_w2: [B, D, H, W] 窗口2 embedding
            pre_w1: [B, D, H, W] 窗口1 pre-norm
            pre_w2: [B, D, H, W] 窗口2 pre-norm
            change_score: [B, 1, H, W] 变化概率图 (0~1)
            diff_feat: [B, D/2, H, W] 差分特征 (可用于辅助监督)
        """
        # 1. 各自 embedding (pre-norm)
        pre_w1 = self.to_embedding(feat_w1)  # [B, D, H, W]
        pre_w2 = self.to_embedding(feat_w2)  # [B, D, H, W]

        # 2. 显式差分编码
        diff_input = torch.cat([pre_w1, pre_w2], dim=1)  # [B, 2D, H, W]
        diff_feat = self.diff_encoder(diff_input)          # [B, D/2, H, W]
        change_score = torch.sigmoid(self.change_gate(diff_feat))  # [B, 1, H, W]

        # 3. 融合: 原始 embedding + 差分特征 → 增强版 embedding
        enhanced_w1 = torch.cat([pre_w1, diff_feat], dim=1)  # [B, 1.5D, H, W]
        enhanced_w2 = torch.cat([pre_w2, diff_feat], dim=1)
        fused_w1 = self.fusion(enhanced_w1)  # [B, D, H, W]
        fused_w2 = self.fusion(enhanced_w2)

        # 4. 应用 L2/VMF (同标准 forward 逻辑)
        emb_w1 = self._apply_norm(fused_w1)
        emb_w2 = self._apply_norm(fused_w2)

        return emb_w1, emb_w2, pre_w1, pre_w2, change_score, diff_feat

    def _apply_norm(self, pre_norm_map: torch.Tensor) -> torch.Tensor:
        """对 pre-norm map 应用 L2/VMF (训练/推理区分)."""
        if self.training and self.skip_l2_training:
            if self.kappa > 0:
                noise = torch.randn_like(pre_norm_map) * math.sqrt(1.0 / self.kappa)
                return pre_norm_map + noise
            return pre_norm_map
        else:
            direction = F.normalize(pre_norm_map, p=2, dim=1)
            if self.kappa > 0:
                return sample_vmf(direction, self.kappa)
            return direction
