"""VMF Bottleneck — V11: 对齐 AEF 原文，训练/推理统一 L2 归一化.

核心设计 (对齐 AEF 论文 Supplemental S2.2):
- 训练时: Conv1x1 → L2 Norm → VMF noise → 球面 embedding
- 推理时: Conv1x1 → L2 Norm → VMF sample → 球面 embedding
- 所有损失 (uniformity/consistency/reconstruction) 都在同一 L2-norm 空间计算
- 这是 AEF 原文的精确实现，区别于 V10 的 skip_l2_training 设计

Difference Module (V10 保留):
- forward_dual_window: 显式编码双窗口差分 + change_gate
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
    """V11 VMF 瓶颈 — 对齐 AEF 原文，训练/推理统一 L2 归一化."""

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        kappa: float = 2000.0,
        skip_l2_training: bool = False,  # V11: 忽略此参数，始终 L2 归一化
    ) -> None:
        super().__init__()
        self.to_embedding = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        # V11: skip_l2_training 不再使用，保留参数仅为兼容旧 checkpoint
        self.skip_l2_training = False

        # ── V10: Difference Module (保留) ──
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim // 2, kernel_size=1),
            nn.GroupNorm(8, embedding_dim // 2),
            nn.GELU(),
        )
        self.change_gate = nn.Conv2d(embedding_dim // 2, 1, kernel_size=1)
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
            embedding_map: [B, D, H, W]  L2-normalized
            embedding_vector: [B, D]  L2-normalized (global mean)
            pre_norm_embedding: [B, D]  与 embedding_vector 相同 (兼容旧接口)
            pre_norm_map: [B, D, H, W]  与 embedding_map 相同 (兼容旧接口)
        """
        pre_norm_map = self.to_embedding(features)  # [B, D, H, W]
        embedding_map = self._apply_norm(pre_norm_map)
        embedding_vector = embedding_map.mean(dim=(-2, -1))
        embedding_vector = F.normalize(embedding_vector, p=2, dim=1)
        # V12.1: 返回真正的 pre-norm（用于 VICReg variance/covariance）
        pre_norm_vector = pre_norm_map.mean(dim=(-2, -1))
        # Dummy: diff_encoder / change_gate / fusion 仅在 dual_window 中使用，
        # 添加 dummy 确保单窗口 forward 时这些参数也有梯度（避免 DDP unused param 错误）
        dummy = torch.tensor(0.0, device=features.device)
        for m in [self.diff_encoder, self.change_gate, self.fusion]:
            for p in m.parameters():
                dummy = dummy + p.sum() * 0.0
        embedding_map = embedding_map + dummy
        pre_norm_map = pre_norm_map + dummy
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
            emb_w1: [B, D, H, W] L2-normalized (window 1)
            emb_w2: [B, D, H, W] L2-normalized (window 2)
            pre_w1: [B, D, H, W] 与 emb_w1 相同 (兼容旧接口)
            pre_w2: [B, D, H, W] 与 emb_w2 相同 (兼容旧接口)
            change_score: [B, 1, H, W] 变化概率图 (0~1)
            diff_feat: [B, D/2, H, W] 差分特征
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

        # 4. 应用 L2/VMF (训练/推理统一)
        emb_w1 = self._apply_norm(fused_w1)
        emb_w2 = self._apply_norm(fused_w2)

        # V11: pre_w 与 emb 相同（兼容旧接口，不再区分 pre-norm 和 L2-norm）
        return emb_w1, emb_w2, emb_w1, emb_w2, change_score, diff_feat

    def _apply_norm(self, pre_norm_map: torch.Tensor) -> torch.Tensor:
        """对 pre-norm map 应用 L2 Norm + VMF 噪声.

        V11 变更: 训练/推理统一处理，始终 L2 归一化。
        VMF 噪声在训练时加在方向向量上（AEF 原文方式）。
        """
        direction = F.normalize(pre_norm_map, p=2, dim=1)
        if self.training and self.kappa > 0:
            # VMF 噪声加在方向向量上（AEF 原文: mean direction + noise on S^63）
            noise = torch.randn_like(direction) * math.sqrt(1.0 / self.kappa)
            noisy = direction + noise
            # 重新归一化保持单位长度
            return F.normalize(noisy, p=2, dim=1)
        elif self.kappa > 0:
            return sample_vmf(direction, self.kappa)
        return direction
