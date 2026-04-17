"""条件解码器 — 从 embedding 重建目标源."""
from __future__ import annotations

import torch
from torch import nn


class ConditionInjector(nn.Module):
    """将条件变量 (window_code, relative_time, metadata) 注入到 embedding."""

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
    ) -> None:
        super().__init__()
        total_cond = window_code_dim + relative_time_code_dim + metadata_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(total_cond, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim * 2),
        )
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        # embedding: [B, C, H, W]
        # conditions: [B, cond_dim]
        cond = torch.cat([window_code, relative_time, metadata], dim=-1)
        cond_features = self.cond_proj(cond)  # [B, C*2]
        gate = self.gate(cond_features)  # [B, C]
        gated = embedding.mean(dim=(-2, -1)) * gate  # [B, C]
        return embedding + gated[:, :, None, None]


class ContinuousDecoder(nn.Module):
    """连续值解码器 (S2, S1, Landsat, DEM 等).

    输入: embedding_map at H/2 x W/2 (64x64 for 128 input)
    输出: 重建 at 相同分辨率 (64x64)
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim, window_code_dim, relative_time_code_dim, metadata_dim,
        )
        hidden = embedding_dim * hidden_mult
        # 不升采样, 输出和 embedding 相同分辨率
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        return self.head(x)


class CategoricalDecoder(nn.Module):
    """类别型解码器 (WorldCover, Dynamic World)."""

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim, window_code_dim, relative_time_code_dim, metadata_dim,
        )
        hidden = embedding_dim * hidden_mult
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        return self.head(x)
