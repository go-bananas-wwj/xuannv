"""条件解码器 — 从 embedding 重建目标源."""
from __future__ import annotations

import torch
from torch import nn


class ConditionInjector(nn.Module):
    """条件注入 — 将时间/窗口/元数据条件注入到 embedding 中.
    
    AEF 对齐: 恢复条件注入，使 decoder 能够利用时间信息生成时间条件化的重建.
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        cond_dim = window_code_dim + relative_time_code_dim + metadata_dim
        if cond_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(cond_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )
        else:
            self.proj = None

    def forward(
        self,
        embedding: torch.Tensor,
        window_code: torch.Tensor | None = None,
        relative_time: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.proj is None:
            return embedding
        
        # embedding: [B, D, H, W]
        B, D, H, W = embedding.shape
        cond_parts = []
        if window_code is not None:
            cond_parts.append(window_code)
        if relative_time is not None:
            cond_parts.append(relative_time)
        if metadata is not None:
            cond_parts.append(metadata)
        
        if len(cond_parts) == 0:
            return embedding
        
        cond = torch.cat(cond_parts, dim=-1)  # [B, total_cond_dim]
        cond_emb = self.proj(cond)  # [B, D]
        # 广播到空间维度: [B, D, 1, 1] + [B, D, H, W]
        return embedding + cond_emb[:, :, None, None]


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
        # V13: 1层decoder — 大幅削弱容量，迫使encoder编码更多信息
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor | None = None,
        relative_time: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.injector(embedding_map)  # V13: 不传递任何条件
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
        # V13: 1层decoder — 大幅削弱容量
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Dropout2d(0.3),
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
