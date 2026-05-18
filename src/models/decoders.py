"""条件解码器 — 从 embedding 重建目标源.

对齐 AEF 论文 Supplemental S2.4:
"Implicit decoders were two-hidden-layer MLPs with a width of 512."
"a small decoder is applied at each pixel embedding"

每个空间位置的 embedding 向量独立过 MLP，输出逐像素重建。
"""
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
    """连续值解码器 — 逐像素 MLP (对齐 AEF 论文 S2.4).

    论文原文: "Implicit decoders were two-hidden-layer MLPs with a width of 512."
    对每个空间位置 (h, w) 的 embedding 向量独立做 MLP 解码。

    输入: embedding_map [B, D, H, W]
    输出: reconstruction [B, out_ch, H, W] (与 embedding_map 同分辨率)
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_width: int = 512,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim, window_code_dim, relative_time_code_dim, metadata_dim,
        )
        # 条件维度拼接后输入 MLP
        cond_dim = window_code_dim + relative_time_code_dim + metadata_dim
        input_dim = embedding_dim + (cond_dim if cond_dim > 0 else 0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, out_channels),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor | None = None,
        relative_time: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. 条件注入
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        # x: [B, D, H, W]

        # 2. 逐像素 MLP
        B, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, D)
        out_flat = self.mlp(x_flat)
        out = out_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class CategoricalDecoder(nn.Module):
    """类别型解码器 — 同样改为逐像素 MLP."""

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_width: int = 512,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim, window_code_dim, relative_time_code_dim, metadata_dim,
        )
        cond_dim = window_code_dim + relative_time_code_dim + metadata_dim
        input_dim = embedding_dim + (cond_dim if cond_dim > 0 else 0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, out_channels),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        B, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, D)
        out_flat = self.mlp(x_flat)
        out = out_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out
