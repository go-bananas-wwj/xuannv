"""Bottleneck — Query Attention → 64维Embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRBottleneck(nn.Module):
    """
    将Encoder输出的高维token压缩为64维全局embedding。
    使用可学习Query做Cross-Attention，比GAP更灵活。
    """

    def __init__(
        self,
        embed_dim: int = 256,
        output_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # 可学习Query: [1, 1, output_dim]
        self.query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

        # Query projection到embed_dim
        self.q_proj = nn.Linear(output_dim, embed_dim, bias=False)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: [B, N, embed_dim]
        Returns:
            embedding: [B, output_dim]
        """
        B, N, D = encoder_output.shape

        # Query
        q = self.query.expand(B, -1, -1)  # [B, 1, output_dim]
        q = self.q_proj(q)  # [B, 1, embed_dim]
        q = q.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, 1, d]

        # Key/Value
        kv = self.kv_proj(encoder_output)  # [B, N, 2*embed_dim]
        kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, H, N, d]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, 1, D)  # [B, 1, embed_dim]
        out = self.out_proj(out)  # [B, 1, output_dim]
        out = self.dropout(out)

        out = out.squeeze(1)  # [B, output_dim]
        out = self.norm(out)
        return out
