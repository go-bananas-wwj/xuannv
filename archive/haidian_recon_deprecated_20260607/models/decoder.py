"""Transformer Decoder — 交叉注意力."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CrossAttentionBlock(nn.Module):
    """Pre-Norm Cross-Attention Block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [B, N_q, D] — mask token query
            kv: [B, N_kv, D] — encoder输出的visible token
        Returns:
            [B, N_q, D]
        """
        B, Nq, D = q.shape
        Nkv = kv.shape[1]

        # Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        q_proj = self.q_proj(q_norm).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_proj = self.kv_proj(kv_norm).reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv_proj[0], kv_proj[1]  # [B, H, Nkv, d]

        attn = (q_proj @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, Nq, D)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        q = q + out

        # MLP
        q = q + self.mlp(self.norm_mlp(q))
        return q


class HRDecoder(nn.Module):
    """
    Transformer Decoder。
    输入:
        mask_tokens: [B, N_mask, D] — 被mask位置的query token
        encoder_output: [B, N_visible, D] — encoder输出
    输出: [B, N_mask, D]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_gradient_checkpointing
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mask_tokens: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = mask_tokens
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, encoder_output)
            else:
                x = layer(x, encoder_output)
        x = self.norm(x)
        return x
