"""OlmoEarth 蒸馏投影头."""
from __future__ import annotations
import torch
import torch.nn as nn


class OlmoEarthProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 512, out_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x_proj = self.proj(x_flat)
        return x_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
