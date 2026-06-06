"""多源重建头 — 将decoder token映射回像素空间."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRReconstructionHead(nn.Module):
    """
    将decoder输出的token映射回各源的像素空间。
    每个源独立的重建头。
    """

    def __init__(
        self,
        embed_dim: int = 256,
        out_channels: int = 6,
        patch_size: int = 8,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = (image_size // patch_size) ** 2

        # 先用MLP将每个token映射到patch_size^2 * out_channels
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * out_channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N_patches, embed_dim]
        Returns:
            reconstruction: [B, out_channels, H, W]
        """
        B, N, D = tokens.shape
        assert N == self.n_patches, f"Expected {self.n_patches} patches, got {N}"

        # [B, N, patch_size^2 * C]
        x = self.mlp(tokens)

        # reshape: [B, N, C, ps, ps]
        x = x.reshape(B, N, self.out_channels, self.patch_size, self.patch_size)

        # 重排为图像: [B, C, H, W]
        # N = grid_h * grid_w
        grid_h = grid_w = self.image_size // self.patch_size
        x = x.reshape(B, grid_h, grid_w, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, grid_h, ps, grid_w, ps]
        x = x.reshape(B, self.out_channels, self.image_size, self.image_size)

        return x
