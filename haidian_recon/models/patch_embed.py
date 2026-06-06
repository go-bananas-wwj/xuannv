"""多源Patch Embedding — 每个源独立stem."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSourcePatchEmbed(nn.Module):
    """
    为每个源独立做Patch Embedding。
    输入: [B, T, C, H, W] — T为时间步数
    输出: [B, T, N_patches, embed_dim]

    若某源分辨率与image_size不同，先通过插值对齐到image_size。
    """

    def __init__(
        self,
        source_channels: dict[str, int],
        embed_dim: int = 256,
        patch_size: int = 8,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.source_channels = source_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = (image_size // patch_size) ** 2

        # 每个源独立的Conv2d做patch embedding
        self.stems = nn.ModuleDict()
        for name, ch in source_channels.items():
            self.stems[name] = nn.Conv2d(
                in_channels=ch,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        source_name: str,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] 或 [B, C, H, W]
            source_name: 源名称

        Returns:
            tokens: [B, T, N_patches, embed_dim] 或 [B, N_patches, embed_dim]
        """
        stem = self.stems[source_name]

        # 处理输入维度
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # [B*T, C, H, W]
            x_flat = x.reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1
            x_flat = x

        # 若尺寸不对，插值到image_size
        if H != self.image_size or W != self.image_size:
            x_flat = F.interpolate(
                x_flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Patch embedding: [B*T, embed_dim, H//ps, W//ps]
        feat = stem(x_flat)
        # Flatten: [B*T, embed_dim, N_patches]
        feat = feat.flatten(2).transpose(1, 2)  # [B*T, N_patches, embed_dim]

        if x.dim() == 5:
            feat = feat.reshape(B, T, self.n_patches, self.embed_dim)
        else:
            feat = feat.reshape(B, self.n_patches, self.embed_dim).unsqueeze(1)  # [B, 1, N, D]

        return feat
