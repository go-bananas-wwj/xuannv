"""Multi-scale Patch Embedding — per-source patch size based on native resolution."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePatchEmbed(nn.Module):
    """
    Per-source patch embedding with resolution-aware patch sizes.

    High-res sources (Planet 3m, SAR 3m): patch_size=4  -> 64x64 tokens -> pool to 32x32
    Mid-res sources (S2 10m):            patch_size=8  -> 32x32 tokens
    Low-res sources (Landsat 30m):       patch_size=16 -> 16x16 tokens -> upsample to 32x32

    All sources output a unified [B, T, 1024, D] token sequence for the encoder.
    """

    def __init__(
        self,
        source_channels: dict[str, int],
        embed_dim: int = 512,
        patch_size: int = 8,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.source_channels = source_channels
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Resolution-aware patch sizes
        self.source_patch_sizes: dict[str, int] = {
            "planet": 4,
            "tianyi_sar": 4,
            "s2": 8,
            "landsat": 16,
        }

        # Target token grid: all sources unified to 32x32 = 1024 tokens
        self.target_grid = image_size // 8  # 256 // 8 = 32
        self.target_n_patches = self.target_grid ** 2  # 1024

        # Per-source stems
        self.stems = nn.ModuleDict()
        for name, ch in source_channels.items():
            ps = self.source_patch_sizes.get(name, patch_size)
            self.stems[name] = nn.Conv2d(
                in_channels=ch,
                out_channels=embed_dim,
                kernel_size=ps,
                stride=ps,
                bias=True,
            )

        # High-res token pooling: 64x64 -> 32x32 (Planet, SAR)
        self.token_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Low-res token upsampling: 16x16 -> 32x32 (Landsat)
        self.token_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        source_name: str,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] or [B, C, H, W]
            source_name: source key
        Returns:
            tokens: [B, T, 1024, embed_dim]  (always 1024 patches)
        """
        stem = self.stems[source_name]

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1
            x_flat = x

        # Resize to image_size if needed
        if H != self.image_size or W != self.image_size:
            x_flat = F.interpolate(
                x_flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Patch embedding
        feat = stem(x_flat)  # [B*T, D, grid_h, grid_w]

        # Resolution-aware token grid adjustment
        if source_name in ("planet", "tianyi_sar"):
            # 64x64 -> 32x32 via pooling
            feat = self.token_pool(feat)
        elif source_name == "landsat":
            # 16x16 -> 32x32 via upsampling
            feat = self.token_upsample(feat)
        # s2: 32x32, no change needed

        # Flatten to [B*T, n_patches, D]
        feat = feat.flatten(2).transpose(1, 2)  # [B*T, 1024, embed_dim]

        if x.dim() == 5:
            feat = feat.reshape(B, T, self.target_n_patches, self.embed_dim)
        else:
            feat = feat.reshape(B, T, self.target_n_patches, self.embed_dim)

        return feat
