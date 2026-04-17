"""STP Block — 三路径时空编码器."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class STPBlock(nn.Module):
    """单个 Space-Time-Precision block.

    三条路径:
    - Time path: 时间轴自注意力 (分辨率 1/8)
    - Space path: 空间自注意力 (分辨率 1/16)
    - Precision path: 3x3 卷积 (分辨率 1/2)
    最后通过 Laplacian pyramid 风格的信息交换融合。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        time_code_dim: int = 64,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Precision path: 高分辨率 3x3 卷积
        self.precision_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

        # Time path: 时间轴注意力 (降采样到 1/8)
        self.time_down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.time_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.time_norm = nn.LayerNorm(channels)
        self.time_up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

        # Space path: 空间自注意力 (降采样到 1/16)
        self.space_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        )
        self.space_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.space_norm = nn.LayerNorm(channels)
        self.space_up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
        )

        # 跨路径融合: 将不同分辨率的特征对齐到 precision 分辨率
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.residual_norm = nn.GroupNorm(8, channels)

    def forward(
        self,
        x: torch.Tensor,
        time_codes: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]  输入特征
            time_codes: [B, T, time_code_dim]  时间编码
            frame_mask: [B, T]  帧掩码
        Returns:
            x: [B, T, C, H, W]  输出特征
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)

        # ── Precision path ──
        x_prec = self.precision_conv(x_flat)  # [B*T, C, H, W]

        # ── Time path (降采样 → 时间注意力 → 上采样) ──
        x_t = self.time_down(x_flat)  # [B*T, C, H/2, W/2]
        _, c_t, h_t, w_t = x_t.shape
        x_t = x_t.reshape(B, T, c_t, h_t * w_t).permute(0, 3, 1, 2)  # [B, H/2*W/2, T, C]
        x_t_reshaped = x_t.reshape(B * h_t * w_t, T, c_t)
        # 时间注意力
        if frame_mask is not None:
            mask_expanded = ~frame_mask  # [B, T] → bool mask (True=忽略)
            mask_per_spatial = mask_expanded.repeat_interleave(h_t * w_t, dim=0)
        else:
            mask_per_spatial = None
        x_t_attn, _ = self.time_attn(x_t_reshaped, x_t_reshaped, x_t_reshaped, key_padding_mask=mask_per_spatial)
        x_t_attn = self.time_norm(x_t_attn)
        x_t_attn = x_t_attn.reshape(B, h_t * w_t, T, c_t).permute(0, 2, 3, 1)  # [B, T, C, H/2*W/2]
        x_t_up = x_t_attn.reshape(B * T, c_t, h_t, w_t)
        x_t_up = self.time_up(x_t_up)  # [B*T, C, H, W]

        # ── Space path (降采样 → 空间注意力 → 上采样) ──
        x_s = self.space_down(x_flat)  # [B*T, C, H/4, W/4]
        _, c_s, h_s, w_s = x_s.shape
        # 空间自注意力: 将空间展平为序列
        x_s_flat = x_s.reshape(B, T, c_s, h_s * w_s).permute(0, 1, 3, 2)  # [B, T, H/4*W/4, C]
        x_s_flat = x_s_flat.reshape(B * T, h_s * w_s, c_s)
        x_s_attn, _ = self.space_attn(x_s_flat, x_s_flat, x_s_flat)
        x_s_attn = self.space_norm(x_s_attn)
        x_s_attn = x_s_attn.reshape(B, T, h_s, w_s, c_s).permute(0, 1, 4, 2, 3)  # [B, T, C, H/4, W/4]
        x_s_up = x_s_attn.reshape(B * T, c_s, h_s, w_s)
        x_s_up = self.space_up(x_s_up)  # [B*T, C, H, W]

        # ── 融合 ──
        fused = self.fusion(torch.cat([x_prec, x_t_up, x_s_up], dim=1))  # [B*T, C, H, W]

        # ── 残差 ──
        residual = self.residual_norm(x_flat)
        out = fused + residual
        return out.reshape(B, T, C, H, W)
