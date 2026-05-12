"""STP Block — 三路径时空编码器.

V13 优化: 手动实现 MultiheadAttention forward，避免 NPU 上 _masked_softmax fallback 到 CPU.
"""
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
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        time_code_dim: int = 64,
        use_space: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_space = use_space

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
        fusion_in = channels * 3 if use_space else channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.residual_norm = nn.GroupNorm(8, channels)

    def _manual_mha(self, attn_module: nn.MultiheadAttention, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """手动实现 MultiheadAttention forward，避免 NPU _masked_softmax fallback.

        Args:
            attn_module: nn.MultiheadAttention 实例（复用其 weight/bias）
            x: [B, T, C]
            mask: [B, T] bool, True=忽略该位置
        Returns:
            out: [B, T, C]
        """
        B, T, C = x.shape
        num_heads = attn_module.num_heads
        head_dim = C // num_heads

        # in_proj: [3*C, C] @ [B, T, C] -> [B, T, 3*C]
        qkv = F.linear(x, attn_module.in_proj_weight, attn_module.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to multi-head: [B, num_heads, T, head_dim]
        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, num_heads, head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # apply mask: True -> -inf
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # reshape back and out_proj
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = F.linear(out, attn_module.out_proj.weight, attn_module.out_proj.bias)
        return out

    def forward(
        self,
        x: torch.Tensor,
        time_codes: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]  输入特征
            time_codes: [B, T, time_code_dim]  时间编码 (未使用，保留接口)
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

        # V13: 手动 MHA 避免 NPU _masked_softmax fallback
        if frame_mask is not None:
            mask_expanded = ~frame_mask  # [B, T] → True=忽略
            mask_per_spatial = mask_expanded.repeat_interleave(h_t * w_t, dim=0)
        else:
            mask_per_spatial = None

        x_t_attn = self._manual_mha(self.time_attn, x_t_reshaped, mask_per_spatial)
        x_t_attn = self.time_norm(x_t_attn)
        x_t_attn = x_t_attn.reshape(B, h_t * w_t, T, c_t).permute(0, 2, 3, 1)  # [B, T, C, H/2*W/2]
        x_t_up = x_t_attn.reshape(B * T, c_t, h_t, w_t)
        x_t_up = self.time_up(x_t_up)  # [B*T, C, H, W]

        # ── Space path (降采样 → 空间注意力 → 上采样) ──
        if self.use_space:
            x_s = self.space_down(x_flat)  # [B*T, C, H/4, W/4]
            _, c_s, h_s, w_s = x_s.shape
            x_s_flat = x_s.reshape(B, T, c_s, h_s * w_s).permute(0, 1, 3, 2)  # [B, T, H/4*W/4, C]
            x_s_flat = x_s_flat.reshape(B * T, h_s * w_s, c_s)

            # Space attention 不涉及 frame mask（空间维度全部有效）
            x_s_attn = self._manual_mha(self.space_attn, x_s_flat, mask=None)
            x_s_attn = self.space_norm(x_s_attn)
            x_s_attn = x_s_attn.reshape(B, T, h_s, w_s, c_s).permute(0, 1, 4, 2, 3)  # [B, T, C, H/4, W/4]
            x_s_up = x_s_attn.reshape(B * T, c_s, h_s, w_s)
            x_s_up = self.space_up(x_s_up)  # [B*T, C, H, W]

        # ── 融合 ──
        if self.use_space:
            fused = self.fusion(torch.cat([x_prec, x_t_up, x_s_up], dim=1))  # [B*T, C, H, W]
        else:
            fused = self.fusion(torch.cat([x_prec, x_t_up], dim=1))  # [B*T, 2*C, H, W]

        # ── 残差 ──
        residual = self.residual_norm(x_flat)
        out = fused + residual
        return out.reshape(B, T, C, H, W)
