"""STP Block — 玄女V2 严格对齐 AEF 论文.

核心设计:
- 三条路径独立存在: Space(1/16L, 1024-dim), Time(1/8L, 512-dim), Precision(1/2L, 128-dim)
- 每个 block 内有显式 6 方向跨尺度交换 (LearnedSpatialResampling)
- 15 个 STP blocks
- Channel-first 接口 [B, T, C, H, W], operator 内部做 permute
- 手动 MHA 实现, 避免 NPU _masked_softmax fallback

参考: Brayden-Zhang/alphaearth-foundations
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


# ────────────────────────────────────────────
# 0. Sinusoidal Time Encoding
# ────────────────────────────────────────────

class SinusoidalTimeEncoding(nn.Module):
    """正弦时间编码 — TimeOperator 内部使用."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestamps_ms: torch.Tensor) -> torch.Tensor:
        """Args: timestamps_ms [B, T] (毫秒). Returns: [B, T, dim]."""
        t = timestamps_ms.float() / (1000.0 * 3600.0 * 24.0 * 365.25)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.dim, 2, device=t.device).float() / self.dim))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        enc = torch.cat([args.sin(), args.cos()], dim=-1)
        if enc.shape[-1] > self.dim:
            enc = enc[..., :self.dim]
        return enc


# ────────────────────────────────────────────
# 1. Learned Spatial Resampling
# ────────────────────────────────────────────

class LearnedSpatialResampling(nn.Module):
    """可学习的空间重采样 — Laplacian pyramid rescaling 的简化实现.

    用于跨尺度信息传递, 将源路径的特征投影到目标路径的分辨率.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float) -> None:
        super().__init__()
        self.scale_factor = scale_factor

        if scale_factor > 1:
            # 上采样: scale=2.0 时, 输出尺寸 = 输入 * 2
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
            )
        elif scale_factor < 1:
            # 下采样: scale=0.5 时, stride=2
            stride = int(1.0 / scale_factor)
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=stride * 2 - 1,
                stride=stride,
                padding=stride - 1,
            )
        else:
            # 同分辨率
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ────────────────────────────────────────────
# 2. Space Operator — 空间自注意力 (1/16L)
# ────────────────────────────────────────────

class SpaceOperator(nn.Module):
    """Space path: ViT-style 空间自注意力.

    输入/输出: [B, T, C, H, W] (channel-first)
    内部: permute -> [B*T, H*W, C] -> LayerNorm + QKV + MHA + MLP -> permute 回来
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        # [B, T, C, H, W] -> [B, T, H, W, C] -> [B*T, H*W, C]
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, C)

        residual = x_flat
        x_norm = self.norm1(x_flat)

        # QKV: [B*T, H*W, 3*C]
        qkv = self.qkv(x_norm)
        # -> [B*T, H*W, 3, num_heads, head_dim]
        qkv = qkv.reshape(B * T, H * W, 3, self.num_heads, self.head_dim)
        # -> [3, B*T, num_heads, H*W, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # [B*T, num_heads, H*W, head_dim] -> [B*T, H*W, C]
        out = out.permute(0, 2, 1, 3).reshape(B * T, H * W, C)
        x_flat = residual + self.proj(out)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # [B*T, H*W, C] -> [B, T, H, W, C] -> [B, T, C, H, W]
        out = x_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        return out


# ────────────────────────────────────────────
# 3. Time Operator — 时间自注意力 (1/8L)
# ────────────────────────────────────────────

class TimeOperator(nn.Module):
    """Time path: 时间轴自注意力.

    输入/输出: [B, T, C, H, W]
    内部: 加 sinusoidal 时间编码 -> permute -> [B*H*W, T, C] -> MHA + MLP
    """

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.time_encoding = SinusoidalTimeEncoding(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps_ms: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
            timestamps_ms: [B, T] 毫秒时间戳
            frame_mask: [B, T] bool, True=有效帧
        Returns:
            [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # 时间编码: [B, T, dim]
        time_enc = self.time_encoding(timestamps_ms)  # [B, T, self.dim]
        # broadcast 到空间维度: [B, T, dim] -> [B, T, dim, 1, 1]
        # x 是 [B, T, C, H, W], C = self.dim
        time_enc = time_enc.unsqueeze(-1).unsqueeze(-1)  # [B, T, dim, 1, 1]
        x = x + time_enc

        # [B, T, C, H, W] -> [B, T, H, W, C] -> [B, H, W, T, C] -> [B*H*W, T, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        residual = x_flat
        x_norm = self.norm1(x_flat)

        # QKV: [B*H*W, T, 3*C]
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B * H * W, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Frame mask: 屏蔽无效帧作为 key（防止有效帧关注到零填充帧）
        # scores: [B*H*W, num_heads, T_query, T_key]
        if frame_mask is not None:
            # frame_mask: [B, T] -> [B*H*W, 1, 1, T]（沿 key 维度屏蔽）
            mask = frame_mask.unsqueeze(1)  # [B, 1, T]
            mask = mask.repeat_interleave(H * W, dim=0)  # [B*H*W, 1, T]
            mask = mask.unsqueeze(2)  # [B*H*W, 1, 1, T] ← key 维度，禁止关注无效帧
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax NaN 防护: 如果某行全 -inf, 替换为 0
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores = torch.where(torch.isinf(scores_max), torch.zeros_like(scores), scores)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # [B*H*W, num_heads, T, head_dim] -> [B*H*W, T, C]
        out = out.permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        x_flat = residual + self.proj(out)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # [B*H*W, T, C] -> [B, H, W, T, C] -> [B, T, C, H, W]
        out = x_flat.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return out


# ────────────────────────────────────────────
# 4. Precision Operator — 3x3 卷积 (1/2L)
# ────────────────────────────────────────────

class PrecisionOperator(nn.Module):
    """Precision path: 3x3 卷积保持高分辨率.

    输入/输出: [B, T, C, H, W]
    内部: permute -> [B*T, C, H, W] -> GroupNorm + Conv + GELU + Conv + 残差
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        num_groups = min(32, dim // 4) if dim >= 4 else 1

        self.norm1 = nn.GroupNorm(num_groups, dim)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, dim * 4)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        # [B, T, C, H, W] -> [B*T, C, H, W]
        x_conv = x.reshape(B * T, C, H, W)

        residual = x_conv
        x_conv = self.conv1(self.norm1(x_conv))
        x_conv = F.gelu(x_conv)
        x_conv = self.conv2(self.norm2(x_conv))
        x_conv = residual + x_conv

        # [B*T, C, H, W] -> [B, T, C, H, W]
        return x_conv.reshape(B, T, C, H, W)


# ────────────────────────────────────────────
# 5. STP Block — 三条路径 + 跨尺度交换
# ────────────────────────────────────────────

class STPBlock(nn.Module):
    """单 STP block — 论文 Figure 2D 实现.

    三条路径独立计算, 然后通过 LearnedSpatialResampling 做跨尺度信息交换.
    """

    def __init__(
        self,
        space_dim: int = 1024,
        time_dim: int = 512,
        precision_dim: int = 128,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim

        self.space_op = SpaceOperator(space_dim, num_heads)
        self.time_op = TimeOperator(time_dim, num_heads)
        self.precision_op = PrecisionOperator(precision_dim)

        # 6 方向跨尺度交换
        # Space -> Time/ Precision: 上采样 (scale > 1)
        self.space_to_time = LearnedSpatialResampling(space_dim, time_dim, 2.0)
        self.space_to_precision = LearnedSpatialResampling(space_dim, precision_dim, 8.0)
        # Time -> Space/ Precision
        self.time_to_space = LearnedSpatialResampling(time_dim, space_dim, 0.5)
        self.time_to_precision = LearnedSpatialResampling(time_dim, precision_dim, 4.0)
        # Precision -> Space/ Time
        self.precision_to_space = LearnedSpatialResampling(precision_dim, space_dim, 0.125)
        self.precision_to_time = LearnedSpatialResampling(precision_dim, time_dim, 0.25)

    def _ensure_shape(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """确保空间尺寸匹配目标."""
        if x.shape[2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x

    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        precision_x: torch.Tensor,
        timestamps_ms: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            space_x:  [B, T, space_dim,     H_s, W_s]
            time_x:   [B, T, time_dim,      H_t, W_t]
            precision_x: [B, T, precision_dim, H_p, W_p]
            timestamps_ms: [B, T]
            frame_mask: [B, T] bool
        Returns:
            space_out, time_out, precision_out (同输入 shape)
        """
        B, T = space_x.shape[:2]

        # ── 各路径独立计算 ──
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps_ms, frame_mask)
        precision_out = self.precision_op(precision_x)

        # 获取各路径空间尺寸
        _, _, _, space_H, space_W = space_out.shape
        _, _, _, time_H, time_W = time_out.shape
        _, _, _, precision_H, precision_W = precision_out.shape

        # 展平为 2D (BT, C, H, W) 用于空间重采样
        space_2d = space_out.reshape(B * T, self.space_dim, space_H, space_W)
        time_2d = time_out.reshape(B * T, self.time_dim, time_H, time_W)
        precision_2d = precision_out.reshape(B * T, self.precision_dim, precision_H, precision_W)

        # ── 跨尺度交换 ──
        # Space 路径接收 Time + Precision 的信息
        time_to_space = self._ensure_shape(
            self.time_to_space(time_2d), space_H, space_W
        )
        precision_to_space = self._ensure_shape(
            self.precision_to_space(precision_2d), space_H, space_W
        )
        space_exchange = space_2d + time_to_space + precision_to_space

        # Time 路径接收 Space + Precision 的信息
        space_to_time = self._ensure_shape(
            self.space_to_time(space_2d), time_H, time_W
        )
        precision_to_time = self._ensure_shape(
            self.precision_to_time(precision_2d), time_H, time_W
        )
        time_exchange = time_2d + space_to_time + precision_to_time

        # Precision 路径接收 Space + Time 的信息
        space_to_precision = self._ensure_shape(
            self.space_to_precision(space_2d), precision_H, precision_W
        )
        time_to_precision = self._ensure_shape(
            self.time_to_precision(time_2d), precision_H, precision_W
        )
        precision_exchange = precision_2d + space_to_precision + time_to_precision

        # 重塑回 (B, T, C, H, W)
        space_out = space_exchange.reshape(B, T, self.space_dim, space_H, space_W)
        time_out = time_exchange.reshape(B, T, self.time_dim, time_H, time_W)
        precision_out = precision_exchange.reshape(B, T, self.precision_dim, precision_H, precision_W)

        return space_out, time_out, precision_out


# ────────────────────────────────────────────
# 6. STP Encoder — 完整编码器
# ────────────────────────────────────────────

class STPEncoder(nn.Module):
    """STP Encoder — 论文完整实现.

    输入: [B, T, C, H, W] (C=precision_dim, H=W=64 即 sensor encoder 输出的 1/2L)
    输出: [B, T, precision_dim, H, W] (融合后的 precision 分辨率特征)
    """

    def __init__(
        self,
        precision_dim: int = 128,
        time_dim: int = 512,
        space_dim: int = 1024,
        num_blocks: int = 15,
        num_heads: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.precision_dim = precision_dim
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.use_checkpoint = use_checkpoint

        # 初始投影到三条路径的维度
        self.space_proj = nn.Conv2d(precision_dim, space_dim, kernel_size=1)
        self.time_proj = nn.Conv2d(precision_dim, time_dim, kernel_size=1)

        # STP blocks
        self.blocks = nn.ModuleList([
            STPBlock(space_dim, time_dim, precision_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # 最终上采样回 precision 分辨率
        self.final_space_up = LearnedSpatialResampling(space_dim, precision_dim, 8.0)
        self.final_time_up = LearnedSpatialResampling(time_dim, precision_dim, 4.0)

        # 输出归一化
        num_groups = min(32, precision_dim // 4) if precision_dim >= 4 else 1
        self.norm = nn.GroupNorm(num_groups, precision_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps_ms: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, precision_dim, H, W] (H=W=64, 即 1/2L)
            timestamps_ms: [B, T] 毫秒时间戳
            frame_mask: [B, T] bool, True=有效帧
        Returns:
            [B, T, precision_dim, H, W]
        """
        B, T, C, H, W = x.shape
        assert C == self.precision_dim, f"输入通道数 {C} != precision_dim {self.precision_dim}"

        # 展平 batch 和时间: [B*T, C, H, W]
        x_flat = x.reshape(B * T, C, H, W)

        # ── 初始化为三条路径 ──
        # Space: project + downsample 到 1/16L (H/8, W/8)
        space = self.space_proj(x_flat)  # [B*T, space_dim, H, W]
        space = F.adaptive_avg_pool2d(space, (H // 8, W // 8))

        # Time: project + downsample 到 1/8L (H/4, W/4)
        time = self.time_proj(x_flat)    # [B*T, time_dim, H, W]
        time = F.adaptive_avg_pool2d(time, (H // 4, W // 4))

        # Precision: 保持 1/2L (H, W)
        precision = x_flat                # [B*T, precision_dim, H, W]

        # 重塑为 (B, T, C, H, W)
        _, _, space_H, space_W = space.shape
        _, _, time_H, time_W = time.shape
        space = space.reshape(B, T, self.space_dim, space_H, space_W)
        time = time.reshape(B, T, self.time_dim, time_H, time_W)
        precision = precision.reshape(B, T, self.precision_dim, H, W)

        # ── 应用 STP blocks ──
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                space, time, precision = torch.utils.checkpoint.checkpoint(
                    block, space, time, precision, timestamps_ms, frame_mask,
                    use_reentrant=False,
                )
            else:
                space, time, precision = block(space, time, precision, timestamps_ms, frame_mask)

        # ── 最终融合: space/time 上采样回 precision 分辨率 ──
        space_flat = space.reshape(B * T, self.space_dim, space_H, space_W)
        time_flat = time.reshape(B * T, self.time_dim, time_H, time_W)
        precision_flat = precision.reshape(B * T, self.precision_dim, H, W)

        space_up = self.final_space_up(space_flat)   # [B*T, precision_dim, ?, ?]
        time_up = self.final_time_up(time_flat)       # [B*T, precision_dim, ?, ?]

        # 确保尺寸匹配 precision 分辨率
        space_up = self._ensure_shape(space_up, H, W)
        time_up = self._ensure_shape(time_up, H, W)

        # 三条路径相加 + 归一化
        out = space_up + time_up + precision_flat
        out = self.norm(out)

        return out.reshape(B, T, self.precision_dim, H, W)

    def _ensure_shape(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if x.shape[2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x
