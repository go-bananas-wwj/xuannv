"""时间编码模块 — 时间戳、窗口、相对时间."""
from __future__ import annotations

import math
import torch
from torch import nn


class TimeCodeEncoder(nn.Module):
    """绝对时间编码: 毫秒时间戳 → sinusoidal + MLP."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestamps_ms: torch.Tensor) -> torch.Tensor:
        # timestamps_ms: [B, T]
        t = timestamps_ms / (1000.0 * 3600.0 * 24.0 * 365.25)  # 归一化到年
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.dim, 2, device=t.device).float() / self.dim))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        codes = torch.cat([args.sin(), args.cos()], dim=-1)
        if codes.shape[-1] > self.dim:
            codes = codes[..., :self.dim]
        return self.mlp(codes)


class WindowCodeEncoder(nn.Module):
    """窗口编码: (valid_start_ms, valid_end_ms) → 窗口特征向量."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(4, dim),  # start_sin, start_cos, end_sin, end_cos
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, valid_start_ms: torch.Tensor, valid_end_ms: torch.Tensor) -> torch.Tensor:
        # valid_start_ms, valid_end_ms: [B]  (never scalar)
        # Use .view(-1) instead of .squeeze() to preserve batch dim
        start = valid_start_ms.view(-1) / (1000.0 * 3600.0 * 24.0 * 365.25)
        end = valid_end_ms.view(-1) / (1000.0 * 3600.0 * 24.0 * 365.25)
        codes = torch.stack([
            (start * 2 * math.pi).sin(),
            (start * 2 * math.pi).cos(),
            (end * 2 * math.pi).sin(),
            (end * 2 * math.pi).cos(),
        ], dim=-1)  # [B, 4]
        return self.mlp(codes)  # [B, window_code_dim]


class RelativeTimeCodeEncoder(nn.Module):
    """相对时间编码: [0,1) 范围内的相对位置 → 编码."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, relative_time: torch.Tensor) -> torch.Tensor:
        # relative_time: [B*T, 1] 范围 [0,1)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.dim, 2, device=relative_time.device).float() / self.dim))
        args = relative_time * freqs.unsqueeze(0)
        codes = torch.cat([args.sin(), args.cos()], dim=-1)
        if codes.shape[-1] > self.dim:
            codes = codes[..., :self.dim]
        return self.mlp(codes)
