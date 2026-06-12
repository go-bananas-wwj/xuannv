"""传感器编码器 — 支持可变输入通道与可选 adapter.

严格按论文设计: 只有带时间戳的图像帧作为输入。
静态源 (DEM/WorldCover 等) 仅作为重建目标，不经过编码器。
"""
from __future__ import annotations

import torch
from torch import nn


class SensorEncoder(nn.Module):
    """单个传感器: 可选 adapter + stem conv + projection.

    当 in_channels != stem_channels 时，会在 stem 前插入 1x1 adapter。
    支持配置 stem_stride / stem_layers，用于多分辨率输入下采样。
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int | None = None,
        stem_dim: int = 64,
        out_dim: int = 128,
        stem_stride: int = 2,
        stem_layers: int = 1,
    ) -> None:
        super().__init__()
        stem_channels = stem_channels if stem_channels is not None else in_channels
        self.use_adapter = in_channels != stem_channels
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.GELU(),
            )
        else:
            self.adapter = nn.Identity()

        layers: list[nn.Module] = []
        in_ch = stem_channels
        for _ in range(stem_layers):
            layers.append(nn.Conv2d(in_ch, stem_dim, kernel_size=3, stride=stem_stride, padding=1))
            layers.append(nn.BatchNorm2d(stem_dim))
            layers.append(nn.GELU())
            in_ch = stem_dim
        self.stem = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Conv2d(stem_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W] → [B*T, out_dim, H', W']
        x = self.adapter(x)
        return self.projection(self.stem(x))


class SensorEncoderBank(nn.Module):
    """多源传感器编码器组.

    支持 per-source 输入通道配置 (source_channels)。
    当配置为空时，退化为所有源统一使用 input_dim 通道。
    仅创建实际配置的输入源编码器，避免 checkpoint 兼容性问题与参数浪费。
    """

    _SOURCE_TYPE_ID: dict[str, int] = {
        "s2": 0, "s1": 1, "landsat": 2, "s2_hr": 3, "s1_hr": 4,
        "tianyi_sar": 5, "planet": 6,
    }

    def __init__(
        self,
        num_sensor_types: int = 16,
        input_dim: int = 6,
        stem_dim: int = 128,
        out_dim: int = 256,
        source_channels: dict[str, int] | None = None,
        stem_channels: int | None = None,
        input_sources: list[str] | None = None,
        source_stem_stride: dict[str, int] | None = None,
        source_stem_layers: dict[str, int] | None = None,
        common_spatial_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.source_channels = source_channels or {}
        stem_ch = stem_channels if stem_channels is not None else input_dim
        self.source_stem_stride = source_stem_stride or {}
        self.source_stem_layers = source_stem_layers or {}
        self.common_spatial_size = common_spatial_size  # (H, W)

        # 确定需要创建编码器的输入源列表
        if input_sources is not None:
            self.input_sources = list(input_sources)
        else:
            # 默认仅创建 3 类基础输入源 (兼容 v1/v2 checkpoint)
            self.input_sources = ["s2", "s1", "landsat"]

        # 每种传感器独立编码器 (仅创建实际需要的)
        self.encoders = nn.ModuleDict()
        for src_name in self.input_sources:
            in_ch = self.source_channels.get(src_name, input_dim)
            self.encoders[src_name] = SensorEncoder(
                in_channels=in_ch,
                stem_channels=stem_ch,
                stem_dim=stem_dim,
                out_dim=out_dim,
                stem_stride=self.source_stem_stride.get(src_name, 2),
                stem_layers=self.source_stem_layers.get(src_name, 1),
            )

    @property
    def source_type_ids(self) -> dict[str, int]:
        return dict(self._SOURCE_TYPE_ID)

    def forward(
        self,
        source_frames: torch.Tensor | list[torch.Tensor],
        source_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_frames:
              - legacy: [B, S, T, C, H, W]
              - multires: List[Tensor], each [B, T, C, H_s, W_s]
            source_type_ids: [B, S]
        Returns:
            encoded: [B, S, T, out_dim, H_c, W_c]
        """
        is_multires = isinstance(source_frames, list)
        source_names = {v: k for k, v in self._SOURCE_TYPE_ID.items()}

        if is_multires:
            S = len(source_frames)
            B = source_frames[0].shape[0]
        else:
            B, S, T, C, H, W = source_frames.shape

        encoded_list = []
        for s_idx in range(S):
            src_type = int(source_type_ids[0, s_idx].item())
            src_name = source_names.get(src_type, "s2")
            encoder = self.encoders[src_name] if src_name in self.encoders else self.encoders["s2"]

            if is_multires:
                frames = source_frames[s_idx]  # [B, T, C, H_s, W_s]
                _, T, C, H, W = frames.shape
                in_ch = self.source_channels.get(src_name, self.input_dim)
                if C > in_ch:
                    frames = frames[:, :, :in_ch, :, :]
                frames = frames.reshape(B * T, in_ch, H, W)
                enc = encoder(frames)  # [B*T, out_dim, H', W']
                _, d, h, w = enc.shape
                enc = enc.reshape(B, T, d, h, w)
                # 对齐到公共空间分辨率
                if self.common_spatial_size is not None and (h, w) != tuple(self.common_spatial_size):
                    enc = torch.nn.functional.interpolate(
                        enc.reshape(B * T, d, h, w),
                        size=tuple(self.common_spatial_size),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(B, T, d, *self.common_spatial_size)
            else:
                frames = source_frames[:, s_idx, :, :, :, :]  # [B, T, C, H, W]
                in_ch = self.source_channels.get(src_name, self.input_dim)
                if C > in_ch:
                    frames = frames[:, :, :in_ch, :, :]
                frames = frames.reshape(B * T, in_ch, H, W)
                enc = encoder(frames)  # [B*T, out_dim, H/2, W/2]
                _, d, h, w = enc.shape
                enc = enc.reshape(B, T, d, h, w)

            encoded_list.append(enc)

        return torch.stack(encoded_list, dim=1)  # [B, S, T, out_dim, H_c, W_c]
