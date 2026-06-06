"""数据预处理 — 独立于src/data/transforms.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError


def read_tif(path: Path | str, image_size: int = 128, resampling: str = "bilinear") -> np.ndarray | None:
    """读取TIFF，resize到image_size，返回(C, H, W) float32."""
    try:
        with rasterio.open(path) as src:
            if image_size <= 0 or (src.width == image_size and src.height == image_size):
                data = src.read().astype(np.float32)
            else:
                from rasterio.enums import Resampling as RioResampling
                rio_mode = RioResampling.nearest if resampling == "nearest" else RioResampling.bilinear
                data = src.read(
                    out_shape=(src.count, image_size, image_size),
                    resampling=rio_mode,
                ).astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data
    except (RasterioIOError, OSError):
        return None


def normalize_source(data: np.ndarray, source_name: str, stats: dict) -> np.ndarray:
    """
    根据源类型做预处理 + z-score归一化。

    Args:
        data: (C, H, W) float32 原始数据
        source_name: 源名称
        stats: {channel_N: {mean, std}} 统计字典
    """
    source_name = source_name.lower().strip()

    # 1. 源特定预处理
    if source_name == "planet":
        # Planet: uint16 DN值 → 反射率缩放 → log变换
        if data.max() > 100:
            data = data / 10000.0
        data = np.log(np.clip(data, 0, None) + 1) / 10.0
    elif source_name in ("s2", "landsat"):
        # 光学源: log(x+1)/10 变换
        if data.max() < 2.0:
            data = data * 10000.0
        data = np.log(np.clip(data, 0, None) + 1) / 10.0
    elif source_name in ("s1", "tianyi_sar"):
        # SAR: dB范围裁剪
        if data.max() > 100:
            data = np.log10(np.clip(data / 10000.0, 1e-10, None)) * 10.0
        data = np.clip(data, -30.0, 10.0)

    # 2. z-score 归一化
    out = np.zeros_like(data, dtype=np.float32)
    for c in range(data.shape[0]):
        key = f"channel_{c}"
        if key in stats:
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            if std > 1e-8:
                out[c] = (data[c] - mean) / std
            else:
                out[c] = data[c] - mean
        else:
            # fallback: 逐通道计算
            mean = float(np.nanmean(data[c]))
            std = float(np.nanstd(data[c]))
            std = max(std, 1e-6)
            out[c] = (np.nan_to_num(data[c], nan=mean) - mean) / std

    # 3. ±6σ 裁剪
    out = np.clip(out, -6.0, 6.0)
    return out
