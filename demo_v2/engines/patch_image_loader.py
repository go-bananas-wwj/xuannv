"""根据 patch_id + 时间窗口加载对应数据源图像."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from PIL import Image

from demo_v2.utils.constants import RAW_DIR


def _extract_date_ms(fname: str) -> float | None:
    """从文件名提取日期并转为毫秒时间戳."""
    m = re.search(r'(\d{8})', fname)
    if m:
        dt = datetime.strptime(m.group(1), "%Y%m%d")
        return dt.timestamp() * 1000.0
    m = re.search(r'(\d{4}Q\d)', fname)
    if m:
        q_str = m.group(1)
        year = int(q_str[:4])
        quarter = int(q_str[-1])
        month = {1: 2, 2: 5, 3: 8, 4: 11}[quarter]
        dt = datetime(year, month, 15)
        return dt.timestamp() * 1000.0
    return None


def _find_best_tif(source_dir: Path, window_start_ms: float, window_end_ms: float) -> Path | None:
    """在时间窗口内找日期最接近窗口中间的 TIFF."""
    if not source_dir.exists():
        return None
    candidates = []
    for f in sorted(source_dir.glob("*.tif")):
        t = _extract_date_ms(f.stem)
        if t is not None and window_start_ms <= t <= window_end_ms:
            candidates.append((f, t))
    if not candidates:
        return None
    mid = (window_start_ms + window_end_ms) / 2.0
    best = min(candidates, key=lambda x: abs(x[1] - mid))
    return best[0]


def load_patch_source_rgb(
    patch_id: str,
    source: str,
    window: Tuple[float, float],
) -> np.ndarray | None:
    """加载指定 patch、数据源、时间窗口的 RGB 图像 [H, W, 3] uint8.

    Args:
        patch_id: patch ID.
        source: 数据源名称, e.g. "s2", "s2_hr", "s1", "landsat".
        window: (start_ms, end_ms).
    """
    source_dir = RAW_DIR / source / patch_id
    tif_path = _find_best_tif(source_dir, window[0], window[1])
    if tif_path is None:
        return None

    try:
        with rasterio.open(str(tif_path)) as ds:
            data = ds.read()
    except Exception:
        return None

    if source in ("s2", "s2_hr", "landsat", "highres"):
        if data.shape[0] >= 3:
            if source in ("s2", "s2_hr") and data.shape[0] >= 4:
                rgb = data[[2, 1, 0]].astype(np.float32)
            else:
                rgb = data[:3].astype(np.float32)
            valid = rgb[rgb > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    rgb = (rgb - p2) / (p98 - p2)
            rgb = np.clip(rgb, 0, 1).transpose(1, 2, 0)
            return (rgb * 255).astype(np.uint8)

    if source in ("s1", "s1_hr"):
        if data.shape[0] >= 2:
            vv = data[0].astype(np.float32)
            vh = data[1].astype(np.float32)
            vv_n = np.clip((vv + 25) / 35, 0, 1)
            vh_n = np.clip((vh + 30) / 35, 0, 1)
            rgb = np.stack([vv_n, vh_n, vv_n / (vh_n + 1e-6) * 0.3], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            return (rgb * 255).astype(np.uint8)
        else:
            band = data[0].astype(np.float32)
            band_n = np.clip((band + 25) / 35, 0, 1)
            rgb = np.stack([band_n, band_n, band_n], axis=-1)
            return (rgb * 255).astype(np.uint8)

    # Fallback: single band grayscale -> RGB
    band = data[0].astype(np.float32)
    valid = band[band > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        if p98 > p2:
            band = (band - p2) / (p98 - p2)
    band = np.clip(band, 0, 1)
    rgb = np.stack([band, band, band], axis=-1)
    return (rgb * 255).astype(np.uint8)


def load_patch_source_raw(
    patch_id: str,
    source: str,
    window: Tuple[float, float],
) -> np.ndarray | None:
    """加载指定 patch、数据源、时间窗口的原始多波段 TIFF 数据 [C, H, W].

    Args:
        patch_id: patch ID.
        source: 数据源名称.
        window: (start_ms, end_ms).
    """
    source_dir = RAW_DIR / source / patch_id
    tif_path = _find_best_tif(source_dir, window[0], window[1])
    if tif_path is None:
        return None
    try:
        with rasterio.open(str(tif_path)) as ds:
            return ds.read()
    except Exception:
        return None


def compute_ndvi_from_s2(s2_data: np.ndarray) -> np.ndarray:
    """从 S2 数据计算 NDVI [H, W].

    S2 波段: B2, B3, B4, B8, B11, B12
    NIR = B8 (index 3), Red = B4 (index 2)
    """
    if s2_data.shape[0] < 4:
        return np.zeros(s2_data.shape[1:], dtype=np.float32)
    nir = s2_data[3].astype(np.float32)
    red = s2_data[2].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi
