"""GeoTIFF 读写工具。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds


RESAMPLING_MAP = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
}


def read_tif(
    path: str | Path,
    image_size: int = -1,
    resampling: str = "bilinear",
) -> np.ndarray | None:
    """
    读取 GeoTIFF，返回 (C, H, W) float32 数组；失败返回 None。

    Args:
        path       : TIFF 文件路径
        image_size : 目标尺寸（-1 表示不 resize）
        resampling : 重采样方法
    """
    try:
        with rasterio.open(path) as src:
            rs = RESAMPLING_MAP.get(resampling, Resampling.bilinear)
            if image_size > 0 and (src.width != image_size or src.height != image_size):
                data = src.read(
                    out_shape=(src.count, image_size, image_size),
                    resampling=rs,
                ).astype(np.float32)
            else:
                data = src.read().astype(np.float32)
        # NaN 检查：若有 NaN 用 0 替换
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data
    except Exception:
        return None


def write_tif(
    path: str | Path,
    data: np.ndarray,
    crs: str,
    bounds: list[float],
    nodata: float | None = None,
    dtype: str = "float32",
) -> None:
    """
    将 (C, H, W) 数组写入 GeoTIFF。

    Args:
        path   : 输出路径
        data   : (C, H, W) numpy 数组
        crs    : 坐标系字符串（如 "EPSG:32652"）
        bounds : [left, bottom, right, top] UTM 坐标
        nodata : nodata 值
        dtype  : 输出数据类型
    """
    if data.ndim == 2:
        data = data[np.newaxis]  # → (1, H, W)
    c, h, w = data.shape
    transform = from_bounds(*bounds, width=w, height=h)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data.astype(dtype))
