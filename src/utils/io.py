"""IO 工具."""
from __future__ import annotations

from pathlib import Path

import rasterio
import numpy as np


def read_tif(path: Path | str, target_size: int | None = None) -> np.ndarray | None:
    """读取 GeoTIFF 文件."""
    try:
        with rasterio.open(str(path)) as src:
            data = src.read().astype(np.float32)
        if target_size is not None and (data.shape[-1] != target_size or data.shape[-2] != target_size):
            import torch.nn.functional as F
            t = torch.from_numpy(data).unsqueeze(0)
            t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
            data = t.squeeze(0).numpy()
        return data
    except Exception:
        return None


def save_tif(data: np.ndarray, path: Path | str, crs: str = "EPSG:32652",
             transform=None, nodata: float = np.nan) -> None:
    """保存为 GeoTIFF."""
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0] if data.ndim == 3 else 1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if data.ndim == 3:
            for i in range(data.shape[0]):
                dst.write(data[i], i + 1)
        else:
            dst.write(data, 1)
