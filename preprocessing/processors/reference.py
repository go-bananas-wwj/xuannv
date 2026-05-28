"""参考数据处理器：DEM / WorldCover / Dynamic World / JRC Water。

这些数据在训练中作为重建目标，不需要时序处理，仅需：
    1. 检查 nodata 值并替换为 0
    2. WorldCover / DW / JRC 需确保是 nearest resampling（分类数据）
    3. 格式标准化：float32 GeoTIFF
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from preprocessing.utils.logging import get_logger
from preprocessing.utils.tiff import write_tif

logger = get_logger(__name__)

# WorldCover ESA 类别编码（保留，不做映射）
# 10=Tree, 20=Shrubland, 30=Grassland, 40=Cropland, 50=Built-up, ...

# DEM 合理高度范围（m）
_DEM_VALID_RANGE = (-500.0, 9000.0)


class ReferenceProcessor:
    """
    参考数据（静态目标）后处理器。

    对 DEM/WorldCover/DynamicWorld/JRCWater 做统一的格式规范化。
    """

    def __init__(self, region_cfg: dict, source_name: str) -> None:
        self.region_cfg = region_cfg
        self.source_name = source_name
        self.source_dir = Path(region_cfg["output_dir"]) / source_name
        self.source_cfg = region_cfg["sources"].get(source_name, {})

    def run(self, workers: int = 1) -> dict:
        if not self.source_dir.exists():
            logger.warning(f"[RefProcessor/{self.source_name}] 目录不存在: {self.source_dir}")
            return {}

        patches = [d for d in self.source_dir.iterdir()
                   if d.is_dir() and d.name.startswith("patch_")]
        stats = {"processed": 0, "skipped": 0, "failed": 0}

        for patch_dir in sorted(patches):
            for tif_path in sorted(patch_dir.glob("*.tif")):
                try:
                    result = self._process_tif(tif_path)
                    stats[result] += 1
                except Exception as e:
                    logger.debug(f"  {tif_path}: {e}")
                    stats["failed"] += 1

        logger.info(f"[RefProcessor/{self.source_name}] 完成: {stats}")
        return stats

    def _process_tif(self, tif_path: Path) -> str:
        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)
            nodata = src.nodata
            crs = src.crs.to_string()
            bounds = [src.bounds.left, src.bounds.bottom,
                      src.bounds.right, src.bounds.top]

        changed = False

        # 替换 nodata / fill values
        if nodata is not None:
            mask = data == nodata
            if mask.any():
                data[mask] = 0.0
                changed = True

        # JRC Water 中的 -128 fill value
        if self.source_name == "jrc_water":
            mask = data < 0
            if mask.any():
                data[mask] = 0.0
                changed = True

        # DEM 高度 clip
        if self.source_name == "dem":
            lo, hi = _DEM_VALID_RANGE
            valid = (data >= lo) & (data <= hi)
            if not valid.all():
                data = np.where(valid, data, 0.0)
                changed = True

        # NaN / Inf 清理
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            changed = True

        if not changed:
            return "skipped"

        write_tif(tif_path, data, crs=crs, bounds=bounds)
        return "processed"
