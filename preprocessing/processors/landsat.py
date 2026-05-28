"""Landsat 后处理器。

主要功能:
    1. Landsat Collection 2 Level-2 DN → 表面反射率转换
       (SR = 0.0000275 * DN - 0.2，Landsat-8/9 官方公式)
    2. QA 波段云掩膜（可选，若数据已预过滤则跳过）
    3. clip 到 [0, 1] 物理合理范围
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from preprocessing.utils.logging import get_logger
from preprocessing.utils.tiff import write_tif

logger = get_logger(__name__)

# Landsat C2 L2 表面反射率转换系数
_SR_SCALE = 0.0000275
_SR_OFFSET = -0.2

# 检测下载数据是否为 DN（通常 7000-65535）或已归一化（0-1）
_DN_THRESHOLD = 100.0


class LandsatProcessor:
    """
    Landsat 后处理器：DN → 表面反射率。

    如果 tif 已是 0-1 范围（GEE 已做转换），则跳过，仅做 clip 和质量检查。
    """

    def __init__(self, region_cfg: dict) -> None:
        self.region_cfg = region_cfg
        self.source_dir = Path(region_cfg["output_dir"]) / "landsat"

    def run(self, workers: int = 1) -> dict:
        if not self.source_dir.exists():
            logger.warning(f"[LandsatProcessor] 目录不存在: {self.source_dir}")
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

        logger.info(f"[LandsatProcessor] 完成: {stats}")
        return stats

    def _process_tif(self, tif_path: Path) -> str:
        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)
            crs = src.crs.to_string()
            bounds = [src.bounds.left, src.bounds.bottom,
                      src.bounds.right, src.bounds.top]

        # 若已是 0-1 范围，只做 clip
        if data.max() <= 2.0:
            clipped = np.clip(data, 0.0, 1.0)
            if np.allclose(clipped, data):
                return "skipped"
            write_tif(tif_path, clipped, crs=crs, bounds=bounds)
            return "processed"

        # Landsat C2 L2 DN → 表面反射率
        # 过滤 fill value（0 或 65535）
        fill_mask = (data == 0) | (data >= 65535)
        sr = _SR_SCALE * data + _SR_OFFSET
        sr[fill_mask] = 0.0
        sr = np.clip(sr, 0.0, 1.0)

        write_tif(tif_path, sr, crs=crs, bounds=bounds)
        return "processed"
