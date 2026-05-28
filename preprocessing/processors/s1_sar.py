"""S1 / 通用 SAR 数据后处理器。

功能:
    1. 检测 S1 数值范围（0-1 线性 or DN），统一转换为 dB 值
    2. 对海淀干涉 SAR（已由 LocalSARImporter 完成振幅提取）做最终质量检查
    3. clip 极端值，避免 ±Inf 进入训练
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from preprocessing.utils.logging import get_logger
from preprocessing.utils.tiff import write_tif

logger = get_logger(__name__)

# S1 RTC 归一化反射率产品的典型范围（线性功率，0-1）
_S1_LINEAR_MAX = 2.0

# dB 值 clip 范围（物理意义：-35 dB 为极暗，5 dB 为金属强散射体）
_DB_MIN = -35.0
_DB_MAX = 5.0


def sar_to_db(data: np.ndarray, source_name: str = "s1") -> np.ndarray:
    """
    将 SAR 数据转换为 dB 值，自适应输入格式。

    支持:
        (A) 0-1 线性功率（Planetary Computer S1 RTC）→ 10*log10(x)
        (B) DN 值（原始 GRD，通常 0-32767）→ 先 /10000 再转 dB
        (C) 已是 dB 值（< 0 且 > -40）→ 直接 clip 返回

    Returns:
        (C, H, W) float32 dB 数组，值域 [_DB_MIN, _DB_MAX]
    """
    data = data.astype(np.float32)
    if data.max() <= _S1_LINEAR_MAX:
        # 线性功率 → dB
        db = 10.0 * np.log10(np.clip(data, 1e-10, None))
    elif data.max() > 100:
        # DN → 线性功率（假设 scale=10000）→ dB
        db = 10.0 * np.log10(np.clip(data / 10000.0, 1e-10, None))
    else:
        # 已是 dB 或振幅
        if data.min() >= 0:
            # 振幅（amplitude）→ dB : 20*log10
            db = 20.0 * np.log10(np.clip(data, 1e-10, None))
        else:
            db = data  # 已是 dB

    return np.clip(db, _DB_MIN, _DB_MAX).astype(np.float32)


class SARProcessor:
    """
    S1 / 通用 SAR 后处理器。

    对 source_dir 内所有 patch 的 .tif 文件执行 dB 转换（in-place 覆盖）。
    可传入 source_name="s1" 或 source_name="sar" 以区分日志。
    """

    def __init__(self, region_cfg: dict, source_name: str = "s1") -> None:
        self.region_cfg = region_cfg
        self.source_name = source_name
        # 对于云筛选后的目录优先用 cloud_filtered_dir（S2），SAR/S1 直接用 output_dir
        base = Path(region_cfg.get("cloud_filtered_dir", region_cfg["output_dir"]))
        # S1 不做云筛选，直接用原始 output_dir
        if source_name == "s1":
            base = Path(region_cfg["output_dir"])
        self.source_dir = base / source_name

    def run(self, workers: int = 1) -> dict:
        """
        遍历所有 patch，对每个 tif 执行 dB 转换（覆盖原文件）。

        Returns:
            {"processed": N, "skipped": N, "failed": N}
        """
        if not self.source_dir.exists():
            logger.warning(f"[SARProcessor/{self.source_name}] 目录不存在: {self.source_dir}")
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

        logger.info(f"[SARProcessor/{self.source_name}] 完成: {stats}")
        return stats

    def _process_tif(self, tif_path: Path) -> str:
        """处理单个 tif，覆盖写入 dB 值。跳过已是 dB 的文件。"""
        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
            crs = src.crs.to_string()
            bounds = [src.bounds.left, src.bounds.bottom,
                      src.bounds.right, src.bounds.top]

        # 已是合理 dB 范围则跳过（避免二次转换）
        if data.min() >= _DB_MIN - 5 and data.max() <= _DB_MAX + 5 and data.min() < 0:
            return "skipped"

        db_data = sar_to_db(data, self.source_name)
        write_tif(tif_path, db_data, crs=crs, bounds=bounds)
        return "processed"
