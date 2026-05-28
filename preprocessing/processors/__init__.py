"""处理器工厂。"""
from __future__ import annotations


def get_processor(region_cfg: dict, source_name: str):
    """根据 source_name 返回合适的处理器实例。

    注意: s2 的后处理（云筛选）由 cloud_filter 步骤单独处理，
    process 步骤跳过 s2（抛出 ValueError 由 orchestrator catch）。
    """
    if source_name == "s2":
        # s2 已在 cloud_filter 步骤处理，process 步骤主动跳过
        raise ValueError("s2 由 cloud_filter 步骤处理，无需 process 步骤后处理")
    elif source_name in ("s1", "sar"):
        from preprocessing.processors.s1_sar import SARProcessor
        return SARProcessor(region_cfg, source_name)
    elif source_name == "landsat":
        from preprocessing.processors.landsat import LandsatProcessor
        return LandsatProcessor(region_cfg)
    elif source_name in ("dem", "worldcover", "dynamic_world", "jrc_water"):
        from preprocessing.processors.reference import ReferenceProcessor
        return ReferenceProcessor(region_cfg, source_name)
    else:
        raise ValueError(f"未知源 '{source_name}'，无对应处理器")
