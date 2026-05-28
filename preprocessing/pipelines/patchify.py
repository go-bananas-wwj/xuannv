"""Patch 切割流水线。

从区域配置生成 patches_meta.json，内容与训练代码期望的格式对齐：
    {
        "region_name": "harbin",
        "crs": "EPSG:32652",
        "patch_size_m": 1280,
        "step_m": 1280,
        "patches": [
            {"id": 0, "utm_bounds": [left, bottom, right, top], "center_lonlat": [lon, lat]},
            ...
        ]
    }

生成文件路径: {output_dir}/patches_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

from preprocessing.utils.geo import bbox_to_utm_patches
from preprocessing.utils.logging import get_logger

logger = get_logger(__name__)


def run_patchify(region_cfg: dict, max_patches: int | None = None) -> list[dict]:
    """
    生成 patch 网格并写入 patches_meta.json。

    Args:
        region_cfg  : 区域配置字典
        max_patches : 可选上限（用于快速调试）

    Returns:
        patch 列表
    """
    patch_cfg = region_cfg["patch"]
    bbox = region_cfg["bbox"]
    crs_override = region_cfg.get("crs")

    patches, crs = bbox_to_utm_patches(
        bbox_deg=bbox,
        patch_size_m=patch_cfg["size_m"],
        step_m=patch_cfg.get("step_m", patch_cfg["size_m"]),
        crs_override=crs_override,
        utm_grid=patch_cfg.get("utm_grid"),
    )

    if max_patches is not None:
        patches = patches[:max_patches]

    meta = {
        "region_name": region_cfg["region_name"],
        "crs": crs,
        "patch_size_m": patch_cfg["size_m"],
        "step_m": patch_cfg.get("step_m", patch_cfg["size_m"]),
        "n_patches": len(patches),
        "patches": patches,
    }

    out_dir = Path(region_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "patches_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    logger.info(
        f"[Patchify/{region_cfg['region_name']}] "
        f"生成 {len(patches)} 个 patches, CRS={crs} → {meta_path}"
    )
    return patches


def load_patches(region_cfg: dict) -> list[dict]:
    """从已有 patches_meta.json 加载 patch 列表。"""
    meta_path = Path(region_cfg["output_dir"]) / "patches_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"patches_meta.json 不存在: {meta_path}\n"
            f"请先运行 patchify 步骤。"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["patches"]
