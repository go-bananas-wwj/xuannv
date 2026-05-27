#!/usr/bin/env python3
"""
准备下游任务的像素级标注数据。

输出:
  /workspace/xuannv/data/labels/water/{patch_id}.npy     → 水体二值mask [128,128]
  /workspace/xuannv/data/labels/building/{patch_id}.npy  → 建筑物二值mask [128,128]
  /workspace/xuannv/data/labels/landuse/{patch_id}.npy   → 土地利用多类 [128,128]
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


def jrc_water_to_mask(jrc_path: Path) -> np.ndarray:
    """JRC Water → 水体二值mask (64x64, 与embedding_map对齐)
    
    JRC Water编码 (v1.4):
    - 0 = No data
    - 1 = Not water
    - 2 = Permanent water
    - 其他值 = 季节性水体等
    """
    with rasterio.open(jrc_path) as src:
        data = src.read(1)  # [43, 43]

    # 上采样到64x64 (最近邻，与embedding_map对齐)
    from scipy.ndimage import zoom
    mask = (data >= 2).astype(np.uint8)
    mask = zoom(mask, (64/43, 64/43), order=0)  # 最近邻上采样
    mask = (mask > 0.5).astype(np.uint8)
    return mask


def worldcover_to_building_mask(wc_path: Path) -> np.ndarray:
    """WorldCover → 建筑物二值mask (64x64, 与embedding_map对齐)"""
    with rasterio.open(wc_path) as src:
        data = src.read(1)  # [128, 128]
    mask = (data == 50).astype(np.uint8)
    # 下采样到64x64 (最近邻保持标签)
    from scipy.ndimage import zoom
    mask = zoom(mask, (0.5, 0.5), order=0)
    return mask


def worldcover_to_landuse(wc_path: Path) -> np.ndarray:
    """WorldCover → 土地利用多类分类 (64x64, 与embedding_map对齐)"""
    CLASS_MAP = {
        10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 6,
    }

    with rasterio.open(wc_path) as src:
        data = src.read(1)  # [128, 128]

    # 先映射到连续索引
    out = np.zeros_like(data, dtype=np.uint8)
    for old_cls, new_cls in CLASS_MAP.items():
        out[data == old_cls] = new_cls

    # 下采样到64x64 (最近邻保持类别标签)
    from scipy.ndimage import zoom
    out = zoom(out, (0.5, 0.5), order=0)
    return out


def process_all_labels(data_root: Path, output_dir: Path, patch_ids: list[str] | None = None):
    """处理所有patch的标注"""

    if patch_ids is None:
        # 从jrc_water目录获取所有patch
        patch_ids = sorted([d.name for d in (data_root / "jrc_water").iterdir() if d.is_dir()])

    print(f"处理 {len(patch_ids)} 个patch的标注...")

    # 创建输出目录
    (output_dir / "water").mkdir(parents=True, exist_ok=True)
    (output_dir / "building").mkdir(parents=True, exist_ok=True)
    (output_dir / "landuse").mkdir(parents=True, exist_ok=True)

    stats = {"water_pixels": 0, "building_pixels": 0, "landuse_pixels": 0}

    for pid in tqdm(patch_ids):
        # JRC Water
        jrc_path = data_root / "jrc_water" / pid / "static.tif"
        if jrc_path.exists():
            mask = jrc_water_to_mask(jrc_path)
            np.save(output_dir / "water" / f"{pid}.npy", mask)
            stats["water_pixels"] += mask.sum()

        # WorldCover → Building
        wc_path = data_root / "worldcover" / pid / "static.tif"
        if wc_path.exists():
            mask = worldcover_to_building_mask(wc_path)
            np.save(output_dir / "building" / f"{pid}.npy", mask)
            stats["building_pixels"] += mask.sum()

            # WorldCover → Landuse
            label = worldcover_to_landuse(wc_path)
            np.save(output_dir / "landuse" / f"{pid}.npy", label)
            stats["landuse_pixels"] += label.size

    print(f"\n标注准备完成!")
    print(f"  水体像素: {stats['water_pixels']}")
    print(f"  建筑物像素: {stats['building_pixels']}")
    print(f"  总土地利用像素: {stats['landuse_pixels']}")
    return stats


def main():
    data_root = Path("/workspace/raw/harbin_scenes_cloud_filtered")
    output_dir = Path("/workspace/xuannv/data/labels")

    # 获取patch列表（从grid geojson）
    import geopandas as gpd
    grid = gpd.read_file("/workspace/index/harbin/grid/harbin_grid.geojson")
    patch_ids = sorted(grid["patch_id"].tolist())

    process_all_labels(data_root, output_dir, patch_ids)


if __name__ == "__main__":
    main()
