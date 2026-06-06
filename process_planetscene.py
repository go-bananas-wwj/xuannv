#!/usr/bin/env python3
"""将 PlanetScene PSScene 瓦片拼接并裁剪为海淀 Patch 格式."""

import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds


def load_patches_meta(meta_path: Path) -> tuple[list[dict], str]:
    """加载 patch 元数据."""
    with open(meta_path) as f:
        meta = json.load(f)
    patches = meta["patches"]
    crs = meta.get("crs", "EPSG:32650")
    return patches, crs


def load_tile_info(raw_dir: Path) -> list[dict]:
    """扫描所有 PSScene 瓦片，提取日期和路径."""
    tiles = []
    for tif_path in sorted(raw_dir.glob("*AnalyticMS_SR_clip.tif")):
        # 文件名: YYYYMMDD_HHMMSS_xx_xxxx_3B_AnalyticMS_SR_clip.tif
        date_str = tif_path.stem.split("_")[0]
        date = datetime.strptime(date_str, "%Y%m%d")
        tiles.append({
            "path": tif_path,
            "date": date,
            "date_str": date_str,
        })
    return tiles


def get_tile_bounds(tile_path: Path) -> rasterio.coords.BoundingBox | None:
    """读取瓦片地理范围."""
    try:
        with rasterio.open(tile_path) as src:
            return src.bounds
    except Exception as e:
        print(f"警告: 无法读取 {tile_path}: {e}")
        return None


def tiles_intersecting_patch(tiles: list[dict], patch_bounds: list[float]) -> list[dict]:
    """找出与 patch 相交的瓦片 (AABB 测试)."""
    pl, pb, pr, pt = patch_bounds
    result = []
    for tile in tiles:
        b = tile.get("bounds")
        if b is None:
            continue
        # 相交测试
        if not (b.right < pl or b.left > pr or b.top < pb or b.bottom > pt):
            result.append(tile)
    return result


def process_patch_date(
    patch: dict,
    tiles: list[dict],
    output_dir: Path,
    crs: str,
    target_res: float | None = None,
) -> bool:
    """处理单个 patch 单个日期: 拼接瓦片并裁剪."""
    patch_id = f"patch_{patch['id']:06d}"
    patch_bounds = patch["utm_bounds"]  # [left, bottom, right, top]
    date_str = tiles[0]["date_str"]

    if not tiles:
        return False

    # 打开所有瓦片
    datasets = []
    for tile in tiles:
        try:
            src = rasterio.open(tile["path"])
            datasets.append(src)
        except Exception as e:
            print(f"  警告: 无法打开 {tile['path']}: {e}")

    if not datasets:
        return False

    # 拼接覆盖 patch 区域的镶嵌图
    # 先计算 patch 区域的 window
    try:
        # merge 所有瓦片（内存友好的小块）
        # 为了节省内存，我们只 merge 覆盖 patch bounds 的区域
        mosaic, mosaic_transform = merge(
            datasets,
            bounds=tuple(patch_bounds),
            resampling=rasterio.enums.Resampling.nearest,
        )
    except Exception as e:
        print(f"  拼接失败: {e}")
        for ds in datasets:
            ds.close()
        return False
    finally:
        for ds in datasets:
            ds.close()

    # mosaic shape: (bands, height, width)
    # 计算输出 transform
    if target_res is None:
        # 保持原始分辨率 (3m)
        out_transform = mosaic_transform
        out_height, out_width = mosaic.shape[1], mosaic.shape[2]
    else:
        # 重采样到 target_res
        out_width = int((patch_bounds[2] - patch_bounds[0]) / target_res)
        out_height = int((patch_bounds[3] - patch_bounds[1]) / target_res)
        out_transform = transform_from_bounds(
            *patch_bounds, out_width, out_height
        )
        # 重采样
        from rasterio.warp import reproject, Resampling
        dst_shape = (mosaic.shape[0], out_height, out_width)
        dst = np.empty(dst_shape, dtype=mosaic.dtype)
        reproject(
            source=mosaic,
            destination=dst,
            src_transform=mosaic_transform,
            src_crs=crs,
            dst_transform=out_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
        mosaic = dst

    # 保存 TIFF
    patch_dir = output_dir / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    out_path = patch_dir / f"{date_str}.tif"

    # 从第一个瓦片读取 profile
    with rasterio.open(tiles[0]["path"]) as src:
        profile = src.profile.copy()

    profile.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
        "crs": crs,
        "transform": out_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)

    return True


def visualize_patch(patch_dir: Path, out_png: Path) -> None:
    """生成 patch 的 RGB 可视化图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tif_files = sorted(patch_dir.glob("*.tif"))
    if not tif_files:
        return

    n = min(len(tif_files), 4)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, tif_path in zip(axes, tif_files[:n]):
        with rasterio.open(tif_path) as src:
            rgb = src.read([3, 2, 1])  # PlanetScope: B, G, R, NIR -> 取 R, G, B (bands 3,2,1?)
            # PlanetScope 波段: 1=Blue, 2=Green, 3=Red, 4=NIR
            # 所以 RGB = bands 3, 2, 1
            rgb = np.clip(rgb / np.percentile(rgb, 98) * 255, 0, 255).astype(np.uint8)
            rgb = np.transpose(rgb, (1, 2, 0))
            ax.imshow(rgb)
            ax.set_title(tif_path.stem)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"可视化图已保存: {out_png}")


def main():
    meta_path = Path("/workspace/xuannv/tools/patches_meta.json")
    raw_dir = Path("/workspace/xuannv/data_raw/beijing/planetscene_raw/PSScene")
    output_dir = Path("/workspace/xuannv/data_raw/beijing/planetscene")
    viz_dir = Path("/workspace/xuannv/data_raw/beijing/viz")

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    patches, crs = load_patches_meta(meta_path)
    print(f"Patch 数: {len(patches)}, CRS: {crs}")

    tiles = load_tile_info(raw_dir)
    print(f"瓦片数: {len(tiles)}")

    # 预读取所有瓦片 bounds
    print("预读取瓦片地理范围...")
    for tile in tiles:
        tile["bounds"] = get_tile_bounds(tile["path"])
    tiles = [t for t in tiles if t["bounds"] is not None]
    print(f"有效瓦片: {len(tiles)}")

    # 按日期分组
    from collections import defaultdict
    date_tiles = defaultdict(list)
    for tile in tiles:
        date_tiles[tile["date_str"]].append(tile)

    print(f"日期数: {len(date_tiles)}")
    for date_str in sorted(date_tiles.keys()):
        print(f"  {date_str}: {len(date_tiles[date_str])} 瓦片")

    # 逐个日期处理
    total_processed = 0
    for date_str in sorted(date_tiles.keys()):
        dtiles = date_tiles[date_str]
        print(f"\n处理日期 {date_str} ({len(dtiles)} 瓦片)...")

        for i, patch in enumerate(patches):
            patch_id = f"patch_{patch['id']:06d}"
            intersecting = tiles_intersecting_patch(dtiles, patch["utm_bounds"])

            if not intersecting:
                continue

            success = process_patch_date(
                patch, intersecting, output_dir, crs, target_res=None
            )
            if success:
                total_processed += 1

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(patches)} patches")

    print(f"\n总计生成 {total_processed} 个 patch-日期组合")

    # 生成可视化图
    print("\n生成可视化图...")
    for patch in patches[:5]:
        patch_id = f"patch_{patch['id']:06d}"
        patch_dir = output_dir / patch_id
        if patch_dir.exists() and any(patch_dir.iterdir()):
            visualize_patch(patch_dir, viz_dir / f"{patch_id}_planetscene.png")


if __name__ == "__main__":
    main()
