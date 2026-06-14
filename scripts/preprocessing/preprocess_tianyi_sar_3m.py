#!/usr/bin/env python3
"""将 ModelScope 下载的天仪 SAR ORG 产品预处理为 3m 对齐 patch。

输入:
    /workspace/xuannv/data_raw/_tmp_sar3m_download/.../*.zip (BC*-SM-ORG-*.zip)
输出:
    /workspace/xuannv/data_raw/haidian/scenes/{patch_id}/tianyi_sar_3m/YYYYMMDD.tif
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_config
from src.data.dataset import HarbinPatchDataset


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--download-dir", default="/workspace/xuannv/data_raw/_tmp_sar3m_download")
    pa.add_argument("--output-root", default="/workspace/xuannv/data_raw/haidian/scenes")
    pa.add_argument("--target-crs", default="EPSG:32650")
    pa.add_argument("--patch-size-m", type=float, default=1280.0)
    pa.add_argument("--target-gsd", type=float, default=3.0)
    pa.add_argument("--workers", type=int, default=4)
    return pa.parse_args()


def get_patch_bounds(cfg) -> dict[str, tuple]:
    """返回 {patch_id: (bounds, crs)}。"""
    cfg.data.preload = False
    cfg.data.num_workers = 0
    ds = HarbinPatchDataset(cfg=cfg)
    out = {}
    for pid in tqdm(ds.patches, desc="loading patch bounds"):
        bounds, crs = ds._compute_patch_bounds(pid)
        out[pid] = (bounds, crs)
    return out


def parse_date_from_zip(zip_path: Path) -> str | None:
    """从文件名解析日期，如 BC4-SM-ORG-2SVV-20250124T143303-... -> 20250124。"""
    m = re.search(r"(\d{8})T\d{6}", zip_path.name)
    if m:
        return m.group(1)
    return None


def extract_org_tiff(zip_path: Path, temp_dir: Path) -> Path | None:
    """解压 zip 中的 .tiff 文件，返回 tiff 路径。"""
    with zipfile.ZipFile(zip_path) as z:
        tiff_members = [m for m in z.namelist() if m.endswith(".tiff") and "ORG" in m.upper()]
        if not tiff_members:
            # fallback: any tiff
            tiff_members = [m for m in z.namelist() if m.endswith(".tiff")]
        if not tiff_members:
            return None
        z.extract(tiff_members[0], temp_dir)
        return next(Path(temp_dir).rglob("*.tiff"))


def reproject_to_patch(src_path: Path, bounds: rasterio.coords.BoundingBox, dst_shape: tuple[int, int],
                       dst_crs: str, dst_path: Path):
    """将单景 SAR 重投影/裁剪到 patch 网格。"""
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata if src.nodata is not None else 0.0

        dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                                    width=dst_shape[1], height=dst_shape[0])
        dst = np.full(dst_shape, src_nodata, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
        )

        # 简单归一化：clip 到 0-1 范围（根据当前 10m SAR 的统计）
        valid = dst != src_nodata
        if valid.any():
            vmin, vmax = np.percentile(dst[valid], [2, 98])
            if vmax > vmin:
                dst[valid] = np.clip((dst[valid] - vmin) / (vmax - vmin), 0.0, 1.0)

        with rasterio.open(
            dst_path,
            "w",
            driver="GTiff",
            height=dst_shape[0],
            width=dst_shape[1],
            count=1,
            dtype=dst.dtype,
            crs=dst_crs,
            transform=dst_transform,
            nodata=src_nodata,
        ) as dst_ds:
            dst_ds.write(dst, 1)


def process_zip(zip_path: Path, patch_bounds: dict, dst_root: Path, dst_shape: tuple[int, int],
                dst_crs: str, pbar: tqdm):
    date_str = parse_date_from_zip(zip_path)
    if date_str is None:
        pbar.write(f"[skip] cannot parse date: {zip_path.name}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tiff_path = extract_org_tiff(zip_path, Path(tmpdir))
        if tiff_path is None:
            pbar.write(f"[skip] no tiff in {zip_path.name}")
            return

        with rasterio.open(tiff_path) as src:
            # 计算场景在目标 CRS 下的 bounds
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            # 近似 bounds
            scene_bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)

        scene_left, scene_bottom, scene_right, scene_top = scene_bounds

        for pid, (bounds, crs) in patch_bounds.items():
            left, bottom, right, top = bounds
            # 判断是否有重叠
            if right < scene_left or left > scene_right or top < scene_bottom or bottom > scene_top:
                continue

            out_dir = dst_root / pid / "tianyi_sar_3m"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{date_str}.tif"
            if out_path.exists():
                continue

            try:
                reproject_to_patch(tiff_path, bounds, dst_shape, dst_crs, out_path)
            except Exception as e:
                pbar.write(f"[error] {pid} {date_str}: {e}")


def process_zip_wrapper(args_tuple):
    zip_path, patch_bounds, dst_root, dst_shape, dst_crs = args_tuple
    try:
        process_zip(zip_path, patch_bounds, dst_root, dst_shape, dst_crs, tqdm(disable=True))
        return (zip_path.name, "ok", None)
    except Exception as e:
        return (zip_path.name, "error", str(e))


def main():
    args = parse_args()
    download_dir = Path(args.download_dir)
    dst_root = Path(args.output_root)
    dst_shape = (427, 427)  # 与 Planet 3m 对齐
    print(f"[info] target shape: {dst_shape}, GSD: {args.target_gsd}m")

    cfg = load_config("configs/config_multires_v1.yaml")
    patch_bounds = get_patch_bounds(cfg)
    print(f"[info] {len(patch_bounds)} patches loaded")

    zips = sorted(download_dir.rglob("*ORG*.zip"))
    print(f"[info] {len(zips)} ORG zip files found")

    task_args = [(z, patch_bounds, dst_root, dst_shape, args.target_crs) for z in zips]
    with mp.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(process_zip_wrapper, task_args), total=len(zips), desc="processing SAR zips"))

    errors = [r for r in results if r[1] == "error"]
    print(f"[done] processed {len(results)} zips, errors: {len(errors)}")
    for name, status, msg in errors[:10]:
        print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
