#!/usr/bin/env python3
"""将 PlanetScope 高分辨率影像与 Haidian patch 对齐，生成 per-patch / per-date 的输入源.

用法:
    cd production/v1_haidian
    python scripts/prepare_planet_haidian.py \
        --planet-dir /workspace/xuannv/data_raw/planet_extracted/PSScene \
        --data-root /workspace/xuannv/data_raw/haidian/scenes \
        --target-size 256 \
        --stats-out /workspace/statistics/haidian_train/planet_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

sys.path.insert(0, "/workspace/xuannv")
from src.data.transforms import read_tif_aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet-dir", required=True, help="解压后 PSScene 目录，含 *_AnalyticMS_SR_clip.tif")
    parser.add_argument("--data-root", required=True, help="Haidian scenes 根目录（patch_id/source_name）")
    parser.add_argument("--target-size", type=int, default=256, help="输出 Planet 图像空间尺寸（正方形）")
    parser.add_argument("--stats-out", required=True, help="输出 planet_stats.json 路径")
    parser.add_argument("--apply-udm2", type=int, default=1, choices=[0, 1], help="是否用 udm2 band1 过滤非 clear 像素")
    parser.add_argument("--dry-run", action="store_true", help="只统计覆盖情况，不写入文件")
    return parser.parse_args()


def _discover_patches(data_root: Path) -> list[str]:
    """发现所有 patch_id（假设目录名为 patch_\d+）."""
    pids = [p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("patch_")]
    return sorted(pids)


def _patch_bounds(data_root: Path, patch_id: str) -> tuple[tuple[float, float, float, float], CRS]:
    """以 s2 / s1 / landsat 中第一个可用 tif 的 bounds 作为 patch 地理范围."""
    for src in ("s2", "s1", "landsat"):
        src_dir = data_root / patch_id / src
        if not src_dir.exists():
            src_dir = data_root / src / patch_id
        if src_dir.exists():
            tifs = sorted(src_dir.glob("*.tif"))
            if tifs:
                with rasterio.open(tifs[0]) as ds:
                    return ds.bounds, ds.crs
    raise FileNotFoundError(f"无法确定 {patch_id} 的地理范围")


def _scene_bounds(scene_path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(scene_path) as ds:
        return ds.bounds


def _intersection_area(a: tuple, b: tuple) -> float:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def _parse_scene_date(scene_name: str) -> str | None:
    """从文件名解析日期，如 20260430_033754_17_251d -> 20260430."""
    parts = scene_name.split("_")
    if parts and len(parts[0]) == 8 and parts[0].isdigit():
        return parts[0]
    return None


def _crop_scene_to_patch(
    scene_path: Path,
    udm2_path: Path | None,
    patch_bounds: tuple,
    dst_crs: CRS,
    target_size: int,
    apply_udm2: bool,
) -> np.ndarray | None:
    """把一张 Planet 场景裁剪/重采样到 patch 范围，返回 (4, H, W) float32."""
    dst_shape = (target_size, target_size)
    data = read_tif_aligned(
        scene_path,
        patch_bounds,
        dst_shape,
        dst_crs,
        resampling="bilinear",
        fill_value=0.0,
    )
    if data is None:
        return None
    if apply_udm2 and udm2_path is not None and udm2_path.exists():
        mask = read_tif_aligned(
            udm2_path,
            patch_bounds,
            dst_shape,
            dst_crs,
            resampling="nearest",
            fill_value=0.0,
        )
        if mask is not None and mask.shape[0] >= 1:
            clear = mask[0] > 0  # band1: clear
            data[:, ~clear] = 0.0
    return data


def _save_tif(path: Path, data: np.ndarray, bounds: tuple, crs: CRS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = data.shape[1], data.shape[2]
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width=w, height=h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data)


def _compute_stats(tif_paths: list[Path], max_samples: int = 500_000) -> dict[str, dict[str, float]]:
    """计算各通道 mean/std（全局统计，用于 z-score）."""
    sums = None
    sumsq = None
    counts = 0
    for p in tqdm(tif_paths, desc="stats"):
        with rasterio.open(p) as ds:
            data = ds.read().astype(np.float32)
        c = data.shape[0]
        if sums is None:
            sums = np.zeros(c, dtype=np.float64)
            sumsq = np.zeros(c, dtype=np.float64)
        flat = data.reshape(c, -1)
        valid = flat != 0.0
        counts += valid.sum(axis=1)
        sums += (flat * valid).sum(axis=1)
        sumsq += (flat ** 2 * valid).sum(axis=1)
    means = sums / np.maximum(counts, 1)
    stds = np.sqrt(np.maximum(sumsq / np.maximum(counts, 1) - means ** 2, 0))
    out: dict[str, dict[str, float]] = {}
    for i in range(len(means)):
        out[f"band_{i}"] = {"mean": float(means[i]), "std": float(stds[i])}
    return out


def main() -> int:
    args = parse_args()
    planet_dir = Path(args.planet_dir)
    data_root = Path(args.data_root)
    stats_out = Path(args.stats_out)

    scenes = sorted(planet_dir.glob("*_AnalyticMS_SR_clip.tif"))
    print(f"[prepare] 发现 Planet 场景: {len(scenes)}")

    # 按日期分组，记录每个场景的范围与 udm2 路径
    scenes_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene_path in scenes:
        date = _parse_scene_date(scene_path.stem)
        if date is None:
            continue
        udm2_path = scene_path.parent / scene_path.name.replace("_AnalyticMS_SR_clip.tif", "_udm2_clip.tif")
        bounds = _scene_bounds(scene_path)
        scenes_by_date[date].append({
            "path": scene_path,
            "udm2": udm2_path if udm2_path.exists() else None,
            "bounds": bounds,
        })
    print(f"[prepare] 日期数: {len(scenes_by_date)}")

    patches = _discover_patches(data_root)
    print(f"[prepare] Haidian patch 数: {len(patches)}")

    patch_info: list[tuple[str, tuple, CRS]] = []
    for pid in patches:
        try:
            bounds, crs = _patch_bounds(data_root, pid)
            patch_info.append((pid, bounds, crs))
        except FileNotFoundError:
            print(f"[warn] 无法确定 {pid} 范围，跳过")

    generated: list[Path] = []
    coverage: dict[str, int] = defaultdict(int)

    for pid, bounds, crs in tqdm(patch_info, desc="patches"):
        out_dir = data_root / pid / "planet"
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for date, scene_list in scenes_by_date.items():
            best = None
            best_area = 0.0
            for sc in scene_list:
                area = _intersection_area(bounds, sc["bounds"])
                if area > best_area:
                    best_area = area
                    best = sc
            if best is None or best_area <= 0:
                continue
            coverage[date] += 1
            if args.dry_run:
                continue
            out_path = out_dir / f"{date}.tif"
            if out_path.exists():
                generated.append(out_path)
                continue
            data = _crop_scene_to_patch(
                best["path"], best["udm2"], bounds, crs, args.target_size, bool(args.apply_udm2)
            )
            if data is None:
                continue
            _save_tif(out_path, data, bounds, crs)
            generated.append(out_path)

    print(f"[prepare] 生成/复用 TIFF 数: {len(generated)}")
    print("[prepare] 每日覆盖 patch 数:")
    for date in sorted(coverage):
        print(f"  {date}: {coverage[date]}")

    if not args.dry_run and generated:
        stats = _compute_stats(generated)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[prepare] 统计量已保存: {stats_out}")
        print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
