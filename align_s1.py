#!/usr/bin/env python3
"""S1 SAR 空间对齐修复脚本。

使用 S2 NIR 波段作为参考，对每个 patch 计算 S1 的最佳整数偏移，
然后应用偏移修复所有 S1 帧。

偏移检测方法：水体 mask IoU
"""
from __future__ import annotations

import rasterio
import numpy as np
from pathlib import Path
from scipy.ndimage import shift


def compute_offset_for_patch(patch_dir: Path) -> tuple[int, int] | None:
    """用 S1 和 S2 共同日期的水体 mask IoU 找最佳整数偏移。"""
    s1_files = {f.stem: f for f in sorted((patch_dir / "s1").glob("*.tif"))}
    s2_files = {f.stem: f for f in sorted((patch_dir / "s2").glob("*.tif"))}
    common = sorted(set(s1_files.keys()) & set(s2_files.keys()))
    if not common:
        return None

    best_offsets = []
    for date in common[:5]:
        with rasterio.open(s1_files[date]) as src:
            s1_data = src.read(1).astype(np.float32)
            s1_nodata = src.nodata
        with rasterio.open(s2_files[date]) as src:
            s2_nir = src.read(4).astype(np.float32)

        # 确保 shape 一致
        if s1_data.shape != s2_nir.shape:
            h = min(s1_data.shape[0], s2_nir.shape[0])
            w = min(s1_data.shape[1], s2_nir.shape[1])
            s1_data = s1_data[:h, :w]
            s2_nir = s2_nir[:h, :w]

        s1 = np.where(
            (s1_data != s1_nodata) & np.isfinite(s1_data) & (s1_data > 0),
            s1_data, np.nan,
        )
        if np.all(np.isnan(s1)):
            continue
        s1_log = np.log10(s1)
        s1_water = s1_log < np.nanpercentile(s1_log, 25)
        s2_water = s2_nir < np.percentile(s2_nir[s2_nir > 0], 30)

        best_iou = -1
        best_dx, best_dy = 0, 0
        for dx in range(-4, 3):
            for dy in range(-3, 4):
                s1_shifted = shift(
                    s1_water.astype(np.float32), shift=(dy, dx),
                    order=0, mode="constant", cval=0,
                )
                inter = np.sum((s1_shifted > 0.5) & s2_water)
                union = np.sum((s1_shifted > 0.5) | s2_water)
                if union > 0:
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_dx, best_dy = dx, dy

        best_offsets.append((best_dx, best_dy, best_iou))

    if not best_offsets:
        return None
    dxs = [o[0] for o in best_offsets]
    dys = [o[1] for o in best_offsets]
    return int(np.median(dxs)), int(np.median(dys))


def align_s1_for_patch(patch_dir: Path, dx: int, dy: int) -> int:
    s1_dir = patch_dir / "s1"
    if not s1_dir.exists():
        return 0
    fixed_count = 0
    for s1_f in sorted(s1_dir.glob("*.tif")):
        with rasterio.open(s1_f) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile
        shifted = shift(
            data, shift=(dy, dx), order=1, mode="constant",
            cval=profile.get("nodata", -9999),
        )
        shifted = shifted.astype(profile["dtype"])
        with rasterio.open(s1_f, "w", **profile) as dst:
            dst.write(shifted, 1)
        fixed_count += 1
    return fixed_count


def main():
    haidian_root = Path("data_raw/haidian/scenes")
    patches = sorted([d for d in haidian_root.iterdir() if d.is_dir() and d.name.startswith("patch_")])
    print(f"共 {len(patches)} 个 patches")

    for i, patch_dir in enumerate(patches):
        patch_id = patch_dir.name
        offset = compute_offset_for_patch(patch_dir)
        if offset is None:
            print(f"[{i+1}/{len(patches)}] {patch_id}: 无共同日期，跳过")
            continue
        dx, dy = offset
        fixed = align_s1_for_patch(patch_dir, dx, dy)
        print(f"[{i+1}/{len(patches)}] {patch_id}: dx={dx:+d}, dy={dy:+d}, 修复 {fixed} 帧")


if __name__ == "__main__":
    main()
