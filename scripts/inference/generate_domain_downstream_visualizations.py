#!/usr/bin/env python3
"""拼接下游任务重建结果为全域大图.

输入:
  /workspace/outputs/aef_qwen_v5_mixed_scale/downstream_recon_2026/{target_name}/

输出:
  /workspace/outputs/aef_qwen_v5_mixed_scale/domain_wide/
    - domain_dem_{month}.png
    - domain_worldcover_{month}.png
    - domain_dynamic_world_{month}.png
    - domain_jrc_water_{month}.png
    - domain_s2_recon_{month}.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import rasterio
import glob

DATA_ROOT = Path("/workspace/raw/harbin_scenes")
RECON_ROOT = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/downstream_recon_2026")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/domain_wide")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
MONTH_LABELS = {
    "2026-01": "Jan", "2026-02": "Feb", "2026-03": "Mar",
    "2026-04": "Apr", "2026-05": "May"
}

PATCH_H, PATCH_W = 133, 134


def load_grid_mapping():
    patches = []
    for i in range(424):
        pid = f"patch_{i:06d}"
        tifs = list((DATA_ROOT / "s1" / pid).glob("*.tif"))
        if tifs:
            with rasterio.open(tifs[0]) as src:
                ulx = src.transform.c
                uly = src.transform.f
                patches.append((pid, ulx, uly))
    unique_y = sorted(set(p[2] for p in patches), reverse=True)
    unique_x = sorted(set(p[1] for p in patches))
    y_to_row = {y: i for i, y in enumerate(unique_y)}
    x_to_col = {x: i for i, x in enumerate(unique_x)}
    grid = {}
    for pid, ulx, uly in patches:
        grid[pid] = (y_to_row[uly], x_to_col[ulx])
    return grid


GRID = load_grid_mapping()
N_ROWS = max(v[0] for v in GRID.values()) + 1
N_COLS = max(v[1] for v in GRID.values()) + 1


def stitch_patches(get_patch_fn, is_rgb=False, fill_value=0.0):
    if is_rgb:
        canvas = np.full((N_ROWS * PATCH_H, N_COLS * PATCH_W, 3), fill_value, dtype=np.float32)
    else:
        canvas = np.full((N_ROWS * PATCH_H, N_COLS * PATCH_W), fill_value, dtype=np.float32)

    for pid, (row, col) in tqdm(GRID.items(), desc="Stitching", leave=False):
        patch_data = get_patch_fn(pid)
        if patch_data is None:
            continue
        y0 = row * PATCH_H
        x0 = col * PATCH_W
        if is_rgb:
            h, w = patch_data.shape[:2]
            canvas[y0:y0+h, x0:x0+w, :] = patch_data[:min(h, PATCH_H), :min(w, PATCH_W), :]
        else:
            h, w = patch_data.shape
            canvas[y0:y0+h, x0:x0+w] = patch_data[:min(h, PATCH_H), :min(w, PATCH_W)]
    return canvas


def save_figure(data, title, out_path, cmap=None, vmin=None, vmax=None, is_rgb=False):
    fig, ax = plt.subplots(1, 1, figsize=(20, 18))
    if is_rgb:
        ax.imshow(np.clip(data, 0, 1))
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("=" * 60)
    print("下游任务全域可视化大图生成")
    print("=" * 60)

    # DEM 重建
    print("\n[1/5] DEM 重建...")
    for month in MONTHS:
        label = MONTH_LABELS[month]
        target_dir = RECON_ROOT / "dem"
        if not target_dir.exists():
            print(f"  Skip {month}: {target_dir} not found")
            continue
        canvas = stitch_patches(lambda pid: np.load(target_dir / f"{pid}_{month}.npy") if (target_dir / f"{pid}_{month}.npy").exists() else None)
        save_figure(canvas, f"DEM Reconstruction: {label} 2026", OUTPUT_DIR / f"domain_dem_{month}.png", cmap="terrain")

    # WorldCover 分类
    print("\n[2/5] WorldCover 分类...")
    # WorldCover 颜色映射 (ESA WorldCover 官方配色)
    wc_colors = {
        0: "#006400",   # Tree cover
        1: "#ffbb22",   # Shrubland
        2: "#ffff4c",   # Grassland
        3: "#f096ff",   # Cropland
        4: "#fa0000",   # Built-up
        5: "#b4b4b4",   # Bare / sparse vegetation
        6: "#f0f0f0",   # Snow and ice
        7: "#0064c8",   # Permanent water bodies
        8: "#0096a0",   # Herbaceous wetland
        9: "#00cf75",   # Mangroves
        10: "#fae6a0",  # Moss and lichen
    }
    wc_cmap = ListedColormap([wc_colors.get(i, "#000000") for i in range(11)])
    for month in MONTHS:
        label = MONTH_LABELS[month]
        target_dir = RECON_ROOT / "worldcover"
        if not target_dir.exists():
            continue
        canvas = stitch_patches(lambda pid: np.load(target_dir / f"{pid}_{month}.npy") if (target_dir / f"{pid}_{month}.npy").exists() else None)
        save_figure(canvas, f"WorldCover Classification: {label} 2026", OUTPUT_DIR / f"domain_worldcover_{month}.png", cmap=wc_cmap, vmin=0, vmax=10)

    # Dynamic World 分类
    print("\n[3/5] Dynamic World 分类...")
    dw_colors = {
        0: "#419bdf",  # Water
        1: "#397d49",  # Trees
        2: "#88b053",  # Grass
        3: "#7a87c6",  # Flooded vegetation
        4: "#e49635",  # Crops
        5: "#dfc35a",  # Shrub and scrub
        6: "#c4281b",  # Built
        7: "#a59b8f",  # Bare
        8: "#b39fe1",  # Snow and ice
    }
    dw_cmap = ListedColormap([dw_colors.get(i, "#000000") for i in range(9)])
    for month in MONTHS:
        label = MONTH_LABELS[month]
        target_dir = RECON_ROOT / "dynamic_world"
        if not target_dir.exists():
            continue
        canvas = stitch_patches(lambda pid: np.load(target_dir / f"{pid}_{month}.npy") if (target_dir / f"{pid}_{month}.npy").exists() else None)
        save_figure(canvas, f"Dynamic World Classification: {label} 2026", OUTPUT_DIR / f"domain_dynamic_world_{month}.png", cmap=dw_cmap, vmin=0, vmax=8)

    # JRC Water
    print("\n[4/5] JRC Water...")
    for month in MONTHS:
        label = MONTH_LABELS[month]
        target_dir = RECON_ROOT / "jrc_water"
        if not target_dir.exists():
            continue
        canvas = stitch_patches(lambda pid: np.load(target_dir / f"{pid}_{month}.npy") if (target_dir / f"{pid}_{month}.npy").exists() else None)
        save_figure(canvas, f"JRC Water: {label} 2026", OUTPUT_DIR / f"domain_jrc_water_{month}.png", cmap="Blues", vmin=0, vmax=1)

    # S2 RGB 重建
    print("\n[5/5] S2 RGB 重建...")
    for month in MONTHS:
        label = MONTH_LABELS[month]
        target_dir = RECON_ROOT / "s2_recon"
        if not target_dir.exists():
            continue
        canvas = stitch_patches(
            lambda pid: np.load(target_dir / f"{pid}_{month}.npy").transpose(1, 2, 0) if (target_dir / f"{pid}_{month}.npy").exists() else None,
            is_rgb=True
        )
        save_figure(canvas, f"S2 RGB Reconstruction: {label} 2026", OUTPUT_DIR / f"domain_s2_recon_{month}.png", is_rgb=True)

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
