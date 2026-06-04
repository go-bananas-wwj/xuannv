#!/usr/bin/env python3
"""生成哈尔滨新区 2026 年 1-5 月全域可视化大图.

包括:
1. 全域 RGB 影像图 (每个月份一张)
2. 全域变化检测图 (每个月度对一张)
3. 全域 Embedding 特征图 (每个月份一张)

输出:
  /workspace/outputs/aef_qwen_v5_mixed_scale/domain_wide/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# ============ 配置 ============
DATA_ROOT = Path("/workspace/raw/harbin_scenes")
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")
CD_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/change_scores_2026")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/domain_wide")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 月份配置
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
MONTH_PAIRS = [
    ("2026-01", "2026-02"),
    ("2026-02", "2026-03"),
    ("2026-03", "2026-04"),
    ("2026-04", "2026-05"),
]
MONTH_LABELS = {
    "2026-01": "Jan", "2026-02": "Feb", "2026-03": "Mar",
    "2026-04": "Apr", "2026-05": "May"
}

# 颜色映射
def make_change_cmap():
    """变化检测热力图: 黑色(无变化) -> 红色(变化)"""
    return LinearSegmentedColormap.from_list("change", ["#000000", "#440000", "#880000", "#cc0000", "#ff0000"])


def make_embedding_cmap():
    """Embedding 幅度图: 深蓝色 -> 青色 -> 黄色 -> 红色"""
    return LinearSegmentedColormap.from_list("emb", ["#000044", "#0088ff", "#00ffcc", "#ffff00", "#ff0000"])


# ============ Grid 映射 ============
def load_grid_mapping():
    """加载 patch 到 grid 位置的映射."""
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

    return grid, len(unique_y), len(unique_x)


GRID, N_ROWS, N_COLS = load_grid_mapping()
print(f"Grid: {N_ROWS} rows x {N_COLS} cols, {len(GRID)} patches")

# 每个 patch 的像素尺寸
PATCH_H, PATCH_W = 133, 134


# ============ 读取函数 ============
def read_patch_rgb(pid: str, month: str) -> np.ndarray | None:
    """读取指定 patch 和月份的 S2 RGB 影像."""
    s2_dir = DATA_ROOT / "s2" / pid
    if not s2_dir.exists():
        return None

    # 找到该月份的 TIFF 文件
    month_prefix = month.replace("-", "")
    tifs = sorted(s2_dir.glob(f"{month_prefix}*.tif"))
    if not tifs:
        # 尝试更宽泛的匹配
        tifs = sorted(s2_dir.glob("*.tif"))
        # 筛选该月份的
        tifs = [t for t in tifs if month_prefix in t.name]
    if not tifs:
        return None

    # 用第一个匹配的文件
    with rasterio.open(tifs[0]) as src:
        data = src.read()  # [C, H, W]

    # RGB: bands 2, 1, 0 (红、绿、蓝)
    rgb = data[[2, 1, 0], :, :].transpose(1, 2, 0)  # [H, W, 3]

    # 归一化到 0-1
    rgb = np.clip(rgb, 0, 1)
    return rgb.astype(np.float32)


def read_patch_change(pid: str, m1: str, m2: str) -> np.ndarray | None:
    """读取指定 patch 和月份对的变化检测分数."""
    path = CD_DIR / f"{pid}_{m1}_to_{m2}.npy"
    if not path.exists():
        return None
    return np.load(path)  # [H, W]


def read_patch_embedding(pid: str, month: str) -> np.ndarray | None:
    """读取指定 patch 和月份的 embedding，返回幅度图."""
    path = EMB_DIR / f"{pid}_{month}.npy"
    if not path.exists():
        return None
    emb = np.load(path)  # [D, H, W]
    # 计算 L2 幅度
    magnitude = np.linalg.norm(emb, axis=0)  # [H, W]
    return magnitude


# ============ 拼接函数 ============
def stitch_patches(get_patch_fn, fill_value=0.0) -> np.ndarray:
    """将所有 patch 按 grid 拼接成一张大图."""
    canvas = np.full((N_ROWS * PATCH_H, N_COLS * PATCH_W), fill_value, dtype=np.float32)

    for pid, (row, col) in tqdm(GRID.items(), desc="Stitching"):
        patch_data = get_patch_fn(pid)
        if patch_data is None:
            continue

        y0 = row * PATCH_H
        x0 = col * PATCH_W

        # 处理单通道或多通道
        if patch_data.ndim == 2:
            h, w = patch_data.shape
            canvas[y0:y0+h, x0:x0+w] = patch_data[:min(h, PATCH_H), :min(w, PATCH_W)]
        else:
            h, w, c = patch_data.shape
            if canvas.ndim == 2:
                canvas = np.stack([canvas] * c, axis=-1)
            canvas[y0:y0+h, x0:x0+w, :] = patch_data[:min(h, PATCH_H), :min(w, PATCH_W), :]

    return canvas


def stitch_rgb_patches(get_rgb_fn) -> np.ndarray:
    """将 RGB patch 拼接成大图."""
    canvas = np.full((N_ROWS * PATCH_H, N_COLS * PATCH_W, 3), 0.0, dtype=np.float32)

    for pid, (row, col) in tqdm(GRID.items(), desc="Stitching RGB"):
        rgb = get_rgb_fn(pid)
        if rgb is None:
            continue

        y0 = row * PATCH_H
        x0 = col * PATCH_W
        h, w = rgb.shape[:2]
        canvas[y0:y0+h, x0:x0+w, :] = rgb[:min(h, PATCH_H), :min(w, PATCH_W), :]

    return canvas


# ============ 保存函数 ============
def save_figure(data, title, out_path, cmap=None, vmin=None, vmax=None, is_rgb=False):
    """保存大图到文件."""
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


# ============ 主流程 ============
def main():
    print("=" * 60)
    print("全域可视化大图生成")
    print("=" * 60)

    # 1. 全域 RGB 影像图 (每个月份)
    print("\n[1/3] 生成全域 RGB 影像图...")
    for month in MONTHS:
        label = MONTH_LABELS[month]
        print(f"  Month: {label}")

        canvas = stitch_rgb_patches(lambda pid: read_patch_rgb(pid, month))
        out_path = OUTPUT_DIR / f"domain_rgb_{month}.png"
        save_figure(canvas, f"Harbin New Area - {label} 2026 RGB", out_path, is_rgb=True)

    # 2. 全域变化检测图 (每个月度对)
    print("\n[2/3] 生成全域变化检测图...")
    change_cmap = make_change_cmap()
    for m1, m2 in MONTH_PAIRS:
        label1 = MONTH_LABELS[m1]
        label2 = MONTH_LABELS[m2]
        print(f"  Period: {label1} → {label2}")

        canvas = stitch_patches(lambda pid: read_patch_change(pid, m1, m2))
        out_path = OUTPUT_DIR / f"domain_change_{m1}_to_{m2}.png"
        save_figure(canvas, f"Change Detection: {label1} → {label2} 2026",
                   out_path, cmap=change_cmap, vmin=0, vmax=1)

    # 3. 全域 Embedding 幅度图 (每个月份)
    print("\n[3/3] 生成全域 Embedding 幅度图...")
    emb_cmap = make_embedding_cmap()
    for month in MONTHS:
        label = MONTH_LABELS[month]
        print(f"  Month: {label}")

        # 先收集所有幅度值来确定全局范围
        magnitudes = []
        for pid in GRID:
            mag = read_patch_embedding(pid, month)
            if mag is not None:
                magnitudes.append(mag)

        if magnitudes:
            all_mags = np.concatenate([m.flatten() for m in magnitudes])
            vmin, vmax = np.percentile(all_mags, [1, 99])
        else:
            vmin, vmax = 0, 1

        canvas = stitch_patches(lambda pid: read_patch_embedding(pid, month))
        out_path = OUTPUT_DIR / f"domain_embedding_{month}.png"
        save_figure(canvas, f"Embedding Magnitude: {label} 2026",
                   out_path, cmap=emb_cmap, vmin=vmin, vmax=vmax)

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
