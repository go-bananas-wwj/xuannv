#!/usr/bin/env python3
"""生成 PlanetScene patch 时序可视化图，展示更多 patch 和所有月份."""

import numpy as np
import rasterio
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stretch_band(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """按分位数拉伸单波段到 0-255."""
    p_low = np.percentile(band, low)
    p_high = np.percentile(band, high)
    if p_high <= p_low:
        p_high = p_low + 1
    stretched = (band - p_low) / (p_high - p_low) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)


def visualize_patch_all_dates(patch_dir: Path, out_png: Path) -> None:
    """生成单个 patch 所有日期的对比图."""
    tif_files = sorted(patch_dir.glob("*.tif"))
    if not tif_files:
        return

    n = len(tif_files)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, tif_path in zip(axes, tif_files):
        with rasterio.open(tif_path) as src:
            # PlanetScope: B1=Blue, B2=Green, B3=Red, B4=NIR
            # RGB = bands 3, 2, 1
            r = src.read(3)
            g = src.read(2)
            b = src.read(1)

            # 对每个波段单独做 2%-98% 拉伸
            r_s = stretch_band(r)
            g_s = stretch_band(g)
            b_s = stretch_band(b)

            rgb = np.stack([r_s, g_s, b_s], axis=2)

        ax.imshow(rgb)
        ax.set_title(tif_path.stem, fontsize=14)
        ax.axis("off")

    # 隐藏多余的子图
    for ax in axes[n:]:
        ax.axis("off")

    patch_name = patch_dir.name
    fig.suptitle(f"PlanetScene 3m - {patch_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {out_png}")


def main():
    planetscene_dir = Path("/workspace/xuannv/data_raw/beijing/planetscene")
    viz_dir = Path("/workspace/xuannv/data_raw/beijing/viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 选择更多 patch：前几个 + 均匀分布
    all_patches = sorted([d for d in planetscene_dir.iterdir() if d.is_dir() and d.name.startswith("patch_")])

    # 选择 12 个 patch：前 3 个 + 每隔 30 个取一个
    selected = all_patches[:3]
    for i in range(30, len(all_patches), 30):
        selected.append(all_patches[i])

    # 去重并限制
    selected = list(dict.fromkeys(selected))[:12]

    print(f"生成 {len(selected)} 个 patch 的可视化图...")
    for patch_dir in selected:
        out_png = viz_dir / f"{patch_dir.name}_all_dates.png"
        visualize_patch_all_dates(patch_dir, out_png)

    print("全部完成")


if __name__ == "__main__":
    main()
