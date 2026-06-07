"""可视化一个 patch 的原始输入数据（所有源）."""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def read_first_tiff(source_dir: str) -> tuple[np.ndarray, str] | tuple[None, str]:
    """读取目录中第一个 TIFF 文件的所有通道."""
    if not os.path.isdir(source_dir):
        return None, f"Dir not found: {source_dir}"
    tiffs = sorted([f for f in os.listdir(source_dir) if f.endswith(".tif")])
    if not tiffs:
        return None, "No TIFF files"
    path = os.path.join(source_dir, tiffs[0])
    try:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # [C, H, W]
            return data, f"{tiffs[0]} ({src.count}ch, {src.shape})"
    except Exception as e:
        return None, str(e)


def percentile_stretch(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Percentile 拉伸到 [0, 1]."""
    p_low, p_high = np.percentile(arr, [low, high])
    if p_high > p_low:
        arr = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
    return arr


def to_rgb(data: np.ndarray, r_idx: int = 0, g_idx: int = 1, b_idx: int = 2) -> np.ndarray:
    """从多通道数据中提取 RGB."""
    c = data.shape[0]
    r = data[r_idx] if r_idx < c else data[0]
    g = data[g_idx] if g_idx < c else data[0]
    b = data[b_idx] if b_idx < c else data[0]
    rgb = np.stack([r, g, b], axis=-1)
    return percentile_stretch(rgb)


def viz_patch(patch_dir: str, output_path: str) -> None:
    """可视化一个 patch 的所有源输入."""
    sources = {
        "tianyi_sar (SAR)": os.path.join(patch_dir, "s1"),
        "sentinel-2 (S2)": os.path.join(patch_dir, "s2"),
        "landsat": os.path.join(patch_dir, "landsat"),
    }

    # Planet 可能在不同根目录
    planet_dir = os.path.join(patch_dir, "planetscene")
    if not os.path.isdir(planet_dir):
        patch_name = os.path.basename(patch_dir)
        planet_dir = f"/workspace/xuannv/data_raw/beijing/planetscene/{patch_name}"
    if os.path.isdir(planet_dir):
        sources["planet"] = planet_dir

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"Patch Inputs: {os.path.basename(patch_dir)}", fontsize=14)

    ax_idx = 0
    for source_name, source_dir in sources.items():
        data, info = read_first_tiff(source_dir)

        row = ax_idx // 4
        col = ax_idx % 4
        ax = axes[row, col]

        if data is None:
            ax.text(0.5, 0.5, f"{source_name}\n{info}", ha="center", va="center", fontsize=10)
            ax.axis("off")
            ax_idx += 1
            continue

        c, h, w = data.shape

        # 显示策略
        if c >= 3 and "planet" in source_name.lower():
            # Planet: BGRN -> 尝试 RGB (B=0, G=1, R=2)
            img = to_rgb(data, r_idx=2, g_idx=1, b_idx=0)
            ax.imshow(img)
        elif c >= 3 and "sentinel" in source_name.lower():
            # S2: 假设前3通道是某种 RGB 顺序，尝试不同的组合
            # 先尝试 (0,1,2) 作为伪 RGB
            img = to_rgb(data, r_idx=2, g_idx=1, b_idx=0)
            ax.imshow(img)
        elif c >= 3:
            img = to_rgb(data, r_idx=0, g_idx=1, b_idx=2)
            ax.imshow(img)
        else:
            # 单/双通道 -> 显示第一个通道灰度
            img = percentile_stretch(data[0])
            ax.imshow(img, cmap="gray")

        ax.set_title(f"{source_name}\n{info}", fontsize=9)
        ax.axis("off")
        ax_idx += 1

    # 如果不足8个，隐藏多余的子图
    for i in range(ax_idx, 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=str, required=True, help="e.g. data_raw/haidian/scenes/patch_000306")
    parser.add_argument("--output", type=str, default="outputs/hre_eval/patch_inputs_viz.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    viz_patch(args.patch_dir, args.output)


if __name__ == "__main__":
    main()
