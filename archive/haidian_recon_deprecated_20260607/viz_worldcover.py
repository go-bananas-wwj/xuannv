"""WorldCover 标签可视化：左图原始影像，右图 WorldCover 标签."""
from __future__ import annotations

import argparse
import os
import random
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

# ESA WorldCover 类别 → 颜色映射
# 参考: https://worldcover2020.esa.int/
WC_COLORS = {
    10:  (0.00, 0.40, 0.00),  # Tree cover — 深绿
    20:  (0.55, 0.80, 0.20),  # Shrubland — 浅绿
    30:  (0.80, 0.90, 0.30),  # Grassland — 黄绿
    40:  (0.95, 0.95, 0.30),  # Cropland — 黄色
    50:  (0.90, 0.20, 0.20),  # Built-up — 红
    60:  (0.80, 0.60, 0.30),  # Bare / sparse vegetation — 棕
    70:  (1.00, 1.00, 1.00),  # Snow and ice — 白
    80:  (0.10, 0.30, 0.80),  # Permanent water bodies — 蓝
    90:  (0.30, 0.70, 0.80),  # Herbaceous wetland — 青
    95:  (0.10, 0.50, 0.50),  # Mangroves — 深青
    100: (0.60, 0.40, 0.70),  # Moss and lichen — 紫
}

WC_NAMES = {
    10:  "Tree cover",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    50:  "Built-up",
    60:  "Bare/sparse",
    70:  "Snow/ice",
    80:  "Water",
    90:  "Wetland",
    95:  "Mangroves",
    100: "Moss/lichen",
}


def worldcover_to_rgb(label_arr: np.ndarray) -> np.ndarray:
    """将 WorldCover 标签数组转为 RGB 图像."""
    h, w = label_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    unique = np.unique(label_arr)
    for val in unique:
        if val in WC_COLORS:
            mask = label_arr == val
            rgb[mask] = WC_COLORS[val]
    return rgb


def read_s2_rgb(patch_dir: str) -> np.ndarray | None:
    """读取 S2 的 RGB 波段作为参考影像."""
    s2_dir = os.path.join(patch_dir, "s2")
    if not os.path.isdir(s2_dir):
        return None
    tiffs = sorted([f for f in os.listdir(s2_dir) if f.endswith(".tif")])
    if not tiffs:
        return None
    # 取第一个时间步
    tiff_path = os.path.join(s2_dir, tiffs[0])
    try:
        with rasterio.open(tiff_path) as src:
            # 假设通道顺序与 Sentinel-2 一致: B2=Blue, B3=Green, B4=Red
            # 如果通道数 >= 3，取前三个作为 RGB
            if src.count >= 3:
                r = src.read(3)
                g = src.read(2)
                b = src.read(1)
            elif src.count >= 1:
                r = g = b = src.read(1)
            else:
                return None
            rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
            # 简单的 percentile 拉伸
            p2, p98 = np.percentile(rgb, [2, 98])
            if p98 > p2:
                rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
            return rgb
    except RasterioIOError:
        return None


def read_worldcover(patch_dir: str) -> np.ndarray | None:
    """读取 WorldCover 标签."""
    wc_dir = os.path.join(patch_dir, "worldcover")
    if not os.path.isdir(wc_dir):
        return None
    tiffs = [f for f in os.listdir(wc_dir) if f.endswith(".tif")]
    if not tiffs:
        return None
    tiff_path = os.path.join(wc_dir, tiffs[0])
    try:
        with rasterio.open(tiff_path) as src:
            arr = src.read(1).astype(np.float32)
            return arr
    except RasterioIOError:
        return None


def visualize_worldcover(data_root: str, output_dir: str, num_samples: int = 5) -> None:
    """随机选 num_samples 个 patch，左图 S2-RGB，右图 WorldCover."""
    os.makedirs(output_dir, exist_ok=True)
    patches = sorted([d for d in os.listdir(data_root) if d.startswith("patch_")])
    if not patches:
        print(f"No patches found in {data_root}")
        return

    random.seed(42)
    selected = random.sample(patches, min(num_samples, len(patches)))

    for patch_name in selected:
        patch_dir = os.path.join(data_root, patch_name)
        s2_rgb = read_s2_rgb(patch_dir)
        wc_label = read_worldcover(patch_dir)

        if s2_rgb is None or wc_label is None:
            print(f"Skip {patch_name}: missing S2 or WorldCover")
            continue

        wc_rgb = worldcover_to_rgb(wc_label)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(s2_rgb)
        axes[0].set_title(f"{patch_name} — S2 RGB (参考)")
        axes[0].axis("off")

        axes[1].imshow(wc_rgb)
        axes[1].set_title(f"{patch_name} — WorldCover 标签")
        axes[1].axis("off")

        # 添加图例
        unique_labels = sorted(np.unique(wc_label))
        legend_texts = []
        for val in unique_labels:
            name = WC_NAMES.get(int(val), f"Class {int(val)}")
            legend_texts.append(f"{int(val)}: {name}")
        fig.text(0.5, 0.02, "  |  ".join(legend_texts), ha="center", fontsize=9)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        out_path = os.path.join(output_dir, f"wc_viz_{patch_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data_raw/haidian/scenes")
    parser.add_argument("--output-dir", type=str, default="outputs/hre_eval/worldcover_viz")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    visualize_worldcover(args.data_root, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
