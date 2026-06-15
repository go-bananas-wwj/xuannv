#!/usr/bin/env python3
"""可视化 CDHead 在施工工地监测上的预测结果.

输入:
    - pred.npz (含 patch_ids, prob_map, label_map)
    - Planet RGB tiffs: data_root/patch_*/planet/20251209.tif, 20260430.tif
输出:
    - 每 patch 一张对比图: 变化前 RGB / 变化后 RGB / 真实 label / 预测概率
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

_LABEL_CMAP = ListedColormap(["#e8e8e8", "#d62728"])
_PRED_CMAP = LinearSegmentedColormap.from_list("pred", ["#ADD8E6", "#8B0000"])


def _rgb_tiff_to_array(tiff_path: Path) -> np.ndarray | None:
    if not tiff_path.exists():
        return None
    with rasterio.open(tiff_path) as src:
        rgb = src.read([1, 2, 3])  # BGR -> use first 3 as RGB
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    for c in range(3):
        band = rgb[..., c]
        low, high = np.percentile(band, [2, 98])
        band = np.clip((band - low) / (high - low + 1e-6), 0, 1)
        rgb[..., c] = band
    return rgb


def visualize_patch(patch_id: str, prob: np.ndarray, label: np.ndarray,
                    before_rgb: np.ndarray, after_rgb: np.ndarray,
                    output_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ["变化前 RGB (2025-12)", "变化后 RGB (2026-04)", "真实 label", "CDHead 预测概率"]
    images = [before_rgb, after_rgb, label, prob]
    cmaps = [None, None, _LABEL_CMAP, _PRED_CMAP]
    vmins = [0, 0, 0, 0]
    vmaxs = [1, 1, 1, 1]

    for ax, img, title, cmap, vmin, vmax in zip(axes, images, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if title in ("真实 label", "CDHead 预测概率"):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pos_ratio = label.mean() * 100
    axes[2].text(
        0.02, 0.98, f"正样本: {int(label.sum())} ({pos_ratio:.2f}%)",
        transform=axes[2].transAxes, fontsize=9, verticalalignment="top",
        color="black", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    axes[3].text(
        0.02, 0.98, f"均值: {prob.mean():.3f}\nmax: {prob.max():.3f}",
        transform=axes[3].transAxes, fontsize=9, verticalalignment="top",
        color="black", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    fig.suptitle(f"施工工地监测 - {patch_id}  (F1={pos_ratio:.1f}% 正样本)", fontsize=13)
    out_path = output_dir / f"{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-npz", default="outputs/eval_phase2best_all/cdhead/shigongjiandu/pred.npz")
    parser.add_argument("--data-root", default="/workspace/xuannv/data_raw/haidian/scenes")
    parser.add_argument("--output-dir", default="visualizations")
    parser.add_argument("--max-patches", type=int, default=6, help="最多可视化多少个含正样本 patch")
    args = parser.parse_args()

    data = np.load(args.pred_npz)
    patch_ids = [str(p) for p in data["patch_ids"]]
    prob_maps = data["prob_map"]
    label_maps = data["label_map"]

    # 按正样本面积排序，优先展示变化明显的 patch
    pos_areas = [lab.sum() for lab in label_maps]
    sorted_indices = sorted(range(len(patch_ids)), key=lambda i: pos_areas[i], reverse=True)
    selected = [i for i in sorted_indices if pos_areas[i] > 0][:args.max_patches]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    for idx in selected:
        pid = patch_ids[idx]
        before_path = data_root / pid / "planet" / "20251209.tif"
        after_path = data_root / pid / "planet" / "20260430.tif"
        before_rgb = _rgb_tiff_to_array(before_path)
        after_rgb = _rgb_tiff_to_array(after_path)
        if before_rgb is None or after_rgb is None:
            print(f"[skip] {pid} 缺少 Planet RGB")
            continue
        visualize_patch(pid, prob_maps[idx], label_maps[idx], before_rgb, after_rgb, output_dir)

    print(f"\n可视化完成: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
