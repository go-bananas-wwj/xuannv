#!/usr/bin/env python3
"""
Stitch 64×64 patch probability maps into 1664×1536 regional heatmaps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
PRED_DIR = Path("/workspace/xuannv/outputs/aef_qwen_v2/patch_predictions")
OUTPUT_DIR = Path("/workspace/xuannv/outputs/aef_qwen_v2/visualization/06_regional_change_maps")

PERIODS = [
    "2025-04~2025-06",
    "2025-06~2025-08",
    "2025-08~2025-09",
    "2025-09~2025-10",
]


def load_grid_mapping() -> dict:
    with open(GRID_PATH) as f:
        data = json.load(f)
    mapping = {}
    for feat in data["features"]:
        props = feat["properties"]
        mapping[props["patch_id"]] = (props["ix"], props["iy"])
    return mapping


def stitch_period(period: str, patch_to_ixiy: dict) -> np.ndarray | None:
    prefix = period.replace("~", "_")
    files = list(PRED_DIR.glob(f"patch_*_{prefix}.npy"))
    if not files:
        print(f"No predictions found for {period}")
        return None

    # Grid dimensions
    ixs = [v[0] for v in patch_to_ixiy.values()]
    iys = [v[1] for v in patch_to_ixiy.values()]
    max_ix = max(ixs)
    max_iy = max(iys)
    n_cols = max_ix + 1  # 26
    n_rows = max_iy + 1  # 24

    canvas = np.full((n_rows * 64, n_cols * 64), np.nan, dtype=np.float32)

    for fpath in files:
        pid = fpath.stem.replace(f"_{prefix}", "")
        if pid not in patch_to_ixiy:
            continue
        ix, iy = patch_to_ixiy[pid]
        # North-up: flip iy so iy=max_iy is at row 0
        row = max_iy - iy
        col = ix
        y0, x0 = row * 64, col * 64
        probs = np.load(fpath)
        canvas[y0:y0 + 64, x0:x0 + 64] = probs

    return canvas


def save_heatmap(canvas: np.ndarray, period: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(16, 14.8))
    # Use a nice colormap; mask NaN as dark gray
    masked = np.ma.masked_invalid(canvas)
    im = ax.imshow(masked, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Regional Change Detection Heatmap — {period}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Patch column (ix)", fontsize=11)
    ax.set_ylabel("Patch row (iy, north-up)", fontsize=11)

    # Grid lines every 64 pixels
    for i in range(0, canvas.shape[1] + 1, 64):
        ax.axvline(i - 0.5, color="white", linewidth=0.3, alpha=0.3)
    for j in range(0, canvas.shape[0] + 1, 64):
        ax.axhline(j - 0.5, color="white", linewidth=0.3, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Change Probability", fontsize=11)

    # Set tick labels to patch coordinates
    n_cols = canvas.shape[1] // 64
    n_rows = canvas.shape[0] // 64
    ax.set_xticks([i * 64 + 31.5 for i in range(0, n_cols, 2)])
    ax.set_xticklabels([str(i) for i in range(0, n_cols, 2)], fontsize=8)
    ax.set_yticks([j * 64 + 31.5 for j in range(0, n_rows, 2)])
    ax.set_yticklabels([str(n_rows - 1 - j) for j in range(0, n_rows, 2)], fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"regional_heatmap_{period.replace('~', '_')}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_to_ixiy = load_grid_mapping()
    print(f"Loaded grid mapping for {len(patch_to_ixiy)} patches")

    for period in PERIODS:
        prefix = period.replace("~", "_")
        canvas = stitch_period(period, patch_to_ixiy)
        if canvas is None:
            continue
        valid_count = np.sum(~np.isnan(canvas))
        total_count = canvas.size
        print(f"{period}: {valid_count}/{total_count} pixels valid ({valid_count / total_count * 100:.1f}%)")
        save_heatmap(canvas, period, OUTPUT_DIR)

    print(f"\nAll heatmaps saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
