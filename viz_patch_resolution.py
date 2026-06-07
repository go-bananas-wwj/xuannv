"""Visualize original resolution vs resized 128x128 for all sources in a patch."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import cv2


def read_tif_raw(path: Path) -> np.ndarray | None:
    """Read raw tif without resize."""
    try:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)
            return data
    except Exception:
        return None


def resize_to(data: np.ndarray, target_size: int = 128) -> np.ndarray:
    """Resize image to target_size x target_size."""
    C, H, W = data.shape
    if H == target_size and W == target_size:
        return data
    result = np.zeros((C, target_size, target_size), dtype=np.float32)
    for c in range(C):
        result[c] = cv2.resize(data[c], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return result


def normalize_for_display(data: np.ndarray, source_name: str) -> np.ndarray:
    """Normalize to 0-1 for display."""
    if data.shape[0] >= 3:
        rgb = data[:3].copy()
    else:
        rgb = np.repeat(data[0:1], 3, axis=0)

    if source_name in ("s2", "landsat"):
        if rgb.max() < 2.0:
            rgb = rgb * 10000.0
        rgb = np.log(np.clip(rgb, 0, None) + 1) / 10.0
    elif source_name == "planet":
        if rgb.max() > 100:
            rgb = rgb / 10000.0
        rgb = np.log(np.clip(rgb, 0, None) + 1) / 10.0
    elif source_name == "tianyi_sar":
        if rgb.max() > 100:
            rgb = np.log10(np.clip(rgb / 10000.0, 1e-10, None)) * 10.0
        rgb = np.clip(rgb, -30.0, 10.0)
        rgb = (rgb + 30) / 40.0

    for i in range(3):
        p2, p98 = np.percentile(rgb[i], [2, 98])
        if p98 > p2:
            rgb[i] = np.clip((rgb[i] - p2) / (p98 - p2), 0, 1)
        else:
            rgb[i] = np.clip(rgb[i], 0, 1)

    rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def main():
    patch_id = "patch_000000"

    sources = {
        "tianyi_sar": {
            "path": Path(f"data_raw/haidian/scenes/{patch_id}/tianyi_sar/20250102.tif"),
            "res_m": 3,
            "channels": 1,
        },
        "s2": {
            "path": Path(f"data_raw/haidian/scenes/{patch_id}/s2/20250210.tif"),
            "res_m": 10,
            "channels": 6,
        },
        "landsat": {
            "path": Path(f"data_raw/haidian/scenes/{patch_id}/landsat/20250214.tif"),
            "res_m": 30,
            "channels": 6,
        },
        "planet": {
            "path": Path(f"data_raw/beijing/planetscene/{patch_id}/20251209.tif"),
            "res_m": 3,
            "channels": 4,
        },
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Patch {patch_id}: Original vs Resized to 128x128", fontsize=14, fontweight="bold")

    for idx, (name, info) in enumerate(sources.items()):
        raw = read_tif_raw(info["path"])
        if raw is None:
            continue

        C, H, W = raw.shape
        resized = resize_to(raw, 128)

        raw_display = normalize_for_display(raw, name)
        resized_display = normalize_for_display(resized, name)

        ax_raw = axes[0, idx]
        ax_raw.imshow(raw_display)
        ax_raw.set_title(
            f"{name.upper()}\n"
            f"Orig: {H}x{W} px\n"
            f"Res: {info['res_m']}m/px\n"
            f"Coverage: {H*info['res_m']}m x {W*info['res_m']}m",
            fontsize=10,
        )
        ax_raw.axis("off")

        ax_resized = axes[1, idx]
        ax_resized.imshow(resized_display)
        ax_resized.set_title(
            f"{name.upper()} (Resized)\n"
            f"128x128 px\n"
            f"Eff. res: {info['res_m'] * H / 128:.1f}m/px\n"
            f"Ratio: {H/128:.1f}x",
            fontsize=10,
        )
        ax_resized.axis("off")

    plt.tight_layout()
    out_path = Path("outputs/viz_patch_resolution_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
