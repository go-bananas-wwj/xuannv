"""Visualize 256x256 patches + multi-scale token grid overlay."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import cv2
import torch

from haidian_recon.models.patch_embed import MultiScalePatchEmbed


def read_tif_raw(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)
            return data
    except Exception:
        return None


def normalize_for_display(data: np.ndarray, source_name: str) -> np.ndarray:
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

    return np.transpose(rgb, (1, 2, 0))


def draw_patch_grid(ax, image_size: int, patch_size: int, color: str, linewidth: float = 0.5):
    """Draw patch grid lines on the image."""
    n = image_size // patch_size
    for i in range(n + 1):
        x = i * patch_size
        ax.axvline(x=x, color=color, linewidth=linewidth, alpha=0.7)
        ax.axhline(y=x, color=color, linewidth=linewidth, alpha=0.7)
    return n * n


def main():
    patch_id = "patch_000000"
    image_size = 256

    sources = {
        "tianyi_sar": {
            "path": Path(f"data_raw/haidian/scenes_256/{patch_id}/tianyi_sar/20250102.tif"),
            "res_m": 3,
            "channels": 1,
            "patch_size": 4,
            "grid_color": "red",
        },
        "s2": {
            "path": Path(f"data_raw/haidian/scenes_256/{patch_id}/s2/20250210.tif"),
            "res_m": 10,
            "channels": 6,
            "patch_size": 8,
            "grid_color": "yellow",
        },
        "landsat": {
            "path": Path(f"data_raw/haidian/scenes_256/{patch_id}/landsat/20250214.tif"),
            "res_m": 30,
            "channels": 6,
            "patch_size": 16,
            "grid_color": "cyan",
        },
        "planet": {
            "path": Path(f"data_raw/beijing/planetscene_256/{patch_id}/20251209.tif"),
            "res_m": 3,
            "channels": 4,
            "patch_size": 4,
            "grid_color": "lime",
        },
    }

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.2)

    fig.suptitle(
        f"Patch {patch_id} @ 256x256 | Multi-Scale Patch Embed | Resolution-Aware Token Grid",
        fontsize=14, fontweight="bold",
    )

    # Row 1: Original 256x256 images
    for idx, (name, info) in enumerate(sources.items()):
        raw = read_tif_raw(info["path"])
        if raw is None:
            continue

        C, H, W = raw.shape
        display = normalize_for_display(raw, name)

        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(display)
        n_tokens = draw_patch_grid(ax, image_size, info["patch_size"], info["grid_color"])
        ax.set_title(
            f"{name.upper()}\n"
            f"Size: {H}x{W} | Res: {info['res_m']}m/px\n"
            f"Patch: {info['patch_size']}x{info['patch_size']} | Tokens: {n_tokens}",
            fontsize=10,
        )
        ax.axis("off")

    # Row 2: Token grid visualization (after pooling/upsampling -> 32x32 = 1024)
    source_channels = {name: info["channels"] for name, info in sources.items()}
    patch_embed = MultiScalePatchEmbed(
        source_channels=source_channels,
        embed_dim=512,
        image_size=image_size,
    )

    for idx, (name, info) in enumerate(sources.items()):
        raw = read_tif_raw(info["path"])
        if raw is None:
            continue

        # Prepare tensor
        x = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        with torch.no_grad():
            tokens = patch_embed(x, name)  # [1, 1, 1024, 512]
        n_tokens = tokens.shape[2]
        grid = int(np.sqrt(n_tokens))  # 32

        display = normalize_for_display(raw, name)

        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(display)
        # Draw unified 32x32 grid (after pool/upsample)
        unified_ps = image_size // grid  # 8
        draw_patch_grid(ax, image_size, unified_ps, "white", linewidth=0.8)
        ax.set_title(
            f"{name.upper()} (Unified Token Grid)\n"
            f"After pool/upsample: {grid}x{grid} = {n_tokens} tokens\n"
            f"Each token = {unified_ps}x{unified_ps} = {unified_ps**2} pixels",
            fontsize=10,
        )
        ax.axis("off")

    # Row 3: Comparison table + summary
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis("off")

    summary_text = (
        "Multi-Scale Patch Embed Summary:\n\n"
        "Source         | Native Res | Patch Size | Raw Grid | Operation      | Unified Grid | Pixels/Token | Ground/Token\n"
        "---------------|------------|------------|----------|----------------|--------------|--------------|-------------\n"
        "Planet (3m)    | 3m/px      | 4x4        | 64x64    | AvgPool 2x2    | 32x32        | 8x8 = 64     | 64x3m = 192m\n"
        "Tianyi SAR(3m) | 3m/px      | 4x4        | 64x64    | AvgPool 2x2    | 32x32        | 8x8 = 64     | 64x3m = 192m\n"
        "S2 (10m)       | 10m/px     | 8x8        | 32x32    | None           | 32x32        | 8x8 = 64     | 64x10m = 640m\n"
        "Landsat (30m)  | 30m/px     | 16x16      | 16x16    | Upsample 2x2   | 32x32        | 8x8 = 64     | 64x30m = 1920m\n"
        "\n"
        "Key Insight: All sources unified to 1024 tokens, but each token covers different ground area.\n"
        "High-res sources (Planet/SAR) preserve more spatial detail via 4x4 patch + pooling.\n"
        "Low-res source (Landsat) upsampled to match grid, but token information is coarser."
    )
    ax_summary.text(
        0.05, 0.5, summary_text,
        transform=ax_summary.transAxes,
        fontsize=10, family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    out_path = Path("outputs/viz_patch_256_tokens.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
