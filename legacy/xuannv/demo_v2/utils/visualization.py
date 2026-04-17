"""可视化工具函数库."""
from __future__ import annotations

import io
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

matplotlib.use("Agg")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# AEF 风格变化检测色带：深蓝 → 红 → 黄
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "change",
    ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ff6b6b", "#ffd93d"],
    N=256,
)


def fig_to_pil(fig: plt.Figure, dpi: int = 150) -> Image.Image:
    """matplotlib Figure → PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, transparent=False)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy array [H, W, 3] → PIL Image."""
    return Image.fromarray(arr)


def colorize_worldcover(wc_arr: np.ndarray, colors: dict) -> np.ndarray:
    """WorldCover 标签 [H, W] → RGB [H, W, 3] uint8."""
    h, w = wc_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for v, color in colors.items():
        rgb[wc_arr == v] = color
    return rgb


def apply_hot_colormap(grayscale: np.ndarray, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "hot") -> np.ndarray:
    """单通道灰度图 [H, W] → colormap RGB [H, W, 3] uint8."""
    norm = np.clip((grayscale - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    rgb = (cmap_obj(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def change_heatmap_fig(
    change_map: np.ndarray,
    title: str = "Change Heatmap",
    cmap=CHANGE_CMAP,
    vmin: float = 0.0,
    vmax: float | None = None,
    figsize: tuple = (5.5, 4.8),
) -> Image.Image:
    """生成带 colorbar 的变化热力图（AEF 风格）."""
    if vmax is None:
        vmax = max(float(np.percentile(change_map, 95)), 0.01)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(change_map, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Change Intensity", fontsize=10)
    fig.tight_layout()
    return fig_to_pil(fig)


def overlay_rgb_heatmap(
    rgb: np.ndarray,
    change_map: np.ndarray,
    alpha: float = 0.5,
    cmap=CHANGE_CMAP,
    vmin: float = 0.0,
    vmax: float | None = None,
    title: str = "Overlay",
    figsize: tuple = (5.5, 4.8),
) -> Image.Image:
    """将热力图叠加到 RGB 上，返回带标题的 PIL Image."""
    if vmax is None:
        vmax = max(float(np.percentile(change_map, 95)), 0.01)

    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        rgb_display = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    else:
        rgb_display = np.clip(rgb, 0, 255).astype(np.uint8)

    norm = np.clip((change_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat_rgba = cmap(norm)  # [H, W, 4]

    alpha_map = heat_rgba[:, :, 3] * alpha
    overlay = rgb_display.astype(np.float32) * (1 - alpha_map[:, :, np.newaxis])
    overlay += heat_rgba[:, :, :3] * 255 * alpha_map[:, :, np.newaxis]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig_to_pil(fig)


def binary_change_map(change_scores: np.ndarray, threshold: float) -> np.ndarray:
    """变化分数 [H, W] → 二值化 RGB [H, W, 3] uint8（红=变化，黑=无变化）."""
    binary = (change_scores >= threshold).astype(np.uint8)
    rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb[binary == 1] = [255, 0, 0]
    return rgb


def ndvi_delta_map(ndvi_before: np.ndarray, ndvi_after: np.ndarray) -> np.ndarray:
    """计算 NDVI 绝对差异图 [H, W]."""
    return np.abs(ndvi_after - ndvi_before)
