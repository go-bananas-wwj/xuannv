"""Tab: AlphaEarth Official Embedding — Global Overview + Harbin Annotations."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from shapely.geometry import Polygon, MultiPolygon

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from demo_v2.utils.visualization import fig_to_pil
from demo_v2.utils.harbin_annotations_v2 import load_harbin_annotations

# ── Paths ──
ALPHA_2023_PATH = Path("/workspace/outputs/alphaearth_harbin/alphaearth_harbin_2023.tif")
ALPHA_2024_PATH = Path("/workspace/outputs/alphaearth_harbin/alphaearth_harbin_2024.tif")
CACHE_PATH = Path("/workspace/outputs/alphaearth_harbin/global_overview.npz")

# ── Annotation colors ──
ANNOT_COLORS = {
    "june.shp": "#FFD700",       # Gold
    "aug.shp": "#FF8C00",        # DarkOrange
    "September.shp": "#FF4500",  # OrangeRed
    "October.shp": "#8A2BE2",    # BlueViolet
    "SAR建筑工地.shp": "#32CD32", # LimeGreen
    "SAR房屋拆除.shp": "#00CED1", # DarkTurquoise
    "SAR疑似违建.shp": "#DC143C", # Crimson
    "SAR非农非粮.shp": "#1E90FF", # DodgerBlue
    "unknown": "#FF69B4",        # HotPink fallback
}

_period_color_map = {
    "2025-04~2025-06": ANNOT_COLORS["june.shp"],
    "2025-06~2025-08": ANNOT_COLORS["aug.shp"],
    "2025-08~2025-09": ANNOT_COLORS["September.shp"],
    "2025-09~2025-10": ANNOT_COLORS["October.shp"],
    "2025-all": "#888888",
}

_ANNOTATIONS_CACHE = load_harbin_annotations()


def _build_annotation_legend() -> list[mpatches.Patch]:
    items = [
        ("Optical 4-6月", ANNOT_COLORS["june.shp"]),
        ("Optical 6-8月", ANNOT_COLORS["aug.shp"]),
        ("Optical 8-9月", ANNOT_COLORS["September.shp"]),
        ("Optical 9-10月", ANNOT_COLORS["October.shp"]),
        ("SAR 建筑工地", ANNOT_COLORS["SAR建筑工地.shp"]),
        ("SAR 房屋拆除", ANNOT_COLORS["SAR房屋拆除.shp"]),
        ("SAR 疑似违建", ANNOT_COLORS["SAR疑似违建.shp"]),
        ("SAR 非农非粮", ANNOT_COLORS["SAR非农非粮.shp"]),
    ]
    return [mpatches.Patch(facecolor="none", edgecolor=c, linewidth=2, label=l) for l, c in items]


def _load_global_arrays():
    if not CACHE_PATH.exists():
        return None, None, None
    data = np.load(CACHE_PATH)
    return data["rgb_2023"], data["rgb_2024"], data["change_score"]


def _draw_global_annotations(ax, ds_transform):
    """Draw all annotation polygons on matplotlib axes using pixel coords."""
    for pid, recs in _ANNOTATIONS_CACHE.items():
        for r in recs:
            geom = r["geometry"]
            if geom is None:
                continue

            source = r.get("source", "")
            remark = r.get("remark", "")
            period = r.get("period", "")
            if source == "sar":
                color = ANNOT_COLORS.get(remark + ".shp", ANNOT_COLORS["unknown"])
            elif source == "optical_shp":
                color = _period_color_map.get(period, ANNOT_COLORS["unknown"])
            else:
                color = _period_color_map.get(period, ANNOT_COLORS["unknown"])

            geoms = []
            if isinstance(geom, MultiPolygon):
                geoms = list(geom.geoms)
            elif isinstance(geom, Polygon):
                geoms = [geom]
            else:
                continue

            for g in geoms:
                coords = []
                for x, y in g.exterior.coords:
                    row, col = rasterio.transform.rowcol(ds_transform, x, y)
                    coords.append((col, row))
                if len(coords) >= 3:
                    poly = mpatches.Polygon(coords, closed=True, fill=False,
                                            edgecolor=color, linewidth=1.8,
                                            linestyle="-", alpha=0.9)
                    ax.add_patch(poly)


def _render_global_overview(view_mode: str):
    rgb_2023, rgb_2024, score = _load_global_arrays()
    if rgb_2023 is None:
        return None, "❌ 全局概览数据尚未预计算，请联系管理员运行 `scripts/precompute_alphaearth_global.py`。"

    H, W = score.shape
    vmax = max(float(np.percentile(score, 95)), 0.001)

    legend_handles = _build_annotation_legend()
    ds_transform = rasterio.Affine.identity()  # Not used directly, but we need the real transform
    # Get real transform from the TIF
    with rasterio.open(ALPHA_2023_PATH) as ds:
        ds_transform = ds.transform

    if view_mode == "2023 Embedding RGB":
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(rgb_2023)
        ax.set_title("AlphaEarth 2023 全域 Embedding RGB", fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        img = fig_to_pil(fig)

    elif view_mode == "2024 Embedding RGB":
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(rgb_2024)
        ax.set_title("AlphaEarth 2024 全域 Embedding RGB", fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        img = fig_to_pil(fig)

    elif view_mode == "Change Score (2023→2024)":
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(score, cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title("AlphaEarth 全域变化强度 (Cosine Distance)", fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change Intensity")
        fig.tight_layout()
        img = fig_to_pil(fig)

    elif view_mode == "Change Score + Annotations":
        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(score, cmap="hot", vmin=0.0, vmax=vmax)
        _draw_global_annotations(ax, ds_transform)
        ax.set_title("AlphaEarth 全域变化强度 + 哈尔滨新区标注", fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change Intensity")
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                  framealpha=0.95, title="Annotations")
        fig.tight_layout()
        img = fig_to_pil(fig)

    elif view_mode == "2024 RGB + Annotations":
        fig, ax = plt.subplots(figsize=(14, 11))
        ax.imshow(rgb_2024)
        _draw_global_annotations(ax, ds_transform)
        ax.set_title("AlphaEarth 2024 全域 RGB + 哈尔滨新区标注", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                  framealpha=0.95, title="Annotations")
        fig.tight_layout()
        img = fig_to_pil(fig)

    else:
        return None, "❌ 未知视图模式"

    # Stats
    n_annotated_patches = len(_ANNOTATIONS_CACHE)
    total_annots = sum(len(v) for v in _ANNOTATIONS_CACHE.values())
    stats = (
        f"### AlphaEarth 官方 Embedding 全域概览\n\n"
        f"| 指标 | 值 |\n|------|-----|\n"
        f"| 图像尺寸 | {W} × {H} px |\n"
        f"| 空间分辨率 | 10 m |\n"
        f"| 覆盖面积 | ~ {W*H*0.0001:.1f} km² |\n"
        f"| 变化分数均值 | {score.mean():.4f} |\n"
        f"| 变化分数最大值 | {score.max():.4f} |\n"
        f"| 变化分数 95% 分位数 | {np.percentile(score, 95):.4f} |\n"
        f"| 有标注 Patch 数 | {n_annotated_patches} |\n"
        f"| 标注总记录数 | {total_annots} |\n\n"
        f"**视图模式**：{view_mode}"
    )
    return img, stats


def build_alphaearth_tab():
    """Build AlphaEarth official embedding global overview tab."""
    gr.Markdown(
        "## 🌐 AlphaEarth Official Embedding — 全域概览\n"
        "展示 Google 官方 AlphaEarth 年度 Embedding（2023 vs 2024）的全域可视化，"
        "并叠加哈尔滨新区变化检测标注（月度光学 + SAR 专题）。"
    )

    view_choices = [
        "2024 RGB + Annotations",
        "Change Score + Annotations",
        "Change Score (2023→2024)",
        "2023 Embedding RGB",
        "2024 Embedding RGB",
    ]

    with gr.Row():
        with gr.Column(scale=1):
            view_select = gr.Dropdown(
                choices=view_choices,
                value="2024 RGB + Annotations",
                label="选择视图",
            )
            btn_gen = gr.Button("🚀 生成全域一览图", variant="primary")

        with gr.Column(scale=4):
            img_global = gr.Image(label="AlphaEarth 全域概览", height=750)
            stats_md = gr.Markdown()

    btn_gen.click(
        fn=_render_global_overview,
        inputs=[view_select],
        outputs=[img_global, stats_md],
    )
