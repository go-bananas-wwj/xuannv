#!/usr/bin/env python3
"""
验证标注与卫星影像对齐情况.
对每个标注源, 将标注几何体绘制在对应 patch 的 Before/After S2 影像上.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping

sys.path.insert(0, "/workspace/xuannv")

from demo_v2.engines.patch_image_loader import _find_best_tif, load_patch_source_rgb
from demo_v2.utils.constants import RAW_DIR, TIME_WINDOWS
from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    load_harbin_annotations,
    rasterize_patch_changes,
    CATEGORY_TO_IDX,
)

CATEGORY_NAMES = {v: k for k, v in CATEGORY_TO_IDX.items()}

OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2/annotation_alignment_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}

SOURCE_COLORS = {
    "optical_excel": "red",
    "optical_shp": "lime",
    "sar": "blue",
}


def load_s2_rgb(patch_id: str, month: str):
    window = TIME_WINDOWS.get(month)
    if window is None:
        return None, None
    img = load_patch_source_rgb(patch_id, "s2", window)
    if img is None:
        return None, None
    
    tif_path = _find_best_tif(RAW_DIR / "s2" / patch_id, window[0], window[1])
    return img, tif_path


def draw_annotation_on_image(ax, img, geom, color, transform):
    """在 ax 上绘制标注几何体."""
    from shapely.geometry import Point, Polygon
    
    if geom is None:
        return
    
    if isinstance(geom, Point):
        # 使用 rasterio transform 将 UTM -> pixel (rowcol 返回 row, col)
        try:
            row, col = rasterio.transform.rowcol(transform, geom.x, geom.y)
            ax.scatter(col, row, c=color, s=80, marker='x', linewidths=2)
        except Exception:
            pass
    elif hasattr(geom, 'exterior'):
        # Polygon/Multipolygon
        try:
            coords = []
            if hasattr(geom, 'geoms'):
                for g in geom.geoms:
                    if hasattr(g, 'exterior'):
                        ring = [(rasterio.transform.rowcol(transform, x, y)[::-1]) 
                                for x, y in g.exterior.coords]
                        coords.append(ring)
            else:
                ring = [(rasterio.transform.rowcol(transform, x, y)[::-1]) 
                        for x, y in geom.exterior.coords]
                coords.append(ring)
            
            for ring in coords:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.plot(xs, ys, c=color, linewidth=2)
        except Exception:
            pass


def verify_patch_source(ax_before, ax_after, patch_id: str, recs: list, period: str):
    bm, am = PERIOD_TO_MONTHS[period]
    img_b, tif_b = load_s2_rgb(patch_id, bm)
    img_a, tif_a = load_s2_rgb(patch_id, am)
    
    if img_b is None or img_a is None or tif_b is None or tif_a is None:
        ax_before.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax_before.transAxes)
        ax_after.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax_after.transAxes)
        return
    
    with rasterio.open(str(tif_b)) as ds:
        transform_b = ds.transform
    with rasterio.open(str(tif_a)) as ds:
        transform_a = ds.transform
    
    ax_before.imshow(img_b, interpolation="bilinear")
    ax_after.imshow(img_a, interpolation="bilinear")
    
    for rec in recs:
        geom = rec["geometry"]
        color = SOURCE_COLORS.get(rec["source"], "yellow")
        draw_annotation_on_image(ax_before, img_b, geom, color, transform_b)
        draw_annotation_on_image(ax_after, img_a, geom, color, transform_a)
    
    ax_before.set_xticks([])
    ax_before.set_yticks([])
    ax_after.set_xticks([])
    ax_after.set_yticks([])


def main():
    annotations = load_harbin_annotations()
    
    # 按 source 分组统计
    source_groups = {}
    for pid, recs in annotations.items():
        for rec in recs:
            src = rec["source"]
            period = rec["period"]
            key = f"{src}_{period}"
            if key not in source_groups:
                source_groups[key] = []
            source_groups[key].append((pid, rec))
    
    print(f"Total annotation groups: {len(source_groups)}")
    for key in sorted(source_groups.keys()):
        print(f"  {key}: {len(source_groups[key])} records")
    
    # 对每个 source+period 生成验证图 (最多前 15 个 patch)
    for key, items in sorted(source_groups.items()):
        src, period = key.rsplit("_", 1)
        if period not in PERIOD_TO_MONTHS:
            continue
        
        # 按 patch 分组
        patch_items = {}
        for pid, rec in items:
            if pid not in patch_items:
                patch_items[pid] = []
            patch_items[pid].append(rec)
        
        patches = sorted(patch_items.keys())[:15]
        n_patches = len(patches)
        if n_patches == 0:
            continue
        
        fig, axes = plt.subplots(n_patches, 2, figsize=(10, n_patches * 4.5))
        if n_patches == 1:
            axes = axes.reshape(1, -1)
        
        for i, pid in enumerate(patches):
            verify_patch_source(axes[i, 0], axes[i, 1], pid, patch_items[pid], period)
            cat_label = ",".join(sorted(set(CATEGORY_NAMES.get(rec.get("category"), rec["category"]) for rec in patch_items[pid])))
            axes[i, 0].set_ylabel(f"{pid}\n{cat_label}", fontsize=9, rotation=0, ha="right", va="center")
        
        axes[0, 0].set_title("Before S2", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("After S2", fontsize=12, fontweight="bold")
        
        # 添加 legend
        legend_elements = [
            plt.Line2D([0], [0], color="red", lw=2, label="optical_excel"),
            plt.Line2D([0], [0], color="lime", lw=2, label="optical_shp"),
            plt.Line2D([0], [0], color="blue", lw=2, label="sar"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
        
        fig.suptitle(f"Annotation Alignment Check — {src} | {period}", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        out_path = OUTPUT_DIR / f"verify_{src}_{period.replace('~', '_')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path} ({n_patches} patches)")


if __name__ == "__main__":
    main()
