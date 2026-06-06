#!/usr/bin/env python3
"""变化检测 Mask 可视化 — 验证标注与 patch 对齐是否正确.

核心改进: 将变化 mask 叠加到 S2 RGB 原图上，可直观验证对齐精度.

用法:
    python visualize_cd_mask.py --period june --output-dir evaluation/results/cd_viz
    python visualize_cd_mask.py --period all --output-dir evaluation/results/cd_viz
"""
from __future__ import annotations

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box, Point
from shapely import vectorized

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── 时间窗口 → 季度影像映射 ────────────────────────────────────────────────
PERIOD_TO_QUARTER = {
    "june": "2025Q2",
    "aug": "2025Q3",
    "September": "2025Q3",
    "October": "2025Q4",
}

PERIODS: dict[str, dict] = {
    "june": {
        "before": (1743436800000, 1746028799000),
        "after":  (1748707200000, 1751299199000),
    },
    "aug": {
        "before": (1748707200000, 1751299199000),
        "after":  (1753977600000, 1756655999000),
    },
    "September": {
        "before": (1753977600000, 1756655999000),
        "after":  (1756656000000, 1759247999000),
    },
    "October": {
        "before": (1756656000000, 1759247999000),
        "after":  (1759248000000, 1761926399000),
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="变化检测 Mask 可视化")
    p.add_argument("--period", default="all", choices=["june", "aug", "September", "October", "all"])
    p.add_argument("--annot-dir", default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
    p.add_argument("--grid", default="/workspace/index/harbin/grid/harbin_grid.geojson")
    p.add_argument("--s2-root", default="/workspace/xuannv/data_raw/harbin_scenes/s2")
    p.add_argument("--output-dir", default="evaluation/results/cd_viz")
    p.add_argument("--max-patches", type=int, default=20, help="最多可视化 patch 数量")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_grid(grid_path: str) -> dict:
    with open(grid_path) as f:
        data = json.load(f)
    bounds: dict[str, tuple] = {}
    for feat in data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return bounds


def load_changes(annot_dir: str, period: str) -> list:
    shp_name = f"{period}.shp"
    try:
        gdf = gpd.read_file(f"{annot_dir}/{shp_name}")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        return [row.geometry for _, row in gdf.iterrows() if row.geometry is not None]
    except Exception as e:
        print(f"  [警告] 无法加载 {shp_name}: {e}")
        return []


def load_s2_rgb(s2_root: str, pid: str, quarter: str) -> np.ndarray | None:
    """读取 S2 季度合成影像并生成 RGB [H, W, 3] (0-1)."""
    if not HAS_RASTERIO:
        return None
    tif_path = Path(s2_root) / pid / f"{quarter}.tif"
    if not tif_path.exists():
        return None
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()  # [C, H, W]
        if data.shape[0] < 3:
            return None
        # S2 6波段: [B2, B3, B4, B8, B11, B12] → RGB=[B4, B3, B2] = idx [2,1,0]
        rgb = data[[2, 1, 0], ...].transpose(1, 2, 0).astype(np.float32)
        # 百分位归一化增强可视化
        p2, p98 = np.percentile(rgb, [2, 98])
        rgb = (rgb - p2) / (p98 - p2 + 1e-6)
        rgb = np.clip(rgb, 0, 1)
        return rgb
    except Exception as e:
        print(f"  [警告] 读取 {tif_path} 失败: {e}")
        return None


def build_mask(changes: list, bounds: tuple, H: int = 128, W: int = 128) -> np.ndarray:
    """构建变化 mask，与 auc_eval.py 使用相同逻辑（向量化加速）."""
    mask = np.zeros((H, W), dtype=bool)
    minx, miny, maxx, maxy = bounds
    patch_box = box(minx, miny, maxx, maxy)

    xs = np.linspace(minx + (maxx - minx) / (2 * W), maxx - (maxx - minx) / (2 * W), W)
    ys = np.linspace(maxy - (maxy - miny) / (2 * H), miny + (maxy - miny) / (2 * H), H)
    xv, yv = np.meshgrid(xs, ys)

    for geom in changes:
        if not patch_box.intersects(geom):
            continue
        try:
            buffered = geom.buffer(1.0)
            inside = vectorized.contains(buffered, xv, yv)
            mask |= inside
        except Exception:
            for y in range(H):
                for x in range(W):
                    px = minx + (x + 0.5) / W * (maxx - minx)
                    py = maxy - (y + 0.5) / H * (maxy - miny)
                    if geom.buffer(1.0).contains(Point(px, py)):
                        mask[y, x] = True
    return mask


def visualize_patch(pid: str, bounds: tuple, changes: list, period: str,
                    s2_root: str, output_dir: Path, dpi: int):
    """为单个 patch 生成可视化图 (4列: RGB原图 / RGB+Mask叠加 / 地理视图 / Mask二值)."""
    H, W = 128, 128
    mask = build_mask(changes, bounds, H, W)
    changed_pixels = int(mask.sum())
    total_pixels = H * W
    ratio = changed_pixels / total_pixels * 100

    quarter = PERIOD_TO_QUARTER.get(period, "2025Q2")
    rgb = load_s2_rgb(s2_root, pid, quarter)
    has_rgb = rgb is not None

    ncols = 4 if has_rgb else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 3:
        axes = [axes[0], axes[1], axes[2]]
    ax_idx = 0

    # 1. RGB 原图
    if has_rgb:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.imshow(rgb)
        ax.set_title(f"S2 RGB ({quarter})\n{pid}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # 2. RGB + Mask 叠加
    if has_rgb:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.imshow(rgb)
        # 红色半透明叠加变化区域
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [1.0, 0.0, 0.0, 0.45]  # R, G, B, alpha
        ax.imshow(overlay)
        ax.set_title(f"RGB + Changed Mask\n{changed_pixels}px ({ratio:.1f}%)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # 3. 地理叠加视图 (始终保留)
    ax = axes[ax_idx]
    ax_idx += 1
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(maxy, miny)
    ax.set_aspect("equal")
    ax.set_title(f"Geographic Overlay\nChanged polygons")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    rect = plt.Rectangle((minx, miny), maxx - minx, maxy - miny,
                         fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(rect)
    for geom in changes:
        if box(minx, miny, maxx, maxy).intersects(geom):
            try:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.4, fc="red", ec="darkred")
            except Exception:
                pass

    # 4. Mask 二值图 (始终保留)
    ax = axes[ax_idx]
    ax_idx += 1
    ax.imshow(mask, cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Changed Mask Only\nRed=Changed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for i in range(0, W + 1, 16):
        ax.axvline(i - 0.5, color="gray", linewidth=0.3)
        ax.axhline(i - 0.5, color="gray", linewidth=0.3)

    fig.suptitle(f"Patch {pid} | Period: {period} | Changed: {changed_pixels}px ({ratio:.1f}%)",
                 fontsize=14, fontweight="bold")

    out_path = output_dir / f"{period}_{pid}_mask.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, changed_pixels, ratio


def visualize_summary(all_results: list, output_dir: Path, dpi: int):
    """生成汇总统计图."""
    periods = sorted(set(r["period"] for r in all_results))
    fig, axes = plt.subplots(1, len(periods), figsize=(6 * len(periods), 5))
    if len(periods) == 1:
        axes = [axes]

    for ax, period in zip(axes, periods):
        stats = [r for r in all_results if r["period"] == period]
        ratios = [r["ratio"] for r in stats]
        pids = [r["pid"] for r in stats]

        ax.bar(range(len(ratios)), ratios, color="coral")
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(pids, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Changed Pixel %")
        ax.set_title(f"{period}\nmean={np.mean(ratios):.2f}% max={np.max(ratios):.1f}%")
        ax.axhline(np.mean(ratios), color="blue", linestyle="--", label=f"mean={np.mean(ratios):.2f}%")
        ax.legend()

    fig.suptitle("Change Detection Mask Coverage by Period", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "summary_coverage.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  汇总图已保存: {out_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  变化检测 Mask 可视化")
    print("=" * 60)

    patch_bounds = load_grid(args.grid)
    print(f"  加载 {len(patch_bounds)} 个 patch 边界")

    periods = [args.period] if args.period != "all" else list(PERIODS.keys())
    all_results = []

    for period in periods:
        print(f"\n  处理时期: {period}")
        changes = load_changes(args.annot_dir, period)
        if not changes:
            print(f"    无变化标注，跳过")
            continue
        print(f"    共 {len(changes)} 个变化多边形")

        affected_pids = []
        for pid, bounds in patch_bounds.items():
            pb = box(*bounds)
            for geom in changes:
                if pb.intersects(geom):
                    affected_pids.append(pid)
                    break

        print(f"    涉及 {len(affected_pids)} 个 patch")

        if len(affected_pids) > args.max_patches:
            print(f"    限制可视化前 {args.max_patches} 个 patch")
            affected_pids = affected_pids[:args.max_patches]

        for pid in sorted(affected_pids):
            out_path, ch_px, ratio = visualize_patch(
                pid, patch_bounds[pid], changes, period,
                args.s2_root, output_dir, args.dpi
            )
            all_results.append({
                "period": period, "pid": pid,
                "changed_pixels": ch_px, "ratio": ratio,
                "path": str(out_path),
            })
            print(f"    {pid}: {ch_px}px changed ({ratio:.2f}%) -> {out_path.name}")

    if all_results:
        visualize_summary(all_results, output_dir, args.dpi)

        print("\n" + "=" * 60)
        print("  统计摘要")
        print("=" * 60)
        for period in periods:
            stats = [r for r in all_results if r["period"] == period]
            if stats:
                ratios = [r["ratio"] for r in stats]
                print(f"  {period:12s}: patches={len(stats)}  mean={np.mean(ratios):.2f}%  "
                      f"median={np.median(ratios):.2f}%  max={np.max(ratios):.1f}%")
        print(f"\n  所有可视化文件保存在: {output_dir}")
    else:
        print("\n  [警告] 未生成任何可视化")


if __name__ == "__main__":
    main()
