#!/usr/bin/env python3
"""变化检测 Mask 可视化 — 验证标注与 patch 对齐是否正确.

用法:
    python visualize_cd_mask.py --period june --output-dir out/cd_viz
    python visualize_cd_mask.py --period all --output-dir out/cd_viz
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

# ── 时间窗口 ────────────────────────────────────────────────────────────────
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
    p.add_argument("--output-dir", default="out/cd_viz")
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


def build_mask(changes: list, bounds: tuple, H: int = 128, W: int = 128) -> np.ndarray:
    """构建变化 mask，与 auc_eval.py 使用相同逻辑（向量化加速）."""
    mask = np.zeros((H, W), dtype=bool)
    minx, miny, maxx, maxy = bounds
    patch_box = box(minx, miny, maxx, maxy)

    # 向量化生成像素中心坐标网格
    xs = np.linspace(minx + (maxx - minx) / (2 * W), maxx - (maxx - minx) / (2 * W), W)
    ys = np.linspace(maxy - (maxy - miny) / (2 * H), miny + (maxy - miny) / (2 * H), H)
    xv, yv = np.meshgrid(xs, ys)

    for geom in changes:
        if not patch_box.intersects(geom):
            continue
        # 使用 buffer(1.0) + contains 检查每个像素中心
        try:
            buffered = geom.buffer(1.0)
            inside = vectorized.contains(buffered, xv, yv)
            mask |= inside
        except Exception:
            # 回退到逐点检查
            for y in range(H):
                for x in range(W):
                    px = minx + (x + 0.5) / W * (maxx - minx)
                    py = maxy - (y + 0.5) / H * (maxy - miny)
                    if geom.buffer(1.0).contains(Point(px, py)):
                        mask[y, x] = True
    return mask


def visualize_patch(pid: str, bounds: tuple, changes: list, period: str, output_dir: Path, dpi: int):
    """为单个 patch 生成可视化图."""
    H, W = 128, 128
    mask = build_mask(changes, bounds, H, W)
    changed_pixels = int(mask.sum())
    total_pixels = H * W
    ratio = changed_pixels / total_pixels * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Mask 二值图
    ax = axes[0]
    ax.imshow(mask, cmap="Reds", interpolation="nearest")
    ax.set_title(f"Changed Mask\n{changed_pixels}/{total_pixels} pixels ({ratio:.1f}%)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 2. 叠加到 patch 边界框的地理视图
    ax = axes[1]
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(maxy, miny)  # y轴翻转，与图像坐标一致
    ax.set_aspect("equal")
    ax.set_title(f"Geographic Overlay\n{bounds[:2]}\n{bounds[2:]}")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # 绘制 patch 边界
    rect = plt.Rectangle((minx, miny), maxx - minx, maxy - miny,
                         fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(rect)

    # 绘制变化多边形
    for geom in changes:
        if box(minx, miny, maxx, maxy).intersects(geom):
            try:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.4, fc="red", ec="darkred")
            except Exception:
                pass

    # 3. 像素级详细视图 + 网格
    ax = axes[2]
    ax.imshow(mask, cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Pixel Grid (128×128)\nRed=Changed, Green=Unchanged")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 添加网格线
    for i in range(0, W + 1, 16):
        ax.axvline(i - 0.5, color="gray", linewidth=0.3)
        ax.axhline(i - 0.5, color="gray", linewidth=0.3)

    # 在图上标出变化像素数量
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

        # 找出与该时期变化相交的 patches
        affected_pids = []
        for pid, bounds in patch_bounds.items():
            pb = box(*bounds)
            for geom in changes:
                if pb.intersects(geom):
                    affected_pids.append(pid)
                    break

        print(f"    涉及 {len(affected_pids)} 个 patch")

        # 限制数量
        if len(affected_pids) > args.max_patches:
            print(f"    限制可视化前 {args.max_patches} 个 patch")
            affected_pids = affected_pids[:args.max_patches]

        for pid in sorted(affected_pids):
            out_path, ch_px, ratio = visualize_patch(
                pid, patch_bounds[pid], changes, period, output_dir, args.dpi
            )
            all_results.append({
                "period": period, "pid": pid,
                "changed_pixels": ch_px, "ratio": ratio,
                "path": str(out_path),
            })
            print(f"    {pid}: {ch_px}px changed ({ratio:.2f}%) -> {out_path.name}")

    if all_results:
        visualize_summary(all_results, output_dir, args.dpi)

        # 输出统计摘要
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
