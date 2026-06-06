#!/usr/bin/env python3
"""多源时间序列可视化：每行一个 sensor，按时间顺序从左到右."""

import re
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def stretch_band(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    p_low = np.percentile(band, low)
    p_high = np.percentile(band, high)
    if p_high <= p_low:
        p_high = p_low + 1
    stretched = (band - p_low) / (p_high - p_low) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)


def read_rgb_s2(tif_path: Path) -> np.ndarray | None:
    """读取 S2 RGB (B4=R, B3=G, B2=B)."""
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read()
            if arr.shape[0] < 4:
                return None
            r = stretch_band(arr[3])
            g = stretch_band(arr[2])
            b = stretch_band(arr[1])
            return np.stack([r, g, b], axis=2)
    except Exception:
        return None


def read_rgb_planet(tif_path: Path) -> np.ndarray | None:
    """读取 PlanetScene RGB (B3=R, B2=G, B1=B)."""
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read()
            if arr.shape[0] < 3:
                return None
            r = stretch_band(arr[2])
            g = stretch_band(arr[1])
            b = stretch_band(arr[0])
            return np.stack([r, g, b], axis=2)
    except Exception:
        return None


def read_s1_vv(tif_path: Path) -> np.ndarray | None:
    """读取 S1 VV 波段，灰度."""
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            stretched = stretch_band(arr)
            return np.stack([stretched, stretched, stretched], axis=2)
    except Exception:
        return None


def read_rgb_landsat(tif_path: Path) -> np.ndarray | None:
    """读取 Landsat RGB."""
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read()
            n = arr.shape[0]
            # 尝试找 RGB 波段
            if n >= 6:
                r_idx, g_idx, b_idx = 3, 2, 1
            elif n >= 3:
                r_idx, g_idx, b_idx = 2, 1, 0
            else:
                return None
            r = stretch_band(arr[r_idx])
            g = stretch_band(arr[g_idx])
            b = stretch_band(arr[b_idx])
            return np.stack([r, g, b], axis=2)
    except Exception:
        return None


def get_timeline(sensor_dir: Path) -> list[tuple[datetime, Path]]:
    """获取 sensor 目录下的时间序列."""
    if not sensor_dir.exists():
        return []
    timeline = []
    for tif_path in sorted(sensor_dir.glob("*.tif")):
        stem = tif_path.stem
        # 尝试解析日期
        m = re.match(r'(\d{8})', stem)
        if m:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
        else:
            m = re.match(r'(\d{4}-\d{2}-\d{2})', stem)
            if m:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d")
            else:
                continue
        timeline.append((dt, tif_path))
    return sorted(timeline)


def subsample_timeline(timeline: list, max_n: int = 8) -> list:
    """均匀采样时间点，最多 max_n 个."""
    if len(timeline) <= max_n:
        return timeline
    indices = np.linspace(0, len(timeline) - 1, max_n, dtype=int)
    return [timeline[i] for i in indices]


def visualize_patch_multi_source(patch_id: str, output_png: Path):
    """生成单个 patch 的多源时间序列可视化图."""
    base_dir = Path("/workspace/xuannv/data_raw/haidian/scenes") / patch_id
    planet_dir = Path("/workspace/xuannv/data_raw/beijing/planetscene") / patch_id

    # 收集各 sensor 的时间线
    sensors = []

    # S2
    s2_timeline = get_timeline(base_dir / "s2")
    if s2_timeline:
        sensors.append(("S2 (10m)", s2_timeline, read_rgb_s2))

    # PlanetScene
    planet_timeline = get_timeline(planet_dir)
    if planet_timeline:
        sensors.append(("PlanetScene (3m)", planet_timeline, read_rgb_planet))

    # S1
    s1_timeline = get_timeline(base_dir / "s1")
    if s1_timeline:
        sensors.append(("S1 SAR", s1_timeline, read_s1_vv))

    # Landsat
    landsat_timeline = get_timeline(base_dir / "landsat")
    if landsat_timeline:
        sensors.append(("Landsat (30m)", landsat_timeline, read_rgb_landsat))

    if not sensors:
        print(f"无数据: {patch_id}")
        return

    # 计算需要的子图布局
    n_rows = len(sensors)
    max_cols = max(len(subsample_timeline(tl)) for _, tl, _ in sensors)

    fig = plt.figure(figsize=(2.5 * max_cols + 1, 2.8 * n_rows + 0.5))
    gs = GridSpec(n_rows, max_cols, figure=fig, wspace=0.05, hspace=0.15,
                  left=0.06, right=0.98, top=0.94, bottom=0.04)

    for row_idx, (sensor_name, timeline, reader) in enumerate(sensors):
        sampled = subsample_timeline(timeline, max_n=8)
        n_cols = len(sampled)

        for col_idx, (dt, tif_path) in enumerate(sampled):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = reader(tif_path)
            if img is not None:
                ax.imshow(img)
            ax.set_title(f"{dt.strftime('%Y-%m-%d')}", fontsize=9)
            ax.axis("off")

            # 第一列加 y 轴标签（sensor 名称）
            if col_idx == 0:
                ax.set_ylabel(sensor_name, fontsize=11, rotation=0, ha="right", va="center")

    fig.suptitle(f"Multi-Source Time Series - {patch_id}", fontsize=14, y=0.98)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {output_png}")


def main():
    viz_dir = Path("/workspace/xuannv/data_raw/beijing/viz_multi_source")
    viz_dir.mkdir(parents=True, exist_ok=True)

    patches = ["patch_000000", "patch_000030", "patch_000090", "patch_000180"]

    for pid in patches:
        out_png = viz_dir / f"{pid}_multi_source.png"
        visualize_patch_multi_source(pid, out_png)

    print("全部完成")


if __name__ == "__main__":
    main()
