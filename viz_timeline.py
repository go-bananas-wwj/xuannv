"""生成横向时间轴多源遥感长图（多patch版）。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
import numpy as np
import rasterio
from datetime import datetime

fm.fontManager.addfont("/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

PATCH_IDS = [
    "patch_000000",
    "patch_000030",
    "patch_000060",
    "patch_000090",
    "patch_000120",
    "patch_000150",
    "patch_000180",
    "patch_000210",
]
HAIDIAN_ROOT = Path("data_raw/haidian/scenes")
PLANET_ROOT_BASE = Path("data_raw/beijing/planetscene")
OUT_DIR = Path("data_raw/haidian/viz_multi_source")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THUMB_SIZE = 120
MAX_COLS = 44  # 天仪SAR约44帧


def parse_date(stem: str) -> datetime:
    return datetime.strptime(stem, "%Y%m%d")


def find_nearest_file(files: list[Path], target: datetime) -> tuple[Path | None, int]:
    """找到最接近目标日期的文件，返回(文件, 天数差)。"""
    if not files:
        return None, 9999
    best = None
    best_diff = float('inf')
    for f in files:
        try:
            d = parse_date(f.stem)
            diff = abs((d - target).days)
            if diff < best_diff:
                best_diff = diff
                best = f
        except Exception:
            continue
    if best_diff > 30:
        return None, int(best_diff)
    return best, int(best_diff)


def collect_range(files: list[Path], band_indices: list[int], is_planet: bool = False):
    all_vals = [[] for _ in band_indices]
    for f in files:
        try:
            with rasterio.open(f) as src:
                data = src.read()
                for i, bi in enumerate(band_indices):
                    if bi < data.shape[0]:
                        band = data[bi].astype(np.float32)
                        if is_planet:
                            band = band / 10000.0
                        band = band[np.isfinite(band)]
                        if len(band) > 0:
                            all_vals[i].extend(band.flatten().tolist())
        except Exception:
            continue
    if not all(all_vals[i] for i in range(len(band_indices))):
        return None
    lows = np.array([np.percentile(all_vals[i], 2) for i in range(len(band_indices))])
    highs = np.array([np.percentile(all_vals[i], 98) for i in range(len(band_indices))])
    return lows, highs


def read_rgb_thumb(f: Path, lows: np.ndarray, highs: np.ndarray, is_planet: bool = False) -> np.ndarray | None:
    try:
        with rasterio.open(f) as src:
            data = src.read()
            if data.shape[0] < 3:
                return None
            rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32)
            if is_planet:
                rgb = rgb / 10000.0
            rgb = np.where(np.isfinite(rgb), rgb, 0)
            for c in range(3):
                lo, hi = lows[c], highs[c]
                if hi > lo:
                    rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
                else:
                    mx, mn = rgb[..., c].max(), rgb[..., c].min()
                    if mx > mn:
                        rgb[..., c] = (rgb[..., c] - mn) / (mx - mn)
            return rgb
    except Exception:
        return None


def read_sar_thumb(f: Path, low: float, high: float, is_s1: bool) -> np.ndarray | None:
    try:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where((data != nodata) & np.isfinite(data), data, np.nan)
            else:
                data = np.where(np.isfinite(data), data, np.nan)
            if is_s1:
                data = np.where(data > 0, np.log10(data), np.nan)
            if high > low:
                data = np.clip((data - low) / (high - low), 0, 1)
            else:
                mx, mn = np.nanmax(data), np.nanmin(data)
                if mx > mn:
                    data = (data - mn) / (mx - mn)
            data = np.where(np.isfinite(data), data, 0)
            rgb = np.stack([data, data, data], axis=-1)
            return rgb
    except Exception:
        return None


def resize_to_thumb(img: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil_img = pil_img.resize((size, size), Image.Resampling.NEAREST)
    return np.array(pil_img) / 255.0


def viz_patch_timeline(patch_id: str) -> bool:
    haidian_dir = HAIDIAN_ROOT / patch_id
    planet_dir = PLANET_ROOT_BASE / patch_id
    out_path = OUT_DIR / f"{patch_id}_timeline.png"

    if not haidian_dir.exists():
        print(f"跳过 {patch_id}: 目录不存在")
        return False

    planet_files = sorted(planet_dir.glob("*.tif")) if planet_dir.exists() else []
    s2_files = sorted((haidian_dir / "s2").glob("*.tif")) if (haidian_dir / "s2").exists() else []
    tianyi_files = sorted((haidian_dir / "tianyi_sar").glob("*.tif")) if (haidian_dir / "tianyi_sar").exists() else []
    s1_files = sorted((haidian_dir / "s1").glob("*.tif")) if (haidian_dir / "s1").exists() else []
    landsat_files = sorted((haidian_dir / "landsat").glob("*.tif")) if (haidian_dir / "landsat").exists() else []

    if not tianyi_files:
        print(f"跳过 {patch_id}: 无天仪SAR数据")
        return False

    tianyi_dates = [parse_date(f.stem) for f in tianyi_files]
    tianyi_dates = sorted(tianyi_dates)

    # 收集全局拉伸范围
    planet_range = collect_range(planet_files, [2, 1, 0], is_planet=True) if planet_files else None
    s2_range = collect_range(s2_files, [2, 1, 0]) if s2_files else None
    landsat_range = collect_range(landsat_files, [2, 1, 0]) if landsat_files else None

    def sar_range(files, is_s1):
        all_vals = []
        for f in files:
            try:
                with rasterio.open(f) as src:
                    data = src.read(1).astype(np.float32)
                    nodata = src.nodata
                    if nodata is not None:
                        data = np.where((data != nodata) & np.isfinite(data), data, np.nan)
                    else:
                        data = np.where(np.isfinite(data), data, np.nan)
                    valid = data[~np.isnan(data)]
                    if is_s1:
                        valid = valid[valid > 0]
                        valid = np.log10(valid)
                    all_vals.extend(valid.tolist())
            except Exception:
                continue
        if not all_vals:
            return None
        arr = np.array(all_vals)
        return np.percentile(arr, 2), np.percentile(arr, 98)

    tianyi_range = sar_range(tianyi_files, False) if tianyi_files else None
    s1_range = sar_range(s1_files, True) if s1_files else None

    sensors = [
        ("PlanetScene\n(3m)", planet_files, "rgb", planet_range, False),
        ("S2\n(10m)", s2_files, "rgb", s2_range, False),
        ("天仪SAR\n(10m)", tianyi_files, "sar", tianyi_range, False),
        ("S1 SAR\n(10m)", s1_files, "sar", s1_range, True),
        ("Landsat\n(30m)", landsat_files, "rgb", landsat_range, False),
    ]

    n_cols = len(tianyi_dates)
    n_rows = len(sensors)

    thumb_w = THUMB_SIZE / 100
    thumb_h = thumb_w
    pad_x = 0.18
    pad_y = 0.45
    label_w = 1.6

    fig_w = label_w + n_cols * (thumb_w + pad_x) + pad_x
    fig_h = n_rows * (thumb_h + pad_y) + pad_y * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (label, files, mode, range_vals, is_s1) in enumerate(sensors):
        for col, target_date in enumerate(tianyi_dates):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            nearest, diff = find_nearest_file(files, target_date)
            actual_date_str = ""
            if nearest:
                actual_date_str = nearest.stem[2:]  # YYMMDD
                if nearest and range_vals:
                    if mode == "rgb":
                        is_planet = "planet" in label.lower()
                        img = read_rgb_thumb(nearest, range_vals[0], range_vals[1], is_planet=is_planet)
                    else:
                        img = read_sar_thumb(nearest, range_vals[0], range_vals[1], is_s1)

                    if img is not None:
                        img = resize_to_thumb(img, THUMB_SIZE)
                        ax.imshow(img)

            # 上方：基准日期（小字，灰色）
            base_str = target_date.strftime("%y%m%d")
            # 下方：实际文件日期
            if actual_date_str and actual_date_str != base_str:
                # 日期不匹配，用红色标注
                caption = f"{base_str}→{actual_date_str}"
                color = "#c0392b"  # 暗红
                fontsize = 5.5
            elif actual_date_str:
                caption = base_str
                color = "#2c3e50"  # 深灰
                fontsize = 6
            else:
                caption = "无数据"
                color = "#95a5a6"  # 浅灰
                fontsize = 6

            ax.set_title(caption, fontsize=fontsize, color=color, pad=2)

            # 在图像下方添加差异天数（如果 > 0）
            if diff > 0 and actual_date_str:
                ax.text(0.5, -0.08, f"±{diff}天", transform=ax.transAxes,
                        fontsize=4.5, color="#e74c3c", ha="center", va="top")

        # 第一列标注数据源名称
        axes[row, 0].set_ylabel(label, fontsize=8, fontweight="bold", rotation=0,
                                 labelpad=45, va="center", ha="right")

    fig.suptitle(f"{patch_id} - 2025~2026 多源时间轴 (11天间隔)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0.04, 0, 1, 0.99], h_pad=0.5, w_pad=0.15)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")
    return True


if __name__ == "__main__":
    success_count = 0
    for pid in PATCH_IDS:
        if viz_patch_timeline(pid):
            success_count += 1
    print(f"\n完成: {success_count}/{len(PATCH_IDS)} 个patch")
