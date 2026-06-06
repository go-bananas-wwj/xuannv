"""
可视化海淀 patch 的多源时间序列（修复版）。
修复内容：
1. PlanetScene DN/10000 → reflectance
2. S1 处理 nodata(-9999)/NaN，log10 转换后固定范围拉伸
3. TianyiSAR 过滤 -9999，固定 dB 范围拉伸
4. 所有 sensor 使用 patch 内全局范围拉伸，保证时间一致性
5. 改进布局：增加间距、标注尺寸
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
import numpy as np
import rasterio

fm.fontManager.addfont("/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

HAIDIAN_ROOT = Path("data_raw/haidian/scenes")
PLANET_ROOT = Path("data_raw/beijing/planetscene")
PATCH_IDS = ["patch_000000", "patch_000030", "patch_000090", "patch_000180"]
OUT_DIR = Path("data_raw/haidian/viz_multi_source")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COLS = 8


def sample_dates(files: list[Path], n: int) -> list[Path]:
    files = sorted(files)
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n, dtype=int)
    return [files[i] for i in idx]


def collect_planetscene_range(files: list[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    """收集 PlanetScene 所有帧的 global 2%/98% percentile（DN/10000 后）。"""
    all_vals = [[] for _ in range(3)]  # R, G, B
    for f in files:
        try:
            with rasterio.open(f) as src:
                data = src.read()  # [C, H, W]
                if data.shape[0] < 3:
                    continue
                # B=0, G=1, R=2 → R, G, B
                for c_idx, band_idx in enumerate([2, 1, 0]):
                    band = data[band_idx].astype(np.float32) / 10000.0
                    band = band[np.isfinite(band)]
                    if len(band) > 0:
                        all_vals[c_idx].extend(band.flatten().tolist())
        except Exception:
            continue
    if not all(all_vals[c] for c in range(3)):
        return None
    lows = np.array([np.percentile(all_vals[c], 2) for c in range(3)])
    highs = np.array([np.percentile(all_vals[c], 98) for c in range(3)])
    return lows, highs


def read_planetscene_rgb(f: Path, lows: np.ndarray, highs: np.ndarray) -> np.ndarray | None:
    try:
        with rasterio.open(f) as src:
            data = src.read()
            if data.shape[0] < 3:
                return None
            rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32) / 10000.0
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
    except Exception as e:
        print(f"  PlanetScene 读取失败: {e}")
        return None


def collect_s2_range(files: list[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    all_vals = [[] for _ in range(3)]
    for f in files:
        try:
            with rasterio.open(f) as src:
                data = src.read()
                if data.shape[0] < 3:
                    continue
                for c_idx, band_idx in enumerate([2, 1, 0]):
                    band = data[band_idx].astype(np.float32)
                    band = band[np.isfinite(band)]
                    if len(band) > 0:
                        all_vals[c_idx].extend(band.flatten().tolist())
        except Exception:
            continue
    if not all(all_vals[c] for c in range(3)):
        return None
    lows = np.array([np.percentile(all_vals[c], 2) for c in range(3)])
    highs = np.array([np.percentile(all_vals[c], 98) for c in range(3)])
    return lows, highs


def read_s2_rgb(f: Path, lows: np.ndarray, highs: np.ndarray) -> np.ndarray | None:
    try:
        with rasterio.open(f) as src:
            data = src.read()
            if data.shape[0] < 3:
                return None
            rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32)
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
    except Exception as e:
        print(f"  S2 读取失败: {e}")
        return None


def collect_sar_range(files: list[Path], is_s1: bool) -> tuple[float, float] | None:
    """收集 SAR 所有帧的 global 2%/98% percentile。S1 先 log10 转换。"""
    all_vals = []
    for f in files:
        try:
            with rasterio.open(f) as src:
                data = src.read(1).astype(np.float32)
                # mask nodata and NaN
                nodata = src.nodata
                if nodata is not None:
                    data = np.where((data != nodata) & np.isfinite(data), data, np.nan)
                else:
                    data = np.where(np.isfinite(data), data, np.nan)
                valid = data[~np.isnan(data)]
                if len(valid) == 0:
                    continue
                if is_s1:
                    # S1 raw values -> log10 for better visualization
                    valid = valid[valid > 0]
                    if len(valid) == 0:
                        continue
                    valid = np.log10(valid)
                # TianyiSAR is already in dB
                all_vals.extend(valid.flatten().tolist())
        except Exception:
            continue
    if not all_vals:
        return None
    arr = np.array(all_vals)
    lo = np.percentile(arr, 2)
    hi = np.percentile(arr, 98)
    return lo, hi


def read_sar(f: Path, low: float, high: float, is_s1: bool) -> np.ndarray | None:
    try:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where((data != nodata) & np.isfinite(data), data, np.nan)
            else:
                data = np.where(np.isfinite(data), data, np.nan)
            if is_s1:
                # log10 transform
                data = np.where(data > 0, np.log10(data), np.nan)
            # normalize to [0, 1]
            if high > low:
                data = np.clip((data - low) / (high - low), 0, 1)
            else:
                mx, mn = np.nanmax(data), np.nanmin(data)
                if mx > mn:
                    data = (data - mn) / (mx - mn)
            # nan -> 0 (black)
            data = np.where(np.isfinite(data), data, 0)
            return data
    except Exception as e:
        print(f"  SAR 读取失败: {e}")
        return None


def collect_landsat_range(files: list[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    all_vals = [[] for _ in range(3)]
    for f in files:
        try:
            with rasterio.open(f) as src:
                data = src.read()
                if data.shape[0] < 3:
                    continue
                for c_idx, band_idx in enumerate([2, 1, 0]):
                    band = data[band_idx].astype(np.float32)
                    band = band[np.isfinite(band)]
                    if len(band) > 0:
                        all_vals[c_idx].extend(band.flatten().tolist())
        except Exception:
            continue
    if not all(all_vals[c] for c in range(3)):
        return None
    lows = np.array([np.percentile(all_vals[c], 2) for c in range(3)])
    highs = np.array([np.percentile(all_vals[c], 98) for c in range(3)])
    return lows, highs


def read_landsat_rgb(f: Path, lows: np.ndarray, highs: np.ndarray) -> np.ndarray | None:
    try:
        with rasterio.open(f) as src:
            data = src.read()
            if data.shape[0] < 3:
                return None
            rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32)
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
    except Exception as e:
        print(f"  Landsat 读取失败: {e}")
        return None


def viz_patch(patch_id: str):
    patch_dir = HAIDIAN_ROOT / patch_id
    planet_dir = PLANET_ROOT / patch_id
    if not patch_dir.exists():
        print(f"跳过 {patch_id}: 目录不存在")
        return

    s2_dir = patch_dir / "s2"
    tianyi_dir = patch_dir / "tianyi_sar"
    s1_dir = patch_dir / "s1"
    landsat_dir = patch_dir / "landsat"

    s2_files = sorted(s2_dir.glob("*.tif")) if s2_dir.exists() else []
    tianyi_files = sorted(tianyi_dir.glob("*.tif")) if tianyi_dir.exists() else []
    s1_files = sorted(s1_dir.glob("*.tif")) if s1_dir.exists() else []
    landsat_files = sorted(landsat_dir.glob("*.tif")) if landsat_dir.exists() else []
    planet_files = sorted(planet_dir.glob("*.tif")) if planet_dir.exists() else []

    s2_sel = sample_dates(s2_files, N_COLS)
    tianyi_sel = sample_dates(tianyi_files, N_COLS)
    s1_sel = sample_dates(s1_files, N_COLS)
    landsat_sel = sample_dates(landsat_files, N_COLS)
    planet_sel = sample_dates(planet_files, N_COLS)

    # 收集 global 拉伸范围
    planet_range = collect_planetscene_range(planet_files) if planet_files else None
    s2_range = collect_s2_range(s2_files) if s2_files else None
    tianyi_range = collect_sar_range(tianyi_files, is_s1=False) if tianyi_files else None
    s1_range = collect_sar_range(s1_files, is_s1=True) if s1_files else None
    landsat_range = collect_landsat_range(landsat_files) if landsat_files else None

    # 读取尺寸用于标注
    planet_shape = ""
    if planet_files:
        with rasterio.open(planet_files[0]) as src:
            planet_shape = f"{src.height}×{src.width}"
    s2_shape = ""
    if s2_files:
        with rasterio.open(s2_files[0]) as src:
            s2_shape = f"{src.height}×{src.width}"
    tianyi_shape = ""
    if tianyi_files:
        with rasterio.open(tianyi_files[0]) as src:
            tianyi_shape = f"{src.height}×{src.width}"
    s1_shape = ""
    if s1_files:
        with rasterio.open(s1_files[0]) as src:
            s1_shape = f"{src.height}×{src.width}"
    landsat_shape = ""
    if landsat_files:
        with rasterio.open(landsat_files[0]) as src:
            landsat_shape = f"{src.height}×{src.width}"

    sensors = [
        (f"PlanetScene\n(3m) {planet_shape}", planet_sel, "planet", planet_range),
        (f"S2 Sentinel-2\n(10m) {s2_shape}", s2_sel, "s2", s2_range),
        (f"天仪 SAR\n(10m) {tianyi_shape}", tianyi_sel, "tianyi", tianyi_range),
        (f"S1 SAR\n(10m) {s1_shape}", s1_sel, "s1", s1_range),
        (f"Landsat\n(30m) {landsat_shape}", landsat_sel, "landsat", landsat_range),
    ]

    fig, axes = plt.subplots(len(sensors), N_COLS, figsize=(2.6 * N_COLS, 2.5 * len(sensors)))
    if len(sensors) == 1:
        axes = axes.reshape(1, -1)

    for row, (label, files, mode, range_vals) in enumerate(sensors):
        for col in range(N_COLS):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("gray")
            if col < len(files):
                f = files[col]
                date_str = f.stem
                if mode == "planet":
                    img = read_planetscene_rgb(f, range_vals[0], range_vals[1]) if range_vals else None
                elif mode == "s2":
                    img = read_s2_rgb(f, range_vals[0], range_vals[1]) if range_vals else None
                elif mode in ("tianyi", "s1"):
                    img = read_sar(f, range_vals[0], range_vals[1], is_s1=(mode=="s1")) if range_vals else None
                    if img is not None:
                        img = plt.cm.gray(img)
                elif mode == "landsat":
                    img = read_landsat_rgb(f, range_vals[0], range_vals[1]) if range_vals else None
                else:
                    img = None
                if img is not None:
                    ax.imshow(img)
                ax.set_title(date_str, fontsize=8, pad=4)
            else:
                ax.axis("off")

        # 竖排标注在行左侧
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold", rotation=0,
                                 labelpad=60, va="center", ha="right")

    fig.suptitle(f"{patch_id} - 多源时间序列对比（全局拉伸）", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0.07, 0, 1, 0.99], h_pad=0.8, w_pad=0.5)
    out_path = OUT_DIR / f"{patch_id}_multisource_v2.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    for pid in PATCH_IDS:
        viz_patch(pid)
    print("完成")
