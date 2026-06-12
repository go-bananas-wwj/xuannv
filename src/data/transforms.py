"""数据预处理变换函数.

包含读写、归一化、增强等通用工具，供 dataset.py 使用.
对齐原版 AEF 预处理流水线 (V7):
  1. 光学 (S2/Landsat): log(x+1)/10 → z-score → ±6σ clip
  2. SAR (S1): clip[-30,10] dB → z-score → ±6σ clip
  3. 分类 (WorldCover/DW): class index → 归一化到 [0,1]
  4. 其他连续值: z-score → ±6σ clip
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_bounds
from rasterio.warp import Resampling as RioResampling, reproject

# 论文设计: 3类输入源
INPUT_SOURCES = ["s2", "s1", "landsat"]

# 7类目标源: (name, loss_type, sensor_src)
# loss_type: 0=MSE  1=CrossEntropy  2=Dice  3=CE+Dice  4=SmoothL1  5=Huber
TARGET_SOURCES = [
    ("s2", 0, "s2"),
    ("s1", 0, "s1"),
    ("landsat", 0, "landsat"),
    ("dem", 4, "dem"),
    ("worldcover", 1, "worldcover"),
    ("dynamic_world", 1, "dynamic_world"),
    ("jrc_water", 0, "jrc_water"),
]

SOURCE_TYPE_MAP = {
    "s2": 0,
    "s1": 1,
    "landsat": 2,
    "dem": 3,
    "worldcover": 4,
    "dynamic_world": 5,
    "jrc_water": 6,
    "tianyi_sar": 7,   # 天仪卫星 X 波段 SAR（1通道 VV，已是 dB 值）
}

# 预处理常量 (对齐原版 AEF)
LOG_TRANSFORM_SOURCES = {"s2", "landsat", "s2_hr"}
SAR_SOURCES = {"s1", "s1_hr", "tianyi_sar"}   # 天仪 SAR 已是 dB 值，走同一 clip[-30,10] 分支
SAR_CLIP_RANGE = (-30.0, 10.0)
CATEGORICAL_SOURCES = {"worldcover", "dynamic_world"}
SIGMA_CLIP = 6.0

# WorldCover / Dynamic World / IO-LULC 类别映射
WC_NUM_CLASSES = 11
DW_NUM_CLASSES = 9
WC_CLASS_MAP = {
    10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6, 80: 7, 90: 8, 95: 9, 100: 10,
}
# Google Dynamic World 原始类别 (0-8, 如使用 GEE 数据)
DW_CLASS_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
}
# IO-LULC Annual V2 类别（1-11 非连续）→ 0-based 9类
# 来源：Esri/Impact Observatory 10m LULC
# 1=Water, 2=Trees, 4=Flooded Veg, 5=Crops, 7=Built, 8=Bare, 9=Snow, 10=Clouds, 11=Rangeland
IOLULC_CLASS_MAP = {
    0: 0,   # background
    1: 1,   # Water
    2: 2,   # Trees
    4: 3,   # Flooded Vegetation
    5: 4,   # Crops
    7: 5,   # Built Area（海淀主导类别）
    8: 6,   # Bare Ground
    9: 7,   # Snow/Ice
    10: 0,  # Clouds → background（忽略）
    11: 8,  # Rangeland
}
# JRC 全球地表水：连续值 0-100（水体出现频率 %）
JRC_WATER_CLIP = (0.0, 100.0)


# ---------------------------------------------------------------------------
# Timestamp / 文件名解析
# ---------------------------------------------------------------------------

def label_to_timestamp_ms(label: str) -> int:
    """兼容单景 (YYYYMMDD) 和季度 (YYYYQN) 文件名 -> 时间戳 ms."""
    label = str(label).strip().upper()
    label = label.replace(".TIF", "").replace(".TIFF", "")

    match = re.match(r"(\d{4})Q(\d)", label)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        month = quarter * 3 - 2
        from datetime import datetime
        dt = datetime(year, month, 15)
        return int(dt.timestamp() * 1000)

    if label.isdigit() and len(label) == 8:
        year = int(label[:4])
        month = int(label[4:6])
        day = int(label[6:8])
        from datetime import datetime
        dt = datetime(year, month, day)
        return int(dt.timestamp() * 1000)

    # 兜底: 对数字直接解释
    if label.isdigit():
        return int(label)

    raise ValueError(f"Cannot parse timestamp from label: {label}")


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def read_tif(path: Path | str, image_size: int, resampling: str = "bilinear") -> np.ndarray | None:
    """读取单张 TIFF, resize 到 image_size × image_size, 返回 (C, H, W) float32; 失败返回 None.

    Args:
        image_size: 目标 H/W; 若 <=0 则返回原始尺寸.
        resampling: "bilinear" 或 "nearest" (分类数据用 nearest).
    """
    try:
        with rasterio.open(path) as src:
            if image_size <= 0 or (src.width == image_size and src.height == image_size):
                data = src.read().astype(np.float32)
            else:
                rio_mode = RioResampling.nearest if resampling == "nearest" else RioResampling.bilinear
                data = src.read(
                    out_shape=(src.count, image_size, image_size),
                    resampling=rio_mode,
                ).astype(np.float32)
            # nodata / nan / inf → 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data
    except (RasterioIOError, OSError):
        return None


def read_tif_aligned(
    path: Path | str,
    dst_bounds: tuple[float, float, float, float],
    dst_shape: tuple[int, int],
    dst_crs: str | rasterio.crs.CRS,
    resampling: str = "bilinear",
    fill_value: float = 0.0,
) -> np.ndarray | None:
    """读取单张 TIFF 并按目标地理 bounds/shape 重投影，保持真实 GSD.

    Args:
        path: TIFF 文件路径.
        dst_bounds: (left, bottom, right, top) 目标地理范围.
        dst_shape: (height, width) 目标像素尺寸.
        dst_crs: 目标坐标参考系.
        resampling: "bilinear"（连续/光学/SAR）或 "nearest"（分类/含 NaN）.
        fill_value: 无数据区域填充值.

    Returns:
        (C, H, W) float32 数组；失败返回 None.
    """
    try:
        rio_mode = RioResampling.nearest if resampling == "nearest" else RioResampling.bilinear
        dst_transform = from_bounds(
            dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3],
            width=dst_shape[1], height=dst_shape[0],
        )
        with rasterio.open(path) as src:
            dst = np.full((src.count, *dst_shape), fill_value, dtype=np.float32)
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rio_mode,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_nodata=fill_value,
            )
            dst = np.nan_to_num(dst, nan=fill_value, posinf=fill_value, neginf=fill_value)
            return dst
    except (RasterioIOError, OSError, rasterio.errors.CRSError):
        return None


# ---------------------------------------------------------------------------
# Normalization (对齐原版 AEF V7)
# ---------------------------------------------------------------------------

def normalize_data(data: np.ndarray, source_name: str, stats: dict, num_classes: int = 11) -> np.ndarray:
    """根据数据源选择对应的归一化策略.

    Args:
        data: 原始数据，形状 (C, H, W).
        source_name: 数据源名称.
        stats: 各通道的 mean/std 统计字典.
        num_classes: 分类目标的类别数（仅对分类源有效）.

    Returns:
        归一化后的 numpy 数组.
    """
    source_name = source_name.lower().strip()

    # 1. 分类源: 类别索引编码, 不做 z-score
    if source_name in CATEGORICAL_SOURCES:
        return _normalize_categorical(data, source_name)

    # 2. SAR 源: dB 范围裁剪
    if source_name in SAR_SOURCES:
        # 修复：检测 PC 下载的 DN 值（sigma0 * 10000 格式），先转 dB
        if data.max() > 100:
            # PC sentinel-1-grd 提供的是 sigma0 * 10000 格式的定点数
            # 先还原到线性 sigma0，再转 dB: 10*log10(sigma0)
            data = np.log10(np.clip(data / 10000.0, 1e-10, None)) * 10.0
        lo, hi = SAR_CLIP_RANGE
        data = np.clip(data, lo, hi)

    # 3. 光学源: log(x+1)/10 变换
    # 修复：检测大庆/海淀的0-1归一化数据，先还原到原始反射率范围
    if source_name in LOG_TRANSFORM_SOURCES:
        if data.max() < 2.0:
            # 数据已经是0-1范围（GEE导出时不同scale），先×10000还原
            data = data * 10000.0
        data = np.log(np.clip(data, 0, None) + 1) / 10.0

    # 4. z-score 归一化
    source_stats = stats.get(source_name)
    if source_stats is not None:
        out = np.zeros_like(data, dtype=np.float32)
        for c in range(data.shape[0]):
            key = f"band_{c}"
            if key not in source_stats:
                key = f"channel_{c}"   # 兼容 compute_statistics.py 的 channel_N 格式
            if key in source_stats:
                mean = source_stats[key]["mean"]
                std = source_stats[key]["std"]
                if std > 1e-8:
                    out[c] = (data[c] - mean) / std
                else:
                    out[c] = data[c] - mean
            else:
                out[c] = data[c]
        data = out
    else:
        # fallback: 逐通道 z-score
        out = np.zeros_like(data, dtype=np.float32)
        for c in range(data.shape[0]):
            mean = float(np.nanmean(data[c]))
            std = float(np.nanstd(data[c]))
            std = std if std > 1e-6 else 1e-6
            out[c] = (np.nan_to_num(data[c], nan=mean) - mean) / std
        data = out

    # 5. ±6σ 裁剪
    data = np.clip(data, -SIGMA_CLIP, SIGMA_CLIP)
    return data


def _normalize_categorical(data: np.ndarray, source_name: str) -> np.ndarray:
    """分类源: class index → one-hot 编码 (num_classes, H, W)."""
    if source_name == "worldcover":
        class_map = WC_CLASS_MAP
        num_classes = WC_NUM_CLASSES
    elif source_name == "dynamic_world":
        # 使用 IO-LULC Annual V2 映射（1-11 非连续 → 0-based 9类）
        class_map = IOLULC_CLASS_MAP
        num_classes = DW_NUM_CLASSES
    else:
        class_map = DW_CLASS_MAP
        num_classes = DW_NUM_CLASSES

    if data.ndim == 3:
        data = data[0:1]
    label = data[0].astype(np.int64)
    H, W = label.shape
    one_hot = np.zeros((num_classes, H, W), dtype=np.float32)
    # 将原始值映射到类别索引
    mapped = np.full_like(label, -1, dtype=np.int64)
    for val, idx in class_map.items():
        mapped[label == val] = idx
    # 只保留有效映射的像素
    valid_mapped = (mapped >= 0) & (mapped < num_classes)
    if valid_mapped.any():
        one_hot[mapped[valid_mapped], np.where(valid_mapped)[0], np.where(valid_mapped)[1]] = 1.0
    return one_hot
