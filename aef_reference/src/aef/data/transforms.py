"""数据预处理工具."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio


def read_tif(
    path: Path,
    image_size: int,
    resize_mode: str | None = None,
) -> np.ndarray | None:
    """读取 TIFF 文件并 resize 到目标尺寸.

    根据源的分辨率差异，自动或显式选择重采样策略：
    - 高分辨率大图 (如 Planet 3m, 427×427): 使用 area/average 下采样到目标尺寸，
      保持完整地理覆盖范围，避免 center crop 导致范围丢失。
    - 低分辨率小图 (如 Landsat 30m, 44×44): 使用 bilinear 上采样到目标尺寸，
      避免 edge pad 导致大面积无效复制值。
    - 接近目标分辨率的图像 (如 S2 10m, 131×129): center crop 到目标尺寸。

    Args:
        path: TIFF 文件路径。
        image_size: 目标空间尺寸 (H, W)。
        resize_mode: 显式指定 resize 模式。可选:
            - 'crop': 强制 center crop。
            - 'bilinear': 强制 bilinear resize (连续数据上采样)。
            - 'area': 强制 area/average resize (连续数据下采样)。
            - 'nearest': 强制 nearest resize (分类数据)。
            - None: 根据图像尺寸自动判断。
    """
    try:
        with rasterio.open(path) as src:
            C = src.count
            H = src.height
            W = src.width

            if H == image_size and W == image_size:
                return src.read().astype(np.float32)

            # 自动判断 resize 模式
            if resize_mode is None:
                # 图像远大于目标尺寸 (>=1.5x): 高分辨率下采样
                if H >= image_size * 1.5 and W >= image_size * 1.5:
                    resize_mode = "area"
                # 图像远小于目标尺寸 (<=0.75x): 低分辨率上采样
                elif H <= image_size * 0.75 or W <= image_size * 0.75:
                    resize_mode = "bilinear"
                else:
                    resize_mode = "crop"

            if resize_mode == "crop":
                data = src.read()  # (C, H, W)
                if H >= image_size and W >= image_size:
                    start_h = (H - image_size) // 2
                    start_w = (W - image_size) // 2
                    data = data[:, start_h:start_h + image_size, start_w:start_w + image_size]
                else:
                    # 居中 pad，保持地理中心对齐
                    pad_h = max(0, image_size - H)
                    pad_w = max(0, image_size - W)
                    pad_top = pad_h // 2
                    pad_left = pad_w // 2
                    pad_bottom = pad_h - pad_top
                    pad_right = pad_w - pad_left
                    data = np.pad(
                        data,
                        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="edge",
                    )
                    data = data[:, :image_size, :image_size]
            else:
                from rasterio.enums import Resampling

                if resize_mode == "area":
                    resampling = Resampling.average
                elif resize_mode == "nearest":
                    resampling = Resampling.nearest
                else:
                    resampling = Resampling.bilinear
                data = src.read(
                    out_shape=(C, image_size, image_size),
                    resampling=resampling,
                )

            return data.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def normalize_source(data: np.ndarray, source_name: str, stats: dict) -> np.ndarray:
    """按源类型归一化."""
    if source_name in ("planet", "s2", "landsat"):
        data = np.log(np.clip(data, 0, None) + 1) / 10.0
    elif source_name in ("s1", "tianyi_sar"):
        data = np.clip(data, -30.0, 10.0)

    if stats and "mean" in stats and "std" in stats:
        mean = np.array(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.array(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.where(std < 1e-6, 1.0, std)
        data = (data - mean) / std

    return data


def parse_date_to_ms(date_str: str) -> float:
    """从日期字符串解析为毫秒时间戳.
    支持格式: YYYYMMDD, YYYY-MM-DD, YYYYMMDDHHMMSS 等.
    """
    # 提取纯数字
    digits = re.sub(r"\D", "", date_str)
    if len(digits) >= 8:
        year = int(digits[:4])
        month = int(digits[4:6])
        day = int(digits[6:8])
        # 简单计算 ms（近似值，不需要精确到毫秒）
        # 使用已知参考点简化计算
        from datetime import datetime, timezone
        try:
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            return dt.timestamp() * 1000.0
        except ValueError:
            pass
    # fallback: 返回基于字符串哈希的伪时间戳（保证一致性）
    import hashlib
    h = int(hashlib.md5(date_str.encode()).hexdigest(), 16)
    return float(h % 1000000000) * 1000.0
