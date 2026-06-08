"""数据预处理工具."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio


def read_tif(path: Path, image_size: int) -> np.ndarray | None:
    """读取 TIFF 文件并 resize 到目标尺寸."""
    try:
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
            if data.shape[1] != image_size or data.shape[2] != image_size:
                # 简单裁剪或填充
                C, H, W = data.shape
                if H >= image_size and W >= image_size:
                    start_h = (H - image_size) // 2
                    start_w = (W - image_size) // 2
                    data = data[:, start_h:start_h + image_size, start_w:start_w + image_size]
                else:
                    pad_h = max(0, image_size - H)
                    pad_w = max(0, image_size - W)
                    data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
                    data = data[:, :image_size, :image_size]
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
