"""设备管理工具."""
from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True, device_str: str | None = None) -> torch.device:
    """获取推理/训练设备.

    Args:
        prefer_cuda: 优先使用 CUDA (若可用).
        device_str: 显式指定设备字符串，如 "cuda:0" 或 "cpu".

    Returns:
        torch.device 实例.
    """
    if device_str is not None:
        return torch.device(device_str)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_cuda_device_index(device: torch.device) -> int:
    """从 torch.device 中提取 CUDA 设备索引，非 CUDA 返回 -1."""
    if device.type == "cuda" and device.index is not None:
        return device.index
    return -1
