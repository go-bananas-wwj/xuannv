"""设备管理工具."""
from __future__ import annotations

import torch
import torch_npu


def get_device(prefer_npu: bool = True, device_str: str | None = None) -> torch.device:
    """获取推理/训练设备.

    Args:
        prefer_npu: 优先使用 NPU (若可用).
        device_str: 显式指定设备字符串，如 "npu:0" 或 "cpu".

    Returns:
        torch.device 实例.
    """
    if device_str is not None:
        return torch.device(device_str)
    if prefer_npu and torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def get_npu_device_index(device: torch.device) -> int:
    """从 torch.device 中提取 NPU 设备索引，非 NPU 返回 -1."""
    if device.type == "npu" and device.index is not None:
        return device.index
    return -1
