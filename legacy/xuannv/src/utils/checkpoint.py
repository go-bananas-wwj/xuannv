"""Checkpoint 加载与保存工具.

封装 torch.load / torch.save 的常见模式，减少脚本层重复代码.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    keys: tuple[str, ...] = ("model_state_dict",),
    weights_only: bool = False,
) -> dict[str, Any]:
    """加载 checkpoint，自动处理多层 key 回退.

    Args:
        path: checkpoint 文件路径.
        device: map_location 目标设备.
        keys: 依次尝试读取的 state_dict key 列表，默认优先 "model_state_dict".
        weights_only: 是否启用 PyTorch 2.6+ 的 weights_only 安全加载.

    Returns:
        加载后的 state_dict (dict).

    Raises:
        FileNotFoundError: checkpoint 不存在.
        KeyError: 所有备选 key 均未找到.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=weights_only)

    for key in keys:
        if key in ckpt:
            return ckpt[key]

    # 如果 ckpt 本身就是一个 state_dict (无外层包装)
    if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    available = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    raise KeyError(f"Checkpoint keys {keys} not found in {path}. Available: {available}")


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """保存训练 checkpoint.

    Args:
        path: 目标文件路径.
        epoch: 当前训练轮次.
        model_state: 模型 state_dict.
        optimizer_state: 优化器 state_dict (可选).
        metrics: 评估指标字典 (可选).
        extra: 其他需要保存的字段 (可选).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    if metrics is not None:
        payload["metrics"] = metrics
    if extra is not None:
        payload.update(extra)

    torch.save(payload, path)
