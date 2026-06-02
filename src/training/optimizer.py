"""优化器与学习率调度工具."""
from __future__ import annotations

import math

import torch

from src.config import Config


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """根据配置构建 AdamW 优化器。

    若 backbone_lr_scale < 1，将 distill_head 以外的参数单独设较低 LR，
    以减缓 backbone 解冻后的表征坍塌。
    """
    t = cfg.training
    scale = getattr(t, 'backbone_lr_scale', 1.0)
    if scale != 1.0:
        backbone_params = [p for n, p in model.named_parameters() if 'distill_head' not in n]
        head_params = [p for n, p in model.named_parameters() if 'distill_head' in n]
        param_groups = [
            {"params": backbone_params, "lr": t.lr * scale, "name": "backbone"},
            {"params": head_params, "lr": t.lr, "name": "head"},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": t.lr, "name": "all"}]
    return torch.optim.AdamW(
        param_groups,
        lr=t.lr,
        weight_decay=t.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> torch.optim.lr_scheduler._LRScheduler | None:
    """根据配置构建学习率调度器."""
    t = cfg.training
    if t.lr_schedule == "cosine_no_restart":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t.epochs,
            eta_min=1e-7,
        )
    return None


def get_cosine_lr(epoch: int, step: int, steps_per_epoch: int, cfg: Config) -> float:
    """计算 Warmup + Cosine Decay 学习率.

    Args:
        epoch: 当前 epoch 索引 (0-based).
        step: 当前 epoch 内的 step 索引 (0-based).
        steps_per_epoch: 每个 epoch 的总步数.
        cfg: 项目配置.

    Returns:
        当前 step 的学习率.
    """
    t = cfg.training
    warmup_steps = t.warmup_epochs * steps_per_epoch
    total_steps = t.epochs * steps_per_epoch
    current = epoch * steps_per_epoch + step

    if current < warmup_steps:
        return t.lr * (current + 1) / warmup_steps
    else:
        progress = (current - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 1e-7 + (t.lr - 1e-7) * 0.5 * (1.0 + math.cos(math.pi * progress))
