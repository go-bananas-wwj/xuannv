"""优化器和调度器."""
from __future__ import annotations

import math

import torch
from torch.optim import AdamW


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    """构建AdamW优化器，对bias和norm参数不设weight decay."""
    no_decay = ["bias", "norm", "ln", "LayerNorm"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=lr, betas=(0.9, 0.999))


class CosineScheduler:
    """Cosine退火 + warmup调度器."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        lr_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
