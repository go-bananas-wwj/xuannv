"""Embedding Memory Bank — 用于扩大 Uniformity Loss 的有效 batch.

核心设计:
- 环形队列存储最近见过的 L2-normalized embedding
- 每个 rank 独立维护（不需要 DDP 同步）
- 计算 uniformity 时，使用 当前 batch + memory bank 的样本
"""
from __future__ import annotations

import torch
from torch import nn


class EmbeddingMemoryBank:
    """Embedding 记忆银行 — 环形队列存储历史 embedding.

    Args:
        K: 队列容量
        dim: embedding 维度
        device: 存储设备
    """

    def __init__(self, K: int, dim: int, device: torch.device) -> None:
        self.K = K
        self.dim = dim
        self.device = device
        self.queue = torch.zeros(K, dim, device=device)
        self.ptr = 0          # 当前写入位置
        self.size = 0         # 当前已填充数量

    def enqueue(self, embeddings: torch.Tensor) -> None:
        """将当前 batch 的 embedding 入队.

        Args:
            embeddings: [B, D] pre-norm embedding（不做 L2 归一化，保持与 gathered_pre 同一空间）
        """
        B = embeddings.shape[0]
        if B == 0:
            return

        # 环形写入
        for i in range(B):
            self.queue[self.ptr] = embeddings[i]
            self.ptr = (self.ptr + 1) % self.K
            self.size = min(self.size + 1, self.K)

    def get_all(self) -> torch.Tensor:
        """返回当前队列中所有有效 embedding.

        Returns:
            [N, D] N = min(self.size, self.K)
        """
        if self.size == 0:
            return torch.zeros(0, self.dim, device=self.device)
        return self.queue[:self.size].clone()

    def clear(self) -> None:
        """清空队列."""
        self.queue.zero_()
        self.ptr = 0
        self.size = 0
