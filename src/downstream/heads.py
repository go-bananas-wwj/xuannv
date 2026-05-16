"""下游任务Heads - 轻量级分割/分类头"""
from __future__ import annotations

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """通用分割Head - 输入embedding map，输出像素级mask
    
    用于: 水体检测、建筑物分割（二分类）
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [B, D, H, W]
        Returns:
            logits: [B, 1, H, W]
        """
        x = F.relu(self.bn1(self.conv1(embedding)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class ClassificationHead(nn.Module):
    """通用分类Head - 输入embedding map，输出像素级多类分类
    
    用于: 土地利用分割（多分类）
    """
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [B, D, H, W]
        Returns:
            logits: [B, num_classes, H, W]
        """
        x = F.relu(self.bn1(self.conv1(embedding)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class ChangeDetectionHeadSimple(nn.Module):
    """简化版变化检测Head - 输入before/after embedding，输出变化mask
    
    比task_heads.py中的版本更轻量，适合快速下游训练
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = embedding_dim * 2  # before + after
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.out = nn.Conv2d(hidden_dim, 1, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_before: [B, D, H, W]
            emb_after: [B, D, H, W]
        Returns:
            logits: [B, 1, H, W]
        """
        x = torch.cat([emb_before, emb_after], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.out(x)
        return x
