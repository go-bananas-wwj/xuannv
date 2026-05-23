"""下游任务 Head — MLP / CNN 像素级分类器.

支持:
- PixelMLPHead: 2-3 层 MLP, 逐像素分类
- PixelConvHead: 1×1/3×3 CNN, 保持空间结构
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelMLPHead(nn.Module):
    """2 层 MLP 像素级分类器.

    输入: [B, D, H, W] 或展平后的 [B*N, D]
    输出: [B, N_cls, H, W] 或 [B*N, N_cls]
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, D, H, W] or [B*N, D]
        Returns:
            logits: [B, C, H, W] or [B*N, C]
        """
        squeeze = False
        if x.dim() == 4:
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
            squeeze = True

        out = F.relu(self.bn1(self.fc1(x)))
        out = self.drop1(out)
        out = self.fc2(out)

        if squeeze:
            out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C, H, W]
        return out


class PixelMLPHeadV2(nn.Module):
    """@deprecated: 仅 archive/ 中使用，活跃代码无引用。保留用于 git history 兼容。"""
    """3 层 MLP 像素级分类器 (更强表达能力)."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 4:
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, D)
            squeeze = True

        out = self.net(x)

        if squeeze:
            out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class PixelConvHead(nn.Module):
    """1×1/3×3 CNN 像素级分类器.

    天然支持批量空间推理 [B, D, H, W] → [B, C, H, W].
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 10,
        kernel_size: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.out = nn.Conv2d(hidden_dim, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return self.out(out)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss for multi-class classification.

    Args:
        logits: [N, C] raw logits
        targets: [N] long class indices
        alpha: [C] class weights
        gamma: focusing parameter
        reduction: "mean" | "sum" | "none"
    """
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce

    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    return focal
