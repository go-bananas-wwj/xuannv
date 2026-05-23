"""轻量级任务头 — 冻结 backbone 后训练."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ChangeDetectionHead(nn.Module):
    """像素级变化检测头.

    输入: before/after embedding maps [B, D, H, W]
    输出: 变化概率 [B, 1, H, W]
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # 输入特征: |e1-e2|, e1*e2, e1, e2  -> 4D
        in_dim = embedding_dim * 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        diff = emb_before - emb_after
        feat = torch.cat([
            torch.abs(diff),
            emb_before * emb_after,
            emb_before,
            emb_after,
        ], dim=1)
        x = self.conv1(feat)
        x = self.conv2(x)
        return self.out(x)


class ChangeDetectionHeadV2(nn.Module):
    """增强版像素级变化检测头 — 参考 Siamese Difference Module 设计.

    输入: before/after embedding maps [B, D, H, W]
    输出: 变化概率 [B, 1, H, W]

    相比原版 ChangeDetectionHead:
    - 更深的差异编码器 (3-layer with residual)
    - 更大的 hidden_dim 以学习复杂变化模式
    - 保持轻量 (< 1M 参数)
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        in_dim = embedding_dim * 4  # |diff|, mul, e1, e2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        diff = emb_before - emb_after
        feat = torch.cat([
            torch.abs(diff),
            emb_before * emb_after,
            emb_before,
            emb_after,
        ], dim=1)
        x = self.encoder(feat)
        x = F.relu(self.res1(x) + x)
        x = F.relu(self.res2(x) + x)
        return self.out(x)


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) module.
    
    轻量级通道注意力，参数量极少 (~100 params)，在遥感变化检测中
    能有效抑制背景、增强变化区域响应。
    """

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0) -> None:
        super().__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class ChangeDetectionHeadV3(nn.Module):
    """V2 + ECA 通道注意力增强版.

    在 residual block 之间插入 ECA，提升特征判别性。
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        in_dim = embedding_dim * 4

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.eca = ECA(hidden_dim)

        self.res2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        diff = emb_before - emb_after
        feat = torch.cat([
            torch.abs(diff),
            emb_before * emb_after,
            emb_before,
            emb_after,
        ], dim=1)
        x = self.encoder(feat)
        x = F.relu(self.res1(x) + x)
        x = self.eca(x)
        x = F.relu(self.res2(x) + x)
        return self.out(x)


class MultiClassChangeDetectionHead(nn.Module):
    """@deprecated: 仅 archive/ 中使用，活跃代码无引用。保留用于 git history 兼容。"""
    """多类别变化检测头.

    输出 3 个独立通道，分别对应:
      0: construction (建筑工地/建造房屋)
      1: demolition   (房屋拆除)
      2: land_conversion (道路/水塘/农田变化)

    训练时: 每类独立 sigmoid + focal/dice loss
    推理时: max(probs) 或 mean(probs) 转为 binary change score
    """

    NUM_CLASSES = 3

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        in_dim = embedding_dim * 4

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.eca = ECA(hidden_dim)

        self.res2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim // 2, self.NUM_CLASSES, 1),
        )

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        diff = emb_before - emb_after
        feat = torch.cat([
            torch.abs(diff),
            emb_before * emb_after,
            emb_before,
            emb_after,
        ], dim=1)
        x = self.encoder(feat)
        x = F.relu(self.res1(x) + x)
        x = self.eca(x)
        x = F.relu(self.res2(x) + x)
        return self.out(x)  # [B, 3, H, W]


class ClassSpecificCDHead(nn.Module):
    """@deprecated: 仅 archive/ 中使用，活跃代码无引用。保留用于 git history 兼容。"""
    """类特定轻量变化检测头 — 每类独立参数，对齐 AlphaEarth 轻量下游头设计.

    所有类别共享轻量差异编码器 (1x1 conv 降维)，
    但每类有独立的 2 层 conv head，避免 construction 主导 demolition/land_conversion。

    Args:
        embedding_dim: backbone 输出维度
        hidden_dim: 共享编码器输出维度 (默认 32，极轻量)
        num_classes: 变化类别数 (默认 3: construction/demolition/land_conversion)
    """

    NUM_CLASSES = 3
    CLASS_NAMES = ["construction", "demolition", "land_conversion"]

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        # 共享差异编码器: 只拼接 before/after，不用复杂的 |diff|/mul
        in_dim = embedding_dim * 2
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 每类独立的轻量 head (2层 3x3 conv)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
            for _ in range(num_classes)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([emb_before, emb_after], dim=1)  # [B, 2D, H, W]
        feat = self.diff_encoder(feat)  # [B, hidden_dim, H, W]

        # 每类独立预测
        logits_list = [head(feat) for head in self.heads]  # list of [B, 1, H, W]
        logits = torch.cat(logits_list, dim=1)  # [B, num_classes, H, W]
        return logits


class PrototypeFewShotHead(nn.Module):
    """@deprecated: 仅 archive/ 中使用，活跃代码无引用。保留用于 git history 兼容。"""
    """可学习原型 Few-Shot 分类头 (像素级).

    输入: before/after embedding maps [B, D, H, W]
    输出: 分类 logits [B, num_classes, H, W]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 64,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        # 投影层: 将 2D 映射到 hidden_dim
        self.proj = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # 可学习原型 [num_classes, hidden_dim]
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim))

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([emb_before, emb_after], dim=1)  # [B, 2D, H, W]
        z = self.proj(feat)  # [B, H, H, W]
        z = F.normalize(z, dim=1)
        w = F.normalize(self.prototypes, dim=-1)  # [num_classes, hidden_dim]
        # 逐像素余弦相似度 -> [B, num_classes, H, W]
        logits = torch.einsum("bdhw,cd->bchw", z, w) * self.temperature
        return logits


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary Dice loss."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """多类别 Dice loss，每类独立计算后平均.

    logits: [B, C, H, W]
    target: [B, C, H, W] float
    """
    pred = torch.sigmoid(logits)
    loss = 0.0
    C = logits.size(1)
    for c in range(C):
        intersection = (pred[:, c] * target[:, c]).sum()
        union = pred[:, c].sum() + target[:, c].sum()
        loss += 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return loss / C


def focal_bce_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE for imbalanced change detection."""
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def multiclass_focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """多类别 Focal BCE，每类独立计算.

    logits: [B, C, H, W]
    target: [B, C, H, W] float
    class_weights: [C] 可选，用于处理类别不平衡
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # [B, C, H, W]
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    if class_weights is not None:
        focal = focal * class_weights.view(1, -1, 1, 1)
    return focal.mean()
