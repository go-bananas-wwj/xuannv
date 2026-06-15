#!/usr/bin/env python3
"""海淀 6 任务的可学习下游头实现.

提供基于 PyTorch 的像素级 MLP 与轻量 U-Net 分割头，
用于与 sklearn LogisticRegression/MLP 做横向比较.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch.utils.data import DataLoader, TensorDataset

# 引入 src 中已实现的 change-detection 头与损失
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.models.heads import ChangeDetectionHeadV2


class _MLPPixelNet(nn.Module):
    """像素级 MLP: [C] -> hidden -> 1."""

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (128,),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C] -> [B]
        return self.net(x).squeeze(-1)


class PixelMLPHead:
    """在摊平后的像素特征上训练 MLP 二分类头.

    输入 X: [N, C], y: [N] (0/1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = (128,),
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 4096,
        device: str = "cpu",
        patience: int = 10,
    ) -> None:
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = _MLPPixelNet(input_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # 按 9:1 随机拆分出验证集，用于 early stopping
        n = len(y_train)
        perm = np.random.permutation(n)
        split = int(n * 0.9)
        tr_idx, val_idx = perm[:split], perm[split:]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        pos_weight = torch.tensor(
            neg / max(pos, 1.0), dtype=torch.float32, device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # validation
            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    val_losses.append(
                        nn.functional.binary_cross_entropy_with_logits(logits, yb).item()
                    )
            self.model.train()
            val_loss = float(np.mean(val_losses))

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        probs: list[np.ndarray] = []
        dataset = TensorDataset(torch.from_numpy(X_test).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)


class PixelMLPHeadV2:
    """改进版像素级 MLP：Focal Loss + Dice Loss，按验证 IoU 早停.

    主要解决原 MLP 假阳性过多的问题：
    - Focal Loss 降低大量简单背景样本的权重；
    - Dice Loss 直接优化重叠区域；
    - 验证 IoU 早停保留最好的 checkpoint。
    """

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = (64, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 4096,
        device: str = "cpu",
        patience: int = 15,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.gamma = gamma
        self.alpha = alpha
        self.model = _MLPPixelNet(input_dim, hidden, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    @staticmethod
    def _focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float,
        alpha: float | None,
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight = (1 - p_t) ** gamma
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            weight = alpha_t * weight
        return (weight * bce).mean()

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1 - (2 * intersection + 1e-6) / (union + 1e-6)

    @staticmethod
    def _iou(
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        intersection = (preds * targets).sum()
        union = (preds + targets).clamp_max(1).sum()
        return (intersection / (union + 1e-6)).item()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n = len(y_train)
        perm = np.random.permutation(n)
        split = int(n * 0.9)
        tr_idx, val_idx = perm[:split], perm[split:]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_iou = -1.0
        patience_counter = 0
        best_state: dict | None = None

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self._focal_loss(logits, yb, self.gamma, self.alpha) + self._dice_loss(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # validation IoU
            self.model.eval()
            val_ious: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    val_ious.append(self._iou(logits, yb))
            self.model.train()
            val_iou = float(np.mean(val_ious))

            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        probs: list[np.ndarray] = []
        dataset = TensorDataset(torch.from_numpy(X_test).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)


class PixelMLPHeadV3:
    """进一步改进的像素级 MLP.

    相比 V2 的改进：
    - 更大的隐藏层 (256, 128) + Dropout，提升拟合能力；
    - WeightedRandomSampler 对正例像素过采样，缓解类别极不平衡；
    - 训练/验证拆分后，在验证集上搜索最优阈值；
    - 仍使用 Focal + Dice 损失。

    适用于 upsampled 高分辨率特征（如 128x128）以及差分特征。
    """

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 8192,
        device: str = "cpu",
        patience: int = 15,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_sample_weight: float = 10.0,
        num_train_samples: int = 100000,
    ) -> None:
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.gamma = gamma
        self.alpha = alpha
        self.pos_sample_weight = pos_sample_weight
        self.num_train_samples = num_train_samples
        self.model = _MLPPixelNet(input_dim, hidden, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.threshold = 0.5

    @staticmethod
    def _focal_dice_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float | None,
        gamma: float,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight = (1 - p_t) ** gamma
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            weight = alpha_t * weight
        focal = (weight * bce).mean()

        pred = torch.sigmoid(logits)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = 1 - (2.0 * intersection + 1.0) / (union + 1.0)
        return focal + dice.mean()

    @staticmethod
    def _best_threshold(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float]:
        probs = torch.sigmoid(logits)
        best_th, best_f1 = 0.5, 0.0
        for th in torch.linspace(0.05, 0.95, 37):
            pred = (probs >= th).float()
            tp = (pred * targets).sum()
            fp = ((pred == 1) & (targets == 0)).sum()
            fn = ((pred == 0) & (targets == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th.item()
        return best_th, best_f1.item()

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampler: torch.utils.data.Sampler | None = None,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        if sampler is not None:
            return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n = len(y_train)
        perm = np.random.permutation(n)
        split = int(n * 0.9)
        tr_idx, val_idx = perm[:split], perm[split:]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        sample_weights = np.where(y_tr == 1, self.pos_sample_weight, 1.0)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=min(self.num_train_samples, len(y_tr)),
            replacement=True,
        )
        train_loader = self._make_loader(X_tr, y_tr, sampler=sampler)
        val_loader = self._make_loader(X_val, y_val, sampler=None)

        best_iou = -1.0
        patience_counter = 0
        best_state: dict | None = None

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self._focal_dice_loss(logits, yb, self.alpha, self.gamma)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_ious: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    intersection = (preds * yb).sum()
                    union = (preds + yb).clamp_max(1).sum()
                    val_ious.append((intersection / (union + 1e-6)).item())
            self.model.train()
            val_iou = float(np.mean(val_ious))

            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 在验证集上选择使 F1 最大的阈值
        self.model.eval()
        val_probs: list[np.ndarray] = []
        val_labels: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(p)
                val_labels.append(yb.numpy())
        val_probs = np.concatenate(val_probs, axis=0).flatten()
        val_labels = np.concatenate(val_labels, axis=0).flatten()
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(val_labels, val_probs)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        best_idx = int(np.argmax(f1s))
        self.threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5
        if np.isnan(self.threshold) or self.threshold <= 0 or self.threshold >= 1:
            self.threshold = 0.5

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        probs: list[np.ndarray] = []
        dataset = TensorDataset(torch.from_numpy(X_test).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)


class _UNet(nn.Module):
    """轻量 U-Net，输入 [B, C, H, W]，输出 [B, H, W] logits."""

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        bc = base_channels
        self.enc1 = self._conv_block(in_channels, bc)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(bc, bc * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(bc * 2, bc * 2)

        self.up2 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2)
        # up2 输出 bc + enc2 输出 2bc = 3bc
        self.dec2 = self._conv_block(bc * 3, bc)
        self.up1 = nn.ConvTranspose2d(bc, bc, kernel_size=2, stride=2)
        # up1 输出 bc + enc1 输出 bc = 2bc
        self.dec1 = self._conv_block(bc * 2, bc)
        self.out = nn.Conv2d(bc, 1, kernel_size=1)

    def _conv_block(self, in_c: int, out_c: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                      # [B, bc, H, W]
        e2 = self.enc2(self.pool1(e1))         # [B, 2bc, H/2, W/2]
        b = self.bottleneck(self.pool2(e2))    # [B, 2bc, H/4, W/4]
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))  # [B, bc, H/2, W/2]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1)) # [B, bc, H, W]
        return self.out(d1).squeeze(1)         # [B, H, W]


class UNetHead:
    """在 spatial embedding map 上训练轻量 U-Net 分割头.

    输入 X: [N, C, H, W], y: [N, H, W] (0/1)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        lr: float = 1e-3,
        epochs: int = 120,
        batch_size: int = 8,
        device: str = "cpu",
        patience: int = 15,
    ) -> None:
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = _UNet(in_channels, base_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.threshold = 0.5

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # 按 patch 维度 9:1 拆分验证集
        n = X_train.shape[0]
        perm = np.random.permutation(n)
        split = max(1, int(n * 0.9))
        tr_idx, val_idx = perm[:split], perm[split:]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        pos = y_tr.sum()
        neg = y_tr.size - pos
        pos_weight = torch.tensor(
            neg / max(pos, 1.0), dtype=torch.float32, device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    val_losses.append(
                        nn.functional.binary_cross_entropy_with_logits(logits, yb).item()
                    )
            self.model.train()
            val_loss = float(np.mean(val_losses))

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 在验证集上选择使 F1 最大的阈值
        self.model.eval()
        val_probs: list[np.ndarray] = []
        val_labels: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(p)
                val_labels.append(yb.numpy())
        val_probs = np.concatenate(val_probs, axis=0).flatten()
        val_labels = np.concatenate(val_labels, axis=0).flatten()
        from sklearn.metrics import precision_recall_curve, f1_score
        prec, rec, thr = precision_recall_curve(val_labels, val_probs)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        best_idx = int(np.argmax(f1s))
        self.threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5
        # 若最佳阈值导致全 0/全 1，回退到 0.5
        if np.isnan(self.threshold) or self.threshold <= 0 or self.threshold >= 1:
            self.threshold = 0.5

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        probs: list[np.ndarray] = []
        dataset = TensorDataset(torch.from_numpy(X_test).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)  # [N, H, W]


class CDHead:
    """基于 before/after embedding map 的变化检测头.

    复用 src.models.heads.ChangeDetectionHeadV2（差异编码 + residual + ECA），
    使用 Focal + Dice 损失，按验证 IoU 早停。

    输入 X_before: [N, D, H, W], X_after: [N, D, H, W], y: [N, H, W]
    输出 prob: [N, H, W]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        epochs: int = 120,
        batch_size: int = 16,
        device: str = "cpu",
        patience: int = 20,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.gamma = gamma
        self.alpha = alpha
        self.model = ChangeDetectionHeadV2(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def _focal_dice_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        gamma: float,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight = (1 - p_t) ** gamma
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            weight = alpha_t * weight
        focal = (weight * bce).mean()

        pred = torch.sigmoid(logits)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = 1 - (2.0 * intersection + 1.0) / (union + 1.0)
        return focal + dice.mean()

    @staticmethod
    def _iou(logits: torch.Tensor, targets: torch.Tensor) -> float:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        intersection = (preds * targets).sum()
        union = (preds + targets).clamp_max(1).sum()
        return (intersection / (union + 1e-6)).item()

    def _make_loader(
        self,
        X_before: np.ndarray,
        X_after: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X_before).float(),
            torch.from_numpy(X_after).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def fit(
        self,
        X_before: np.ndarray,
        X_after: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        # 按 patch 维度 9:1 拆分验证集（保持同一张图前后时相成对）
        n = X_before.shape[0]
        perm = np.random.permutation(n)
        split = max(1, int(n * 0.9))
        tr_idx, val_idx = perm[:split], perm[split:]

        train_loader = self._make_loader(
            X_before[tr_idx], X_after[tr_idx], y_train[tr_idx], shuffle=True
        )
        val_loader = self._make_loader(
            X_before[val_idx], X_after[val_idx], y_train[val_idx], shuffle=False
        )

        best_iou = -1.0
        patience_counter = 0
        best_state: dict | None = None

        self.model.train()
        for epoch in range(self.epochs):
            for xb_bef, xb_aft, yb in train_loader:
                xb_bef = xb_bef.to(self.device)
                xb_aft = xb_aft.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb_bef, xb_aft).squeeze(1)
                loss = self._focal_dice_loss(logits, yb, self.alpha, self.gamma)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_ious: list[float] = []
            with torch.no_grad():
                for xb_bef, xb_aft, yb in val_loader:
                    xb_bef = xb_bef.to(self.device)
                    xb_aft = xb_aft.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb_bef, xb_aft).squeeze(1)
                    val_ious.append(self._iou(logits, yb))
            self.model.train()
            val_iou = float(np.mean(val_ious))
            print(f"[CDHead] epoch {epoch+1}/{self.epochs} val_iou={val_iou:.4f} best={best_iou:.4f} patience={patience_counter}", flush=True)
            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X_before: np.ndarray, X_after: np.ndarray) -> np.ndarray:
        self.model.eval()
        probs: list[np.ndarray] = []
        dataset = TensorDataset(
            torch.from_numpy(X_before).float(),
            torch.from_numpy(X_after).float(),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for xb_bef, xb_aft in loader:
                xb_bef = xb_bef.to(self.device)
                xb_aft = xb_aft.to(self.device)
                logits = self.model(xb_bef, xb_aft).squeeze(1)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)  # [N, H, W]
