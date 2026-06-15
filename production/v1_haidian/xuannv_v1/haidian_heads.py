#!/usr/bin/env python3
"""海淀 6 任务的可学习下游头实现.

提供基于 PyTorch 的像素级 MLP 与轻量 U-Net 分割头，
用于与 sklearn LogisticRegression/MLP 做横向比较.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from torch.utils.data import DataLoader, TensorDataset


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
        return np.concatenate(probs, axis=0)  # [N, H, W]
