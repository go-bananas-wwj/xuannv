#!/usr/bin/env python3
"""
下游任务Head训练脚本 — 冻结backbone，使用预计算embedding。

支持4类下游任务:
1. change_detection  → 输入before/after embedding，输出变化mask
2. water_detection   → 输入单月embedding，输出水体mask
3. building_segmentation → 输入单月embedding，输出建筑物mask
4. landuse_segmentation → 输入单月embedding，输出土地利用分类

用法:
  python scripts/downstream/train_downstream.py \
    --embedding-dir /path/to/embeddings \
    --task change_detection --period june \
    --epochs 20 --lr 1e-3 --device npu:0
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.downstream.heads import SegmentationHead, ClassificationHead, ChangeDetectionHeadSimple
from src.models.heads import ChangeDetectionHead, ChangeDetectionHeadV2
from src.models.downstream_heads import PixelConvHead
from src.utils.device import get_device


# ============ 数据集定义 ============

class ChangeDetectionDataset(Dataset):
    """变化检测数据集 — 加载before/after embedding + 变化mask"""

    def __init__(self, embedding_dir: Path, mask_dir: Path, period: str,
                 before_month: str, after_month: str, patch_ids: list[str] | None = None):
        self.embedding_dir = embedding_dir
        self.mask_dir = mask_dir / period
        self.before_month = before_month
        self.after_month = after_month

        # 获取有mask的patch列表
        if patch_ids is None:
            mask_files = list(self.mask_dir.glob("*.npy"))
            self.patch_ids = sorted([f.stem for f in mask_files])
        else:
            self.patch_ids = [p for p in patch_ids if (self.mask_dir / f"{p}.npy").exists()]

        print(f"ChangeDetectionDataset ({period}): {len(self.patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        emb_before = np.load(self.embedding_dir / f"{pid}_{self.before_month}.npy")  # [D, H, W]
        emb_after = np.load(self.embedding_dir / f"{pid}_{self.after_month}.npy")
        mask = np.load(self.mask_dir / f"{pid}.npy")  # [H, W]

        return {
            "emb_before": torch.from_numpy(emb_before).float(),
            "emb_after": torch.from_numpy(emb_after).float(),
            "mask": torch.from_numpy(mask).long(),
        }


class SegmentationDataset(Dataset):
    """分割数据集 — 加载单月embedding + 分割mask

    用于: 水体检测、建筑物分割
    """

    def __init__(self, embedding_dir: Path, label_dir: Path, month: str,
                 patch_ids: list[str] | None = None):
        self.embedding_dir = embedding_dir
        self.label_dir = label_dir
        self.month = month

        if patch_ids is None:
            self.patch_ids = sorted([f.stem for f in self.label_dir.glob("*.npy")])
        else:
            self.patch_ids = [p for p in patch_ids if (self.label_dir / f"{p}.npy").exists()]

        print(f"SegmentationDataset ({month}): {len(self.patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        emb = np.load(self.embedding_dir / f"{pid}_{self.month}.npy")  # [D, H, W]
        label = np.load(self.label_dir / f"{pid}.npy")  # [H, W]

        return {
            "embedding": torch.from_numpy(emb).float(),
            "label": torch.from_numpy(label).long(),
        }


class ClassificationDataset(Dataset):
    """分类数据集 — 加载单月embedding + 多类分类mask

    用于: 土地利用分割
    """

    def __init__(self, embedding_dir: Path, label_dir: Path, month: str,
                 patch_ids: list[str] | None = None):
        self.embedding_dir = embedding_dir
        self.label_dir = label_dir
        self.month = month

        if patch_ids is None:
            self.patch_ids = sorted([f.stem for f in self.label_dir.glob("*.npy")])
        else:
            self.patch_ids = [p for p in patch_ids if (self.label_dir / f"{p}.npy").exists()]

        print(f"ClassificationDataset ({month}): {len(self.patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        emb = np.load(self.embedding_dir / f"{pid}_{self.month}.npy")
        label = np.load(self.label_dir / f"{pid}.npy")

        return {
            "embedding": torch.from_numpy(emb).float(),
            "label": torch.from_numpy(label).long(),
        }


# ============ 训练逻辑 ============

def train_change_detection(
    embedding_dir: Path,
    mask_dir: Path,
    period: str,
    before_month: str,
    after_month: str,
    output_dir: Path,
    device: torch.device,
    embedding_dim: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    """训练变化检测head"""
    print(f"\n{'='*60}")
    print(f"训练变化检测: {period} ({before_month} → {after_month})")
    print(f"{'='*60}")

    dataset = ChangeDetectionDataset(embedding_dir, mask_dir, period, before_month, after_month)
    if len(dataset) == 0:
        print("❌ 无数据，跳过")
        return None

    # 划分train/val
    n = len(dataset)
    n_train = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型
    head = ChangeDetectionHeadV2(embedding_dim=embedding_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0
    history = []

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            emb_b = batch["emb_before"].to(device)
            emb_a = batch["emb_after"].to(device)
            mask = batch["mask"].to(device)  # [B, H, W]

            logits = head(emb_b, emb_a)  # [B, 1, H, W]

            # BCE + Dice
            bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), mask.float())
            pred = torch.sigmoid(logits.squeeze(1))
            dice = 1.0 - (2.0 * (pred * mask.float()).sum() + 1e-5) / (pred.sum() + mask.float().sum() + 1e-5)
            loss = bce + dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * emb_b.size(0)
            train_samples += emb_b.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(train_samples, 1)

        # Validation
        head.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                emb_b = batch["emb_before"].to(device)
                emb_a = batch["emb_after"].to(device)
                mask = batch["mask"].to(device)

                logits = head(emb_b, emb_a)
                pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()

                intersection = (pred * mask.float()).sum()
                union = ((pred + mask.float()) > 0).float().sum()
                iou = intersection / (union + 1e-5)

                val_iou += iou.item() * emb_b.size(0)
                val_samples += emb_b.size(0)

        avg_val_iou = val_iou / max(val_samples, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_iou": avg_val_iou,
        })

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
                "val_iou": best_val_iou,
            }, output_dir / f"cd_{period}_best.pt")

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Best: {best_val_iou:.4f}")

    with open(output_dir / f"cd_{period}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ 变化检测 {period} 完成, Best Val IoU: {best_val_iou:.4f}")
    return {"period": period, "best_val_iou": best_val_iou}


def train_segmentation(
    embedding_dir: Path,
    label_dir: Path,
    task_name: str,
    month: str,
    output_dir: Path,
    device: torch.device,
    embedding_dim: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    """训练分割head（水体/建筑物）"""
    print(f"\n{'='*60}")
    print(f"训练分割任务: {task_name} ({month})")
    print(f"{'='*60}")

    dataset = SegmentationDataset(embedding_dir, label_dir, month)
    if len(dataset) == 0:
        print("❌ 无数据，跳过")
        return None

    n = len(dataset)
    n_train = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    head = SegmentationHead(embedding_dim=embedding_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0
    history = []

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            emb = batch["embedding"].to(device)  # [B, D, H, W]
            label = batch["label"].to(device)  # [B, H, W]

            logits = head(emb).squeeze(1)  # [B, H, W]
            bce = F.binary_cross_entropy_with_logits(logits, label.float())
            pred = torch.sigmoid(logits)
            dice = 1.0 - (2.0 * (pred * label.float()).sum() + 1e-5) / (pred.sum() + label.float().sum() + 1e-5)
            loss = bce + dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * emb.size(0)
            train_samples += emb.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(train_samples, 1)

        # Validation
        head.eval()
        val_iou = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embedding"].to(device)
                label = batch["label"].to(device)

                logits = head(emb).squeeze(1)
                pred = (torch.sigmoid(logits) > 0.5).float()

                intersection = (pred * label.float()).sum()
                union = ((pred + label.float()) > 0).float().sum()
                iou = intersection / (union + 1e-5)

                val_iou += iou.item() * emb.size(0)
                val_samples += emb.size(0)

        avg_val_iou = val_iou / max(val_samples, 1)
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_iou": avg_val_iou})

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
                "val_iou": best_val_iou,
            }, output_dir / f"{task_name}_best.pt")

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Best: {best_val_iou:.4f}")

    with open(output_dir / f"{task_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ {task_name} 完成, Best Val IoU: {best_val_iou:.4f}")
    return {"task": task_name, "best_val_iou": best_val_iou}


def train_classification(
    embedding_dir: Path,
    label_dir: Path,
    task_name: str,
    month: str,
    num_classes: int,
    output_dir: Path,
    device: torch.device,
    embedding_dim: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    """训练分类head（土地利用）"""
    print(f"\n{'='*60}")
    print(f"训练分类任务: {task_name} ({month}), {num_classes} classes")
    print(f"{'='*60}")

    dataset = ClassificationDataset(embedding_dir, label_dir, month)
    if len(dataset) == 0:
        print("❌ 无数据，跳过")
        return None

    n = len(dataset)
    n_train = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    head = ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            emb = batch["embedding"].to(device)
            label = batch["label"].to(device)  # [B, H, W]

            logits = head(emb)  # [B, C, H, W]
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * emb.size(0)
            train_samples += emb.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(train_samples, 1)

        # Validation
        head.eval()
        val_acc = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embedding"].to(device)
                label = batch["label"].to(device)

                logits = head(emb)
                pred = logits.argmax(dim=1)
                acc = (pred == label).float().mean()

                val_acc += acc.item() * emb.size(0)
                val_samples += emb.size(0)

        avg_val_acc = val_acc / max(val_samples, 1)
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_acc": avg_val_acc})

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
            }, output_dir / f"{task_name}_best.pt")

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Best: {best_val_acc:.4f}")

    with open(output_dir / f"{task_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ {task_name} 完成, Best Val Acc: {best_val_acc:.4f}")
    return {"task": task_name, "best_val_acc": best_val_acc}


# ============ 主函数 ============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True, help="Embedding目录")
    parser.add_argument("--task", required=True, choices=["change_detection", "water_detection", "building_segmentation", "landuse_segmentation"])
    parser.add_argument("--period", default="june", help="变化检测时间段")
    parser.add_argument("--mask-dir", default="/workspace/xuannv/data/change_masks", help="变化mask目录")
    parser.add_argument("--label-dir", default="/workspace/xuannv/data/labels", help="分割/分类标注目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--device", default="npu:0", help="Device")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "change_detection":
        period_map = {
            "june": ("2025-04", "2025-06"),
            "aug": ("2025-06", "2025-08"),
            "september": ("2025-08", "2025-09"),
            "october": ("2025-09", "2025-10"),
        }
        before, after = period_map[args.period]
        result = train_change_detection(
            embedding_dir, Path(args.mask_dir), args.period, before, after,
            output_dir, device, args.embedding_dim, args.epochs, args.lr, args.batch_size
        )

    elif args.task == "water_detection":
        result = train_segmentation(
            embedding_dir, Path(args.label_dir) / "water", "water_detection", "2024-06",
            output_dir, device, args.embedding_dim, args.epochs, args.lr, args.batch_size
        )

    elif args.task == "building_segmentation":
        result = train_segmentation(
            embedding_dir, Path(args.label_dir) / "building", "building_segmentation", "2024-06",
            output_dir, device, args.embedding_dim, args.epochs, args.lr, args.batch_size
        )

    elif args.task == "landuse_segmentation":
        result = train_classification(
            embedding_dir, Path(args.label_dir) / "landuse", "landuse_segmentation", "2024-06",
            11, output_dir, device, args.embedding_dim, args.epochs, args.lr, args.batch_size
        )

    if result:
        with open(output_dir / f"{args.task}_result.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
