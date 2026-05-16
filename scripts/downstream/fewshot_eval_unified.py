#!/usr/bin/env python3
"""
统一 Few-Shot 下游任务评估 — 参考 AEF 论文方法。

评估3个下游任务:
1. 变化检测 (Change Detection) — 4个月份段合并为统一二分类数据
2. 水体检测 (Water Detection) — JRC Water 二分类
3. 土地利用分割 (Land Use Segmentation) — WorldCover 多分类

方法:
- 冻结 backbone，提取 embedding
- K-shot: 用 K 个 patch 训练轻量下游 head
- N-splits: 每次随机划分，取平均
- 变化检测 head: 2-layer Conv (|diff|, mul, e1, e2)
- 分割 head: 2-layer Conv (single embedding)
- 分类 head: 2-layer Conv + 1x1 (single embedding)

用法:
    python scripts/downstream/fewshot_eval_unified.py \
        --config configs/v2_vicreg_recon.yaml \
        --checkpoint /workspace/outputs/v2_vicreg_recon_10ep/epoch_best_xxx.pt \
        --tasks cd,water,landuse \
        --k-shots 5,10,20,50 \
        --n-splits 5 \
        --device npu:0 \
        --output /workspace/outputs/v2_vicreg_recon_10ep/fewshot_results.json
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============ 轻量下游 Heads ============

class SimpleCDHead(nn.Module):
    """轻量变化检测 Head — AEF风格 2-layer Conv."""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        in_dim = embedding_dim * 4  # |diff|, mul, e1, e2
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

    def forward(self, emb_before, emb_after):
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


class SimpleSegHead(nn.Module):
    """轻量分割 Head — 2-layer Conv for binary segmentation."""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, emb):
        x = self.conv1(emb)
        x = self.conv2(x)
        return self.out(x)


class SimpleClsHead(nn.Module):
    """轻量分类 Head — 2-layer Conv for multi-class segmentation."""
    def __init__(self, embedding_dim: int = 128, num_classes: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, emb):
        x = self.conv1(emb)
        x = self.conv2(x)
        return self.out(x)


# ============ 数据加载 ============

def load_backbone(config_path: str, checkpoint_path: str, device: str):
    """加载冻结的 backbone."""
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    cfg = load_config(config_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, cfg


def extract_embedding_for_window(model, dataset, pidx, month_str: str, device: str):
    """提取指定月份的 embedding map."""
    import time
    from src.inference.engine import extract_embedding_map

    # 月份 → 时间窗口 (ms)
    year, mon = month_str.split("-")
    year, mon = int(year), int(mon)
    import calendar
    start_sec = time.mktime((year, mon, 1, 0, 0, 0, 0, 0, 0))
    last_day = calendar.monthrange(year, mon)[1]
    end_sec = time.mktime((year, mon, last_day, 23, 59, 59, 0, 0, 0))
    valid_start, valid_end = int(start_sec * 1000), int(end_sec * 1000)

    emb = extract_embedding_map(model, dataset, pidx, valid_start, valid_end, device, normalize=True)
    return emb  # [D, H, W] numpy


def load_change_detection_data(embedding_dir: Path):
    """加载变化检测数据 — 4个月份段合并."""
    data = []
    mask_dir = Path("/workspace/xuannv/data/change_masks")

    periods = [
        ("june", "2024-04", "2024-06"),
        ("aug", "2024-06", "2024-08"),
        ("september", "2024-08", "2024-09"),
        ("october", "2024-09", "2024-10"),
    ]

    for period, before_month, after_month in periods:
        period_mask_dir = mask_dir / period
        if not period_mask_dir.exists():
            continue
        for mask_file in period_mask_dir.glob("*.npy"):
            pid = mask_file.stem
            emb_b_path = embedding_dir / f"{pid}_{before_month}.npy"
            emb_a_path = embedding_dir / f"{pid}_{after_month}.npy"
            if not emb_b_path.exists() or not emb_a_path.exists():
                continue

            emb_b = np.load(emb_b_path)  # [D, H, W]
            emb_a = np.load(emb_a_path)
            mask = np.load(mask_file)  # [H, W]

            # 只保留有变化的patch（正样本）或明确无变化的patch（负样本）
            # 所有mask都是有效的（有变化=1，无变化=0）
            data.append({
                "pid": pid,
                "eb": torch.from_numpy(emb_b).float(),
                "ea": torch.from_numpy(emb_a).float(),
                "mask": torch.from_numpy(mask).float(),
                "period": period,
            })

    print(f"  变化检测数据: {len(data)} 个窗口组合")
    return data


def load_water_data(embedding_dir: Path):
    """加载水体检测数据."""
    data = []
    label_dir = Path("/workspace/xuannv/data/labels/water")
    month = "2024-06"  # 用6月的embedding

    for label_file in label_dir.glob("*.npy"):
        pid = label_file.stem
        emb_path = embedding_dir / f"{pid}_{month}.npy"
        if not emb_path.exists():
            continue

        emb = np.load(emb_path)
        label = np.load(label_file)

        data.append({
            "pid": pid,
            "emb": torch.from_numpy(emb).float(),
            "mask": torch.from_numpy(label).float(),
        })

    print(f"  水体检测数据: {len(data)} 个patch")
    return data


def load_landuse_data(embedding_dir: Path):
    """加载土地利用分割数据."""
    data = []
    label_dir = Path("/workspace/xuannv/data/labels/landuse")
    month = "2024-06"

    for label_file in label_dir.glob("*.npy"):
        pid = label_file.stem
        emb_path = embedding_dir / f"{pid}_{month}.npy"
        if not emb_path.exists():
            continue

        emb = np.load(emb_path)
        label = np.load(label_file)

        data.append({
            "pid": pid,
            "emb": torch.from_numpy(emb).float(),
            "label": torch.from_numpy(label).long(),
        })

    print(f"  土地利用数据: {len(data)} 个patch")
    return data


# ============ Few-Shot 训练与评估 ============

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def train_cd_head(head, train_data, device, epochs=30, lr=1e-3):
    """训练变化检测head — 使用类别权重处理极端不平衡."""
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    # 计算正负样本权重
    total_pos = sum(item["mask"].sum().item() for item in train_data)
    total_neg = sum((1 - item["mask"]).sum().item() for item in train_data)
    pos_weight = torch.tensor([total_neg / max(total_pos, 1.0)], device=device)
    print(f"    CD pos_weight={pos_weight.item():.1f} (pos={total_pos:.0f}, neg={total_neg:.0f})")

    for epoch in range(epochs):
        head.train()
        np.random.shuffle(train_data)
        epoch_loss = 0.0
        for item in train_data:
            eb = item["eb"].unsqueeze(0).to(device)
            ea = item["ea"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).unsqueeze(0).to(device)

            pred = head(eb, ea)
            loss_bce = F.binary_cross_entropy_with_logits(pred, mask, pos_weight=pos_weight)
            loss_dice = dice_loss(pred, mask)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

    return head


def train_seg_head(head, train_data, device, epochs=30, lr=1e-3):
    """训练分割head — 使用类别权重."""
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    # 计算正负样本权重
    total_pos = sum(item["mask"].sum().item() for item in train_data)
    total_neg = sum((1 - item["mask"]).sum().item() for item in train_data)
    pos_weight = torch.tensor([total_neg / max(total_pos, 1.0)], device=device)
    print(f"    Water pos_weight={pos_weight.item():.1f}")

    for epoch in range(epochs):
        head.train()
        np.random.shuffle(train_data)
        epoch_loss = 0.0
        for item in train_data:
            emb = item["emb"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).unsqueeze(0).to(device)

            pred = head(emb)
            loss_bce = F.binary_cross_entropy_with_logits(pred, mask, pos_weight=pos_weight)
            loss_dice = dice_loss(pred, mask)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

    return head


def train_cls_head(head, train_data, device, num_classes=7, epochs=30, lr=1e-3):
    """训练分类head — 使用类别权重处理不平衡."""
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    # 类别权重 (逆频率加权)
    all_labels = torch.cat([item["label"].flatten() for item in train_data])
    class_counts = torch.bincount(all_labels, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    print(f"    LandUse class_weights: {class_weights.cpu().numpy().round(2)}")

    for epoch in range(epochs):
        head.train()
        np.random.shuffle(train_data)
        epoch_loss = 0.0
        for item in train_data:
            emb = item["emb"].unsqueeze(0).to(device)
            label = item["label"].unsqueeze(0).to(device)

            pred = head(emb)
            loss = F.cross_entropy(pred, label, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

    return head


def evaluate_cd(head, test_data, device):
    head.eval()
    all_preds = []
    all_masks = []
    per_patch_aucs = []

    with torch.no_grad():
        for item in test_data:
            eb = item["eb"].unsqueeze(0).to(device)
            ea = item["ea"].unsqueeze(0).to(device)
            mask = item["mask"].numpy()

            pred = head(eb, ea)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]

            all_preds.extend(pred_prob.flatten().tolist())
            all_masks.extend(mask.flatten().tolist())

            # Patch-level AUC
            if mask.sum() > 0 and (1 - mask).sum() > 0:
                patch_auc = roc_auc_score(mask.flatten(), pred_prob.flatten())
                per_patch_aucs.append(patch_auc)

    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    global_auc = roc_auc_score(all_masks, all_preds) if len(np.unique(all_masks)) > 1 else 0.0
    mean_patch_auc = np.mean(per_patch_aucs) if per_patch_aucs else 0.0

    # IoU @ threshold=0.5
    pred_binary = (all_preds > 0.5).astype(np.uint8)
    tp = np.logical_and(pred_binary, all_masks).sum()
    fp = np.logical_and(pred_binary, 1 - all_masks).sum()
    fn = np.logical_and(1 - pred_binary, all_masks).sum()
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "global_auc": float(global_auc),
        "mean_patch_auc": float(mean_patch_auc),
        "iou": float(iou),
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall": float(tp / (tp + fn + 1e-8)),
        "f1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
    }


def evaluate_seg(head, test_data, device):
    head.eval()
    all_preds = []
    all_masks = []
    per_patch_aucs = []

    with torch.no_grad():
        for item in test_data:
            emb = item["emb"].unsqueeze(0).to(device)
            mask = item["mask"].numpy()

            pred = head(emb)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]

            all_preds.extend(pred_prob.flatten().tolist())
            all_masks.extend(mask.flatten().tolist())

            if mask.sum() > 0 and (1 - mask).sum() > 0:
                patch_auc = roc_auc_score(mask.flatten(), pred_prob.flatten())
                per_patch_aucs.append(patch_auc)

    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    global_auc = roc_auc_score(all_masks, all_preds) if len(np.unique(all_masks)) > 1 else 0.0
    mean_patch_auc = np.mean(per_patch_aucs) if per_patch_aucs else 0.0

    pred_binary = (all_preds > 0.5).astype(np.uint8)
    tp = np.logical_and(pred_binary, all_masks).sum()
    fp = np.logical_and(pred_binary, 1 - all_masks).sum()
    fn = np.logical_and(1 - pred_binary, all_masks).sum()
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "global_auc": float(global_auc),
        "mean_patch_auc": float(mean_patch_auc),
        "iou": float(iou),
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall": float(tp / (tp + fn + 1e-8)),
        "f1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
    }


def evaluate_cls(head, test_data, device, num_classes=7):
    head.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for item in test_data:
            emb = item["emb"].unsqueeze(0).to(device)
            label = item["label"].numpy()

            pred = head(emb)
            pred_cls = pred.argmax(dim=1).cpu().numpy()[0]

            all_preds.extend(pred_cls.flatten().tolist())
            all_labels.extend(label.flatten().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Pixel accuracy
    pixel_acc = (all_preds == all_labels).mean()

    # Per-class IoU and F1
    ious = []
    f1s = []
    present_classes = np.unique(all_labels)
    for c in range(num_classes):
        pred_c = (all_preds == c)
        label_c = (all_labels == c)
        tp = np.logical_and(pred_c, label_c).sum()
        fp = np.logical_and(pred_c, ~label_c).sum()
        fn = np.logical_and(~pred_c, label_c).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        ious.append(iou)
        f1s.append(f1)

    # Balanced accuracy (macro)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        "pixel_accuracy": float(pixel_acc),
        "balanced_accuracy": float(bacc),
        "f1_macro": float(f1_macro),
        "mean_iou": float(np.mean(ious)),
        "per_class_iou": [float(x) for x in ious],
        "per_class_f1": [float(x) for x in f1s],
    }


def run_fewshot_cd(all_data, k_shot, n_splits, embedding_dim, device):
    """Few-shot 变化检测."""
    n_total = len(all_data)
    if k_shot >= n_total:
        k_shot = max(1, n_total // 2)

    split_results = []
    for split_idx in range(n_splits):
        indices = list(range(n_total))
        np.random.shuffle(indices)
        train_idx = indices[:k_shot]
        test_idx = indices[k_shot:]

        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        head = SimpleCDHead(embedding_dim=embedding_dim, hidden_dim=64)
        head = train_cd_head(head, train_data, device, epochs=50, lr=1e-3)
        result = evaluate_cd(head, test_data, device)
        result["split"] = split_idx
        split_results.append(result)

    return {
        "k_shot": k_shot,
        "global_auc_mean": float(np.mean([r["global_auc"] for r in split_results])),
        "global_auc_std": float(np.std([r["global_auc"] for r in split_results])),
        "iou_mean": float(np.mean([r["iou"] for r in split_results])),
        "iou_std": float(np.std([r["iou"] for r in split_results])),
        "f1_mean": float(np.mean([r["f1"] for r in split_results])),
        "f1_std": float(np.std([r["f1"] for r in split_results])),
        "splits": split_results,
    }


def run_fewshot_water(all_data, k_shot, n_splits, embedding_dim, device):
    """Few-shot 水体检测."""
    n_total = len(all_data)
    if k_shot >= n_total:
        k_shot = max(1, n_total // 2)

    split_results = []
    for split_idx in range(n_splits):
        indices = list(range(n_total))
        np.random.shuffle(indices)
        train_idx = indices[:k_shot]
        test_idx = indices[k_shot:]

        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        head = SimpleSegHead(embedding_dim=embedding_dim, hidden_dim=64)
        head = train_seg_head(head, train_data, device, epochs=50, lr=1e-3)
        result = evaluate_seg(head, test_data, device)
        result["split"] = split_idx
        split_results.append(result)

    return {
        "k_shot": k_shot,
        "global_auc_mean": float(np.mean([r["global_auc"] for r in split_results])),
        "global_auc_std": float(np.std([r["global_auc"] for r in split_results])),
        "iou_mean": float(np.mean([r["iou"] for r in split_results])),
        "iou_std": float(np.std([r["iou"] for r in split_results])),
        "f1_mean": float(np.mean([r["f1"] for r in split_results])),
        "f1_std": float(np.std([r["f1"] for r in split_results])),
        "splits": split_results,
    }


def run_fewshot_landuse(all_data, k_shot, n_splits, embedding_dim, device):
    """Few-shot 土地利用分割."""
    n_total = len(all_data)
    if k_shot >= n_total:
        k_shot = max(1, n_total // 2)

    split_results = []
    for split_idx in range(n_splits):
        indices = list(range(n_total))
        np.random.shuffle(indices)
        train_idx = indices[:k_shot]
        test_idx = indices[k_shot:]

        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        head = SimpleClsHead(embedding_dim=embedding_dim, num_classes=7, hidden_dim=64)
        head = train_cls_head(head, train_data, device, num_classes=7, epochs=50, lr=1e-3)
        result = evaluate_cls(head, test_data, device, num_classes=7)
        result["split"] = split_idx
        split_results.append(result)

    return {
        "k_shot": k_shot,
        "pixel_accuracy_mean": float(np.mean([r["pixel_accuracy"] for r in split_results])),
        "pixel_accuracy_std": float(np.std([r["pixel_accuracy"] for r in split_results])),
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in split_results])),
        "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in split_results])),
        "f1_macro_mean": float(np.mean([r["f1_macro"] for r in split_results])),
        "f1_macro_std": float(np.std([r["f1_macro"] for r in split_results])),
        "mean_iou_mean": float(np.mean([r["mean_iou"] for r in split_results])),
        "mean_iou_std": float(np.std([r["mean_iou"] for r in split_results])),
        "splits": split_results,
    }


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Backbone config")
    parser.add_argument("--checkpoint", required=True, help="Backbone checkpoint")
    parser.add_argument("--embedding-dir", help="预计算embedding目录（如有）")
    parser.add_argument("--tasks", default="cd,water,landuse", help="任务列表")
    parser.add_argument("--k-shots", default="5,10,20,50", help="K-shot列表")
    parser.add_argument("--n-splits", type=int, default=5, help="随机split次数")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    k_shots = [int(x.strip()) for x in args.k_shots.split(",")]
    tasks = [x.strip() for x in args.tasks.split(",")]

    print("=" * 70)
    print(f" 统一 Few-Shot 下游评估")
    print(f" Config: {args.config}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Tasks: {tasks}")
    print(f" K-shots: {k_shots}")
    print(f" N-splits: {args.n_splits}")
    print("=" * 70)

    # 加载 backbone
    print("\n[1/3] 加载 backbone...")
    model, dataset, cfg = load_backbone(args.config, args.checkpoint, device)
    embedding_dim = cfg.model.embedding_dim
    print(f"  Embedding dim: {embedding_dim}")

    # 提取或加载 embedding
    if args.embedding_dir:
        embedding_dir = Path(args.embedding_dir)
        print(f"\n[2/3] 使用预计算 embedding: {embedding_dir}")
    else:
        # 在线提取
        embedding_dir = Path("/workspace/xuannv/data/embeddings") / Path(args.checkpoint).parent.name
        embedding_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[2/3] 提取 embedding 到: {embedding_dir}")
        # 这里简化：假设已有预计算embedding
        # 实际运行时，先运行 extract_embeddings.py
        print("  注意: 未提供预计算embedding，请先运行 extract_embeddings.py")
        return

    # 评估各任务
    all_results = {}

    if "cd" in tasks:
        print("\n[3/3] 评估: 变化检测 (CD)")
        cd_data = load_change_detection_data(embedding_dir)
        if len(cd_data) >= min(k_shots):
            cd_results = []
            for k in k_shots:
                if k >= len(cd_data):
                    continue
                print(f"\n  --- K={k} ---")
                result = run_fewshot_cd(cd_data, k, args.n_splits, embedding_dim, device)
                print(f"    Global AUC: {result['global_auc_mean']:.4f} ± {result['global_auc_std']:.4f}")
                print(f"    IoU: {result['iou_mean']:.4f} ± {result['iou_std']:.4f}")
                print(f"    F1: {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
                cd_results.append(result)
            all_results["change_detection"] = cd_results
        else:
            print(f"  数据不足 ({len(cd_data)} < {min(k_shots)})")

    if "water" in tasks:
        print("\n[3/3] 评估: 水体检测 (Water)")
        water_data = load_water_data(embedding_dir)
        if len(water_data) >= min(k_shots):
            water_results = []
            for k in k_shots:
                if k >= len(water_data):
                    continue
                print(f"\n  --- K={k} ---")
                result = run_fewshot_water(water_data, k, args.n_splits, embedding_dim, device)
                print(f"    Global AUC: {result['global_auc_mean']:.4f} ± {result['global_auc_std']:.4f}")
                print(f"    IoU: {result['iou_mean']:.4f} ± {result['iou_std']:.4f}")
                print(f"    F1: {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
                water_results.append(result)
            all_results["water_detection"] = water_results
        else:
            print(f"  数据不足 ({len(water_data)} < {min(k_shots)})")

    if "landuse" in tasks:
        print("\n[3/3] 评估: 土地利用分割 (LandUse)")
        landuse_data = load_landuse_data(embedding_dir)
        if len(landuse_data) >= min(k_shots):
            landuse_results = []
            for k in k_shots:
                if k >= len(landuse_data):
                    continue
                print(f"\n  --- K={k} ---")
                result = run_fewshot_landuse(landuse_data, k, args.n_splits, embedding_dim, device)
                print(f"    Pixel Acc: {result['pixel_accuracy_mean']:.4f} ± {result['pixel_accuracy_std']:.4f}")
                print(f"    Balanced Acc: {result['balanced_accuracy_mean']:.4f} ± {result['balanced_accuracy_std']:.4f}")
                print(f"    F1 Macro: {result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}")
                print(f"    mIoU: {result['mean_iou_mean']:.4f} ± {result['mean_iou_std']:.4f}")
                landuse_results.append(result)
            all_results["landuse_segmentation"] = landuse_results
        else:
            print(f"  数据不足 ({len(landuse_data)} < {min(k_shots)})")

    # 保存结果
    output_data = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "tasks": tasks,
        "k_shots": k_shots,
        "n_splits": args.n_splits,
        "results": all_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f" 评估完成！结果保存: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
