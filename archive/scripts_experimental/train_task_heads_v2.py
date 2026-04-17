#!/usr/bin/env python3
"""冻结 backbone，训练轻量级任务头 v2 — 改进正则化减少全局高偏置.

用法:
    CUDA_VISIBLE_DEVICES=2 python scripts/train_task_heads_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, "/workspace/xuannv")

from src.models.heads import ChangeDetectionHead
from scripts.train_task_heads import EmbeddingChangeDataset, CACHE_DIR, evaluate_cd_head, build_masks, DEVICE, OUTPUT_DIR

BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 200
PATIENCE = 30


def focal_bce_v2(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, gamma: float = 1.0) -> torch.Tensor:
    """Focal BCE with milder down-weighting of easy negatives."""
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_sig = torch.sigmoid(pred)
    pred_flat = pred_sig.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def boundary_aware_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """惩罚在负样本上的平均高响应（防止全局高偏置）."""
    pred_sig = torch.sigmoid(pred)
    neg_mask = (target == 0).float()
    neg_count = neg_mask.sum() + 1e-8
    neg_mean = (pred_sig * neg_mask).sum() / neg_count
    # 惩罚负样本上的高概率
    return F.relu(neg_mean - 0.1) ** 2


def main():
    print(f"[TrainV2] Using device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / "embeddings.json"
    before_emb = np.load(CACHE_DIR / "before_embeddings.npy")
    after_emb = np.load(CACHE_DIR / "after_embeddings.npy")
    with open(cache_path) as f:
        meta = json.load(f)

    binary_masks, cat_masks = build_masks(meta["records"])
    print(f"[Data] Masks: positive ratio = {binary_masks.mean():.4f}")

    N = len(meta["records"])
    train_idx, val_idx = train_test_split(
        list(range(N)), test_size=0.2, random_state=42, stratify=binary_masks.max(axis=(1, 2))
    )
    print(f"[Data] Train={len(train_idx)}, Val={len(val_idx)}")

    train_ds = EmbeddingChangeDataset(before_emb, after_emb, binary_masks, cat_masks, train_idx)
    val_ds = EmbeddingChangeDataset(before_emb, after_emb, binary_masks, cat_masks, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cd_head = ChangeDetectionHead(128, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(cd_head.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        cd_head.train()
        epoch_losses = []
        for batch in train_loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = cd_head(emb_b, emb_a).squeeze(1)
            loss_focal = focal_bce_v2(logits, mask)
            loss_dice = dice_loss(logits, mask)
            loss_neg = boundary_aware_loss(logits, mask)
            loss = loss_focal + 0.3 * loss_dice + 0.5 * loss_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        metrics = evaluate_cd_head(cd_head, val_loader, DEVICE)
        scheduler.step(metrics["auc"])

        avg_loss = np.mean(epoch_losses)
        print(
            f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | "
            f"Val AUC={metrics['auc']:.4f} BA={metrics['ba']:.4f} F1={metrics['f1']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = cd_head.state_dict()
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "cd_head": cd_head.state_dict(),
                "metrics": metrics,
            }, OUTPUT_DIR / "best_cd_head_v2.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[TrainV2] Early stopping at epoch {epoch}")
                break

    print(f"[TrainV2] Best Val AUC: {best_auc:.4f}")
    if best_state is not None:
        cd_head.load_state_dict(best_state)

    # 保存统一的 task_heads.pt
    torch.save({
        "cd_head": cd_head.state_dict(),
        "config": {"embedding_dim": 128, "hidden_dim": 64},
    }, OUTPUT_DIR / "task_heads.pt")
    print(f"[TrainV2] Saved task heads to {OUTPUT_DIR / 'task_heads.pt'}")


if __name__ == "__main__":
    main()
