#!/usr/bin/env python3
"""
对照实验: 用原版 ChangeDetectionHead 在月度 embedding 上训练.
与 train_monthly_cd_head.py 完全一致，只是 head 换成原版.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
)
from src.models.heads import (
    ChangeDetectionHead,
    dice_loss,
    focal_bce_loss,
)

EMBEDDING_DIR = Path("/workspace/outputs/aef_qwen_v2/monthly_embeddings_2025")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2/monthly_cd_head_baseline")

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
DEVICE = torch.device("npu:0" if torch.npu.is_available() else "cpu")

PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
    "2025-all": ("2025-04", "2025-10"),
}


def boundary_aware_loss(pred, target):
    pred_sig = torch.sigmoid(pred)
    neg_mask = (target == 0).float()
    neg_count = neg_mask.sum() + 1e-8
    neg_mean = (pred_sig * neg_mask).sum() / neg_count
    return F.relu(neg_mean - 0.1) ** 2


class MonthlyEmbeddingDataset(Dataset):
    def __init__(self, records, indices):
        self.records = records
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rec = self.records[self.indices[idx]]
        emb_before = np.load(rec["before_path"])
        emb_after = np.load(rec["after_path"])
        mask = rec["mask"]
        return {
            "emb_before": torch.from_numpy(emb_before).float(),
            "emb_after": torch.from_numpy(emb_after).float(),
            "mask": torch.from_numpy(mask).long(),
        }


def collate_fn(batch):
    return {
        "emb_before": torch.stack([b["emb_before"] for b in batch]),
        "emb_after": torch.stack([b["emb_after"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


def build_records():
    annotated = get_annotated_patches()
    records = []
    for pid in annotated:
        period = get_period_for_patch(pid)
        if period is None or period not in PERIOD_TO_MONTHS:
            continue
        bm, am = PERIOD_TO_MONTHS[period]
        bpath = EMBEDDING_DIR / f"{pid}_{bm}.npy"
        apath = EMBEDDING_DIR / f"{pid}_{am}.npy"
        if not bpath.exists() or not apath.exists():
            continue
        mask, _ = rasterize_patch_changes(pid, grid_size=64)
        if mask.sum() == 0:
            continue
        records.append({
            "patch_id": pid,
            "period": period,
            "before_path": str(bpath),
            "after_path": str(apath),
            "mask": mask.astype(np.int32),
        })
    return records


def evaluate(head, loader):
    head.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = head(emb_b, emb_a).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            targets = mask.cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_targets.append(targets)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    auc = roc_auc_score(all_targets, all_probs)
    preds = (all_probs > 0.5).astype(int)
    ba = balanced_accuracy_score(all_targets, preds)
    f1 = f1_score(all_targets, preds, zero_division=0)
    return {"auc": auc, "ba": ba, "f1": f1}


def main():
    print("=" * 70)
    print("  月度 Embedding + 原版 ChangeDetectionHead (Baseline)")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    periods = [r["period"] for r in records]
    train_idx, val_idx = train_test_split(
        list(range(len(records))), test_size=0.2, random_state=42, stratify=periods
    )
    print(f"Train={len(train_idx)}, Val={len(val_idx)}")

    train_ds = MonthlyEmbeddingDataset(records, train_idx)
    val_ds = MonthlyEmbeddingDataset(records, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    head = ChangeDetectionHead(embedding_dim=128, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=15, verbose=True)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        epoch_losses = []
        for batch in train_loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = head(emb_b, emb_a).squeeze(1)
            loss = focal_bce_loss(logits, mask, alpha=0.5, gamma=2.0) + 0.5 * dice_loss(logits, mask) + 0.3 * boundary_aware_loss(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        metrics = evaluate(head, val_loader)
        scheduler.step(metrics["auc"])

        if epoch % 10 == 0 or metrics["auc"] > best_auc:
            print(f"Epoch {epoch:03d} | Loss={np.mean(epoch_losses):.4f} | Val AUC={metrics['auc']:.4f} BA={metrics['ba']:.4f} F1={metrics['f1']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = head.state_dict().copy()
            patience_counter = 0
            torch.save({"epoch": epoch, "cd_head": head.state_dict(), "metrics": metrics}, OUTPUT_DIR / "best_baseline.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best Val AUC: {best_auc:.4f}")
    if best_state is not None:
        head.load_state_dict(best_state)
        torch.save({"cd_head": head.state_dict(), "config": {"embedding_dim": 128, "hidden_dim": 64}}, OUTPUT_DIR / "baseline_head.pt")


if __name__ == "__main__":
    main()
