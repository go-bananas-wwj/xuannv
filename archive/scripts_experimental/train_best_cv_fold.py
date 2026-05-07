#!/usr/bin/env python3
"""Train and save the best CV fold (Fold 3 from V3+OHEM, Val AUC 0.7259)."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
)
from src.models.heads import ChangeDetectionHeadV3, dice_loss, focal_bce_loss

EMBEDDING_DIR = Path("/workspace/outputs/aef_qwen_v2/monthly_embeddings_2025")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2/monthly_cd_head")
DEVICE = torch.device("npu:0" if torch.npu.is_available() else "cpu")

PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 120
PATIENCE = 30


def boundary_aware_loss(pred, target):
    pred_sig = torch.sigmoid(pred)
    neg_mask = (target == 0).float()
    neg_count = neg_mask.sum() + 1e-8
    neg_mean = (pred_sig * neg_mask).sum() / neg_count
    return F.relu(neg_mean - 0.1) ** 2


def ohem_focal_bce_loss(logits, target, ohem_ratio=0.25, alpha=0.5, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    pos_mask = target > 0.5
    neg_mask = ~pos_mask
    pos_loss = focal[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=logits.device)
    neg_loss = focal[neg_mask]
    if neg_loss.numel() > 0:
        k = max(1, int(neg_loss.numel() * ohem_ratio))
        hard_neg_loss = torch.topk(neg_loss, k).values.mean()
    else:
        hard_neg_loss = torch.tensor(0.0, device=logits.device)
    return pos_loss + hard_neg_loss


class EmbeddingAugment:
    def __init__(self, noise_std=0.02, channel_dropout=0.1):
        self.noise_std = noise_std
        self.channel_dropout = channel_dropout

    def __call__(self, emb_before, emb_after, mask):
        if random.random() < 0.5:
            emb_before = torch.flip(emb_before, dims=[-1])
            emb_after = torch.flip(emb_after, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < 0.5:
            emb_before = torch.flip(emb_before, dims=[-2])
            emb_after = torch.flip(emb_after, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if random.random() < 0.5:
            k = random.randint(1, 3)
            emb_before = torch.rot90(emb_before, k=k, dims=[-2, -1])
            emb_after = torch.rot90(emb_after, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
        if self.noise_std > 0:
            emb_before = emb_before + torch.randn_like(emb_before) * self.noise_std
            emb_after = emb_after + torch.randn_like(emb_after) * self.noise_std
        if self.channel_dropout > 0:
            ch_mask = torch.rand(emb_before.size(0), device=emb_before.device) > self.channel_dropout
            emb_before = emb_before * ch_mask.view(-1, 1, 1)
            emb_after = emb_after * ch_mask.view(-1, 1, 1)
        return emb_before, emb_after, mask


class MonthlyPatchDataset(Dataset):
    def __init__(self, records, indices, augment=None):
        self.records = records
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rec = self.records[self.indices[idx]]
        emb_b = np.load(rec["before_path"])
        emb_a = np.load(rec["after_path"])
        mask = torch.from_numpy(rec["mask"]).long()
        emb_b = torch.from_numpy(emb_b).float()
        emb_a = torch.from_numpy(emb_a).float()
        if self.augment is not None:
            emb_b, emb_a, mask = self.augment(emb_b, emb_a, mask)
        return {"emb_before": emb_b, "emb_after": emb_a, "mask": mask}


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


def evaluate_head(head, loader):
    head.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            eb = batch["emb_before"].to(DEVICE)
            ea = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = head(eb, ea).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            targets = mask.cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_targets.append(targets)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    auc = roc_auc_score(all_targets, all_probs)
    ba = balanced_accuracy_score(all_targets, (all_probs > 0.5).astype(int))
    f1 = f1_score(all_targets, (all_probs > 0.5).astype(int), zero_division=0)
    return {"auc": auc, "ba": ba, "f1": f1}


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    records = build_records()
    train_idx = np.load("/tmp/fold3_train_idx.npy").tolist()
    val_idx = np.load("/tmp/fold3_val_idx.npy").tolist()

    augment = EmbeddingAugment(noise_std=0.02, channel_dropout=0.1)
    train_ds = MonthlyPatchDataset(records, train_idx, augment=augment)
    val_ds = MonthlyPatchDataset(records, val_idx, augment=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    head = ChangeDetectionHeadV3(embedding_dim=128, hidden_dim=64, dropout=0.4).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        for batch in train_loader:
            eb = batch["emb_before"].to(DEVICE)
            ea = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = head(eb, ea).squeeze(1)
            loss_focal = ohem_focal_bce_loss(logits, mask, ohem_ratio=0.25, alpha=0.5, gamma=2.0)
            loss_dice = dice_loss(logits, mask)
            loss_neg = boundary_aware_loss(logits, mask)
            loss = loss_focal + 0.5 * loss_dice + 0.3 * loss_neg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        metrics = evaluate_head(head, val_loader)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = head.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    print(f"Best Val AUC: {best_auc:.4f}")
    if best_state is not None:
        head.load_state_dict(best_state)
        torch.save({
            "cd_head": head.state_dict(),
            "metrics": {"auc": best_auc},
            "config": {"embedding_dim": 128, "hidden_dim": 64, "dropout": 0.4},
        }, OUTPUT_DIR / "best_cv_fold3_head.pt")
        print("Saved to best_cv_fold3_head.pt")


if __name__ == "__main__":
    main()
