#!/usr/bin/env python3
"""
月度 Embedding + ChangeDetectionHead 的 5-Fold Patch-Level Cross-Validation.
支持 V2/V3/MultiClass Head，支持数据增强和 OHEM。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
torch.set_num_threads(4)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
    rasterize_patch_categories,
)
from src.models.heads import (
    ChangeDetectionHeadV2,
    ChangeDetectionHeadV3,
    MultiClassChangeDetectionHead,
    dice_loss,
    focal_bce_loss,
    multiclass_dice_loss,
    multiclass_focal_bce_loss,
)

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

CATEGORY_TO_MC = {
    0: -1,
    1: 0,   # construction
    2: 1,   # demolition
    3: 2,   # road -> land_conversion
    4: 2,   # water_change -> land_conversion
    5: 2,   # farmland -> land_conversion
}


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


def multiclass_ohem_focal_bce_loss(logits, target, ohem_ratio=0.25, alpha=0.5, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce

    C = logits.size(1)
    total_loss = 0.0
    for c in range(C):
        focal_c = focal[:, c]
        target_c = target[:, c]
        pos_mask = target_c > 0.5
        neg_mask = ~pos_mask

        pos_loss = focal_c[pos_mask].mean() if pos_mask.any() else 0.0
        neg_loss = focal_c[neg_mask]
        if neg_loss.numel() > 0:
            k = max(1, int(neg_loss.numel() * ohem_ratio))
            hard_neg_loss = torch.topk(neg_loss, k).values.mean()
        else:
            hard_neg_loss = 0.0
        total_loss += pos_loss + hard_neg_loss

    return total_loss / C


class EmbeddingAugment:
    def __init__(self, noise_std=0.02, channel_dropout=0.1):
        self.noise_std = noise_std
        self.channel_dropout = channel_dropout

    def __call__(self, emb_before, emb_after, mask, extra_mask=None):
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        rotate = random.random() < 0.5
        if rotate:
            k = random.randint(1, 3)

        if flip_h:
            emb_before = torch.flip(emb_before, dims=[-1])
            emb_after = torch.flip(emb_after, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
            if extra_mask is not None:
                extra_mask = torch.flip(extra_mask, dims=[-1])
        if flip_v:
            emb_before = torch.flip(emb_before, dims=[-2])
            emb_after = torch.flip(emb_after, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
            if extra_mask is not None:
                extra_mask = torch.flip(extra_mask, dims=[-2])
        if rotate:
            emb_before = torch.rot90(emb_before, k=k, dims=[-2, -1])
            emb_after = torch.rot90(emb_after, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
            if extra_mask is not None:
                extra_mask = torch.rot90(extra_mask, k=k, dims=[-2, -1])

        if self.noise_std > 0:
            emb_before = emb_before + torch.randn_like(emb_before) * self.noise_std
            emb_after = emb_after + torch.randn_like(emb_after) * self.noise_std

        if self.channel_dropout > 0:
            ch_mask = torch.rand(emb_before.size(0), device=emb_before.device) > self.channel_dropout
            emb_before = emb_before * ch_mask.view(-1, 1, 1)
            emb_after = emb_after * ch_mask.view(-1, 1, 1)

        if extra_mask is not None:
            return emb_before, emb_after, mask, extra_mask
        return emb_before, emb_after, mask


class MonthlyPatchDataset(Dataset):
    def __init__(self, records, indices, augment=None, use_category_mask=False):
        self.records = records
        self.indices = indices
        self.augment = augment
        self.use_category_mask = use_category_mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rec = self.records[self.indices[idx]]
        emb_b = np.load(rec["before_path"])
        emb_a = np.load(rec["after_path"])
        mask = torch.from_numpy(rec["mask"]).long()
        emb_b = torch.from_numpy(emb_b).float()
        emb_a = torch.from_numpy(emb_a).float()

        cat_mask = None
        if self.use_category_mask and "category_mask" in rec:
            cat_mask = torch.from_numpy(rec["category_mask"]).long()

        if self.augment is not None:
            if cat_mask is not None:
                emb_b, emb_a, mask, cat_mask = self.augment(emb_b, emb_a, mask, cat_mask)
            else:
                emb_b, emb_a, mask = self.augment(emb_b, emb_a, mask)

        out = {
            "emb_before": emb_b,
            "emb_after": emb_a,
            "mask": mask,
        }
        if cat_mask is not None:
            out["category_mask"] = cat_mask
        return out


def collate_fn(batch):
    out = {
        "emb_before": torch.stack([b["emb_before"] for b in batch]),
        "emb_after": torch.stack([b["emb_after"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }
    if "category_mask" in batch[0]:
        out["category_mask"] = torch.stack([b["category_mask"] for b in batch])
    return out


def build_records(use_category_mask=False):
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
        rec = {
            "patch_id": pid,
            "period": period,
            "before_path": str(bpath),
            "after_path": str(apath),
            "mask": mask.astype(np.int32),
        }
        if use_category_mask:
            cat_mask, _ = rasterize_patch_categories(pid, grid_size=64)
            rec["category_mask"] = cat_mask.astype(np.int32)
        records.append(rec)
    return records


def mc_target_from_category_mask(cat_mask, num_classes=3):
    B, H, W = cat_mask.shape
    target = torch.zeros(B, num_classes, H, W, device=cat_mask.device, dtype=torch.float32)
    for b in range(B):
        for cat_idx, mc_idx in CATEGORY_TO_MC.items():
            if cat_idx == 0 or mc_idx < 0:
                continue
            target[b, mc_idx, cat_mask[b] == cat_idx] = 1.0
    return target


def evaluate_head(head, loader, use_mc=False):
    head.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            eb = batch["emb_before"].to(DEVICE)
            ea = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            if use_mc:
                logits = head(eb, ea)  # [B, 3, H, W]
                probs = torch.sigmoid(logits).max(dim=1)[0]  # max-rule
            else:
                logits = head(eb, ea).squeeze(1)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy().reshape(-1))
            all_targets.append(mask.cpu().numpy().reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    auc = roc_auc_score(all_targets, all_probs)
    preds = (all_probs > 0.5).astype(int)
    ba = balanced_accuracy_score(all_targets, preds)
    f1 = f1_score(all_targets, preds, zero_division=0)
    return {"auc": auc, "ba": ba, "f1": f1}


def create_head(head_type, hidden_dim, dropout):
    if head_type == "v2":
        return ChangeDetectionHeadV2(embedding_dim=128, hidden_dim=hidden_dim).to(DEVICE)
    elif head_type == "v3":
        return ChangeDetectionHeadV3(embedding_dim=128, hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    elif head_type == "mc":
        return MultiClassChangeDetectionHead(embedding_dim=128, hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


def train_fold(train_records, val_records, fold_id, args):
    print(f"\n--- Fold {fold_id+1}/5 ---")
    use_mc = args.head_type == "mc"
    augment = EmbeddingAugment(noise_std=args.noise_std, channel_dropout=args.channel_dropout)
    train_ds = MonthlyPatchDataset(train_records, list(range(len(train_records))), augment=augment, use_category_mask=use_mc)
    val_ds = MonthlyPatchDataset(val_records, list(range(len(val_records))), augment=None, use_category_mask=use_mc)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    head = create_head(args.head_type, args.hidden_dim, args.dropout)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        for batch in train_loader:
            eb = batch["emb_before"].to(DEVICE)
            ea = batch["emb_after"].to(DEVICE)

            if use_mc:
                cat_mask = batch["category_mask"].to(DEVICE)
                mc_target = mc_target_from_category_mask(cat_mask, num_classes=MultiClassChangeDetectionHead.NUM_CLASSES)
                logits = head(eb, ea)
                if args.ohem:
                    loss_focal = multiclass_ohem_focal_bce_loss(logits, mc_target, ohem_ratio=args.ohem_ratio, alpha=0.5, gamma=2.0)
                else:
                    class_weights = torch.tensor([0.8, 1.0, 1.0], device=DEVICE)
                    loss_focal = multiclass_focal_bce_loss(logits, mc_target, alpha=0.5, gamma=2.0, class_weights=class_weights)
                loss_dice = multiclass_dice_loss(logits, mc_target)
                loss = loss_focal + 0.5 * loss_dice
            else:
                mask = batch["mask"].to(DEVICE)
                logits = head(eb, ea).squeeze(1)
                if args.ohem:
                    loss_focal = ohem_focal_bce_loss(logits, mask, ohem_ratio=args.ohem_ratio, alpha=0.5, gamma=2.0)
                else:
                    loss_focal = focal_bce_loss(logits, mask, alpha=0.5, gamma=2.0)
                loss_dice = dice_loss(logits, mask)
                loss_neg = boundary_aware_loss(logits, mask)
                loss = loss_focal + 0.5 * loss_dice + 0.3 * loss_neg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        metrics = evaluate_head(head, val_loader, use_mc=use_mc)
        if epoch % 10 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch:03d} | Val AUC={metrics['auc']:.4f} BA={metrics['ba']:.4f} F1={metrics['f1']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = head.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    print(f"Fold {fold_id+1} Best Val AUC: {best_auc:.4f}")
    return best_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_type", type=str, default="v2", choices=["v2", "v3", "mc"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--channel_dropout", type=float, default=0.1)
    parser.add_argument("--ohem", action="store_true")
    parser.add_argument("--ohem_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. npu:0")
    return parser.parse_args()


def main():
    global DEVICE
    args = parse_args()
    if args.device:
        DEVICE = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[Config] head_type={args.head_type}, hidden_dim={args.hidden_dim}, dropout={args.dropout}, ohem={args.ohem}")

    records = build_records(use_category_mask=(args.head_type == "mc"))
    print(f"Total records: {len(records)}")

    periods = [r["period"] for r in records]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_aucs = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(records)), periods)):
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        auc = train_fold(train_recs, val_recs, fold_id, args)
        fold_aucs.append(auc)

    print("\n" + "=" * 60)
    print("  5-Fold Cross-Validation Results")
    print("=" * 60)
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Median AUC: {np.median(fold_aucs):.4f}")

    out_name = args.head_type
    if args.ohem:
        out_name += "_ohem"
    save_path = OUTPUT_DIR / f"crossval_results_{out_name}.json"
    with open(save_path, "w") as f:
        json.dump({
            "fold_aucs": fold_aucs,
            "mean": float(np.mean(fold_aucs)),
            "std": float(np.std(fold_aucs)),
            "median": float(np.median(fold_aucs)),
            "args": vars(args),
        }, f, indent=2)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()
