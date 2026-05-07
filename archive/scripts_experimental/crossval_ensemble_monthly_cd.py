#!/usr/bin/env python3
"""
月度 Embedding + Deep Ensemble 的 5-Fold Patch-Level Cross-Validation.
每一 fold 训练多个异构 head，验证时做 late fusion (mean prob).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

gpu_id = "6"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        gpu_id = sys.argv[i + 1]
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
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
    ChangeDetectionHead,
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
    1: 0,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
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
                logits = head(eb, ea)
                probs = torch.sigmoid(logits).max(dim=1)[0]
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
    f1 = f1_score(all_targets, all_probs > 0.5, zero_division=0)
    return {"auc": auc, "ba": ba, "f1": f1}


def train_single_head(head_name, head, train_loader, val_loader, use_mc, ohem):
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
            if use_mc:
                cat_mask = batch["category_mask"].to(DEVICE)
                mc_target = mc_target_from_category_mask(cat_mask, num_classes=MultiClassChangeDetectionHead.NUM_CLASSES)
                logits = head(eb, ea)
                if ohem:
                    loss_focal = multiclass_ohem_focal_bce_loss(logits, mc_target, ohem_ratio=0.25, alpha=0.5, gamma=2.0)
                else:
                    class_weights = torch.tensor([0.8, 1.0, 1.0], device=DEVICE)
                    loss_focal = multiclass_focal_bce_loss(logits, mc_target, alpha=0.5, gamma=2.0, class_weights=class_weights)
                loss_dice = multiclass_dice_loss(logits, mc_target)
                loss = loss_focal + 0.5 * loss_dice
            else:
                mask = batch["mask"].to(DEVICE)
                logits = head(eb, ea).squeeze(1)
                if ohem:
                    loss_focal = ohem_focal_bce_loss(logits, mask, ohem_ratio=0.25, alpha=0.5, gamma=2.0)
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
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = head.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return best_auc, head


def evaluate_ensemble(heads, use_mcs, loader, member_aucs=None):
    """Late fusion: average probabilities from all heads."""
    for head in heads:
        head.eval()
    all_probs_mean = []
    all_probs_weighted = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            eb = batch["emb_before"].to(DEVICE)
            ea = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            probs_list = []
            for head, use_mc in zip(heads, use_mcs):
                if use_mc:
                    logits = head(eb, ea)
                    probs = torch.sigmoid(logits).max(dim=1)[0]
                else:
                    logits = head(eb, ea).squeeze(1)
                    probs = torch.sigmoid(logits)
                probs_list.append(probs)
            stacked = torch.stack(probs_list)
            all_probs_mean.append(stacked.mean(dim=0).cpu().numpy().reshape(-1))
            if member_aucs is not None:
                weights = torch.tensor([max(0.0, a - 0.5) for a in member_aucs], device=stacked.device)
                weights = weights / (weights.sum() + 1e-8)
                weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
                all_probs_weighted.append(weighted.cpu().numpy().reshape(-1))
            all_targets.append(mask.cpu().numpy().reshape(-1))
    all_targets = np.concatenate(all_targets)
    results = {}
    for name, probs in [("mean", np.concatenate(all_probs_mean))]:
        auc = roc_auc_score(all_targets, probs)
        ba = balanced_accuracy_score(all_targets, (probs > 0.5).astype(int))
        f1 = f1_score(all_targets, (probs > 0.5).astype(int), zero_division=0)
        results[name] = {"auc": auc, "ba": ba, "f1": f1}
    if member_aucs is not None and all_probs_weighted:
        probs = np.concatenate(all_probs_weighted)
        auc = roc_auc_score(all_targets, probs)
        ba = balanced_accuracy_score(all_targets, (probs > 0.5).astype(int))
        f1 = f1_score(all_targets, (probs > 0.5).astype(int), zero_division=0)
        results["weighted"] = {"auc": auc, "ba": ba, "f1": f1}
    return results


def train_fold(train_records, val_records, fold_id):
    print(f"\n--- Fold {fold_id+1}/5 ---")
    augment = EmbeddingAugment(noise_std=0.02, channel_dropout=0.1)
    no_aug = None

    # Data loaders
    train_ds_binary = MonthlyPatchDataset(train_records, list(range(len(train_records))), augment=augment, use_category_mask=False)
    train_ds_mc = MonthlyPatchDataset(train_records, list(range(len(train_records))), augment=augment, use_category_mask=True)
    val_ds = MonthlyPatchDataset(val_records, list(range(len(val_records))), augment=None, use_category_mask=False)
    val_mc_ds = MonthlyPatchDataset(val_records, list(range(len(val_records))), augment=None, use_category_mask=True)

    train_binary_loader = DataLoader(train_ds_binary, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    train_mc_loader = DataLoader(train_ds_mc, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    val_mc_loader = DataLoader(val_mc_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Define ensemble members for this fold
    members = [
        ("E1_baseline", ChangeDetectionHead(embedding_dim=128, hidden_dim=64).to(DEVICE), train_binary_loader, val_loader, False, False),
        ("E2_v2", ChangeDetectionHeadV2(embedding_dim=128, hidden_dim=64).to(DEVICE), train_binary_loader, val_loader, False, False),
        ("E3_v3_aug", ChangeDetectionHeadV3(embedding_dim=128, hidden_dim=64, dropout=0.4).to(DEVICE), train_binary_loader, val_loader, False, False),
        ("E4_v3_ohem", ChangeDetectionHeadV3(embedding_dim=128, hidden_dim=64, dropout=0.4).to(DEVICE), train_binary_loader, val_loader, False, True),
        ("E5_mc", MultiClassChangeDetectionHead(embedding_dim=128, hidden_dim=64, dropout=0.4).to(DEVICE), train_mc_loader, val_mc_loader, True, False),
    ]

    trained_heads = []
    use_mcs = []
    member_aucs = {}
    for name, head, tr_loader, vl_loader, use_mc, ohem in members:
        auc, trained_head = train_single_head(name, head, tr_loader, vl_loader, use_mc, ohem)
        member_aucs[name] = auc
        trained_heads.append(trained_head)
        use_mcs.append(use_mc)
        print(f"  {name} Best AUC: {auc:.4f}")

    # Ensemble evaluations: all heads
    ensemble_metrics = evaluate_ensemble(trained_heads, use_mcs, val_loader, member_aucs=list(member_aucs.values()))
    for k, v in ensemble_metrics.items():
        print(f"  Ensemble ({k}) AUC: {v['auc']:.4f}")

    # Ensemble evaluations: binary heads only (exclude MultiClass)
    binary_heads = [h for h, mc in zip(trained_heads, use_mcs) if not mc]
    binary_use_mcs = [mc for mc in use_mcs if not mc]
    binary_aucs = [a for a, mc in zip(list(member_aucs.values()), use_mcs) if not mc]
    binary_ensemble = evaluate_ensemble(binary_heads, binary_use_mcs, val_loader, member_aucs=binary_aucs)
    for k, v in binary_ensemble.items():
        print(f"  Ensemble-binary ({k}) AUC: {v['auc']:.4f}")
    ensemble_metrics["binary"] = binary_ensemble["mean"]
    if "weighted" in binary_ensemble:
        ensemble_metrics["binary_weighted"] = binary_ensemble["weighted"]

    return member_aucs, ensemble_metrics


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    records = build_records(use_category_mask=True)
    print(f"Total records: {len(records)}")

    periods = [r["period"] for r in records]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    ensemble_aucs = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(records)), periods)):
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        member_aucs, ensemble_metrics = train_fold(train_recs, val_recs, fold_id)
        fold_results.append({"member_aucs": member_aucs, "ensemble": ensemble_metrics})

    print("\n" + "=" * 70)
    print("  Deep Ensemble 5-Fold Cross-Validation Results")
    print("=" * 70)

    # Show per-member averages
    member_names = list(fold_results[0]["member_aucs"].keys())
    for name in member_names:
        vals = [f["member_aucs"][name] for f in fold_results]
        print(f"  {name} Mean: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Show ensemble results for each strategy
    for strategy in ["mean", "weighted", "binary", "binary_weighted"]:
        aucs = [f["ensemble"].get(strategy, {}).get("auc") for f in fold_results]
        aucs = [a for a in aucs if a is not None]
        if aucs:
            print(f"  Ensemble-{strategy} Mean: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    save_path = OUTPUT_DIR / "crossval_results_ensemble.json"
    with open(save_path, "w") as f:
        json.dump({
            "fold_results": fold_results,
        }, f, indent=2)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()
