#!/usr/bin/env python3
"""
Partial Backbone Fine-tuning for Change Detection with Hinge Margin Loss.

解冻最后 N 层 STP blocks + bottleneck，直接优化月度 embedding 的判别性。
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
)
from src.config import load_config
from src.data.dataset import HarbinPatchDataset
from src.models.model import AEFModel

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/epoch_499.pt"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2/backbone_finetune")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}

MONTHLY_WINDOWS = {
    "2025-04": (1743465600000.0, 1746057600000.0),
    "2025-06": (1748736000000.0, 1751328000000.0),
    "2025-08": (1754006400000.0, 1756684800000.0),
    "2025-09": (1756684800000.0, 1759267200000.0),
    "2025-10": (1759267200000.0, 1761945600000.0),
}


def set_requires_grad(model: torch.nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad


def load_model(unfreeze_stp_last_n: int = 2, unfreeze_bottleneck: bool = True):
    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Freeze everything first
    set_requires_grad(model, False)

    # Unfreeze last N STP blocks
    total_stp = len(model.stp_blocks)
    for i in range(max(0, total_stp - unfreeze_stp_last_n), total_stp):
        set_requires_grad(model.stp_blocks[i], True)
        print(f"[Unfreeze] stp_blocks[{i}]")

    # Unfreeze bottleneck
    if unfreeze_bottleneck:
        set_requires_grad(model.bottleneck, True)
        print("[Unfreeze] bottleneck")

    model.train()
    return model


def extract_embedding_map(model, dataset, patch_idx, window_start_ms, window_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(window_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(window_end_ms, dtype=torch.float64)
    batch_dev = {
        k: (v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    output = model(
        source_frames=batch_dev["source_frames"],
        source_timestamps_ms=batch_dev["source_timestamps_ms"],
        source_frame_mask=batch_dev["source_frame_mask"],
        source_input_mask=batch_dev["source_input_mask"],
        source_type_ids=batch_dev["source_type_ids"],
        valid_start_ms=batch_dev["valid_start_ms"],
        valid_end_ms=batch_dev["valid_end_ms"],
        target_relative_time=batch_dev["target_relative_time"],
        target_metadata=batch_dev["target_metadata"],
    )
    emb = output.embedding_map  # [1, D, H, W]
    emb = F.normalize(emb, p=2, dim=1)
    return emb


def hinge_margin_loss(emb_before, emb_after, mask, margin_pos=1.0, margin_neg=0.3):
    """
    emb: [B, D, H, W], L2 normalized
    mask: [B, H, W] binary change mask
    """
    distance = torch.norm(emb_before - emb_after, p=2, dim=1)  # [B, H, W]
    mask_f = mask.float()

    # Positive (change): encourage distance > margin_pos
    loss_pos = (mask_f * F.relu(margin_pos - distance) ** 2).sum() / (mask_f.sum() + 1e-8)

    # Negative (no-change): encourage distance < margin_neg
    loss_neg = ((1 - mask_f) * F.relu(distance - margin_neg) ** 2).sum() / ((1 - mask_f).sum() + 1e-8)

    return loss_pos + loss_neg, distance


class ChangeEmbeddingDataset(Dataset):
    def __init__(self, records, model, dataset):
        self.records = records
        self.model = model
        self.base_dataset = dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        pidx = self.base_dataset.patches.index(rec["patch_id"])
        ws_b, we_b = MONTHLY_WINDOWS[rec["before_month"]]
        ws_a, we_a = MONTHLY_WINDOWS[rec["after_month"]]

        with torch.no_grad():
            emb_b = extract_embedding_map(self.model, self.base_dataset, pidx, ws_b, we_b).squeeze(0).cpu()
            emb_a = extract_embedding_map(self.model, self.base_dataset, pidx, ws_a, we_a).squeeze(0).cpu()

        mask = torch.from_numpy(rec["mask"]).long()
        return {
            "emb_before": emb_b,
            "emb_after": emb_a,
            "mask": mask,
        }


def collate_fn(batch):
    return {
        "emb_before": torch.stack([b["emb_before"] for b in batch]),
        "emb_after": torch.stack([b["emb_after"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


def evaluate_distance(emb_before, emb_after, mask):
    with torch.no_grad():
        distance = torch.norm(emb_before - emb_after, p=2, dim=1)  # [B, H, W]
        # Use distance as score for AUC (higher = more change)
        probs = distance.cpu().numpy().reshape(-1)
        targets = mask.cpu().numpy().reshape(-1)
    return roc_auc_score(targets, probs)


def build_records():
    annotated = get_annotated_patches()
    records = []
    for pid in annotated:
        period = get_period_for_patch(pid)
        if period is None or period not in PERIOD_TO_MONTHS:
            continue
        before_m, after_m = PERIOD_TO_MONTHS[period]
        mask, _ = rasterize_patch_changes(pid, grid_size=64)
        if mask.sum() == 0:
            continue
        records.append({
            "patch_id": pid,
            "period": period,
            "before_month": before_m,
            "after_month": after_m,
            "mask": mask.astype(np.int32),
        })
    return records


def train_fold(train_records, val_records, fold_id, args):
    print(f"\n--- Fold {fold_id+1}/5 ---")

    model = load_model(unfreeze_stp_last_n=args.unfreeze_stp_last_n, unfreeze_bottleneck=args.unfreeze_bottleneck)
    base_dataset = HarbinPatchDataset(load_config(CONFIG_PATH))
    base_dataset.training = False
    base_dataset._spatial_augmentation = False

    # Pre-extract embeddings for fast validation
    val_embs = []
    for rec in val_records:
        pidx = base_dataset.patches.index(rec["patch_id"])
        ws_b, we_b = MONTHLY_WINDOWS[rec["before_month"]]
        ws_a, we_a = MONTHLY_WINDOWS[rec["after_month"]]
        with torch.no_grad():
            emb_b = extract_embedding_map(model, base_dataset, pidx, ws_b, we_b).squeeze(0).cpu()
            emb_a = extract_embedding_map(model, base_dataset, pidx, ws_a, we_a).squeeze(0).cpu()
        val_embs.append({
            "emb_before": emb_b,
            "emb_after": emb_a,
            "mask": torch.from_numpy(rec["mask"]).long(),
        })

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        # Shuffle train records each epoch
        indices = list(range(len(train_records)))
        random.shuffle(indices)

        for i in range(0, len(indices), args.batch_size):
            batch_recs = [train_records[idx] for idx in indices[i:i+args.batch_size]]
            emb_before_list = []
            emb_after_list = []
            mask_list = []

            for rec in batch_recs:
                pidx = base_dataset.patches.index(rec["patch_id"])
                ws_b, we_b = MONTHLY_WINDOWS[rec["before_month"]]
                ws_a, we_a = MONTHLY_WINDOWS[rec["after_month"]]

                emb_b = extract_embedding_map(model, base_dataset, pidx, ws_b, we_b)
                emb_a = extract_embedding_map(model, base_dataset, pidx, ws_a, we_a)
                mask = torch.from_numpy(rec["mask"]).long().to(DEVICE)

                emb_before_list.append(emb_b)
                emb_after_list.append(emb_a)
                mask_list.append(mask)

            emb_before = torch.cat(emb_before_list, dim=0)
            emb_after = torch.cat(emb_after_list, dim=0)
            mask = torch.stack(mask_list, dim=0)

            loss, _ = hinge_margin_loss(emb_before, emb_after, mask, margin_pos=args.margin_pos, margin_neg=args.margin_neg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for ve in val_embs:
                eb = ve["emb_before"].unsqueeze(0).to(DEVICE)
                ea = ve["emb_after"].unsqueeze(0).to(DEVICE)
                distance = torch.norm(eb - ea, p=2, dim=1)
                all_probs.append(distance.cpu().numpy().reshape(-1))
                all_targets.append(ve["mask"].cpu().numpy().reshape(-1))

        val_auc = roc_auc_score(np.concatenate(all_targets), np.concatenate(all_probs))
        avg_loss = np.mean(epoch_losses)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d} | Loss={avg_loss:.4f} | Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"  Fold {fold_id+1} Best Val AUC: {best_auc:.4f}")
    return best_auc, best_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfreeze_stp_last_n", type=int, default=2)
    parser.add_argument("--unfreeze_bottleneck", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--margin_pos", type=float, default=1.0)
    parser.add_argument("--margin_neg", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("  Partial Backbone Fine-tuning with Hinge Margin Loss")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    print(f"[Data] Total records: {len(records)}")

    periods = [r["period"] for r in records]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_aucs = []
    best_states = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(records)), periods)):
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        auc, state = train_fold(train_recs, val_recs, fold_id, args)
        fold_aucs.append(auc)
        best_states.append(state)

    print("\n" + "=" * 70)
    print("  5-Fold Cross-Validation Results")
    print("=" * 70)
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Median AUC: {np.median(fold_aucs):.4f}")

    # Save best fold checkpoint
    best_fold = int(np.argmax(fold_aucs))
    save_path = OUTPUT_DIR / f"best_finetuned_fold{best_fold}.pt"
    torch.save({
        "model_state_dict": best_states[best_fold],
        "args": vars(args),
        "fold_aucs": fold_aucs,
    }, save_path)
    print(f"Saved best fold {best_fold} checkpoint to {save_path}")

    with open(OUTPUT_DIR / "finetune_results.json", "w") as f:
        json.dump({
            "fold_aucs": fold_aucs,
            "mean": float(np.mean(fold_aucs)),
            "std": float(np.std(fold_aucs)),
            "median": float(np.median(fold_aucs)),
            "args": vars(args),
        }, f, indent=2)


if __name__ == "__main__":
    main()
