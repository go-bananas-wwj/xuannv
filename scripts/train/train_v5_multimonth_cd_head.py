#!/usr/bin/env python3
"""
多月份 Embedding 融合 + ChangeDetectionHead 训练.

核心思想:
- 对 before 窗口内的多个月份 embedding 做 mean/std/max 融合
- 对 after 窗口内的多个月份 embedding 做 mean/std/max 融合
- 用融合后的 embedding 训练 CD Head
- 预期效果: 多月份融合提供更稳定的时序上下文，减少单月份噪声
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
)
from src.models.heads import ChangeDetectionHeadV3

# V5 官方 embedding 目录 (L2-normalized)
EMBEDDING_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head_multimonth")

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 120
PATIENCE = 30
DEVICE = torch.device("npu:0" if torch.npu.is_available() else "cpu")

# period -> before_months, after_months (列表，支持多月份融合)
PERIOD_TO_MONTHS = {
    "2025-04~2025-06": (["2025-04"], ["2025-06"]),
    "2025-06~2025-08": (["2025-06"], ["2025-08"]),
    "2025-08~2025-09": (["2025-08"], ["2025-09"]),
    "2025-09~2025-10": (["2025-09"], ["2025-10"]),
}

# 扩展: 使用相邻月份作为上下文
PERIOD_TO_MONTHS_EXTENDED = {
    "2025-04~2025-06": (["2025-04"], ["2025-06", "2025-08"]),
    "2025-06~2025-08": (["2025-04", "2025-06"], ["2025-08", "2025-09"]),
    "2025-08~2025-09": (["2025-06", "2025-08"], ["2025-09", "2025-10"]),
    "2025-09~2025-10": (["2025-08", "2025-09"], ["2025-10"]),
}


def fuse_embeddings(monthly_embs: list[np.ndarray]) -> np.ndarray:
    """对多个月份 embedding 做 mean/std/max 融合.
    
    Args:
        monthly_embs: list of [D, H, W] arrays
    
    Returns:
        [3D, H, W] fused embedding
    """
    embs = np.stack(monthly_embs, axis=0)  # [N, D, H, W]
    mean_emb = embs.mean(axis=0)   # [D, H, W]
    std_emb = embs.std(axis=0)     # [D, H, W]
    max_emb = embs.max(axis=0)     # [D, H, W]
    return np.concatenate([mean_emb, std_emb, max_emb], axis=0)  # [3D, H, W]


def boundary_aware_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_sig = torch.sigmoid(pred)
    neg_mask = (target == 0).float()
    neg_count = neg_mask.sum() + 1e-8
    neg_mean = (pred_sig * neg_mask).sum() / neg_count
    return F.relu(neg_mean - 0.1) ** 2


def focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    intersection = (pred * target.float()).sum()
    union = pred.sum() + target.float().sum()
    return 1.0 - (2.0 * intersection + 1e-8) / (union + 1e-8)


class EmbeddingAugment:
    def __init__(self, noise_std: float = 0.02, channel_dropout: float = 0.1):
        self.noise_std = noise_std
        self.channel_dropout = channel_dropout

    def __call__(self, emb_before: torch.Tensor, emb_after: torch.Tensor, mask: torch.Tensor):
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        rotate = random.random() < 0.5
        if rotate:
            k = random.randint(1, 3)

        if flip_h:
            emb_before = torch.flip(emb_before, dims=[-1])
            emb_after = torch.flip(emb_after, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if flip_v:
            emb_before = torch.flip(emb_before, dims=[-2])
            emb_after = torch.flip(emb_after, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if rotate:
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


class MultiMonthCDDataset(Dataset):
    """加载多月份融合 embedding 对."""
    def __init__(self, records: list[dict], indices: list[int], augment=None):
        self.records = records
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        rec = self.records[self.indices[idx]]
        emb_before = rec["emb_before"]  # [3D, H, W]
        emb_after = rec["emb_after"]
        mask = rec["mask"]  # [H, W]

        emb_before = torch.from_numpy(emb_before).float()
        emb_after = torch.from_numpy(emb_after).float()
        mask = torch.from_numpy(mask).long()

        if self.augment is not None:
            emb_before, emb_after, mask = self.augment(emb_before, emb_after, mask)

        return {
            "emb_before": emb_before,
            "emb_after": emb_after,
            "mask": mask,
            "patch_id": rec["patch_id"],
            "period": rec["period"],
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "emb_before": torch.stack([b["emb_before"] for b in batch]),
        "emb_after": torch.stack([b["emb_after"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_id": [b["patch_id"] for b in batch],
        "period": [b["period"] for b in batch],
    }


def build_records(use_extended: bool = False) -> list[dict]:
    """构建所有可用的 (patch, fused_before, fused_after, mask) 记录."""
    annotated = get_annotated_patches()
    records = []
    period_map = PERIOD_TO_MONTHS_EXTENDED if use_extended else PERIOD_TO_MONTHS

    for pid in annotated:
        period = get_period_for_patch(pid)
        if period is None or period not in period_map:
            continue
        before_ms, after_ms = period_map[period]

        # 加载并融合 before 月份
        before_embs = []
        for m in before_ms:
            path = EMBEDDING_DIR / f"{pid}_{m}.npy"
            if path.exists():
                before_embs.append(np.load(path))
        if not before_embs:
            continue

        # 加载并融合 after 月份
        after_embs = []
        for m in after_ms:
            path = EMBEDDING_DIR / f"{pid}_{m}.npy"
            if path.exists():
                after_embs.append(np.load(path))
        if not after_embs:
            continue

        mask, _ = rasterize_patch_changes(pid, grid_size=64)
        if mask.sum() == 0:
            continue

        fused_before = fuse_embeddings(before_embs)
        fused_after = fuse_embeddings(after_embs)

        records.append({
            "patch_id": pid,
            "period": period,
            "before_months": before_ms,
            "after_months": after_ms,
            "emb_before": fused_before,
            "emb_after": fused_after,
            "mask": mask.astype(np.int32),
        })

    print(f"[Data] Total valid records: {len(records)}")
    if records:
        pos_ratio = np.mean([r["mask"].mean() for r in records])
        print(f"[Data] Average positive pixel ratio: {pos_ratio:.4f}")
    return records


def evaluate(head: nn.Module, loader: DataLoader) -> dict:
    head.eval()
    all_probs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = head(emb_b, emb_a).squeeze(1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy().reshape(-1))
            all_targets.append(mask.cpu().numpy().reshape(-1))
            all_preds.append((probs > 0.5).cpu().numpy().reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(all_targets, all_probs)
    ba = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return {"auc": auc, "ba": ba, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extended", action="store_true", help="使用扩展的多月份上下文")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    global DEVICE
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"  V5 多月份融合 CD Head 训练")
    print(f"  Embedding dir: {EMBEDDING_DIR}")
    print(f"  Extended context: {args.extended}")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records(use_extended=args.extended)
    if not records:
        print("❌ No valid records found.")
        return

    # stratify by period
    periods = [r["period"] for r in records]
    train_idx, val_idx = train_test_split(
        list(range(len(records))),
        test_size=0.2,
        random_state=args.seed,
        stratify=periods,
    )
    print(f"[Data] Train={len(train_idx)}, Val={len(val_idx)}")

    augment = EmbeddingAugment(noise_std=0.02, channel_dropout=0.1)
    train_ds = MultiMonthCDDataset(records, train_idx, augment=augment)
    val_ds = MultiMonthCDDataset(records, val_idx, augment=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 注意: 多月份融合后 embedding 维度变为 3*128=384
    # 但 CD Head 的输入是 concat(|diff|, before*after, before, after) = [B, 4*in_dim, H, W]
    # 所以需要创建适配的 head
    in_dim = 384  # 3 * 128
    
    # 自定义多月份融合 CD Head
    class MultiMonthCDHead(nn.Module):
        def __init__(self, in_dim=384, hidden_dim=128, dropout=0.4):
            super().__init__()
            feat_dim = in_dim * 4  # |diff|, before*after, before, after
            
            self.encoder = nn.Sequential(
                nn.Conv2d(feat_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            )
            
            self.res1 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
            )
            
            self.res2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
            )
            
            self.out = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(hidden_dim // 2, 1, 1),
            )
        
        def forward(self, emb_before, emb_after):
            diff = emb_before - emb_after
            feat = torch.cat([
                torch.abs(diff),
                emb_before * emb_after,
                emb_before,
                emb_after,
            ], dim=1)
            x = self.encoder(feat)
            x = F.relu(self.res1(x) + x)
            x = F.relu(self.res2(x) + x)
            return self.out(x)

    head = MultiMonthCDHead(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[Model] MultiMonthCDHead parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        head.train()
        epoch_losses = []

        for batch in train_loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = head(emb_b, emb_a).squeeze(1)
            loss_focal = focal_bce_loss(logits, mask, alpha=0.5, gamma=2.0)
            loss_dice = dice_loss(logits, mask)
            loss_neg = boundary_aware_loss(logits, mask)
            loss = loss_focal + 0.5 * loss_dice + 0.3 * loss_neg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        metrics = evaluate(head, val_loader)
        avg_loss = np.mean(epoch_losses)

        print(
            f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | "
            f"Val AUC={metrics['auc']:.4f} BA={metrics['ba']:.4f} F1={metrics['f1']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = head.state_dict().copy()
            best_metrics = metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    print(f"[Train] Best Val AUC: {best_auc:.4f}")
    if best_state is not None:
        head.load_state_dict(best_state)
        save_path = OUTPUT_DIR / "multimonth_cd_head.pt"
        torch.save(
            {
                "cd_head": head.state_dict(),
                "metrics": best_metrics,
                "config": {"in_dim": in_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout},
            },
            save_path,
        )
        print(f"[Train] Saved final head to {save_path}")
        
        # 对比 baseline
        print(f"\n  对比 V5 CD Head (AUC 0.9555):")
        delta = best_auc - 0.9555
        print(f"  MultiMonth → V5: {best_auc:.4f} vs 0.9555 ({'↑+' if delta > 0 else '↓'}{abs(delta):.4f})")


if __name__ == "__main__":
    main()
