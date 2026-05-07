#!/usr/bin/env python3
"""
冻结 V4 backbone，用 2025 年单月份 embedding 对训练 ChangeDetectionHead.

基于 V2 脚本修改，使用 V4 官方月度 embedding.

标注 -> 月份对映射:
  2025-04~2025-06 -> before=2025-04, after=2025-06
  2025-06~2025-08 -> before=2025-06, after=2025-08
  2025-08~2025-09 -> before=2025-08, after=2025-09
  2025-09~2025-10 -> before=2025-09, after=2025-10

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=6,7 python scripts/train/train_v4_monthly_cd_head.py --head_type v3 --ohem
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

# V4 官方 embedding 目录
EMBEDDING_DIR = Path("/workspace/outputs/aef_qwen_v4_official/monthly_embeddings_2025")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v4_official/monthly_cd_head")

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 120
PATIENCE = 30
DEVICE = torch.device("npu:0" if torch.npu.is_available() else "cpu")

# period -> (before_month, after_month)
PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}

# 多类别映射: construction=0, demolition=1, land_conversion=2
CATEGORY_TO_MC = {
    0: -1,   # unchanged -> ignore
    1: 0,    # construction
    2: 1,    # demolition
    3: 2,    # road -> land_conversion
    4: 2,    # water_change -> land_conversion
    5: 2,    # farmland -> land_conversion
}


def boundary_aware_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """惩罚负样本上的平均高响应（防止全局高偏置）."""
    pred_sig = torch.sigmoid(pred)
    neg_mask = (target == 0).float()
    neg_count = neg_mask.sum() + 1e-8
    neg_mean = (pred_sig * neg_mask).sum() / neg_count
    return F.relu(neg_mean - 0.1) ** 2


def ohem_focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, ohem_ratio: float = 0.25, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:
    """OHEM 版本的 Focal BCE：保留所有正例 + top-k hardest negatives."""
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


def multiclass_ohem_focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, ohem_ratio: float = 0.25, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:
    """多类别 OHEM Focal BCE，按类分别计算后平均."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # [B, C, H, W]
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
    """Embedding 空间数据增强."""

    def __init__(self, noise_std: float = 0.02, channel_dropout: float = 0.1):
        self.noise_std = noise_std
        self.channel_dropout = channel_dropout

    def __call__(
        self,
        emb_before: torch.Tensor,
        emb_after: torch.Tensor,
        mask: torch.Tensor,
        extra_mask: torch.Tensor | None = None,
    ):
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


class MonthlyEmbeddingDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        indices: list[int],
        augment: EmbeddingAugment | None = None,
        use_category_mask: bool = False,
    ):
        self.records = records
        self.indices = indices
        self.augment = augment
        self.use_category_mask = use_category_mask

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        rec = self.records[self.indices[idx]]
        emb_before = np.load(rec["before_path"])  # [D, H, W]
        emb_after = np.load(rec["after_path"])
        mask = rec["mask"]  # [H, W]

        emb_before = torch.from_numpy(emb_before).float()
        emb_after = torch.from_numpy(emb_after).float()
        mask = torch.from_numpy(mask).long()

        cat_mask = None
        if self.use_category_mask and "category_mask" in rec:
            cat_mask = torch.from_numpy(rec["category_mask"]).long()

        if self.augment is not None:
            if cat_mask is not None:
                emb_before, emb_after, mask, cat_mask = self.augment(emb_before, emb_after, mask, cat_mask)
            else:
                emb_before, emb_after, mask = self.augment(emb_before, emb_after, mask)

        out = {
            "emb_before": emb_before,
            "emb_after": emb_after,
            "mask": mask,
            "patch_id": rec["patch_id"],
            "period": rec["period"],
        }

        if cat_mask is not None:
            out["category_mask"] = cat_mask

        return out


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "emb_before": torch.stack([b["emb_before"] for b in batch]),
        "emb_after": torch.stack([b["emb_after"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_id": [b["patch_id"] for b in batch],
        "period": [b["period"] for b in batch],
    }
    if "category_mask" in batch[0]:
        out["category_mask"] = torch.stack([b["category_mask"] for b in batch])
    return out


def build_records(use_category_mask: bool = False) -> list[dict]:
    """构建所有可用的 (patch, before_month, after_month, mask) 记录."""
    annotated = get_annotated_patches()
    records = []

    for pid in annotated:
        period = get_period_for_patch(pid)
        if period is None or period not in PERIOD_TO_MONTHS:
            continue
        before_m, after_m = PERIOD_TO_MONTHS[period]

        before_path = EMBEDDING_DIR / f"{pid}_{before_m}.npy"
        after_path = EMBEDDING_DIR / f"{pid}_{after_m}.npy"
        if not before_path.exists() or not after_path.exists():
            continue

        mask, _ = rasterize_patch_changes(pid, grid_size=64)
        if mask.sum() == 0:
            continue

        rec = {
            "patch_id": pid,
            "period": period,
            "before_month": before_m,
            "after_month": after_m,
            "before_path": str(before_path),
            "after_path": str(after_path),
            "mask": mask.astype(np.int32),
        }

        if use_category_mask:
            cat_mask, _ = rasterize_patch_categories(pid, grid_size=64)
            rec["category_mask"] = cat_mask.astype(np.int32)

        records.append(rec)

    print(f"[Data] Total valid records: {len(records)}")
    if records:
        pos_ratio = np.mean([r["mask"].mean() for r in records])
        print(f"[Data] Average positive pixel ratio: {pos_ratio:.4f}")
    return records


def mc_target_from_category_mask(cat_mask: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """将 category_mask [B, H, W] 转为多类 one-hot target [B, C, H, W]."""
    B, H, W = cat_mask.shape
    target = torch.zeros(B, num_classes, H, W, device=cat_mask.device, dtype=torch.float32)
    for b in range(B):
        for cat_idx, mc_idx in CATEGORY_TO_MC.items():
            if cat_idx == 0 or mc_idx < 0:
                continue
            target[b, mc_idx, cat_mask[b] == cat_idx] = 1.0
    return target


def evaluate_binary(head: nn.Module, loader: DataLoader) -> dict:
    head.eval()
    all_probs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = head(emb_b, emb_a).squeeze(1)  # [B, H, W]
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


def evaluate_multiclass(head: nn.Module, loader: DataLoader) -> dict:
    """评估多类别 head，返回 binary AUC (max-rule) 和各类 AUC."""
    head.eval()
    all_probs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            cat_mask = batch["category_mask"].to(DEVICE)

            logits = head(emb_b, emb_a)  # [B, 3, H, W]
            probs = torch.sigmoid(logits)
            binary_probs = probs.max(dim=1)[0]  # [B, H, W]

            all_probs.append(binary_probs.cpu().numpy().reshape(-1))
            all_targets.append(mask.cpu().numpy().reshape(-1))
            all_preds.append((binary_probs > 0.5).cpu().numpy().reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(all_targets, all_probs)
    ba = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return {"auc": auc, "ba": ba, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_type", type=str, default="v3", choices=["v2", "v3", "mc"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--channel_dropout", type=float, default=0.1)
    parser.add_argument("--ohem", action="store_true", help="使用 OHEM")
    parser.add_argument("--ohem_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", type=str, default=None, help="输出文件名后缀")
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

    print("=" * 70)
    print(f"  V4 月度 Embedding Change Detection Head 训练  [{args.head_type}]")
    print(f"  Embedding dir: {EMBEDDING_DIR}")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_mc = args.head_type == "mc"
    records = build_records(use_category_mask=use_mc)
    if not records:
        print("❌ No valid records found. Run extract_v4_monthly_embeddings.py first.")
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

    augment = EmbeddingAugment(noise_std=args.noise_std, channel_dropout=args.channel_dropout)
    train_ds = MonthlyEmbeddingDataset(records, train_idx, augment=augment, use_category_mask=use_mc)
    val_ds = MonthlyEmbeddingDataset(records, val_idx, augment=None, use_category_mask=use_mc)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # 创建 head
    if args.head_type == "v2":
        head = ChangeDetectionHeadV2(embedding_dim=128, hidden_dim=args.hidden_dim).to(DEVICE)
    elif args.head_type == "v3":
        head = ChangeDetectionHeadV3(embedding_dim=128, hidden_dim=args.hidden_dim, dropout=args.dropout).to(DEVICE)
    elif args.head_type == "mc":
        head = MultiClassChangeDetectionHead(embedding_dim=128, hidden_dim=args.hidden_dim, dropout=args.dropout).to(DEVICE)
    else:
        raise ValueError(f"Unknown head_type: {args.head_type}")

    n_params = sum(p.numel() for p in head.parameters())
    print(f"[Model] {head.__class__.__name__} parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

            if use_mc:
                cat_mask = batch["category_mask"].to(DEVICE)
                mc_target = mc_target_from_category_mask(cat_mask, num_classes=MultiClassChangeDetectionHead.NUM_CLASSES)
                logits = head(emb_b, emb_a)  # [B, 3, H, W]

                if args.ohem:
                    loss_focal = multiclass_ohem_focal_bce_loss(logits, mc_target, ohem_ratio=args.ohem_ratio, alpha=0.5, gamma=2.0)
                else:
                    class_weights = torch.tensor([0.8, 1.0, 1.0], device=DEVICE)
                    loss_focal = multiclass_focal_bce_loss(logits, mc_target, alpha=0.5, gamma=2.0, class_weights=class_weights)
                loss_dice = multiclass_dice_loss(logits, mc_target)
                loss = loss_focal + 0.5 * loss_dice
            else:
                mask = batch["mask"].to(DEVICE)
                logits = head(emb_b, emb_a).squeeze(1)  # [B, H, W]

                if args.ohem:
                    loss_focal = ohem_focal_bce_loss(logits, mask, ohem_ratio=args.ohem_ratio, alpha=0.5, gamma=2.0)
                else:
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

        if use_mc:
            metrics = evaluate_multiclass(head, val_loader)
        else:
            metrics = evaluate_binary(head, val_loader)

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
        out_name = args.output_name or args.head_type
        save_path = OUTPUT_DIR / f"monthly_cd_head_{out_name}.pt"
        torch.save(
            {
                "cd_head": head.state_dict(),
                "metrics": best_metrics,
                "config": {
                    "head_type": args.head_type,
                    "embedding_dim": 128,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
                "args": vars(args),
            },
            save_path,
        )
        print(f"[Train] Saved final head to {save_path}")


if __name__ == "__main__":
    main()
