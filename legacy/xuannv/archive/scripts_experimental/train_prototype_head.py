#!/usr/bin/env python3
"""冻结 backbone，训练 PrototypeFewShotHead (Binary & Multi-class).

用法:
    CUDA_VISIBLE_DEVICES=2 python scripts/train_prototype_head.py
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

from src.models.heads import PrototypeFewShotHead
from scripts.train_task_heads import EmbeddingChangeDataset, CACHE_DIR, evaluate_cd_head, build_masks, DEVICE, OUTPUT_DIR
from demo_v2.engines.model_engine import ModelEngine
from demo_v2.utils.harbin_annotations_v2 import get_annotated_patches, get_period_for_patch, PERIODS

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 200
PATIENCE = 30


class PrototypeDataset(Dataset):
    def __init__(
        self,
        before_emb: np.ndarray,
        after_emb: np.ndarray,
        binary_masks: np.ndarray,
        cat_masks: np.ndarray,
        indices: list[int] | None = None,
    ):
        self.before = torch.from_numpy(before_emb).float()
        self.after = torch.from_numpy(after_emb).float()
        self.binary = torch.from_numpy(binary_masks).long()
        self.category = torch.from_numpy(cat_masks).long()
        self.indices = indices if indices is not None else list(range(len(self.before)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "emb_before": self.before[i],
            "emb_after": self.after[i],
            "mask": self.binary[i],
            "cat_mask": self.category[i],
        }


def evaluate_proto_head(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """评估 PrototypeFewShotHead (binary) 在验证集上的 AUC / BA / F1."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            emb_b = batch["emb_before"].to(device)
            emb_a = batch["emb_after"].to(device)
            mask = batch["mask"].to(device)
            logits = model(emb_b, emb_a)  # [B, 2, H, W]
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # changed prob
            all_probs.append(probs)
            all_labels.append(mask.cpu().numpy())

    probs = np.concatenate([p.reshape(-1) for p in all_probs])
    labels = np.concatenate([l.reshape(-1) for l in all_labels])

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    preds = (probs > 0.5).astype(int)
    ba = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    return {"auc": auc, "ba": ba, "f1": f1}


def main():
    print(f"[Train] Using device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载缓存 embedding（复用 CD head 的缓存）
    cache_path = CACHE_DIR / "embeddings.json"
    before_emb = np.load(CACHE_DIR / "before_embeddings.npy")
    after_emb = np.load(CACHE_DIR / "after_embeddings.npy")
    with open(cache_path) as f:
        meta = json.load(f)

    binary_masks, cat_masks = build_masks(meta["records"])
    print(f"[Data] Masks: positive ratio = {binary_masks.mean():.4f}")

    # 2. 划分训练/验证集
    N = len(meta["records"])
    train_idx, val_idx = train_test_split(
        list(range(N)), test_size=0.2, random_state=42, stratify=binary_masks.max(axis=(1, 2))
    )
    print(f"[Data] Train={len(train_idx)}, Val={len(val_idx)}")

    train_ds = PrototypeDataset(before_emb, after_emb, binary_masks, cat_masks, train_idx)
    val_ds = PrototypeDataset(before_emb, after_emb, binary_masks, cat_masks, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化 head (binary: 2 classes)
    proto_head = PrototypeFewShotHead(
        embedding_dim=128,
        num_classes=2,
        hidden_dim=64,
        temperature=10.0,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(proto_head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        proto_head.train()
        epoch_losses = []
        for batch in train_loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = proto_head(emb_b, emb_a)  # [B, 2, H, W]
            # focal CE for imbalance
            ce = F.cross_entropy(logits, mask, reduction="none")
            pt = torch.exp(-ce)
            alpha = 0.25
            gamma = 2.0
            focal = (alpha * (1 - pt) ** gamma * ce).mean()
            epoch_losses.append(focal.item())

            optimizer.zero_grad()
            focal.backward()
            optimizer.step()

        metrics = evaluate_proto_head(proto_head, val_loader, DEVICE)
        scheduler.step(metrics["auc"])

        avg_loss = np.mean(epoch_losses)
        print(
            f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | "
            f"Val AUC={metrics['auc']:.4f} BA={metrics['ba']:.4f} F1={metrics['f1']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = proto_head.state_dict()
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "proto_head": proto_head.state_dict(),
                "metrics": metrics,
            }, OUTPUT_DIR / "best_proto_head.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    print(f"[Train] Best Val AUC: {best_auc:.4f}")
    if best_state is not None:
        proto_head.load_state_dict(best_state)

    # 保存最终 head
    torch.save({
        "proto_head": proto_head.state_dict(),
        "config": {"embedding_dim": 128, "num_classes": 2, "hidden_dim": 64, "temperature": 10.0},
    }, OUTPUT_DIR / "proto_head.pt")
    print(f"[Train] Saved proto head to {OUTPUT_DIR / 'proto_head.pt'}")

    # 更新统一的 task_heads.pt
    cd_ckpt = torch.load(OUTPUT_DIR / "task_heads.pt", map_location=DEVICE, weights_only=False)
    torch.save({
        "cd_head": cd_ckpt["cd_head"],
        "proto_head": proto_head.state_dict(),
        "config": {"embedding_dim": 128, "hidden_dim": 64, "num_classes": 2, "temperature": 10.0},
    }, OUTPUT_DIR / "task_heads.pt")
    print("[Train] Updated unified task_heads.pt with both heads.")


if __name__ == "__main__":
    main()
