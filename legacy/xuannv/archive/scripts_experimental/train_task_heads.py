#!/usr/bin/env python3
"""冻结 backbone，训练轻量级任务头 (ChangeDetection + PrototypeFewShot).

用法:
    CUDA_VISIBLE_DEVICES=2 python scripts/train_task_heads.py
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

from src.config import load_config
from src.models.model import AEFModel
from src.models.heads import ChangeDetectionHead, PrototypeFewShotHead, dice_loss, focal_bce_loss
from demo_v2.engines.model_engine import ModelEngine
from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
    rasterize_patch_categories,
    PERIODS,
)

# ── 配置 ──
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VERSION = "v2"
CFG_PATH = "/workspace/xuannv/configs/qwen_v2_temporal.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/best.pt"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2_taskheads")
CACHE_DIR = OUTPUT_DIR / "embedding_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 128
GRID_SIZE = 64
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 200
PATIENCE = 30


def extract_and_cache_embeddings(model_engine: ModelEngine, patch_ids: list[str]) -> dict:
    """为所有标注 patch 提取所有 period 的 before/after embedding."""
    cache_path = CACHE_DIR / "embeddings.json"
    npy_paths = {
        "before": CACHE_DIR / "before_embeddings.npy",
        "after": CACHE_DIR / "after_embeddings.npy",
    }

    if cache_path.exists() and npy_paths["before"].exists() and npy_paths["after"].exists():
        print("[Cache] Loading cached embeddings...")
        with open(cache_path) as f:
            meta = json.load(f)
        before_emb = np.load(npy_paths["before"])
        after_emb = np.load(npy_paths["after"])
        return meta, before_emb, after_emb

    print("[Cache] Extracting embeddings for all annotated patches...")
    records = []
    before_list = []
    after_list = []

    for pid in tqdm(patch_ids, desc="Extract"):
        period = get_period_for_patch(pid)
        if period is None or period not in PERIODS:
            continue
        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0

        emb_before = model_engine.extract_embedding(pid, bs, mid)
        emb_after = model_engine.extract_embedding(pid, mid, be)
        if emb_before is None or emb_after is None:
            continue

        before_list.append(emb_before)
        after_list.append(emb_after)
        records.append({"patch_id": pid, "period": period})

    before_emb = np.stack(before_list, axis=0)  # [N, D, H, W]
    after_emb = np.stack(after_list, axis=0)

    meta = {"records": records, "N": len(records)}
    with open(cache_path, "w") as f:
        json.dump(meta, f)
    np.save(npy_paths["before"], before_emb)
    np.save(npy_paths["after"], after_emb)
    print(f"[Cache] Saved {len(records)} embeddings to {CACHE_DIR}")
    return meta, before_emb, after_emb


def build_masks(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """为所有记录构建 binary mask 和 category mask."""
    binary_masks = []
    cat_masks = []
    for rec in records:
        pid = rec["patch_id"]
        bmask, _ = rasterize_patch_changes(pid, grid_size=GRID_SIZE)
        cmask, _ = rasterize_patch_categories(pid, grid_size=GRID_SIZE)
        binary_masks.append(bmask)
        cat_masks.append(cmask)
    return np.stack(binary_masks, axis=0), np.stack(cat_masks, axis=0)


class EmbeddingChangeDataset(Dataset):
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


def evaluate_cd_head(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """评估 ChangeDetectionHead 在验证集上的 AUC / BA / F1."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            emb_b = batch["emb_before"].to(device)
            emb_a = batch["emb_after"].to(device)
            mask = batch["mask"].to(device)
            logits = model(emb_b, emb_a)
            probs = torch.sigmoid(logits).cpu().numpy()
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

    # 1. 加载 backbone 并提取 embedding
    model_engine = ModelEngine(VERSION, device=str(DEVICE))
    patch_ids = get_annotated_patches()
    print(f"[Data] {len(patch_ids)} annotated patches found.")

    meta, before_emb, after_emb = extract_and_cache_embeddings(model_engine, patch_ids)
    binary_masks, cat_masks = build_masks(meta["records"])
    print(f"[Data] Masks: positive ratio = {binary_masks.mean():.4f}")

    # 2. 划分训练/验证集
    N = len(meta["records"])
    train_idx, val_idx = train_test_split(list(range(N)), test_size=0.2, random_state=42, stratify=binary_masks.max(axis=(1,2)))
    print(f"[Data] Train={len(train_idx)}, Val={len(val_idx)}")

    train_ds = EmbeddingChangeDataset(before_emb, after_emb, binary_masks, cat_masks, train_idx)
    val_ds = EmbeddingChangeDataset(before_emb, after_emb, binary_masks, cat_masks, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化 heads
    cd_head = ChangeDetectionHead(EMBEDDING_DIM, hidden_dim=64).to(DEVICE)
    # fewshot_head = PrototypeFewShotHead(EMBEDDING_DIM, num_classes=6, hidden_dim=64).to(DEVICE)
    # 先只训练 CD head，效果最直接
    optimizer = torch.optim.Adam(cd_head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # 4. 训练循环
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

            logits = cd_head(emb_b, emb_a).squeeze(1)  # [B, H, W]
            loss_bce = focal_bce_loss(logits, mask)
            loss_dice = dice_loss(logits, mask)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # 验证
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
            }, OUTPUT_DIR / "best_cd_head.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    print(f"[Train] Best Val AUC: {best_auc:.4f}")
    if best_state is not None:
        cd_head.load_state_dict(best_state)

    # 5. 保存最终 head + 注册到 demo
    torch.save({
        "cd_head": cd_head.state_dict(),
        "config": {"embedding_dim": EMBEDDING_DIM, "hidden_dim": 64},
    }, OUTPUT_DIR / "task_heads.pt")
    print(f"[Train] Saved task heads to {OUTPUT_DIR / 'task_heads.pt'}")


if __name__ == "__main__":
    main()
