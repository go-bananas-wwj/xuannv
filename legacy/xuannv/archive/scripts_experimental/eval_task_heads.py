#!/usr/bin/env python3
"""快速验证训练好的 Task Heads 效果."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/xuannv")

from src.models.heads import ChangeDetectionHead
from scripts.train_task_heads import EmbeddingChangeDataset, CACHE_DIR

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2_taskheads")


def main():
    head_path = OUTPUT_DIR / "task_heads.pt"
    if not head_path.exists():
        print(f"❌ Head not found: {head_path}")
        return

    ckpt = torch.load(head_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {"embedding_dim": 128, "hidden_dim": 64})
    cd_head = ChangeDetectionHead(cfg["embedding_dim"], cfg["hidden_dim"]).to(DEVICE)
    cd_head.load_state_dict(ckpt["cd_head"])
    cd_head.eval()

    # 加载数据
    cache_path = CACHE_DIR / "embeddings.json"
    before_emb = np.load(CACHE_DIR / "before_embeddings.npy")
    after_emb = np.load(CACHE_DIR / "after_embeddings.npy")
    with open(cache_path) as f:
        meta = json.load(f)

    from scripts.train_task_heads import build_masks
    binary_masks, cat_masks = build_masks(meta["records"])

    # 用同样的划分
    from sklearn.model_selection import train_test_split
    N = len(meta["records"])
    train_idx, val_idx = train_test_split(
        list(range(N)), test_size=0.2, random_state=42, stratify=binary_masks.max(axis=(1, 2))
    )

    val_ds = EmbeddingChangeDataset(before_emb, after_emb, binary_masks, cat_masks, val_idx)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            emb_b = batch["emb_before"].to(DEVICE)
            emb_a = batch["emb_after"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = cd_head(emb_b, emb_a).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(mask.cpu().numpy())

    probs = np.concatenate([p.reshape(-1) for p in all_probs])
    labels = np.concatenate([l.reshape(-1) for l in all_labels])

    auc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    ba = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"✅ Val AUC={auc:.4f} | BA={ba:.4f} | F1={f1:.4f}")
    print(f"   Positive ratio in val: {labels.mean():.4f}")

    # 保存下游 benchmark 格式结果
    benchmark = {
        "model": "v2_taskheads",
        "checkpoint": str(head_path),
        "n_val": len(val_idx),
        "val_auc": auc,
        "val_ba": ba,
        "val_f1": f1,
    }
    with open(OUTPUT_DIR / "eval_result.json", "w") as f:
        json.dump(benchmark, f, indent=2)


if __name__ == "__main__":
    main()
