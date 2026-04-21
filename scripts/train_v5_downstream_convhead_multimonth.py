#!/usr/bin/env python3
"""V5 多月份 Embedding 融合 + PixelConvHead 下游训练.

Phase 3 未执行的设计:
- 对每个像素计算 5 个月 embedding 的统计量: mean, std, max
- 拼接为 [3D, H, W] 输入给 PixelConvHead
- 时序稳定性/变化度本身作为分类特征

预期效果:
- 稳定地物 (建筑、水体): 5 个月 std 低
- 季节性地物 (农田、植被): 5 个月 std 高
- 多月份融合可能比单月份更鲁棒
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, "/workspace/xuannv")

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025_prenorm")
LABEL_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_labels_2025")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MONTHS = ["2025-04", "2025-06", "2025-08", "2025-09", "2025-10"]

# ──────────────────────────────────────────
# PixelConvHead (与 Phase 3 相同)
# ──────────────────────────────────────────
class PixelConvHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        return x

# ──────────────────────────────────────────
# 多月份融合 Dataset
# ──────────────────────────────────────────
class MultiMonthFusionDataset(Dataset):
    """加载 5 个月 pre-norm embedding，计算 mean/std/max 融合."""

    def __init__(self, emb_dir, label_dir, task_name, months=MONTHS, class_remap=None):
        self.emb_dir = Path(emb_dir)
        self.label_dir = Path(label_dir)
        self.task_name = task_name
        self.months = months
        self.class_remap = class_remap

        # 找所有 patch_id (从第一个月的文件推断)
        first_month_files = sorted(self.emb_dir.glob(f"*_{months[0]}.npy"))
        self.patch_ids = [f.stem.rsplit(f"_{months[0]}", 1)[0] for f in first_month_files]

        # 检查所有 patch 是否有完整的 5 个月数据
        self.valid_patches = []
        for pid in self.patch_ids:
            has_all = all((self.emb_dir / f"{pid}_{m}.npy").exists() for m in months)
            has_label = (self.label_dir / f"{pid}_{self.task_name}.npy").exists()
            if has_all and has_label:
                self.valid_patches.append(pid)

        print(f"[MultiMonthFusion] Task={task_name}, valid patches={len(self.valid_patches)}/{len(self.patch_ids)}")

        # 加载一个样本推断维度
        sample_emb = np.load(self.emb_dir / f"{self.valid_patches[0]}_{months[0]}.npy")
        self.D = sample_emb.shape[0]
        self.H = sample_emb.shape[1]
        self.W = sample_emb.shape[2]

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        pid = self.valid_patches[idx]

        # 加载 5 个月 embedding [5, D, H, W]
        embs = []
        for m in self.months:
            e = np.load(self.emb_dir / f"{pid}_{m}.npy")  # [D, H, W]
            embs.append(e)
        embs = np.stack(embs, axis=0)  # [5, D, H, W]

        # 计算统计量
        mean_emb = embs.mean(axis=0)   # [D, H, W]
        std_emb = embs.std(axis=0)     # [D, H, W]
        max_emb = embs.max(axis=0)     # [D, H, W]

        # 拼接为 [3D, H, W]
        fused = np.concatenate([mean_emb, std_emb, max_emb], axis=0)  # [3D, H, W]
        fused_t = torch.from_numpy(fused).float()

        # 加载标签
        lbl = np.load(self.label_dir / f"{pid}_{self.task_name}.npy")
        if self.class_remap is not None:
            lbl = np.vectorize(self.class_remap.get)(lbl)
        lbl_t = torch.from_numpy(lbl).long()

        return fused_t, lbl_t


# ──────────────────────────────────────────
# 训练函数
# ──────────────────────────────────────────
def compute_class_weights(y_true, n_classes):
    counts = np.bincount(y_true, minlength=n_classes)
    total = counts.sum()
    weights = total / (n_classes * (counts + 1e-6))
    return torch.tensor(np.clip(weights, 0.1, 20.0), dtype=torch.float32)


def train_multimonth_conv_head(task_name, ds, n_classes, is_binary=False, epochs=30):
    print(f"\n{'='*60}")
    print(f"  Training MultiMonth Fusion PixelConvHead: {task_name}")
    print(f"  Input dim: {ds.D * 3}, Classes: {n_classes}, Binary: {is_binary}")
    print(f"{'='*60}")

    # Train/val split by patch
    n_total = len(ds)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 计算类别权重 (从训练集)
    all_labels = []
    for i in range(n_train):
        _, lbl = train_ds[i]
        all_labels.extend(lbl[lbl >= 0].flatten().tolist())
    y_true = np.array(all_labels)
    class_weights = compute_class_weights(y_true, n_classes).to(DEVICE)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    # 模型
    model = PixelConvHead(in_dim=ds.D * 3, hidden_dim=128, num_classes=n_classes, kernel_size=3, dropout=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric = 0.0
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_steps = 0
        for emb, lbl in train_loader:
            emb = emb.to(DEVICE)
            lbl = lbl.to(DEVICE)
            optimizer.zero_grad()
            logits = model(emb)  # [B, N, H, W]
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
        scheduler.step()

        # Val
        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(DEVICE)
                lbl = lbl.to(DEVICE)
                logits = model(emb)
                loss = criterion(logits, lbl)
                val_loss += loss.item()
                val_steps += 1
                preds = logits.argmax(dim=1).cpu().numpy().flatten()
                targets = lbl.cpu().numpy().flatten()
                mask = targets >= 0
                all_preds.extend(preds[mask])
                all_targets.extend(targets[mask])

        y_pred = np.array(all_preds)
        y_true_val = np.array(all_targets)

        # 计算指标
        from sklearn.metrics import balanced_accuracy_score, f1_score
        bacc = balanced_accuracy_score(y_true_val, y_pred)

        if is_binary:
            f1 = f1_score(y_true_val, y_pred, pos_label=1, zero_division=0)
            metric = f1
            metric_name = "F1"
        else:
            f1 = f1_score(y_true_val, y_pred, average="weighted", zero_division=0)
            metric = bacc
            metric_name = "BAcc"

        print(f"  Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss/train_steps:.4f} "
              f"val_loss={val_loss/val_steps:.4f} BAcc={bacc:.4f} {metric_name}={metric:.4f}")

        # Early stopping
        if metric > best_metric:
            best_metric = metric
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  Best {metric_name}: {best_metric:.4f}")
    return {"task": task_name, "best_metric": float(best_metric), "metric_name": metric_name, "bacc": float(bacc)}


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    tasks = []
    if args.task == "all":
        tasks = ["WorldCover", "DynamicWorld", "JRC_Water", "OSM_Buildings"]
    else:
        tasks = [args.task]

    results = []
    for task_name in tasks:
        if task_name == "WorldCover":
            # 先扫描所有 label 文件确定类别
            all_classes = set()
            for f in sorted(LABEL_DIR.glob(f"*_WorldCover.npy")):
                lbl = np.load(f)
                all_classes.update(np.unique(lbl[lbl >= 0]).tolist())
            wc_classes = sorted(all_classes)
            wc_remap = {c: i for i, c in enumerate(wc_classes)}
            ds = MultiMonthFusionDataset(EMB_DIR, LABEL_DIR, "WorldCover", class_remap=wc_remap)
            result = train_multimonth_conv_head("WorldCover", ds, n_classes=len(wc_classes), epochs=args.epochs)
        elif task_name == "DynamicWorld":
            ds = MultiMonthFusionDataset(EMB_DIR, LABEL_DIR, "DynamicWorld")
            result = train_multimonth_conv_head("DynamicWorld", ds, n_classes=9, epochs=args.epochs)
        elif task_name == "JRC_Water":
            ds = MultiMonthFusionDataset(EMB_DIR, LABEL_DIR, "JRC_Water")
            result = train_multimonth_conv_head("JRC_Water", ds, n_classes=2, is_binary=True, epochs=args.epochs)
        elif task_name == "OSM_Buildings":
            ds = MultiMonthFusionDataset(EMB_DIR, LABEL_DIR, "OSM_Buildings")
            result = train_multimonth_conv_head("OSM_Buildings", ds, n_classes=2, is_binary=True, epochs=args.epochs)
        else:
            print(f"Unknown task: {task_name}")
            continue
        results.append(result)

    print("\n" + "=" * 60)
    print("  MultiMonth Fusion Results Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['task']:15s} | {r['metric_name']}={r['best_metric']:.4f} BAcc={r['bacc']:.4f}")

    # 对比 Phase 3 单月份结果
    print("\n  对比 Phase 3 单月份 PixelConvHead:")
    print("  WorldCover    | BAcc 0.5648")
    print("  DynamicWorld  | BAcc 0.5734")
    print("  JRC_Water     | BAcc 0.8631")
    print("  OSM_Buildings | BAcc 0.8942")


if __name__ == "__main__":
    main()
