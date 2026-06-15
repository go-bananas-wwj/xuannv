#!/usr/bin/env python3
"""海淀标注下游任务 MLP 评估（5-Fold Patch-Stratified CV）.

用法:
    python scripts/eval/train_haidian_mlp_heads.py \
        --embedding-file /workspace/xuannv/out/haidian_label/embeddings_v2d53.npz \
        --label-dir /workspace/xuannv/haidian_label/labeljson \
        --output-dir /workspace/xuannv/out/haidian_label/results_v2d53 \
        --device npu:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TASKS = ["chachu", "daolubianhuo", "gongdi", "jiazhudongdi", "nongyongdi", "weijian"]


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_embeddings(npz_path: str):
    data = np.load(npz_path)
    patch_ids = data["patch_ids"]
    emb_dec = data["emb_dec"]  # [N, D, H, W]
    emb_apr = data["emb_apr"]
    return patch_ids, emb_dec, emb_apr


def discover_label_json(label_dir: str, patch_id: str) -> str | None:
    """找一个 patch 对应的 labelme JSON（优先无 _patch_xxx 后缀的版本）."""
    d = Path(label_dir)
    cand1 = d / f"{patch_id}_20260430_rgb_uint8.json"
    if cand1.exists():
        return str(cand1)
    cands = sorted(d.glob(f"{patch_id}_*.json"))
    return str(cands[0]) if cands else None


def rasterize_task_mask(json_path: str, task: str, img_h: int, img_w: int) -> np.ndarray:
    """将 labelme polygon 栅格化为二值 mask [img_h, img_w]."""
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    h = ann.get("imageHeight", img_h)
    w = ann.get("imageWidth", img_w)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for shape in ann.get("shapes", []):
        if shape.get("label") != task:
            continue
        pts = [(float(p[0]), float(p[1])) for p in shape.get("points", [])]
        if len(pts) < 3:
            continue
        draw.polygon(pts, fill=1)
    arr = np.array(mask, dtype=np.uint8)
    if arr.shape != (img_h, img_w):
        arr = np.array(
            Image.fromarray(arr).resize((img_w, img_h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return arr


def build_task_data(
    patch_ids: np.ndarray,
    emb_dec: np.ndarray,
    emb_apr: np.ndarray,
    label_dir: str,
    task: str,
    use_diff: bool = False,
    target_size: int = 128,
):
    """构建某个任务的全量像素级特征与标签.

    将 embedding 双线性上采样到 label 原始分辨率（约 427x427），
    避免栅格化标签时正例被最近邻压缩丢失。

    Returns:
        X: [N_pixels, D*2 or D*3]
        y: [N_pixels]
        patch_idx: [N_pixels]
    """
    N, D, H, W = emb_dec.shape
    X_list, y_list, pidx_list = [], [], []
    for i, pid in enumerate(patch_ids):
        json_path = discover_label_json(label_dir, pid)
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        img_h = ann.get("imageHeight", 427)
        img_w = ann.get("imageWidth", 427)

        # 先在原分辨率栅格化，再 resize 到 target_size
        mask = rasterize_task_mask(json_path, task, img_h, img_w)
        mask = np.array(
            Image.fromarray(mask).resize((target_size, target_size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        ).reshape(-1)

        # embedding 上采样到 target_size
        dec_t = torch.from_numpy(emb_dec[i]).float().unsqueeze(0)  # [1, D, H, W]
        apr_t = torch.from_numpy(emb_apr[i]).float().unsqueeze(0)
        dec_up = F.interpolate(dec_t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        apr_up = F.interpolate(apr_t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        dec = dec_up.squeeze(0).permute(1, 2, 0).reshape(-1, D).numpy()
        apr = apr_up.squeeze(0).permute(1, 2, 0).reshape(-1, D).numpy()

        if use_diff:
            x = np.concatenate([dec, apr, apr - dec], axis=1)
        else:
            x = np.concatenate([dec, apr], axis=1)
        X_list.append(x)
        y_list.append(mask)
        pidx_list.append(np.full(x.shape[0], i, dtype=np.int64))
    if not X_list:
        return None, None, None
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    pidx = np.concatenate(pidx_list, axis=0)
    return X, y, pidx


def compute_pos_weight(y_train: np.ndarray) -> float:
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return 1.0
    # 用 sqrt 折中，避免 pos_weight 过大导致全部预测为正
    return min(max(np.sqrt(n_neg / (n_pos + 1e-8)), 1.0), 100.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """在验证集上搜索使 F1 最大的阈值."""
    best_th, best_f1 = 0.5, 0.0
    for th in np.linspace(0.05, 0.95, 37):
        pred = (scores >= th).astype(np.int64)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_th, best_f1 = th, f1
    return best_th, best_f1


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    hidden_dim: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4096,
    patience: int = 10,
    num_train_samples: int = 200000,
) -> tuple[MLPHead, dict]:
    in_dim = X_train.shape[1]
    model = MLPHead(in_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    # WeightedRandomSampler：让正例像素被更频繁采样
    sample_weights = np.where(y_train == 1, 10.0, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=min(num_train_samples, len(y_train)),
        replacement=True,
    )
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        sampler=sampler,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        _, val_f1 = find_best_threshold(y_val, val_scores)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_scores = torch.sigmoid(val_logits).cpu().numpy()
    best_th, _ = find_best_threshold(y_val, val_scores)
    val_pred = (val_scores >= best_th).astype(np.int64)
    metrics = evaluate(y_val, val_pred)
    metrics["threshold"] = float(best_th)
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="海淀标注下游 MLP 5-Fold CV")
    parser.add_argument("--embedding-file", required=True)
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use-diff", action="store_true", help="加入 apr-dec 差分特征")
    parser.add_argument("--target-size", type=int, default=128,
                        help="将 label 和 embedding 统一到的空间分辨率（默认 128）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    patch_ids, emb_dec, emb_apr = load_embeddings(args.embedding_file)
    print(f"[Data] patches={len(patch_ids)}, emb_shape={emb_dec.shape}")

    summary = {}
    for task in TASKS:
        print(f"\n[Task] {task}")
        X, y, pidx = build_task_data(
            patch_ids, emb_dec, emb_apr, args.label_dir, task,
            use_diff=args.use_diff, target_size=args.target_size
        )
        if X is None:
            print(f"[Task] {task} 无数据")
            continue
        print(f"[Task] {task} pixels={len(y)} pos={y.sum()} neg={len(y)-y.sum()}")

        # 按 patch 是否有正例做 StratifiedKFold，保证每折都有正负 patch
        patch_has_pos = np.array([y[pidx == i].sum() > 0 for i in range(len(patch_ids))], dtype=np.int64)
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        fold_results = []
        for fold, (train_pidx, val_pidx) in enumerate(skf.split(np.arange(len(patch_ids)), patch_has_pos)):
            train_mask = np.isin(pidx, train_pidx)
            val_mask = np.isin(pidx, val_pidx)
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            print(f"  Fold {fold}: train={len(y_train)} pos={y_train.sum()} val={len(y_val)} pos={y_val.sum()}")
            _, metrics = train_one_fold(
                X_train, y_train, X_val, y_val, device,
                hidden_dim=args.hidden_dim, epochs=args.epochs,
            )
            print(f"  Fold {fold}: {metrics}")
            fold_results.append(metrics)

        # 汇总 mean/std
        keys = ["accuracy", "precision", "recall", "f1", "iou"]
        task_summary = {
            "folds": fold_results,
            "mean": {k: float(np.mean([f[k] for f in fold_results])) for k in keys},
            "std": {k: float(np.std([f[k] for f in fold_results])) for k in keys},
        }
        summary[task] = task_summary
        with open(out_dir / f"{task}.json", "w", encoding="utf-8") as f:
            json.dump(task_summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[Summary]")
    print(f"{'Task':<18} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    for task, res in summary.items():
        print(
            f"{task:<18} "
            f"{res['mean']['iou']:.3f}±{res['std']['iou']:.3f}  "
            f"{res['mean']['f1']:.3f}±{res['std']['f1']:.3f}  "
            f"{res['mean']['precision']:.3f}±{res['std']['precision']:.3f}  "
            f"{res['mean']['recall']:.3f}±{res['std']['recall']:.3f}  "
            f"{res['mean']['accuracy']:.3f}±{res['std']['accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
