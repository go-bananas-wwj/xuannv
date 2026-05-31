#!/usr/bin/env python
"""海淀数据集 MLP 微调下游评估。

冻结 backbone，提取像素级 spatial embedding，用轻量 MLP 评估三个任务：
  A. WorldCover 6 类土地覆盖分类（主力）
  B. JRC Water 二值水体检测（occurrence > 50 = water）
  C. Dynamic World 9 类分类（参考）

推荐用法（先用 extract_embeddings.py 提取，再用 NPZ 评估，最快）：
    # 1. 先提取（eval_v25 tmux 中已运行）
    python scripts/eval/extract_embeddings.py --config configs/config_haidian_v25.yaml \\
        --checkpoint epoch_best_epoch78.pt --output-dir out/eval_v25_0531/ --format npz

    # 2. 再评估（CPU 即可，秒级完成）
    python scripts/eval/mlp_eval_haidian.py \\
        --embedding-file out/eval_v25_0531/patch_embeddings_shard0.npz \\
        --output-dir out/eval_v25_0531/mlp/

内联提取用法（首次 NPU JIT 编译约 5-10 分钟）：
    python scripts/eval/mlp_eval_haidian.py \\
        --config configs/config_haidian_v25.yaml \\
        --checkpoint epoch_best_epoch78.pt --device npu:0 \\
        --output-dir out/eval_v25_0531/mlp/
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time as time_mod
import calendar
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, jaccard_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

# ── 数据路径 ─────────────────────────────────────────────────────────────────

HAIDIAN_ROOT = Path("/workspace/raw/haidian_train/haidian")

# ── 标注配置 ─────────────────────────────────────────────────────────────────

WC_MAPPING = {10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5}
WC_NAMES   = ["Tree", "Grassland", "Cropland", "BuiltUp", "Bare", "Water"]
WC_NUM_CLASSES = 6

JRC_THRESHOLD   = 50   # occurrence > 50 → water
JRC_NUM_CLASSES = 2

DW_NUM_CLASSES = 9  # Dynamic World 0–8


# ── MLP 模型 ─────────────────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """轻量像素级 MLP 分类器。"""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 标注加载 ─────────────────────────────────────────────────────────────────

def load_worldcover(patch_id: str) -> np.ndarray | None:
    """加载 WorldCover 标注，返回 int8 类别数组（-1 = invalid）。"""
    d = HAIDIAN_ROOT / "worldcover" / patch_id
    tifs = sorted(d.glob("*.tif"))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        arr = src.read(1).astype(np.float32)
    out = np.full(arr.shape, -1, dtype=np.int8)
    for raw, mapped in WC_MAPPING.items():
        out[arr == raw] = mapped
    return out


def load_jrc_water(patch_id: str) -> np.ndarray | None:
    """加载 JRC Water，返回 int8 二值数组（0=陆地, 1=水体, -1=invalid）。"""
    d = HAIDIAN_ROOT / "jrc_water" / patch_id
    tifs = sorted(d.glob("*.tif"))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    out = np.zeros(arr.shape, dtype=np.int8)
    out[arr > JRC_THRESHOLD] = 1
    if nodata is not None:
        out[arr == nodata] = -1
    return out


def load_dynamic_world(patch_id: str) -> np.ndarray | None:
    """加载 Dynamic World，返回 int8 类别数组（round 后 0-8，-1=invalid）。"""
    d = HAIDIAN_ROOT / "dynamic_world" / patch_id
    tifs = sorted(d.glob("*.tif"))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        arr = src.read(1).astype(np.float32)
    rounded = np.round(arr).astype(np.int8)
    out = np.where((rounded >= 0) & (rounded < DW_NUM_CLASSES), rounded, np.int8(-1))
    return out


def resize_label_nearest(label: np.ndarray, h: int, w: int) -> np.ndarray:
    """将标注 resize 到 (h, w)，使用 nearest neighbor。"""
    t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    r = F.interpolate(t, size=(h, w), mode="nearest").squeeze().numpy()
    return r.astype(label.dtype)


# ── Embedding 提取 ────────────────────────────────────────────────────────────

def full_year_window() -> tuple[int, int]:
    """使用 2025-01-01 ~ 2026-06-30 作为全年时间窗口。"""
    vs = int(time_mod.mktime((2025, 1, 1, 0, 0, 0, 0, 0, 0))) * 1000
    ve = int(time_mod.mktime((2026, 6, 30, 23, 59, 59, 0, 0, 0))) * 1000
    return vs, ve


def extract_spatial_map(model, dataset, cfg, sample_map, device, patch_id):
    """提取单 patch 的空间 embedding map [D, H, W]。

    使用覆盖全部数据的时间窗口，让模型聚合所有可用帧。
    """
    # 选最接近年中 (6月) 的月份样本，若无则尝试其他月份
    target_months = [(2025, 6), (2025, 8), (2025, 4), (2025, 10),
                     (2025, 7), (2025, 5), (2025, 9), (2025, 3),
                     (2025, 11), (2025, 2), (2025, 12), (2025, 1),
                     (2026, 1), (2026, 2), (2026, 3)]
    for year, month in target_months:
        key = (patch_id, year, month)
        if key not in sample_map:
            continue
        try:
            item = dataset[sample_map[key]]
        except Exception:
            continue

        vs, ve = full_year_window()

        def _to(x):
            return x.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                source_frames        = _to(item["source_frames"]),
                source_timestamps_ms = _to(item["source_timestamps_ms"]),
                source_frame_mask    = _to(item["source_frame_mask"]),
                source_input_mask    = _to(item["source_input_mask"]),
                source_type_ids      = _to(item["source_type_ids"]),
                valid_start_ms       = torch.tensor([vs], dtype=torch.int64, device=device),
                valid_end_ms         = torch.tensor([ve], dtype=torch.int64, device=device),
                target_relative_time = torch.zeros(1, cfg.data.num_target_sources, device=device),
                target_metadata      = torch.zeros(1, cfg.data.num_target_sources,
                                                    cfg.data.metadata_dim, device=device),
                skip_decoder=True,
            )
        emb_map = F.normalize(out.embedding_map, p=2, dim=1)  # [1, D, H, W]
        return emb_map.squeeze(0).cpu().numpy()                 # [D, H, W]
    return None


# ── 数据集构建 ────────────────────────────────────────────────────────────────

def build_pixel_dataset(spatial_maps: np.ndarray,
                        patch_ids: list[str],
                        label_fn,
                        num_classes: int,
                        task_name: str) -> tuple[np.ndarray, np.ndarray]:
    """从 spatial_maps 和标注构建像素级特征/标签矩阵。

    Args:
        spatial_maps: [N, D, H, W]
        patch_ids: 长度 N
        label_fn: 接受 patch_id → ndarray [H_orig, W_orig] | None
        num_classes: 类别数
    Returns:
        X [N_pixels, D], y [N_pixels]
    """
    _, D, H, W = spatial_maps.shape
    all_X, all_y = [], []

    for i, pid in enumerate(patch_ids):
        label = label_fn(pid)
        if label is None:
            continue
        label_r = resize_label_nearest(label, H, W)  # [H, W]
        feat = spatial_maps[i]                         # [D, H, W]

        label_flat = label_r.flatten().astype(np.int16)
        feat_flat  = feat.reshape(D, -1).T             # [H*W, D]

        # 过滤无效像素
        valid = (label_flat >= 0) & (label_flat < num_classes)
        if not valid.any():
            continue

        all_X.append(feat_flat[valid])
        all_y.append(label_flat[valid])

    if not all_X:
        return np.zeros((0, spatial_maps.shape[1])), np.zeros(0, dtype=np.int16)

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ── 训练 MLP ─────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              num_classes: int, device: torch.device,
              epochs: int = 50, lr: float = 1e-3,
              batch_size: int = 2048) -> dict:
    """训练 TinyMLP 并返回测试集指标。"""
    in_dim = X_train.shape[1]
    mlp = TinyMLP(in_dim=in_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cw = compute_class_weights(y_train, num_classes)
    weight_t = torch.from_numpy(cw).to(device)

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train.astype(np.int64)).to(device)

    for epoch in range(epochs):
        mlp.train()
        idx = torch.randperm(len(Xt), device=device)
        for start in range(0, len(Xt), batch_size):
            b = idx[start:start + batch_size]
            logits = mlp(Xt[b])
            loss = F.cross_entropy(logits, yt[b], weight=weight_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # 评估
    mlp.eval()
    Xe = torch.from_numpy(X_test).float()
    preds = []
    with torch.no_grad():
        for start in range(0, len(Xe), batch_size):
            xb = Xe[start:start + batch_size].to(device)
            preds.append(mlp(xb).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)

    oa = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1_per = f1_score(y_test, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    f1_macro = float(np.mean(f1_per))
    try:
        iou_per = jaccard_score(y_test, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
        miou = float(np.mean(iou_per))
    except Exception:
        iou_per = np.zeros(num_classes)
        miou = 0.0

    return {
        "oa": float(oa),
        "ba": float(ba),
        "macro_f1": f1_macro,
        "miou": miou,
        "f1_per_class": f1_per.tolist(),
        "iou_per_class": iou_per.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ── KNN 基线 ─────────────────────────────────────────────────────────────────

_KNN_MAX_TRAIN = 50_000  # 训练像素上限，避免全量 cdist OOM（1M×1M 需数十 GB）


def knn_eval(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             num_classes: int, k: int = 5,
             device: torch.device | None = None) -> dict:
    """sklearn BallTree KNN 基线（自动降采样到 _KNN_MAX_TRAIN）。

    原 torch.cdist 实现在 1M 训练像素时每批产生 4GB 中间矩阵，无法完成。
    改用 sklearn NearestNeighbors(algorithm='ball_tree') + 分层随机降采样。
    """
    from sklearn.neighbors import NearestNeighbors

    # 分层降采样：每类均匀采样，总量不超过 _KNN_MAX_TRAIN
    if len(X_train) > _KNN_MAX_TRAIN:
        classes = np.unique(y_train)
        per_class = max(1, _KNN_MAX_TRAIN // len(classes))
        sel = []
        for c in classes:
            idx_c = np.where(y_train == c)[0]
            n = min(per_class, len(idx_c))
            sel.append(np.random.choice(idx_c, n, replace=False))
        sel = np.concatenate(sel)
        X_tr_sub = X_train[sel]
        y_tr_sub = y_train[sel]
        print(f"    [KNN] 降采样 {len(X_train)} → {len(X_tr_sub)} 训练像素")
    else:
        X_tr_sub, y_tr_sub = X_train, y_train

    nn = NearestNeighbors(n_neighbors=min(k, len(X_tr_sub)),
                          algorithm="ball_tree", n_jobs=-1)
    nn.fit(X_tr_sub)
    _, idx = nn.kneighbors(X_test)
    y_pred = np.array([
        np.bincount(y_tr_sub[idx[i]].astype(np.int64)).argmax()
        for i in range(len(X_test))
    ])

    oa = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        miou = jaccard_score(y_test, y_pred, average="macro", zero_division=0)
    except Exception:
        miou = 0.0
    return {"oa": float(oa), "macro_f1": float(f1_macro), "miou": float(miou)}


# ── 线性探测（方式一，跟随 pipeline.py 标准）────────────────────────────────────

def eval_linear_probe(X: np.ndarray, y: np.ndarray,
                      num_classes: int,
                      task_type: str = "multi",
                      n_folds: int = 3,
                      max_train: int = 30_000) -> dict:
    """LogisticRegression + K-Fold(3) + balanced_accuracy。

    与 pipeline.py 的 evaluate_semantic_task / evaluate_binary_task 保持一致：
      - 多分类：multi_class='multinomial', solver='lbfgs'
      - 二分类：默认 LR
      - 每折训练集最多 max_train 像素（分层随机采样）
      - 主指标：balanced_accuracy（不受类别不平衡影响）
    """
    present_classes = np.unique(y)
    if len(present_classes) < 2:
        return {"error": f"only {len(present_classes)} class"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    ba_scores, f1m_scores, miou_scores = [], [], []
    f1_bin, iou_bin, auc_bin = [], [], []   # 二分类专用

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 分层降采样到 max_train
        if len(y_tr) > max_train:
            classes = np.unique(y_tr)
            per_class = max(1, max_train // len(classes))
            sel = []
            for c in classes:
                idx_c = np.where(y_tr == c)[0]
                n = min(per_class, len(idx_c))
                sel.append(np.random.choice(idx_c, n, replace=False))
            sel = np.concatenate(sel)
            X_tr, y_tr = X_tr[sel], y_tr[sel]

        if task_type == "binary":
            clf = LogisticRegression(max_iter=300, n_jobs=4, random_state=42)
        else:
            clf = LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=4, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        ba_scores.append(balanced_accuracy_score(y_te, y_pred))
        f1m_scores.append(f1_score(y_te, y_pred, average="macro",
                                   labels=present_classes, zero_division=0))

        ious = []
        for c in present_classes:
            inter = int(((y_pred == c) & (y_te == c)).sum())
            union = int(((y_pred == c) | (y_te == c)).sum())
            ious.append(inter / max(union, 1))
        miou_scores.append(float(np.mean(ious)))

        if task_type == "binary":
            f1_bin.append(f1_score(y_te, y_pred, zero_division=0))
            inter = int(((y_pred == 1) & (y_te == 1)).sum())
            union = int(((y_pred == 1) | (y_te == 1)).sum())
            iou_bin.append(inter / max(union, 1))
            try:
                y_prob = clf.predict_proba(X_te)[:, 1]
                auc_bin.append(float(roc_auc_score(y_te, y_prob)))
            except ValueError:
                auc_bin.append(0.5)

        print(f"    Fold {fold_idx+1}/{n_folds}: BAcc={ba_scores[-1]:.4f}  "
              f"mF1={f1m_scores[-1]:.4f}  mIoU={miou_scores[-1]:.4f}"
              + (f"  AUC={auc_bin[-1]:.4f}" if task_type == "binary" else ""))

    result: dict = {
        "balanced_accuracy": float(np.mean(ba_scores)),
        "f1_macro": float(np.mean(f1m_scores)),
        "miou": float(np.mean(miou_scores)),
        "n_classes": int(len(present_classes)),
        "n_pixels": int(len(y)),
    }
    if task_type == "binary":
        result["f1_water"] = float(np.mean(f1_bin))
        result["iou_water"] = float(np.mean(iou_bin))
        result["auc"] = float(np.mean(auc_bin))
    return result


# ── Few-shot 实验 ─────────────────────────────────────────────────────────────

def fewshot_experiment(X_train_full: np.ndarray, y_train_full: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       num_classes: int, device: torch.device,
                       pixel_budgets: list[int]) -> dict:
    """用不同数量的训练像素训练 MLP，绘制 few-shot 学习曲线。"""
    results = {}
    done_full = False
    for budget in pixel_budgets:
        if budget >= len(X_train_full):
            if done_full:
                continue  # 跳过重复的"全量"实验
            X_tr, y_tr = X_train_full, y_train_full
            label = "full"
            done_full = True
        else:
            # 按类别均匀采样（每类 budget // num_classes，不足则用实际有的）
            selected = []
            classes = np.unique(y_train_full)
            per_class = max(1, budget // len(classes))
            for c in classes:
                idx_c = np.where(y_train_full == c)[0]
                n = min(per_class, len(idx_c))
                chosen = np.random.choice(idx_c, n, replace=False)
                selected.append(chosen)
            sel = np.concatenate(selected)
            X_tr = X_train_full[sel]
            y_tr = y_train_full[sel]
            label = str(budget)

        if len(X_tr) < 10:
            continue

        m = train_mlp(X_tr, y_tr, X_test, y_test, num_classes, device, epochs=50)
        results[label] = {
            "n_train": len(X_tr),
            "oa": m["oa"],
            "macro_f1": m["macro_f1"],
            "miou": m.get("miou", 0.0),
        }
        print(f"    budget={label:>6}  n={len(X_tr):>7}  OA={m['oa']:.4f}  mF1={m['macro_f1']:.4f}  mIoU={m.get('miou', 0.):.4f}")
    return results


# ── 从 NPZ 加载预提取 embedding ───────────────────────────────────────────────

def load_spatial_maps_from_npz(npz_path: str | Path,
                               month_str: str = "2025-06") -> tuple[np.ndarray, list[str]]:
    """从 extract_embeddings.py 生成的 NPZ 加载 spatial maps。

    NPZ 格式：spatial_maps [N, N_months, D, H, W], patch_ids [N], month_labels [N_months]
    返回：单月 spatial_maps [N, D, H, W]（L2 normalized），patch_ids
    """
    data = np.load(npz_path, allow_pickle=True)
    sm     = data["spatial_maps"].astype(np.float32)   # [N, N_months, D, H, W]
    pids   = [str(p) for p in data["patch_ids"]]
    mlabels = [str(m) for m in data["month_labels"]]   # e.g. ['2025-04', ..., '2025-10']

    if month_str in mlabels:
        midx = mlabels.index(month_str)
    else:
        midx = 2  # fallback: 第3个月（通常 2025-06）
        print(f"  [警告] month_str={month_str!r} 不在 {mlabels}，fallback → index {midx} ({mlabels[midx]})")

    sm_sel = sm[:, midx]  # [N, D, H, W]

    # L2 normalize
    norm = np.linalg.norm(sm_sel, axis=1, keepdims=True).clip(min=1e-8)
    sm_norm = sm_sel / norm

    print(f"[加载 NPZ] {npz_path}")
    print(f"  原始: {sm.shape}  →  选月 '{mlabels[midx]}' (index={midx}): {sm_norm.shape}  patches={len(pids)}")
    return sm_norm, pids


# ── 主评估流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    # 快速路径：直接加载已提取的 NPZ（推荐）
    parser.add_argument("--embedding-file", default=None,
                        help="extract_embeddings.py 输出的 NPZ 文件（推荐，无需加载模型）")
    # 内联提取路径（首次 NPU JIT 编译慢）
    parser.add_argument("--config", default="configs/config_haidian_v25.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="npu:0",
                        help="MLP 训练设备（cpu/npu:0）；内联提取时也用于模型推理")
    parser.add_argument("--output-dir", default="out/eval_v25_0531/mlp/")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-patches", type=int, default=0, help="限制 patch 数（0=全量）")
    parser.add_argument("--month", type=str, default="2025-06",
                        help="选择哪个月的 embedding（NPZ 中的 month_label，如 '2025-06'）")
    parser.add_argument("--method", type=str, default="both",
                        choices=["linear", "mlp", "both"],
                        help="linear=线性探测(方式一) mlp=MLP微调(方式二) both=两者都做")
    args = parser.parse_args()

    if args.embedding_file is None and args.checkpoint is None:
        parser.error("必须提供 --embedding-file 或 --checkpoint")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 路径 A：从 NPZ 直接加载（推荐，快） ──────────────────────────────────
    if args.embedding_file:
        spatial_maps, valid_patches = load_spatial_maps_from_npz(args.embedding_file,
                                                                   month_str=args.month)
        checkpoint_label = args.embedding_file
    else:
        # ── 路径 B：内联提取（需 NPU，首次 JIT 编译约 5-10 min） ─────────────
        print("[加载] 配置和数据集（内联提取模式）...")
        from src.config import load_config
        from src.data.dataset import HarbinPatchDataset
        from src.models.model import AEFModel

        cfg = load_config(args.config)
        cfg.data.preload = False

        dataset = HarbinPatchDataset(cfg=cfg)
        sample_map = {
            (pid, y, m): idx
            for idx, (pid, y, m) in enumerate(dataset.monthly_samples)
        }
        all_patches = sorted({pid for pid, _, _ in dataset.monthly_samples})

        if args.num_patches > 0:
            all_patches = all_patches[:args.num_patches]
        print(f"[信息] patch 数: {len(all_patches)}")

        print(f"[加载] 模型到 {device}  (首次 NPU JIT 约 5-10 min，请耐心等待)...")
        model = AEFModel(cfg=cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()
        print(f"[加载] checkpoint: {args.checkpoint}")

        print(f"\n[提取] {len(all_patches)} patches  (每 patch ~2-10s)...")
        _maps, _pids = [], []
        t0 = time_mod.time()
        for i, pid in enumerate(all_patches):
            emb_map = extract_spatial_map(model, dataset, cfg, sample_map, device, pid)
            if emb_map is None:
                print(f"  [跳过] {pid}")
                continue
            _maps.append(emb_map)
            _pids.append(pid)
            if (i + 1) % 20 == 0:
                el = time_mod.time() - t0
                eta = el / (i + 1) * (len(all_patches) - i - 1)
                print(f"  [{i+1}/{len(all_patches)}] {el:.0f}s  ETA={eta:.0f}s")

        spatial_maps   = np.stack(_maps, axis=0)
        valid_patches  = _pids
        checkpoint_label = args.checkpoint
        print(f"[完成] spatial_maps={spatial_maps.shape}")

    # ── 计算 erank ──────────────────────────────────────────────────────────
    D, H, W = spatial_maps.shape[1], spatial_maps.shape[2], spatial_maps.shape[3]
    emb_flat = spatial_maps.reshape(len(valid_patches), D, -1).mean(-1)  # [N, D]
    emb_t = torch.from_numpy(emb_flat).float()
    emb_c = emb_t - emb_t.mean(0, keepdim=True)
    try:
        S = torch.linalg.svdvals(emb_c)
    except Exception:
        S = torch.svd(emb_c, compute_uv=False).S
    S = S.clamp(min=1e-8)
    p = S / S.sum()
    erank = float((-(p * p.log()).sum()).exp())
    print(f"\n[诊断] erank = {erank:.2f} ({'✅ 正常' if erank > 8 else '⚠️ 偏低（目标>8）'})")

    # ── 方式一：线性探测（LogisticRegression + K-Fold + balanced_accuracy）──
    linear_results: dict = {}
    if args.method in ("linear", "both"):
        print("\n" + "="*60)
        print("【方式一：线性探测 — LogisticRegression + K-Fold(3)】")
        print("="*60)

        # WorldCover（全量 patches）
        print("\n  [线性] WorldCover 6 类...")
        X_wc_all, y_wc_all = build_pixel_dataset(spatial_maps, valid_patches,
                                                  load_worldcover, WC_NUM_CLASSES, "worldcover")
        print(f"    像素总量: {len(X_wc_all)}, 类别分布: {np.bincount(y_wc_all, minlength=WC_NUM_CLASSES).tolist()}")
        linear_results["worldcover"] = eval_linear_probe(X_wc_all, y_wc_all, WC_NUM_CLASSES, task_type="multi")
        lw = linear_results["worldcover"]
        print(f"    BAcc={lw['balanced_accuracy']:.4f}  mF1={lw['f1_macro']:.4f}  mIoU={lw['miou']:.4f}")

        # JRC Water
        print("\n  [线性] JRC Water 二值...")
        X_jrc_all, y_jrc_all = build_pixel_dataset(spatial_maps, valid_patches,
                                                    load_jrc_water, JRC_NUM_CLASSES, "jrc_water")
        cnts = np.bincount(y_jrc_all, minlength=2)
        print(f"    像素总量: {len(X_jrc_all)}, land={cnts[0]} water={cnts[1]} ratio={cnts[1]/len(y_jrc_all):.3f}")
        linear_results["jrc_water"] = eval_linear_probe(X_jrc_all, y_jrc_all, JRC_NUM_CLASSES, task_type="binary")
        lj = linear_results["jrc_water"]
        print(f"    BAcc={lj['balanced_accuracy']:.4f}  F1-water={lj['f1_water']:.4f}  IoU-water={lj['iou_water']:.4f}  AUC={lj['auc']:.4f}")

        # Dynamic World
        print("\n  [线性] Dynamic World 9 类...")
        X_dw_all, y_dw_all = build_pixel_dataset(spatial_maps, valid_patches,
                                                  load_dynamic_world, DW_NUM_CLASSES, "dynamic_world")
        print(f"    像素总量: {len(X_dw_all)}, 类别分布: {np.bincount(y_dw_all, minlength=DW_NUM_CLASSES).tolist()}")
        if len(X_dw_all) > 0:
            linear_results["dynamic_world"] = eval_linear_probe(X_dw_all, y_dw_all, DW_NUM_CLASSES, task_type="multi")
            ld = linear_results["dynamic_world"]
            print(f"    BAcc={ld['balanced_accuracy']:.4f}  mF1={ld['f1_macro']:.4f}  mIoU={ld['miou']:.4f}")
        else:
            print("    [跳过] 像素数不足")
            linear_results["dynamic_world"] = {"error": "no pixels"}

    # ── 跳过 MLP 如果 --method linear ────────────────────────────────────────
    if args.method == "linear":
        print("\n[方式一完成，跳过 MLP 微调]")
        out_file = output_dir / "eval_results.json"
        out_data = {"linear_probe": linear_results, "erank": erank}
        with open(out_file, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n[保存] 结果已写入 {out_file}")
        return

    # ── 方式二：MLP 微调（神经网络 + 80/20 patch split）──────────────────────
    print("\n" + "="*60)
    print("【方式二：MLP 微调 — PixelMLP + 80/20 split】")
    print("="*60)
    n = len(valid_patches)
    n_train = int(n * args.train_ratio)
    idx_perm = np.random.permutation(n)
    train_idx = idx_perm[:n_train]
    test_idx  = idx_perm[n_train:]

    train_patches = [valid_patches[i] for i in train_idx]
    test_patches  = [valid_patches[i] for i in test_idx]
    train_maps = spatial_maps[train_idx]
    test_maps  = spatial_maps[test_idx]

    print(f"[划分] train={len(train_patches)} patches, test={len(test_patches)} patches")

    results: dict = {
        "checkpoint": checkpoint_label,
        "n_patches": len(valid_patches),
        "n_train_patches": len(train_patches),
        "n_test_patches": len(test_patches),
        "erank": erank,
        "spatial_map_shape": [D, H, W],
        "linear_probe": linear_results,
    }

    # ── 任务 A：WorldCover ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[任务 A] WorldCover 6 类分类")
    print("="*60)

    X_tr_wc, y_tr_wc = build_pixel_dataset(train_maps, train_patches, load_worldcover, WC_NUM_CLASSES, "worldcover")
    X_te_wc, y_te_wc = build_pixel_dataset(test_maps,  test_patches,  load_worldcover, WC_NUM_CLASSES, "worldcover")
    print(f"  train pixels={len(X_tr_wc)}, test pixels={len(X_te_wc)}")
    print(f"  train 类别分布: {np.bincount(y_tr_wc, minlength=WC_NUM_CLASSES).tolist()}")
    print(f"  test  类别分布: {np.bincount(y_te_wc, minlength=WC_NUM_CLASSES).tolist()}")

    print("\n[任务 A] MLP 全量训练 (epochs={})...".format(args.epochs))
    wc_mlp_full = train_mlp(X_tr_wc, y_tr_wc, X_te_wc, y_te_wc, WC_NUM_CLASSES, device, epochs=args.epochs)
    print(f"  OA={wc_mlp_full['oa']:.4f}  mF1={wc_mlp_full['macro_f1']:.4f}  mIoU={wc_mlp_full['miou']:.4f}")
    for ci, cn in enumerate(WC_NAMES):
        f1 = wc_mlp_full['f1_per_class'][ci]
        iou = wc_mlp_full['iou_per_class'][ci]
        print(f"    {cn:12s}: F1={f1:.4f}  IoU={iou:.4f}")

    print("\n[任务 A] Few-shot 学习曲线...")
    pixel_budgets = [100, 1000, 10000, 50000]
    wc_fewshot = fewshot_experiment(
        X_tr_wc, y_tr_wc, X_te_wc, y_te_wc,
        WC_NUM_CLASSES, device, pixel_budgets
    )

    results["worldcover"] = {
        "mlp_full": wc_mlp_full,
        "fewshot": wc_fewshot,
    }

    # ── 任务 B：JRC Water ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[任务 B] JRC Water 二值分类 (occurrence > 50 = water)")
    print("="*60)

    X_tr_jrc, y_tr_jrc = build_pixel_dataset(train_maps, train_patches, load_jrc_water, JRC_NUM_CLASSES, "jrc_water")
    X_te_jrc, y_te_jrc = build_pixel_dataset(test_maps,  test_patches,  load_jrc_water, JRC_NUM_CLASSES, "jrc_water")
    print(f"  train pixels={len(X_tr_jrc)}, test pixels={len(X_te_jrc)}")
    cnts_tr = np.bincount(y_tr_jrc, minlength=2)
    cnts_te = np.bincount(y_te_jrc, minlength=2)
    print(f"  train: land={cnts_tr[0]} water={cnts_tr[1]} water_ratio={cnts_tr[1]/len(y_tr_jrc):.3f}")
    print(f"  test:  land={cnts_te[0]} water={cnts_te[1]} water_ratio={cnts_te[1]/len(y_te_jrc):.3f}")

    print("\n[任务 B] MLP 全量训练...")
    jrc_mlp = train_mlp(X_tr_jrc, y_tr_jrc, X_te_jrc, y_te_jrc, JRC_NUM_CLASSES, device, epochs=args.epochs)
    f1_water = jrc_mlp['f1_per_class'][1]
    print(f"  OA={jrc_mlp['oa']:.4f}  F1-water={f1_water:.4f}  mF1={jrc_mlp['macro_f1']:.4f}")

    results["jrc_water"] = {
        "mlp_full": jrc_mlp,
    }

    # ── 任务 C：Dynamic World ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("[任务 C] Dynamic World 9 类分类（参考，标注连续值 round 取整）")
    print("="*60)

    X_tr_dw, y_tr_dw = build_pixel_dataset(train_maps, train_patches, load_dynamic_world, DW_NUM_CLASSES, "dynamic_world")
    X_te_dw, y_te_dw = build_pixel_dataset(test_maps,  test_patches,  load_dynamic_world, DW_NUM_CLASSES, "dynamic_world")
    print(f"  train pixels={len(X_tr_dw)}, test pixels={len(X_te_dw)}")
    print(f"  train 类别分布: {np.bincount(y_tr_dw, minlength=DW_NUM_CLASSES).tolist()}")

    if len(X_tr_dw) > 0 and len(X_te_dw) > 0:
        print("\n[任务 C] MLP 全量训练...")
        dw_mlp = train_mlp(X_tr_dw, y_tr_dw, X_te_dw, y_te_dw, DW_NUM_CLASSES, device, epochs=args.epochs)
        print(f"  OA={dw_mlp['oa']:.4f}  mF1={dw_mlp['macro_f1']:.4f}")
        results["dynamic_world"] = {"mlp_full": dw_mlp}
    else:
        print("  [跳过] 像素数不足")
        results["dynamic_world"] = None

    # ── 汇总输出 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("【评估汇总】")
    print("="*60)
    print(f"  erank = {erank:.2f}  月份 = {args.month}")

    if linear_results:
        print("\n  ── 方式一：线性探测（BAcc = balanced_accuracy）──")
        lw = linear_results.get("worldcover", {})
        lj = linear_results.get("jrc_water", {})
        ld = linear_results.get("dynamic_world", {})
        if lw and "balanced_accuracy" in lw:
            print(f"  WorldCover 6类:   BAcc={lw['balanced_accuracy']:.4f}  mF1={lw['f1_macro']:.4f}  mIoU={lw['miou']:.4f}")
        if lj and "balanced_accuracy" in lj:
            print(f"  JRC Water 二值:   BAcc={lj['balanced_accuracy']:.4f}  F1-water={lj['f1_water']:.4f}  AUC={lj['auc']:.4f}")
        if ld and "balanced_accuracy" in ld:
            print(f"  Dynamic World 9类: BAcc={ld['balanced_accuracy']:.4f}  mF1={ld['f1_macro']:.4f}  mIoU={ld['miou']:.4f}")

    print(f"\n  ── 方式二：MLP 微调（OA + BAcc）──")
    print(f"  WorldCover 6类:   OA={wc_mlp_full['oa']:.4f}  BAcc={wc_mlp_full['ba']:.4f}  mF1={wc_mlp_full['macro_f1']:.4f}  mIoU={wc_mlp_full['miou']:.4f}")
    print(f"  JRC Water 二值:   OA={jrc_mlp['oa']:.4f}  BAcc={jrc_mlp['ba']:.4f}  F1-water={f1_water:.4f}")
    if results.get("dynamic_world"):
        dw = results["dynamic_world"]
        print(f"  Dynamic World 9类: OA={dw['mlp_full']['oa']:.4f}  BAcc={dw['mlp_full']['ba']:.4f}  mF1={dw['mlp_full']['macro_f1']:.4f}")

    print("\n  WorldCover Few-shot 学习曲线:")
    for budget_key, v in wc_fewshot.items():
        print(f"    {budget_key:>8} pix | OA={v['oa']:.4f}  mF1={v['macro_f1']:.4f}  mIoU={v['miou']:.4f}")

    # 保存结果
    out_file = output_dir / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[保存] 结果已写入 {out_file}")


if __name__ == "__main__":
    main()
