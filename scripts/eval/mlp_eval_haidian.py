#!/usr/bin/env python
"""海淀数据集 MLP 微调下游评估。

冻结 backbone，提取像素级 spatial embedding，用轻量 MLP 评估三个任务：
  A. WorldCover 6 类土地覆盖分类（主力）
  B. JRC Water 二值水体检测（occurrence > 50 = water）
  C. Dynamic World 9 类分类（参考）

用法：
    python scripts/eval/mlp_eval_haidian.py \\
        --config configs/config_haidian_v25.yaml \\
        --checkpoint /workspace/outputs/exp_v25_haidian_loss_opt_0530/epoch_best_epoch78.pt \\
        --device npu:0 \\
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
    accuracy_score, f1_score, jaccard_score, confusion_matrix, classification_report
)

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
        "macro_f1": f1_macro,
        "miou": miou,
        "f1_per_class": f1_per.tolist(),
        "iou_per_class": iou_per.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ── KNN 基线 ─────────────────────────────────────────────────────────────────

def knn_eval(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             num_classes: int, k: int = 5,
             device: torch.device | None = None) -> dict:
    """PyTorch KNN 基线。"""
    dev = device or torch.device("cpu")
    X_tr = torch.from_numpy(X_train).float().to(dev)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(dev)
    preds = []
    batch = 1024
    with torch.no_grad():
        for i in range(0, len(X_test), batch):
            q = torch.from_numpy(X_test[i:i + batch]).float().to(dev)
            dist = torch.cdist(q, X_tr)
            _, idx = dist.topk(min(k, len(X_train)), largest=False, dim=1)
            nbr = y_tr[idx]
            for j in range(nbr.shape[0]):
                vals, cnts = torch.unique(nbr[j], return_counts=True)
                preds.append(vals[cnts.argmax()].item())
    y_pred = np.array(preds)
    oa = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        miou = jaccard_score(y_test, y_pred, average="macro", zero_division=0)
    except Exception:
        miou = 0.0
    return {"oa": float(oa), "macro_f1": float(f1_macro), "miou": float(miou)}


# ── Few-shot 实验 ─────────────────────────────────────────────────────────────

def fewshot_experiment(X_train_full: np.ndarray, y_train_full: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       num_classes: int, device: torch.device,
                       pixel_budgets: list[int]) -> dict:
    """用不同数量的训练像素训练 MLP，绘制 few-shot 学习曲线。"""
    results = {}
    for budget in pixel_budgets:
        if budget >= len(X_train_full):
            X_tr, y_tr = X_train_full, y_train_full
            label = "full"
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


# ── 主评估流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_haidian_v25.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output-dir", default="out/eval_v25_0531/mlp/")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-patches", type=int, default=0, help="限制评估 patch 数（0=全量）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型和数据集 ────────────────────────────────────────────────────
    print("[加载] 配置和数据集...")
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
    print(f"[信息] 总 patch 数: {len(all_patches)}")

    if args.num_patches > 0:
        all_patches = all_patches[: args.num_patches]
        print(f"[信息] 限制为前 {len(all_patches)} 个 patch")

    print("[加载] 模型...")
    model = AEFModel(cfg=cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[加载] checkpoint: {args.checkpoint}")

    # ── 提取空间 embedding ──────────────────────────────────────────────────
    print(f"\n[提取] {len(all_patches)} patches 的 spatial embedding map...")
    spatial_maps = []
    valid_patches = []
    t0 = time_mod.time()

    for i, pid in enumerate(all_patches):
        emb_map = extract_spatial_map(model, dataset, cfg, sample_map, device, pid)
        if emb_map is None:
            print(f"  [跳过] {pid}: 无法提取")
            continue
        spatial_maps.append(emb_map)
        valid_patches.append(pid)
        if (i + 1) % 50 == 0:
            elapsed = time_mod.time() - t0
            eta = elapsed / (i + 1) * (len(all_patches) - i - 1)
            print(f"  [{i+1}/{len(all_patches)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    spatial_maps = np.stack(spatial_maps, axis=0)  # [N, D, H, W]
    D, H, W = spatial_maps.shape[1], spatial_maps.shape[2], spatial_maps.shape[3]
    print(f"[完成] 提取 {len(valid_patches)} patches，spatial_maps={spatial_maps.shape}")

    # ── 计算 erank ──────────────────────────────────────────────────────────
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

    # ── 训练/测试 split ────────────────────────────────────────────────────
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
        "checkpoint": args.checkpoint,
        "n_patches": len(valid_patches),
        "n_train_patches": len(train_patches),
        "n_test_patches": len(test_patches),
        "erank": erank,
        "spatial_map_shape": [D, H, W],
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

    print("\n[任务 A] KNN k=5 基线...")
    wc_knn = knn_eval(X_tr_wc, y_tr_wc, X_te_wc, y_te_wc, WC_NUM_CLASSES, k=5, device=device)
    print(f"  OA={wc_knn['oa']:.4f}  mF1={wc_knn['macro_f1']:.4f}  mIoU={wc_knn['miou']:.4f}")

    print("\n[任务 A] Few-shot 学习曲线...")
    pixel_budgets = [100, 1000, 10000, 50000]
    wc_fewshot = fewshot_experiment(
        X_tr_wc, y_tr_wc, X_te_wc, y_te_wc,
        WC_NUM_CLASSES, device, pixel_budgets
    )

    results["worldcover"] = {
        "mlp_full": wc_mlp_full,
        "knn_k5": wc_knn,
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

    print("\n[任务 B] KNN k=5 基线...")
    jrc_knn = knn_eval(X_tr_jrc, y_tr_jrc, X_te_jrc, y_te_jrc, JRC_NUM_CLASSES, k=5, device=device)
    print(f"  OA={jrc_knn['oa']:.4f}  mF1={jrc_knn['macro_f1']:.4f}")

    results["jrc_water"] = {
        "mlp_full": jrc_mlp,
        "knn_k5": jrc_knn,
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

        print("\n[任务 C] KNN k=5 基线...")
        dw_knn = knn_eval(X_tr_dw, y_tr_dw, X_te_dw, y_te_dw, DW_NUM_CLASSES, k=5, device=device)
        print(f"  OA={dw_knn['oa']:.4f}  mF1={dw_knn['macro_f1']:.4f}")
        results["dynamic_world"] = {"mlp_full": dw_mlp, "knn_k5": dw_knn}
    else:
        print("  [跳过] 像素数不足")
        results["dynamic_world"] = None

    # ── 汇总输出 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("【评估汇总】")
    print("="*60)
    print(f"  erank = {erank:.2f}")
    print(f"\n  WorldCover 6类:")
    print(f"    KNN k=5   | OA={wc_knn['oa']:.4f}  mF1={wc_knn['macro_f1']:.4f}  mIoU={wc_knn['miou']:.4f}")
    print(f"    MLP 全量  | OA={wc_mlp_full['oa']:.4f}  mF1={wc_mlp_full['macro_f1']:.4f}  mIoU={wc_mlp_full['miou']:.4f}")
    print(f"\n  JRC Water 二值:")
    print(f"    KNN k=5   | OA={jrc_knn['oa']:.4f}  mF1={jrc_knn['macro_f1']:.4f}")
    print(f"    MLP 全量  | OA={jrc_mlp['oa']:.4f}  F1-water={f1_water:.4f}  mF1={jrc_mlp['macro_f1']:.4f}")

    if results.get("dynamic_world"):
        dw = results["dynamic_world"]
        print(f"\n  Dynamic World 9类（参考）:")
        print(f"    KNN k=5   | OA={dw['knn_k5']['oa']:.4f}  mF1={dw['knn_k5']['macro_f1']:.4f}")
        print(f"    MLP 全量  | OA={dw['mlp_full']['oa']:.4f}  mF1={dw['mlp_full']['macro_f1']:.4f}")

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
