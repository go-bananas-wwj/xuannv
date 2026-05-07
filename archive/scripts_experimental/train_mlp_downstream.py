#!/usr/bin/env python3
"""
MLP 下游变化检测 — 为 3 个类别分别训练可学习的 MLP head
- 冻结 V2 backbone
- 特征: concat(before, after, diff, mul) = 512 维
- 5-fold patch-level CV
- 输出新的综合大图
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, Point
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

# ── 配置 ─────────────────────────────────────────────────────────
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH   = "/workspace/outputs/aef_qwen_v2/best.pt"
DEVICE      = torch.device("npu:0" if torch.npu.is_available() else "cpu")

SHP_DIR     = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
GRID_PATH   = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
Q_S2_ROOT   = Path("/workspace/raw/harbin/s2")
M_S2_ROOT   = Path("/workspace/raw/harbin_scenes/s2")

OUTPUT_DIR  = Path("/workspace/outputs/aef_qwen_v2/mlp_downstream")
OUTPUT_Q    = Path("/workspace/outputs/aef_qwen_v2/shp_maps_mlp_quarterly")
OUTPUT_M    = Path("/workspace/outputs/aef_qwen_v2/shp_maps_mlp_monthly")

def _date_to_ms(date_str: str) -> float:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return float(dt.timestamp() * 1000)


PERIODS = {
    "2025-04~2025-06": (_date_to_ms("2025-04-01"), _date_to_ms("2025-06-30")),
    "2025-06~2025-08": (_date_to_ms("2025-06-01"), _date_to_ms("2025-08-31")),
    "2025-08~2025-09": (_date_to_ms("2025-08-01"), _date_to_ms("2025-09-30")),
    "2025-09~2025-10": (_date_to_ms("2025-09-01"), _date_to_ms("2025-10-31")),
    "2025-all":        (_date_to_ms("2025-01-01"), _date_to_ms("2025-12-31")),
}

QUARTER_MAP = {
    "2025-04~2025-06": ("2025Q1", "2025Q2"),
    "2025-06~2025-08": ("2025Q2", "2025Q3"),
    "2025-08~2025-09": ("2025Q2", "2025Q3"),
    "2025-09~2025-10": ("2025Q3", "2025Q4"),
    "2025-all":        ("2025Q2", "2025Q3"),
}

SHP_FILES = {
    "june.shp":             {"period": "2025-04~2025-06", "category": "mixed"},
    "aug.shp":              {"period": "2025-06~2025-08", "category": "mixed"},
    "September.shp":        {"period": "2025-08~2025-09", "category": "mixed"},
    "October.shp":          {"period": "2025-09~2025-10", "category": "mixed"},
    "SAR建筑工地.shp":      {"period": "2025-all",        "category": "construction"},
    "SAR房屋拆除.shp":      {"period": "2025-all",        "category": "demolition"},
    "SAR非农非粮.shp":      {"period": "2025-all",        "category": "farmland"},
    "SAR疑似违建.shp":      {"period": "2025-all",        "category": "construction"},
}

CATEGORY_TO_CLASS = {
    "construction": 0,
    "demolition":   1,
    "farmland":     2,
}

PATCHES_PER_PAGE = 4

# ── 辅助函数 ─────────────────────────────────────────────────────
def load_model():
    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset


def extract_embedding(model, dataset, patch_id: str, w_start: float, w_end: float):
    if patch_id not in dataset.patches:
        return None
    pidx = dataset.patches.index(patch_id)
    batch = dataset[pidx]
    batch["valid_start_ms"] = torch.tensor(w_start, dtype=torch.float64)
    batch["valid_end_ms"]   = torch.tensor(w_end,   dtype=torch.float64)
    batch_dev = {k: v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        out = model(
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
    emb = out.embedding_map[0].cpu().numpy()  # [D, H, W]
    # L2 normalize
    norms = np.linalg.norm(emb, axis=0, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


# ── MLP 定义 ──────────────────────────────────────────────────────
class ChangeMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── 数据集构建 ────────────────────────────────────────────────────
def build_features(emb_before, emb_after):
    """方案 B: concat(before, after, diff, mul)"""
    D, H, W = emb_before.shape
    diff = emb_before - emb_after
    mul = emb_before * emb_after
    feat = np.concatenate([emb_before, emb_after, diff, mul], axis=0)  # [4D, H, W]
    return feat.reshape(feat.shape[0], -1).T  # [H*W, 4D]


def load_grid():
    return gpd.read_file(str(GRID_PATH))


def rasterize_shp_for_patch(patch_geom, shp_gdf, size=64):
    minx, miny, maxx, maxy = patch_geom.bounds
    transform = from_bounds(minx, miny, maxx, maxy, size, size)
    geoms = []
    for _, row in shp_gdf.iterrows():
        geom = row.geometry
        if geom is not None and geom.intersects(patch_geom):
            inter = geom.intersection(patch_geom)
            if not inter.is_empty:
                geoms.append((inter, 1))
    if not geoms:
        return np.zeros((size, size), dtype=np.uint8)
    mask = rasterize(geoms, out_shape=(size, size), transform=transform, fill=0, dtype=np.uint8)
    return mask


def gather_annotations():
    """按类别组织标注 patch."""
    from demo_v2.utils.harbin_annotations_v2 import load_harbin_annotations
    ann = load_harbin_annotations()
    cat_patches = defaultdict(set)
    for pid, recs in ann.items():
        for r in recs:
            cat = r.get("category")
            if cat in CATEGORY_TO_CLASS:
                cat_patches[cat].add(pid)
    return {k: sorted(v) for k, v in cat_patches.items()}


# ── 训练 ──────────────────────────────────────────────────────────
def train_mlp_for_category(cat_name, patch_ids, model, dataset, grid, epochs=100, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"Training MLP for category: {cat_name} ({len(patch_ids)} patches)")
    print(f"{'='*60}")

    # Collect all relevant SHP GDFs for this category
    relevant_shps = [k for k, v in SHP_FILES.items() if v["category"] == cat_name or v["category"] == "mixed"]
    shp_gdfs = {}
    for shp_name in relevant_shps:
        gdf = gpd.read_file(str(SHP_DIR / shp_name))
        if gdf.crs is None or gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        shp_gdfs[shp_name] = gdf

    # Extract embeddings and labels per patch
    patch_data = {}
    for pid in patch_ids:
        period = None
        for shp_name in relevant_shps:
            if SHP_FILES[shp_name]["category"] == cat_name:
                # SAR annotations are all-year
                period = SHP_FILES[shp_name]["period"]
                break
        if period is None:
            # Fallback: find from annotation period
            from demo_v2.utils.harbin_annotations_v2 import get_period_for_patch
            period = get_period_for_patch(pid)
        if period is None or period not in PERIODS:
            continue

        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0

        emb_before = extract_embedding(model, dataset, pid, bs, mid)
        emb_after  = extract_embedding(model, dataset, pid, mid, be)
        if emb_before is None or emb_after is None:
            continue

        H, W = emb_before.shape[1], emb_before.shape[2]
        features = build_features(emb_before, emb_after)  # [H*W, 512]

        # Build GT mask by unioning all relevant SHPs
        patch_row = grid[grid["patch_id"] == pid]
        if len(patch_row) == 0:
            continue
        patch_geom = patch_row.geometry.values[0]

        gt_mask = np.zeros((H, W), dtype=np.uint8)
        for shp_name, gdf in shp_gdfs.items():
            # For mixed SHPs, only count if they actually overlap this patch
            mask_part = rasterize_shp_for_patch(patch_geom, gdf, size=H)
            gt_mask = np.maximum(gt_mask, mask_part)

        labels = gt_mask.flatten()
        patch_data[pid] = {"features": features, "labels": labels}
        print(f"  {pid}: pos={labels.sum()}, neg={len(labels)-labels.sum()}")

    if len(patch_data) < 3:
        print(f"  [!] Not enough patches for {cat_name}")
        return None

    pids = list(patch_data.keys())
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_fold_state = None
    best_fold_auc = 0.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(pids)):
        train_pids = [pids[i] for i in train_idx]
        val_pids   = [pids[i] for i in val_idx]

        # Build train/val datasets
        X_train_list, y_train_list = [], []
        for pid in train_pids:
            d = patch_data[pid]
            pos_idx = np.where(d["labels"] == 1)[0]
            neg_idx = np.where(d["labels"] == 0)[0]
            n_neg = min(len(pos_idx) * 5, len(neg_idx))
            if len(pos_idx) == 0 or n_neg == 0:
                continue
            rng = np.random.RandomState(42 + fold)
            neg_sample = rng.choice(neg_idx, n_neg, replace=False)
            sel = np.concatenate([pos_idx, neg_sample])
            X_train_list.append(d["features"][sel])
            y_train_list.append(d["labels"][sel])

        X_val_list, y_val_list = [], []
        for pid in val_pids:
            d = patch_data[pid]
            pos_idx = np.where(d["labels"] == 1)[0]
            neg_idx = np.where(d["labels"] == 0)[0]
            n_neg = min(len(pos_idx) * 5, len(neg_idx))
            if len(pos_idx) == 0 or n_neg == 0:
                continue
            rng = np.random.RandomState(42 + fold)
            neg_sample = rng.choice(neg_idx, n_neg, replace=False)
            sel = np.concatenate([pos_idx, neg_sample])
            X_val_list.append(d["features"][sel])
            y_val_list.append(d["labels"][sel])

        if not X_train_list or not X_val_list:
            print(f"  Fold {fold+1}: insufficient data")
            continue

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_val   = np.concatenate(X_val_list,   axis=0)
        y_val   = np.concatenate(y_val_list,   axis=0)

        print(f"  Fold {fold+1}: train={len(X_train)} (pos={y_train.sum()}), val={len(X_val)} (pos={y_val.sum()})")

        # Dataloaders
        pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)
        pos_weight = min(pos_weight, 20.0)

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False)

        mlp = ChangeMLP(input_dim=512).to(DEVICE)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))

        best_val_auc = 0.0
        best_state = None
        patience = 15
        no_improve = 0

        for epoch in range(epochs):
            mlp.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = mlp(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            # Validation
            mlp.eval()
            val_probs_all = []
            val_labels_all = []
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = mlp(xb)
                    loss = criterion(logits, yb)
                    val_losses.append(loss.item())
                    val_probs_all.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels_all.append(yb.cpu().numpy())

            val_probs = np.concatenate(val_probs_all)
            val_labels = np.concatenate(val_labels_all)
            val_pred = (val_probs > 0.5).astype(float)

            val_auc = roc_auc_score(val_labels, val_probs)
            val_ba  = balanced_accuracy_score(val_labels, val_pred)
            val_f1  = f1_score(val_labels, val_pred, zero_division=0)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, val_auc={val_auc:.4f}, val_f1={val_f1:.4f}")

            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        print(f"  Fold {fold+1} best val AUC: {best_val_auc:.4f}")
        fold_results.append({
            "fold": fold + 1,
            "best_auc": float(best_val_auc),
            "val_ba": float(val_ba),
            "val_f1": float(val_f1),
        })

        if best_val_auc > best_fold_auc:
            best_fold_auc = best_val_auc
            best_fold_state = best_state

    # Save best head
    if best_fold_state:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / f"{cat_name}_best.pt"
        torch.save(best_fold_state, save_path)
        print(f"  Saved best head: {save_path}")

    summary = {
        "category": cat_name,
        "n_patches": len(patch_data),
        "fold_results": fold_results,
        "mean_auc": float(np.mean([r["best_auc"] for r in fold_results])) if fold_results else 0.0,
        "std_auc": float(np.std([r["best_auc"] for r in fold_results])) if fold_results else 0.0,
    }
    print(f"  {cat_name} mean AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    return summary, best_fold_state, patch_data


# ── 推理 + 可视化 ────────────────────────────────────────────────
def _read_s2_rgb(path: Path, target_size: int = 64) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with rasterio.open(str(path)) as src:
            data = src.read(out_shape=(src.count, target_size, target_size), resampling=Resampling.bilinear)
        if data.shape[0] < 3:
            return None
        rgb = data[[2, 1, 0]].astype(np.float32)
        valid = rgb[rgb > 0]
        if len(valid) == 0:
            return None
        p2, p98 = np.percentile(valid, [2, 98])
        if p98 > p2:
            rgb = (rgb - p2) / (p98 - p2)
        rgb = np.clip(rgb, 0, 1)
        return rgb.transpose(1, 2, 0)
    except Exception:
        return None


def load_rgb_quarterly(patch_id: str, period: str, target_size: int = 64):
    q_before, q_after = QUARTER_MAP[period]
    b = _read_s2_rgb(Q_S2_ROOT / patch_id / f"{q_before}.tif", target_size)
    a = _read_s2_rgb(Q_S2_ROOT / patch_id / f"{q_after}.tif", target_size)
    return b, a


def _load_scene_median(patch_id: str, start_ms: float, end_ms: float, target_size: int):
    s2_dir = M_S2_ROOT / patch_id
    if not s2_dir.exists():
        return None
    files = sorted(s2_dir.glob("*.tif"))
    valid_files = []
    for f in files:
        stem = f.stem
        if len(stem) >= 8 and stem.isdigit():
            ts = float(datetime.strptime(stem[:8], "%Y%m%d").timestamp() * 1000)
            if start_ms <= ts <= end_ms:
                valid_files.append(f)
    if not valid_files and files:
        candidates = []
        for f in files:
            stem = f.stem
            if len(stem) >= 8 and stem.isdigit():
                ts = float(datetime.strptime(stem[:8], "%Y%m%d").timestamp() * 1000)
                candidates.append((abs(ts - (start_ms + end_ms) / 2), f))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            valid_files = [candidates[0][1]]
    if not valid_files:
        return None
    frames = []
    for f in valid_files[:5]:
        rgb = _read_s2_rgb(f, target_size)
        if rgb is not None:
            frames.append(rgb)
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0)


def load_rgb_monthly(patch_id: str, period: str, target_size: int = 64):
    bs, be = PERIODS[period]
    mid = (bs + be) / 2.0
    b = _load_scene_median(patch_id, bs, mid, target_size)
    a = _load_scene_median(patch_id, mid, be, target_size)
    return b, a


def compute_unsupervised_score(emb_before, emb_after):
    D, H, W = emb_before.shape
    fb = emb_before.reshape(D, -1)
    fa = emb_after.reshape(D, -1)
    cos_sim = np.sum(fb * fa, axis=0)
    score = ((1.0 - cos_sim) / 2.0).reshape(H, W)
    return score


def embedding_pca_rgb(emb_before, emb_after):
    D, H, W = emb_before.shape
    concat = np.concatenate([emb_before, emb_after], axis=0)
    flat = concat.reshape(concat.shape[0], -1).T
    pca = PCA(n_components=3)
    pc = pca.fit_transform(flat)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        ch = pc[:, c]
        lo, hi = np.percentile(ch, (2, 98))
        if hi > lo:
            rgb[:, :, c] = ((ch - lo) / (hi - lo)).reshape(H, W)
        else:
            rgb[:, :, c] = 0.5
    rgb = np.clip(rgb, 0, 1)
    return rgb


def draw_page(patches_data, page_idx, shp_name, out_dir, mode_label):
    n = len(patches_data)
    fig, axes = plt.subplots(n, 6, figsize=(24, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    cols = ["Before RGB", "After RGB", "GT Mask", "MLP Prob", "Unsupervised", "Embedding PCA"]
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=12, fontweight="bold")

    for i, d in enumerate(patches_data):
        if d["rgb_before"] is not None:
            axes[i, 0].imshow(d["rgb_before"])
        axes[i, 0].set_ylabel(d["patch_id"], fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        if d["rgb_after"] is not None:
            axes[i, 1].imshow(d["rgb_after"])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        axes[i, 2].imshow(d["gt_mask"], cmap="Reds", interpolation="nearest")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        if d["prob_map"] is not None:
            im = axes[i, 3].imshow(d["prob_map"], cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

        im2 = axes[i, 4].imshow(d["score_map"], cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[i, 4], fraction=0.046)
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])

        axes[i, 5].imshow(d["pca_rgb"], interpolation="nearest")
        mask_overlay = d["gt_mask"].astype(float)
        mask_overlay[d["gt_mask"] == 0] = np.nan
        axes[i, 5].imshow(mask_overlay, cmap=ListedColormap(["red"]), alpha=0.3, interpolation="nearest")
        axes[i, 5].set_xticks([])
        axes[i, 5].set_yticks([])

    fig.suptitle(f"{shp_name} ({mode_label}) — Page {page_idx + 1}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = out_dir / shp_name.replace(".shp", "") / f"page_{page_idx + 1:02d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def inference_and_draw(model, dataset, grid, mlp_heads, mode="quarterly"):
    print(f"\n{'='*60}")
    print(f"Generating result maps ({mode})")
    print(f"{'='*60}")

    out_dir = OUTPUT_Q if mode == "quarterly" else OUTPUT_M

    # Load all SHP GDFs
    shp_gdfs = {}
    for shp_name in SHP_FILES:
        gdf = gpd.read_file(str(SHP_DIR / shp_name))
        if gdf.crs is None or gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        shp_gdfs[shp_name] = gdf

    for shp_name, info in SHP_FILES.items():
        period = info["period"]
        cat_name = info["category"]
        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0

        # Find intersecting patches
        joined = gpd.sjoin(grid, shp_gdfs[shp_name], how="inner", predicate="intersects")
        pids = joined["patch_id"].unique().tolist()
        if not pids:
            continue
        print(f"\n{shp_name}: {len(pids)} patches")

        # Select MLP head
        head = None
        if cat_name in mlp_heads and mlp_heads[cat_name] is not None:
            head = ChangeMLP(input_dim=512).to(DEVICE)
            head.load_state_dict(mlp_heads[cat_name])
            head.eval()

        all_data = []
        for pid in pids:
            patch_row = grid[grid["patch_id"] == pid]
            if len(patch_row) == 0:
                continue
            patch_geom = patch_row.geometry.values[0]

            emb_before = extract_embedding(model, dataset, pid, bs, mid)
            emb_after  = extract_embedding(model, dataset, pid, mid, be)
            if emb_before is None or emb_after is None:
                continue
            H, W = emb_before.shape[1], emb_after.shape[2]

            features = build_features(emb_before, emb_after)  # [H*W, 512]

            # MLP inference
            prob_map = None
            if head is not None:
                with torch.no_grad():
                    probs = []
                    batch_size = 4096
                    for i in range(0, len(features), batch_size):
                        xb = torch.FloatTensor(features[i:i+batch_size]).to(DEVICE)
                        probs.append(torch.sigmoid(head(xb)).cpu().numpy())
                    prob_map = np.concatenate(probs).reshape(H, W)

            # GT mask
            gt_mask = rasterize_shp_for_patch(patch_geom, shp_gdfs[shp_name], size=H)

            # Unsupervised score
            score_map = compute_unsupervised_score(emb_before, emb_after)

            # PCA
            pca_rgb = embedding_pca_rgb(emb_before, emb_after)

            # RGB
            if mode == "quarterly":
                rgb_before, rgb_after = load_rgb_quarterly(pid, period, target_size=H)
            else:
                rgb_before, rgb_after = load_rgb_monthly(pid, period, target_size=H)

            all_data.append({
                "patch_id": pid,
                "rgb_before": rgb_before,
                "rgb_after": rgb_after,
                "gt_mask": gt_mask,
                "prob_map": prob_map,
                "score_map": score_map,
                "pca_rgb": pca_rgb,
            })

        for page_idx in range(0, len(all_data), PATCHES_PER_PAGE):
            page_data = all_data[page_idx:page_idx + PATCHES_PER_PAGE]
            out_path = draw_page(page_data, page_idx // PATCHES_PER_PAGE, shp_name, out_dir, mode)
            print(f"  Saved: {out_path}")


# ── 主程序 ───────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_Q.mkdir(parents=True, exist_ok=True)
    OUTPUT_M.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, dataset = load_model()
    print(f"Model on {DEVICE}")

    print("Loading grid...")
    grid = load_grid()

    cat_patches = gather_annotations()

    all_summaries = {}
    mlp_heads = {}

    for cat_name in ["construction", "demolition", "farmland"]:
        pids = cat_patches.get(cat_name, [])
        if not pids:
            continue
        result = train_mlp_for_category(cat_name, pids, model, dataset, grid)
        if result is not None:
            summary, state, _ = result
            all_summaries[cat_name] = summary
            mlp_heads[cat_name] = state

    # Save summary
    with open(OUTPUT_DIR / "mlp_training_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Generate maps
    inference_and_draw(model, dataset, grid, mlp_heads, mode="quarterly")
    inference_and_draw(model, dataset, grid, mlp_heads, mode="monthly")

    print("\nAll done.")


if __name__ == "__main__":
    main()
