#!/usr/bin/env python3
"""V5 下游任务 PyTorch MLP/CNN Head 训练 (时间对齐 + Focal Loss).

对比对象: sklearn Linear Probe
升级点:
  1. 2-Layer MLP / 1x1 CNN Head
  2. Focal Loss 处理类别不平衡
  3. 类别权重自适应
  4. mIoU + Per-class F1 评估
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    jaccard_score,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.downstream_heads import PixelMLPHead, PixelConvHead, focal_loss

# ── Paths ──
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025")
RAW_DIR = Path("/workspace/raw/harbin_scenes")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/downstream_torch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(42)
MAX_SAMPLES_PER_PATCH = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── DW 月份→季度映射 (时间对齐) ──
MONTH_TO_DW_QUARTER = {
    "2025-01": "2025Q1", "2025-02": "2025Q1", "2025-03": "2025Q1",
    "2025-04": "2025Q2", "2025-05": "2025Q2", "2025-06": "2025Q2",
    "2025-07": "2025Q3", "2025-08": "2025Q3", "2025-09": "2025Q3",
    "2025-10": "2025Q4", "2025-11": "2025Q4", "2025-12": "2025Q4",
}


# ── 颜色表 ──
WORLDCOVER_CLASSES = {
    10: "Tree", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare", 70: "Snow", 80: "Water", 90: "Wetland",
    95: "Mangroves", 100: "Moss",
}
WORLDCOVER_COLORS = {
    10: (65, 155, 223), 20: (57, 125, 73), 30: (136, 176, 83),
    40: (255, 187, 34), 50: (255, 255, 76), 60: (187, 85, 29),
    70: (222, 222, 222), 80: (170, 170, 170), 90: (120, 80, 20),
    95: (140, 140, 140), 100: (100, 100, 100),
}

DYNAMIC_WORLD_CLASSES = {
    0: "Water", 1: "Trees", 2: "Grass", 3: "Flooded Veg",
    4: "Crops", 5: "Shrub/Scrub", 6: "Built", 7: "Bare", 8: "Snow/Ice",
}
DYNAMIC_WORLD_COLORS = {
    0: (0, 100, 200), 1: (0, 100, 0), 2: (136, 176, 83),
    3: (120, 180, 160), 4: (255, 187, 34), 5: (255, 150, 50),
    6: (250, 0, 0), 7: (180, 180, 180), 8: (222, 222, 222),
}


# ── 数据加载 ──
def _load_all_embeddings() -> tuple[list[str], list[str], np.ndarray]:
    files = sorted(EMB_DIR.glob("patch_*.npy"))
    patch_month_map: dict[str, list[str]] = {}
    for f in files:
        stem = f.stem
        parts = stem.split("_")
        pid = "_".join(parts[:2])
        month = parts[2]
        patch_month_map.setdefault(pid, []).append(month)

    patch_ids = sorted(patch_month_map.keys())
    months = sorted(set(m for ms in patch_month_map.values() for m in ms))

    sample = np.load(files[0])
    D, H, W = sample.shape
    Np = len(patch_ids)
    Nm = len(months)

    emb_array = np.zeros((Np, Nm, D, H, W), dtype=np.float32)
    for i, pid in enumerate(patch_ids):
        for j, month in enumerate(months):
            path = EMB_DIR / f"{pid}_{month}.npy"
            if path.exists():
                emb_array[i, j] = np.load(path)

    print(f"[Embedding] Loaded {Np} patches x {Nm} months, shape={emb_array.shape}")
    return patch_ids, months, emb_array


def _resample_label(lbl: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if lbl.shape == (target_h, target_w):
        return lbl
    lbl_pil = PILImage.fromarray(lbl.astype(np.int32))
    lbl_pil = lbl_pil.resize((target_w, target_h), PILImage.NEAREST)
    return np.array(lbl_pil, dtype=np.int32)


def _load_label_tif(label_dir: Path, pid: str) -> np.ndarray | None:
    tif_dir = label_dir / pid
    if not tif_dir.exists():
        return None
    tifs = sorted(tif_dir.glob("*.tif"))
    if not tifs:
        return None
    with rasterio.open(str(tifs[0])) as src:
        return src.read(1)


def _load_dynamic_world_label(pid: str, month: str) -> np.ndarray | None:
    tif_dir = RAW_DIR / "dynamic_world" / pid
    if not tif_dir.exists():
        return None
    target_q = MONTH_TO_DW_QUARTER.get(month)
    if target_q is None:
        return None
    target_path = tif_dir / (target_q + ".tif")
    if target_path.exists():
        with rasterio.open(str(target_path)) as src:
            return src.read(1)
    # fallback
    available = sorted(tif_dir.glob("*Q*.tif"))
    if available:
        with rasterio.open(str(available[-1])) as src:
            return src.read(1)
    return None


# ── 数据集 ──
class PixelDataset(Dataset):
    """逐像素采样数据集."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        max_samples: int | None = None,
    ):
        """Args:
            embeddings: [N, D] float32
            labels: [N] int64
            max_samples: 若为None则使用全部
        """
        self.embeddings = embeddings
        self.labels = labels
        self.n_total = len(embeddings)
        if max_samples is not None and max_samples < self.n_total:
            indices = RNG.choice(self.n_total, max_samples, replace=False)
            self.embeddings = embeddings[indices]
            self.labels = labels[indices]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def build_dataset(
    task_name: str,
    patch_ids: list[str],
    months: list[str],
    emb_array: np.ndarray,
    class_map: dict[int, str],
    is_binary: bool = False,
    jrc_mode: bool = False,
    use_temporal_dw: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """构建 (X, y, cls_to_idx) 数据集."""
    sorted_classes = sorted(class_map.keys())
    cls_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    n_classes = len(sorted_classes)
    D, H, W = emb_array.shape[2], emb_array.shape[3], emb_array.shape[4]

    all_emb, all_lbl = [], []

    for i, pid in enumerate(tqdm(patch_ids, desc=f"Loading {task_name}", leave=False)):
        if use_temporal_dw:
            for j, month in enumerate(months):
                lbl_raw = _load_dynamic_world_label(pid, month)
                if lbl_raw is None:
                    continue
                mapped = np.full_like(lbl_raw, fill_value=-1, dtype=np.int32)
                for orig, idx in cls_to_idx.items():
                    mapped[lbl_raw == orig] = idx
                emb_map = emb_array[i, j]
                lbl_rs = _resample_label(mapped, H, W)
                valid = lbl_rs >= 0
                if valid.sum() == 0:
                    continue
                flat_emb = emb_map[:, valid].T
                flat_lbl = lbl_rs[valid]
                n = flat_emb.shape[0]
                if n > MAX_SAMPLES_PER_PATCH:
                    idx = RNG.choice(n, MAX_SAMPLES_PER_PATCH, replace=False)
                    flat_emb = flat_emb[idx]
                    flat_lbl = flat_lbl[idx]
                all_emb.append(flat_emb)
                all_lbl.append(flat_lbl)
        else:
            if task_name == "OSM Buildings":
                lbl_raw = _load_label_tif(RAW_DIR / "osm_buildings", pid)
            elif jrc_mode:
                lbl_raw = _load_label_tif(RAW_DIR / "jrc_water", pid)
            else:
                lbl_raw = _load_label_tif(RAW_DIR / task_name.lower().replace(" ", "_"), pid)

            if lbl_raw is None:
                continue

            raw_int = lbl_raw.astype(np.int32)
            if jrc_mode:
                mapped = np.full_like(raw_int, fill_value=-1)
                mapped[raw_int == -128] = -1
                mapped[raw_int <= 0] = 0
                mapped[raw_int > 0] = 1
            elif task_name == "OSM Buildings":
                mapped = np.full_like(raw_int, fill_value=-1)
                mapped[raw_int == 1] = 1
                mapped[raw_int == 0] = 0
            else:
                mapped = np.full_like(raw_int, fill_value=-1)
                for orig, idx in cls_to_idx.items():
                    mapped[raw_int == orig] = idx

            for j, month in enumerate(months):
                emb_map = emb_array[i, j]
                lbl_rs = _resample_label(mapped, H, W)
                valid = lbl_rs >= 0
                if valid.sum() == 0:
                    continue
                flat_emb = emb_map[:, valid].T
                flat_lbl = lbl_rs[valid]
                n = flat_emb.shape[0]
                if n > MAX_SAMPLES_PER_PATCH:
                    idx = RNG.choice(n, MAX_SAMPLES_PER_PATCH, replace=False)
                    flat_emb = flat_emb[idx]
                    flat_lbl = flat_lbl[idx]
                all_emb.append(flat_emb)
                all_lbl.append(flat_lbl)

    if not all_emb:
        return None, None, cls_to_idx

    X = np.concatenate(all_emb)
    y = np.concatenate(all_lbl)
    print(f"  Total samples: {len(X):,} | Classes: {n_classes}")
    class_counts = np.bincount(y, minlength=n_classes)
    for c, count in enumerate(class_counts):
        print(f"    Class {c}: {count:,} ({count/len(y)*100:.1f}%)")
    return X, y, cls_to_idx


# ── 训练 ──
def train_task(
    task_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    head_type: str = "mlp",
    use_focal: bool = True,
    gamma: float = 2.0,
) -> dict:
    """训练单个下游任务."""
    print(f"\n{'='*60}")
    print(f"Training: {task_name} [{head_type}] focal={use_focal}")
    print(f"{'='*60}")

    # 类别权重
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = torch.tensor(
        [1.0 / max(c, 1) for c in class_counts],
        dtype=torch.float32,
        device=DEVICE,
    )
    class_weights = class_weights / class_weights.sum() * n_classes
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    # 构建 DataLoader
    train_ds = PixelDataset(X_train, y_train)
    val_ds = PixelDataset(X_val, y_val)

    # Weighted sampler for training
    sample_weights = class_weights.cpu().numpy()[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=4096, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

    # 创建 Head
    if head_type == "mlp":
        head = PixelMLPHead(in_dim=128, hidden_dim=256, num_classes=n_classes, dropout=0.3).to(DEVICE)
    elif head_type == "mlp_v2":
        head = PixelMLPHeadV2(in_dim=128, hidden_dims=[256, 128], num_classes=n_classes, dropout=0.4).to(DEVICE)
    elif head_type == "conv":
        head = PixelConvHead(in_dim=128, hidden_dim=64, num_classes=n_classes, kernel_size=1, dropout=0.3).to(DEVICE)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    n_params = sum(p.numel() for p in head.parameters())
    print(f"  Head params: {n_params:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, 51):
        head.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = head(batch_x)
            if use_focal:
                loss = focal_loss(logits, batch_y, alpha=class_weights, gamma=gamma, reduction="mean")
            else:
                loss = F.cross_entropy(logits, batch_y, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation
        head.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                logits = head(batch_x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(batch_y.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        bacc = balanced_accuracy_score(all_targets, all_preds)
        avg_type = "binary" if n_classes == 2 else "macro"
        f1 = f1_score(all_targets, all_preds, average=avg_type, zero_division=0)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d} | Loss={np.mean(epoch_losses):.4f} | Val BAcc={bacc:.4f} F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = head.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Final eval with best model
    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    all_probs = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            logits = head(batch_x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    bacc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average=avg_type, zero_division=0)

    # mIoU
    try:
        miou = jaccard_score(all_targets, all_preds, average=avg_type, zero_division=0)
    except Exception:
        miou = 0.0

    # Per-class F1
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    print(f"  Best Val BAcc={bacc:.4f} F1={f1:.4f} mIoU={miou:.4f}")

    # Save model
    task_id = task_name.lower().replace(" ", "_").replace("/", "_")
    save_path = OUT_DIR / f"{task_id}_{head_type}_focal{use_focal}.pt"
    torch.save({
        "head": head.state_dict(),
        "head_type": head_type,
        "n_classes": n_classes,
        "class_weights": class_weights.cpu().numpy(),
        "metrics": {"bacc": bacc, "f1": f1, "miou": miou, "per_class_f1": per_class_f1.tolist()},
    }, save_path)
    print(f"  Saved: {save_path}")

    return {
        "task": task_name,
        "n_samples": int(len(X_train) + len(X_val)),
        "n_classes": n_classes,
        "balanced_accuracy": float(bacc),
        "f1_score": float(f1),
        "miou": float(miou),
        "per_class_f1": per_class_f1.tolist(),
    }


def main():
    print("=" * 60)
    print("V5 下游任务 PyTorch MLP Head + Focal Loss 训练")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    patch_ids, months, emb_array = _load_all_embeddings()

    # 70/30 split indices
    n_total_patches = len(patch_ids)
    n_train = int(n_total_patches * 0.7)
    indices = RNG.permutation(n_total_patches)
    train_pids = [patch_ids[i] for i in indices[:n_train]]
    val_pids = [patch_ids[i] for i in indices[n_train:]]

    print(f"\nPatch split: Train={len(train_pids)}, Val={len(val_pids)}")

    results = []

    # Task 1: WorldCover
    wc_present = set()
    for pid in patch_ids[:50]:
        lbl = _load_label_tif(RAW_DIR / "worldcover", pid)
        if lbl is not None:
            wc_present.update(lbl.flatten())
    wc_present = {c for c in wc_present if c in WORLDCOVER_CLASSES}
    wc_map = {c: WORLDCOVER_CLASSES[c] for c in sorted(wc_present)}

    X, y, cls_idx = build_dataset("WorldCover", patch_ids, months, emb_array, wc_map)
    if X is not None:
        train_mask = np.isin([patch_ids.index(p) for p in patch_ids], indices[:n_train])
        # Actually we need per-sample split, not per-patch... let's do random split
        n_total = len(X)
        n_tr = int(n_total * 0.7)
        split_idx = RNG.permutation(n_total)
        X_tr, y_tr = X[split_idx[:n_tr]], y[split_idx[:n_tr]]
        X_val, y_val = X[split_idx[n_tr:]], y[split_idx[n_tr:]]
        results.append(train_task("WorldCover", X_tr, y_tr, X_val, y_val, len(wc_map), head_type="mlp", use_focal=True))

    # Task 2: Dynamic World
    dw_present = set()
    for pid in patch_ids[:50]:
        for month in months:
            lbl = _load_dynamic_world_label(pid, month)
            if lbl is not None:
                dw_present.update(lbl.flatten())
    dw_present = {c for c in dw_present if c in DYNAMIC_WORLD_CLASSES}
    dw_map = {c: DYNAMIC_WORLD_CLASSES[c] for c in sorted(dw_present)}

    X, y, cls_idx = build_dataset("Dynamic World", patch_ids, months, emb_array, dw_map, use_temporal_dw=True)
    if X is not None:
        n_total = len(X)
        n_tr = int(n_total * 0.7)
        split_idx = RNG.permutation(n_total)
        X_tr, y_tr = X[split_idx[:n_tr]], y[split_idx[:n_tr]]
        X_val, y_val = X[split_idx[n_tr:]], y[split_idx[n_tr:]]
        results.append(train_task("Dynamic World", X_tr, y_tr, X_val, y_val, len(dw_map), head_type="mlp", use_focal=True))

    # Task 3: JRC Water
    X, y, cls_idx = build_dataset("JRC Water", patch_ids, months, emb_array, {0: "Non-water", 1: "Water"}, is_binary=True, jrc_mode=True)
    if X is not None:
        n_total = len(X)
        n_tr = int(n_total * 0.7)
        split_idx = RNG.permutation(n_total)
        X_tr, y_tr = X[split_idx[:n_tr]], y[split_idx[:n_tr]]
        X_val, y_val = X[split_idx[n_tr:]], y[split_idx[n_tr:]]
        results.append(train_task("JRC Water", X_tr, y_tr, X_val, y_val, 2, head_type="mlp", use_focal=True))

    # Task 4: OSM Buildings
    X, y, cls_idx = build_dataset("OSM Buildings", patch_ids, months, emb_array, {0: "Non-building", 1: "Building"})
    if X is not None:
        n_total = len(X)
        n_tr = int(n_total * 0.7)
        split_idx = RNG.permutation(n_total)
        X_tr, y_tr = X[split_idx[:n_tr]], y[split_idx[:n_tr]]
        X_val, y_val = X[split_idx[n_tr:]], y[split_idx[n_tr:]]
        results.append(train_task("OSM Buildings", X_tr, y_tr, X_val, y_val, 2, head_type="mlp", use_focal=True))

    # Save metrics
    metrics_path = OUT_DIR / "metrics_torch.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll done. Metrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PyTorch MLP + Focal Loss 下游任务汇总")
    print("=" * 60)
    print(f"{'Task':<20} {'Samples':>10} {'Classes':>8} {'BAcc':>8} {'F1':>8} {'mIoU':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['task']:<20} {r['n_samples']:>10,} {r['n_classes']:>8} {r['balanced_accuracy']:>8.4f} {r['f1_score']:>8.4f} {r['miou']:>8.4f}")

    # Compare with Linear Probe baseline
    print("\n" + "=" * 60)
    print("Linear Probe (v2) vs PyTorch MLP + Focal Loss 对比")
    print("=" * 60)
    baseline = {
        "WorldCover": (0.5199, 0.4706),
        "Dynamic World": (0.4523, 0.4019),
        "JRC Water": (0.8055, 0.7061),
        "OSM Buildings": (0.8676, 0.1556),
    }
    print(f"{'Task':<20} {'LR BAcc':>10} {'MLP BAcc':>10} {'Δ':>8} {'LR F1':>8} {'MLP F1':>8} {'Δ':>8}")
    print("-" * 80)
    for r in results:
        task = r['task']
        lr_b, lr_f = baseline.get(task, (0, 0))
        mlp_b, mlp_f = r['balanced_accuracy'], r['f1_score']
        print(f"{task:<20} {lr_b:>10.4f} {mlp_b:>10.4f} {mlp_b-lr_b:>+8.4f} {lr_f:>8.4f} {mlp_f:>8.4f} {mlp_f-lr_f:>+8.4f}")


if __name__ == "__main__":
    main()
