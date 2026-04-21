#!/usr/bin/env python3
"""V5 下游任务 PixelConvHead + Pre-norm Embedding 训练.

设计原则（吸取 MLP 实验失败教训）:
  1. Pre-norm embedding (保留幅度差异)
  2. BatchNorm2d (等效 StandardScaler，逐通道归一化)
  3. 纯 CE loss + class_weight（不叠加 Focal/WeightedSampler）
  4. 仅用空间邻域的 PixelConvHead（3x3 + 1x1）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.downstream_heads import PixelConvHead

# ── Paths ──
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025_prenorm")
RAW_DIR = Path("/workspace/raw/harbin_scenes")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/downstream_convhead")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(42)
MONTH_TO_DW_QUARTER = {
    "2025-01": "2025Q1", "2025-02": "2025Q1", "2025-03": "2025Q1",
    "2025-04": "2025Q2", "2025-05": "2025Q2", "2025-06": "2025Q2",
    "2025-07": "2025Q3", "2025-08": "2025Q3", "2025-09": "2025Q3",
    "2025-10": "2025Q4", "2025-11": "2025Q4", "2025-12": "2025Q4",
}

# ── Dataset ──
class PatchEmbeddingDataset(Dataset):
    """加载单个 patch 的所有月份 embedding + 标签."""
    def __init__(self, emb_dir, label_dir, task_name, use_temporal_dw=False, jrc_mode=False, is_osm=False, class_remap=None):
        super().__init__()
        self.task_name = task_name
        self.use_temporal_dw = use_temporal_dw
        self.jrc_mode = jrc_mode
        self.is_osm = is_osm
        self.label_dir = label_dir
        self.class_remap = class_remap
        
        files = sorted(emb_dir.glob("patch_*.npy"))
        self.patch_ids = sorted({f.stem.rsplit("_", 1)[0] for f in files})
        self.months = sorted({f.stem.rsplit("_", 1)[1] for f in files})
        
        # Pre-load all embeddings
        sample = np.load(files[0])
        D, H, W = sample.shape
        Np = len(self.patch_ids)
        Nm = len(self.months)
        self.emb_array = np.zeros((Np, Nm, D, H, W), dtype=np.float32)
        for i, pid in enumerate(self.patch_ids):
            for j, month in enumerate(self.months):
                path = emb_dir / f"{pid}_{month}.npy"
                if path.exists():
                    self.emb_array[i, j] = np.load(path)
        
        self.D, self.H, self.W = D, H, W
        self.length = Np * Nm
        
    def _load_label(self, pid, month=None):
        if self.use_temporal_dw:
            tif_dir = self.label_dir / pid
            if not tif_dir.exists():
                return None
            target_q = MONTH_TO_DW_QUARTER.get(month)
            if target_q is None:
                return None
            target_path = tif_dir / (target_q + ".tif")
            if target_path.exists():
                with rasterio.open(str(target_path)) as src:
                    return src.read(1)
            available = sorted(tif_dir.glob("*Q*.tif"))
            if available:
                with rasterio.open(str(available[-1])) as src:
                    return src.read(1)
            return None
        else:
            tif_dir = self.label_dir / pid
            if not tif_dir.exists():
                return None
            tifs = sorted(tif_dir.glob("*.tif"))
            if not tifs:
                return None
            with rasterio.open(str(tifs[0])) as src:
                return src.read(1)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        i = idx // len(self.months)
        j = idx % len(self.months)
        pid = self.patch_ids[i]
        month = self.months[j]
        
        emb = torch.from_numpy(self.emb_array[i, j])  # [D, H, W]
        lbl = self._load_label(pid, month)
        if lbl is None:
            lbl = np.zeros((self.H, self.W), dtype=np.int64)
            # Return valid mask = all zeros (will be filtered out by collate)
        
        # Resample label
        if lbl.shape != (self.H, self.W):
            lbl_pil = PILImage.fromarray(lbl.astype(np.int32))
            lbl_pil = lbl_pil.resize((self.W, self.H), PILImage.NEAREST)
            lbl = np.array(lbl_pil, dtype=np.int32)
        
        # Map label
        if self.jrc_mode:
            mapped = np.full_like(lbl, fill_value=-1)
            mapped[lbl == -128] = -1
            mapped[lbl <= 0] = 0
            mapped[lbl > 0] = 1
        elif self.is_osm:
            mapped = np.full_like(lbl, fill_value=-1)
            mapped[lbl == 1] = 1
            mapped[lbl == 0] = 0
        else:
            mapped = lbl
        
        # Apply class remap for non-contiguous labels (e.g., WorldCover ESA codes)
        if self.class_remap is not None:
            remapped = np.full_like(mapped, fill_value=-1)
            for old_val, new_val in self.class_remap.items():
                remapped[mapped == old_val] = new_val
            mapped = remapped
        
        lbl_t = torch.from_numpy(mapped).long()
        return emb, lbl_t


def compute_class_weights(y_true, n_classes):
    """计算 Smooth CE weight，避免极端过加权."""
    counts = np.bincount(y_true, minlength=n_classes)
    total = counts.sum()
    weights = total / (n_classes * (counts + 1e-6))
    # Cap at 20x to prevent extreme imbalance
    weights = np.clip(weights, 0.1, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_pixel_conv_head(task_name, ds, n_classes, is_binary=False, epochs=30, batch_size=4, lr=1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train/val split by patch
    patch_size = len(ds.months)
    n_patches = len(ds.patch_ids)
    patch_indices = list(range(n_patches))
    train_patches, val_patches = train_test_split(patch_indices, test_size=0.2, random_state=42)
    
    train_indices = []
    for p in train_patches:
        for m in range(len(ds.months)):
            train_indices.append(p * len(ds.months) + m)
    val_indices = []
    for p in val_patches:
        for m in range(len(ds.months)):
            val_indices.append(p * len(ds.months) + m)
    
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Compute class weights from first batch to avoid loading all
    print(f"  [{task_name}] Computing class weights...")
    all_labels = []
    for emb, lbl in train_loader:
        valid = lbl >= 0
        if valid.any():
            all_labels.extend(lbl[valid].cpu().numpy().tolist())
        if len(all_labels) > 50000:
            break
    if len(all_labels) == 0:
        all_labels = [0]
    y_arr = np.array(all_labels)
    class_weights = compute_class_weights(y_arr, n_classes).to(device)
    print(f"  [{task_name}] Class weights: {class_weights.cpu().numpy().round(3)}")
    
    # Model
    model = PixelConvHead(in_dim=ds.D, hidden_dim=128, num_classes=n_classes, kernel_size=3, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_bacc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for emb, lbl in tqdm(train_loader, desc=f"{task_name} E{epoch+1}/{epochs}", leave=False):
            emb = emb.to(device)
            lbl = lbl.to(device)
            
            logits = model(emb)  # [B, C, H, W]
            loss = criterion(logits, lbl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(device)
                lbl = lbl.to(device)
                logits = model(emb)
                pred = logits.argmax(dim=1)
                valid = lbl >= 0
                if valid.any():
                    all_preds.extend(pred[valid].cpu().numpy().tolist())
                    all_labels.extend(lbl[valid].cpu().numpy().tolist())
        
        if len(all_labels) == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val=empty")
            continue
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        bacc = balanced_accuracy_score(y_true, y_pred)
        avg_f1 = "binary" if is_binary else "macro"
        f1 = f1_score(y_true, y_pred, average=avg_f1, zero_division=0)
        
        if bacc > best_bacc:
            best_bacc = bacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, BAcc={bacc:.4f}, F1={f1:.4f}")
    
    # Final eval with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb, lbl in val_loader:
            emb = emb.to(device)
            lbl = lbl.to(device)
            logits = model(emb)
            pred = logits.argmax(dim=1)
            valid = lbl >= 0
            if valid.any():
                all_preds.extend(pred[valid].cpu().numpy().tolist())
                all_labels.extend(lbl[valid].cpu().numpy().tolist())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=avg_f1, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    print(f"  [{task_name}] Final: BAcc={bacc:.4f}, F1={f1:.4f}")
    print(f"  Per-class F1: {per_class_f1.round(3)}")
    
    # Save
    task_id = task_name.lower().replace(" ", "_").replace("/", "_")
    save_path = OUT_DIR / f"{task_id}_convhead.pt"
    torch.save({
        "model": model.state_dict(),
        "class_weights": class_weights.cpu(),
        "n_classes": n_classes,
        "task": task_name,
    }, save_path)
    
    return {
        "task": task_name, "n_classes": n_classes,
        "balanced_accuracy": float(bacc), "f1_score": float(f1),
        "per_class_f1": per_class_f1.tolist(),
    }


def main():
    print("=" * 60)
    print("V5 下游任务 PixelConvHead + Pre-norm Embedding")
    print(f"Embedding: {EMB_DIR}")
    print("=" * 60)
    
    if not EMB_DIR.exists():
        print(f"ERROR: {EMB_DIR} does not exist. Run extract_v5_prenorm_embeddings.py first.")
        return
    
    results = []
    
    # JRC Water (binary, easiest)
    print("\n--- JRC Water ---")
    ds_jrc = PatchEmbeddingDataset(EMB_DIR, RAW_DIR / "jrc_water", "JRC Water", jrc_mode=True)
    results.append(train_pixel_conv_head("JRC Water", ds_jrc, n_classes=2, is_binary=True, epochs=30))
    
    # OSM Buildings (binary, highly imbalanced)
    print("\n--- OSM Buildings ---")
    ds_osm = PatchEmbeddingDataset(EMB_DIR, RAW_DIR / "osm_buildings", "OSM Buildings", is_osm=True)
    results.append(train_pixel_conv_head("OSM Buildings", ds_osm, n_classes=2, is_binary=True, epochs=50))
    
    # Dynamic World (9 classes)
    print("\n--- Dynamic World ---")
    ds_dw = PatchEmbeddingDataset(EMB_DIR, RAW_DIR / "dynamic_world", "Dynamic World", use_temporal_dw=True)
    results.append(train_pixel_conv_head("Dynamic World", ds_dw, n_classes=9, is_binary=False, epochs=40))
    
    # WorldCover
    print("\n--- WorldCover ---")
    ds_wc_temp = PatchEmbeddingDataset(EMB_DIR, RAW_DIR / "worldcover", "WorldCover")
    # Count classes and build remap
    cls_set = set()
    for i in range(min(50, len(ds_wc_temp))):
        _, lbl = ds_wc_temp[i]
        cls_set.update(lbl[lbl >= 0].numpy().tolist())
    wc_classes = sorted(cls_set)
    wc_n = len(wc_classes)
    wc_remap = {c: i for i, c in enumerate(wc_classes)}
    print(f"  WorldCover classes: {wc_classes}")
    print(f"  WorldCover remap: {wc_remap}")
    ds_wc = PatchEmbeddingDataset(EMB_DIR, RAW_DIR / "worldcover", "WorldCover", class_remap=wc_remap)
    results.append(train_pixel_conv_head("WorldCover", ds_wc, n_classes=wc_n, is_binary=False, epochs=40))
    
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll done. Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("ConvHead 下游任务汇总")
    print("=" * 60)
    print(f"{'Task':<20} {'Classes':>8} {'BAcc':>8} {'F1':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['task']:<20} {r['n_classes']:>8} {r['balanced_accuracy']:>8.4f} {r['f1_score']:>8.4f}")


if __name__ == "__main__":
    main()
