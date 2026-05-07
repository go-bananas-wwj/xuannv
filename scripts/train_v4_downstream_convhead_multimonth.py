#!/usr/bin/env python3
"""V4 多月份 Embedding 融合 + PixelConvHead 下游训练（对比 V5）."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image as PILImage
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.downstream_heads import PixelConvHead

# ── Paths ──
EMB_DIR = Path("/workspace/outputs/aef_qwen_v4_official/monthly_embeddings_2025")
RAW_DIR = Path("/workspace/raw/harbin_scenes")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v4_official/downstream_convhead_multimonth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTH_TO_DW_QUARTER = {
    "2025-01": "2025Q1", "2025-02": "2025Q1", "2025-03": "2025Q1",
    "2025-04": "2025Q2", "2025-05": "2025Q2", "2025-06": "2025Q2",
    "2025-07": "2025Q3", "2025-08": "2025Q3", "2025-09": "2025Q3",
    "2025-10": "2025Q4", "2025-11": "2025Q4", "2025-12": "2025Q4",
}


class MultiMonthPatchDataset(Dataset):
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
        self.length = Np

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
        pid = self.patch_ids[idx]
        month = self.months[0]

        embs = self.emb_array[idx]
        mean_emb = embs.mean(axis=0)
        std_emb = embs.std(axis=0)
        max_emb = embs.max(axis=0)
        fused = np.concatenate([mean_emb, std_emb, max_emb], axis=0)
        emb = torch.from_numpy(fused)

        lbl = self._load_label(pid, month)
        if lbl is None:
            lbl = np.zeros((self.H, self.W), dtype=np.int64)

        if lbl.shape != (self.H, self.W):
            lbl_pil = PILImage.fromarray(lbl.astype(np.int32))
            lbl_pil = lbl_pil.resize((self.W, self.H), PILImage.NEAREST)
            lbl = np.array(lbl_pil, dtype=np.int32)

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

        if self.class_remap is not None:
            remapped = np.full_like(mapped, fill_value=-1)
            for old_val, new_val in self.class_remap.items():
                remapped[mapped == old_val] = new_val
            mapped = remapped

        lbl_t = torch.from_numpy(mapped).long()
        return emb, lbl_t


def compute_class_weights(y_true, n_classes):
    counts = np.bincount(y_true, minlength=n_classes)
    total = counts.sum()
    weights = total / (n_classes * (counts + 1e-6))
    weights = np.clip(weights, 0.1, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_pixel_conv_head(task_name, ds, n_classes, is_binary=False, epochs=30):
    print(f"\n{'='*60}")
    print(f"  MultiMonth Fusion PixelConvHead: {task_name}")
    print(f"  Input dim: {ds.D * 3}, Classes: {n_classes}, Binary: {is_binary}")
    print(f"{'='*60}")

    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    all_labels = []
    for i in train_idx:
        _, lbl = ds[i]
        all_labels.extend(lbl[lbl >= 0].flatten().tolist())
    y_true_arr = np.array(all_labels)
    class_weights = compute_class_weights(y_true_arr, n_classes).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    model = PixelConvHead(in_dim=ds.D * 3, hidden_dim=128, num_classes=n_classes, kernel_size=3, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric = 0.0
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for emb, lbl in train_loader:
            emb = emb.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad()
            logits = model(emb)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
        scheduler.step()

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(device)
                lbl = lbl.to(device)
                logits = model(emb)
                preds = logits.argmax(dim=1).cpu().numpy().flatten()
                targets = lbl.cpu().numpy().flatten()
                mask = targets >= 0
                all_preds.extend(preds[mask])
                all_targets.extend(targets[mask])

        y_pred = np.array(all_preds)
        y_true_val = np.array(all_targets)
        bacc = balanced_accuracy_score(y_true_val, y_pred)

        if is_binary:
            f1 = f1_score(y_true_val, y_pred, pos_label=1, zero_division=0)
            metric = f1
            metric_name = "F1"
        else:
            f1 = f1_score(y_true_val, y_pred, average="weighted", zero_division=0)
            metric = bacc
            metric_name = "BAcc"

        print(f"  Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss/train_steps:.4f} BAcc={bacc:.4f} {metric_name}={metric:.4f}")

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

    save_path = OUT_DIR / f"{task_name}_multimonth.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "task": task_name,
        "n_classes": n_classes,
        "in_dim": ds.D * 3,
    }, save_path)
    print(f"  Saved to {save_path}")
    print(f"\n  Best {metric_name}: {best_metric:.4f}")
    return {"task": task_name, "best_metric": float(best_metric), "metric_name": metric_name, "bacc": float(bacc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
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
            all_classes = set()
            for pid in sorted({f.stem.rsplit("_", 1)[0] for f in EMB_DIR.glob("patch_*.npy")}):
                tif_dir = RAW_DIR / "worldcover" / pid
                if tif_dir.exists():
                    tifs = sorted(tif_dir.glob("*.tif"))
                    if tifs:
                        with rasterio.open(str(tifs[0])) as src:
                            lbl = src.read(1)
                            all_classes.update(np.unique(lbl[lbl >= 0]).tolist())
            wc_classes = sorted(all_classes)
            wc_remap = {c: i for i, c in enumerate(wc_classes)}
            ds = MultiMonthPatchDataset(EMB_DIR, RAW_DIR / "worldcover", "WorldCover", class_remap=wc_remap)
            result = train_pixel_conv_head("WorldCover", ds, n_classes=len(wc_classes), epochs=args.epochs)
        elif task_name == "DynamicWorld":
            ds = MultiMonthPatchDataset(EMB_DIR, RAW_DIR / "dynamic_world", "DynamicWorld", use_temporal_dw=True)
            result = train_pixel_conv_head("DynamicWorld", ds, n_classes=9, epochs=args.epochs)
        elif task_name == "JRC_Water":
            ds = MultiMonthPatchDataset(EMB_DIR, RAW_DIR / "jrc_water", "JRC_Water", jrc_mode=True)
            result = train_pixel_conv_head("JRC_Water", ds, n_classes=2, is_binary=True, epochs=args.epochs)
        elif task_name == "OSM_Buildings":
            ds = MultiMonthPatchDataset(EMB_DIR, RAW_DIR / "osm_buildings", "OSM_Buildings", is_osm=True)
            result = train_pixel_conv_head("OSM_Buildings", ds, n_classes=2, is_binary=True, epochs=args.epochs)
        else:
            print(f"Unknown task: {task_name}")
            continue
        results.append(result)

    print("\n" + "=" * 60)
    print("  V4 MultiMonth Fusion Results Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['task']:15s} | {r['metric_name']}={r['best_metric']:.4f} BAcc={r['bacc']:.4f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
