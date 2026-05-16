#!/usr/bin/env python3
"""V5 下游任务 Linear Probe — 使用 PRE-NORM embedding (不 L2 归一化).

与 train_v5_downstream_probes_v2.py 的区别:
  - EMB_DIR 指向 pre-norm embedding 目录
  - 用于快速验证 pre-norm vs L2-normalized 的效果差异
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

# ── Paths ──
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025_prenorm")
RAW_DIR = Path("/workspace/raw/harbin_scenes")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/downstream_prenorm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(42)
MAX_SAMPLES_PER_PATCH = 300

# ── DW 月份→季度映射 ──
MONTH_TO_DW_QUARTER = {
    "2025-01": "2025Q1", "2025-02": "2025Q1", "2025-03": "2025Q1",
    "2025-04": "2025Q2", "2025-05": "2025Q2", "2025-06": "2025Q2",
    "2025-07": "2025Q3", "2025-08": "2025Q3", "2025-09": "2025Q3",
    "2025-10": "2025Q4", "2025-11": "2025Q4", "2025-12": "2025Q4",
}

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

    print(f"[Pre-norm Embedding] Loaded {Np} patches x {Nm} months, shape={emb_array.shape}")
    return patch_ids, months, emb_array


def _resample_label(lbl, target_h, target_w):
    if lbl.shape == (target_h, target_w):
        return lbl
    lbl_pil = PILImage.fromarray(lbl.astype(np.int32))
    lbl_pil = lbl_pil.resize((target_w, target_h), PILImage.NEAREST)
    return np.array(lbl_pil, dtype=np.int32)


def _load_label_tif(label_dir, pid):
    tif_dir = label_dir / pid
    if not tif_dir.exists():
        return None
    tifs = sorted(tif_dir.glob("*.tif"))
    if not tifs:
        return None
    with rasterio.open(str(tifs[0])) as src:
        return src.read(1)


def _load_dynamic_world_label(pid, month):
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
    available = sorted(tif_dir.glob("*Q*.tif"))
    if available:
        with rasterio.open(str(available[-1])) as src:
            return src.read(1)
    return None


def _train_task(task_name, label_source, class_map, color_map, patch_ids, months, emb_array,
                is_binary=False, jrc_mode=False, use_temporal_dw=False):
    print(f"\n{'='*60}")
    print(f"Training: {task_name} (PRE-NORM)")
    print(f"{'='*60}")

    sorted_classes = sorted(class_map.keys())
    cls_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    n_classes = len(sorted_classes)

    all_emb, all_lbl = [], []
    D, H, W = emb_array.shape[2], emb_array.shape[3], emb_array.shape[4]

    for i, pid in enumerate(tqdm(patch_ids, desc=f"{task_name} patches", leave=False)):
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
        return {"error": "no data"}

    X = np.concatenate(all_emb)
    y = np.concatenate(all_lbl)
    print(f"  Total samples: {len(X):,} | Classes: {n_classes}")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_s, y)

    n_total = len(X)
    n_train = int(n_total * 0.7)
    split_idx = RNG.permutation(n_total)
    lr_eval = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr_eval.fit(X_s[split_idx[:n_train]], y[split_idx[:n_train]])
    y_pred = lr_eval.predict(X_s[split_idx[n_train:]])

    bacc = balanced_accuracy_score(y[split_idx[n_train:]], y_pred)
    avg = "binary" if is_binary else "macro"
    f1 = f1_score(y[split_idx[n_train:]], y_pred, average=avg, zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y[split_idx[n_train:]], y_pred, average=None, zero_division=0)

    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  F1 ({avg}): {f1:.4f}")
    print(f"  Per-class F1: {per_class_f1.round(3)}")

    task_id = task_name.lower().replace(" ", "_").replace("/", "_")
    out_path = OUT_DIR / f"{task_id}_linear_probe.pkl"
    joblib.dump({
        "scaler": scaler, "model": lr,
        "classes": sorted_classes,
        "class_names": [class_map[c] for c in sorted_classes],
        "colors": [color_map[c] for c in sorted_classes],
    }, out_path)
    print(f"  Saved: {out_path}")

    return {
        "task": task_name, "n_samples": int(len(X)), "n_classes": n_classes,
        "balanced_accuracy": float(bacc), "f1_score": float(f1),
        "f1_average": avg, "per_class_f1": per_class_f1.tolist(),
    }


def main():
    print("=" * 60)
    print("V5 下游任务 Linear Probe — PRE-NORM Embedding")
    print(f"Embedding: {EMB_DIR}")
    print("=" * 60)

    if not EMB_DIR.exists():
        print(f"ERROR: {EMB_DIR} does not exist. Run extract_v5_prenorm_embeddings.py first.")
        return

    patch_ids, months, emb_array = _load_all_embeddings()
    results = []

    # WorldCover
    wc_present = set()
    for pid in patch_ids[:50]:
        lbl = _load_label_tif(RAW_DIR / "worldcover", pid)
        if lbl is not None:
            wc_present.update(lbl.flatten())
    wc_present = {c for c in wc_present if c in WORLDCOVER_CLASSES}
    wc_map = {c: WORLDCOVER_CLASSES[c] for c in sorted(wc_present)}
    results.append(_train_task("WorldCover", "worldcover", wc_map, WORLDCOVER_COLORS, patch_ids, months, emb_array))

    # Dynamic World
    dw_present = set()
    for pid in patch_ids[:50]:
        for month in months:
            lbl = _load_dynamic_world_label(pid, month)
            if lbl is not None:
                dw_present.update(lbl.flatten())
    dw_present = {c for c in dw_present if c in DYNAMIC_WORLD_CLASSES}
    dw_map = {c: DYNAMIC_WORLD_CLASSES[c] for c in sorted(dw_present)}
    results.append(_train_task("Dynamic World", "dynamic_world", dw_map, DYNAMIC_WORLD_COLORS, patch_ids, months, emb_array, use_temporal_dw=True))

    # JRC Water
    results.append(_train_task("JRC Water", "jrc_water", {0: "Non-water", 1: "Water"}, {0: (180, 180, 180), 1: (0, 100, 200)}, patch_ids, months, emb_array, is_binary=True, jrc_mode=True))

    # OSM Buildings
    results.append(_train_task("OSM Buildings", "osm_buildings", {0: "Non-building", 1: "Building"}, {0: (100, 100, 100), 1: (250, 0, 0)}, patch_ids, months, emb_array))

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll done. Metrics saved to {metrics_path}")

    # Summary + comparison
    print("\n" + "=" * 60)
    print("Pre-norm 下游任务汇总")
    print("=" * 60)
    print(f"{'Task':<20} {'Samples':>10} {'Classes':>8} {'BAcc':>8} {'F1':>8}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            print(f"{r['task']:<20} {r['n_samples']:>10,} {r['n_classes']:>8} {r['balanced_accuracy']:>8.4f} {r['f1_score']:>8.4f}")

    print("\n" + "=" * 60)
    print("L2-normalized vs PRE-NORM Linear Probe 对比")
    print("=" * 60)
    baseline = {
        "WorldCover": (0.5199, 0.4706),
        "Dynamic World": (0.4523, 0.4019),
        "JRC Water": (0.8055, 0.7061),
        "OSM Buildings": (0.8676, 0.1556),
    }
    print(f"{'Task':<20} {'L2 BAcc':>10} {'Pre BAcc':>10} {'Δ':>8} {'L2 F1':>8} {'Pre F1':>8} {'Δ':>8}")
    print("-" * 80)
    for r in results:
        if "error" not in r:
            task = r['task']
            l2_b, l2_f = baseline.get(task, (0, 0))
            pre_b, pre_f = r['balanced_accuracy'], r['f1_score']
            print(f"{task:<20} {l2_b:>10.4f} {pre_b:>10.4f} {pre_b-l2_b:>+8.4f} {l2_f:>8.4f} {pre_f:>8.4f} {pre_f-l2_f:>+8.4f}")


if __name__ == "__main__":
    main()
