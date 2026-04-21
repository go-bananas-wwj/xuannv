#!/usr/bin/env python3
"""训练 V5 下游任务的 Linear Probe 模型。

使用 V5 预计算 embedding + 标签数据，产出 .pkl 模型文件 + metrics.json。
支持: WorldCover, Dynamic World, JRC Water, OSM Buildings.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

# ── Paths ──
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025")
RAW_DIR = Path("/workspace/raw/harbin_scenes")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/downstream_probes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(42)
MAX_SAMPLES_PER_PATCH = 300

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


def _load_all_embeddings() -> tuple[list[str], list[str], np.ndarray]:
    """加载所有 embedding 文件，返回 (patch_ids, months, emb_array).

    emb_array: [N_patch, N_month, D, H, W]
    """
    files = sorted(EMB_DIR.glob("patch_*.npy"))
    patch_month_map: dict[str, list[str]] = {}
    for f in files:
        stem = f.stem  # patch_000000_2025-04
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


def _load_label_tif(label_dir: Path, pid: str) -> np.ndarray | None:
    """加载指定 patch 的标签 GeoTIFF."""
    tif_dir = label_dir / pid
    if not tif_dir.exists():
        return None
    tifs = sorted(tif_dir.glob("*.tif"))
    if not tifs:
        return None
    with rasterio.open(str(tifs[0])) as src:
        return src.read(1)


def _resample_label(lbl: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """将标签上/下采样到目标尺寸（最近邻）."""
    if lbl.shape == (target_h, target_w):
        return lbl
    lbl_pil = PILImage.fromarray(lbl.astype(np.int32))
    lbl_pil = lbl_pil.resize((target_w, target_h), PILImage.NEAREST)
    return np.array(lbl_pil, dtype=np.int32)


def _train_task(
    task_name: str,
    label_source: str,
    class_map: dict[int, str],
    color_map: dict[int, tuple[int, int, int]],
    patch_ids: list[str],
    months: list[str],
    emb_array: np.ndarray,
    is_binary: bool = False,
    jrc_mode: bool = False,
) -> dict:
    """训练单个任务的 Linear Probe，返回 metrics dict."""
    print(f"\n{'='*60}")
    print(f"Training: {task_name}")
    print(f"{'='*60}")

    sorted_classes = sorted(class_map.keys())
    cls_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    n_classes = len(sorted_classes)

    all_emb, all_lbl = [], []
    D, H, W = emb_array.shape[2], emb_array.shape[3], emb_array.shape[4]

    for i, pid in enumerate(tqdm(patch_ids, desc=f"{task_name} patches", leave=False)):
        lbl_raw = _load_label_tif(RAW_DIR / label_source, pid)
        if lbl_raw is None:
            continue

        raw_int = lbl_raw.astype(np.int32)
        if np.issubdtype(lbl_raw.dtype, np.floating):
            nan_mask = np.isnan(lbl_raw)
            raw_int[nan_mask] = -1

        if jrc_mode:
            mapped = np.full_like(raw_int, fill_value=-1)
            mapped[raw_int == -128] = -1
            mapped[raw_int <= 0] = 0
            mapped[raw_int > 0] = 1
        else:
            mapped = np.full_like(raw_int, fill_value=-1)
            for orig, idx in cls_to_idx.items():
                mapped[raw_int == orig] = idx

        for j, month in enumerate(months):
            emb_map = emb_array[i, j]  # [D, H, W]
            lbl_rs = _resample_label(mapped, H, W)

            valid = lbl_rs >= 0
            if valid.sum() == 0:
                continue

            flat_emb = emb_map[:, valid].T  # [N_valid, D]
            flat_lbl = lbl_rs[valid]          # [N_valid]

            n = flat_emb.shape[0]
            if n > MAX_SAMPLES_PER_PATCH:
                idx = RNG.choice(n, MAX_SAMPLES_PER_PATCH, replace=False)
                flat_emb = flat_emb[idx]
                flat_lbl = flat_lbl[idx]

            all_emb.append(flat_emb)
            all_lbl.append(flat_lbl)

    if not all_emb:
        print(f"  No valid label data")
        return {"error": "no data"}

    X = np.concatenate(all_emb)  # [N_total, D]
    y = np.concatenate(all_lbl)  # [N_total]
    print(f"  Total samples: {len(X):,}  |  Classes: {n_classes}")

    # 全局训练
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_s, y)

    # 评估（70/30 split）
    n_total = len(X)
    n_train = int(n_total * 0.7)
    split_idx = RNG.permutation(n_total)
    lr_eval = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr_eval.fit(X_s[split_idx[:n_train]], y[split_idx[:n_train]])
    y_pred = lr_eval.predict(X_s[split_idx[n_train:]])

    bacc = balanced_accuracy_score(y[split_idx[n_train:]], y_pred)
    avg = "binary" if is_binary else "macro"
    f1 = f1_score(y[split_idx[n_train:]], y_pred, average=avg, zero_division=0)

    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  F1 ({avg}): {f1:.4f}")

    # 保存模型
    task_id = task_name.lower().replace(" ", "_").replace("/", "_")
    out_path = OUT_DIR / f"{task_id}_linear_probe.pkl"
    joblib.dump({
        "scaler": scaler,
        "model": lr,
        "classes": sorted_classes,
        "class_names": [class_map[c] for c in sorted_classes],
        "colors": [color_map[c] for c in sorted_classes],
    }, out_path)
    print(f"  Saved: {out_path}")

    return {
        "task": task_name,
        "n_samples": int(len(X)),
        "n_classes": n_classes,
        "balanced_accuracy": float(bacc),
        "f1_score": float(f1),
        "f1_average": avg,
    }


def _train_osm_buildings(
    patch_ids: list[str],
    months: list[str],
    emb_array: np.ndarray,
) -> dict:
    """训练 OSM Buildings Linear Probe."""
    task_name = "OSM Buildings"
    print(f"\n{'='*60}")
    print(f"Training: {task_name}")
    print(f"{'='*60}")

    all_emb, all_lbl = [], []
    D, H, W = emb_array.shape[2], emb_array.shape[3], emb_array.shape[4]

    for i, pid in enumerate(tqdm(patch_ids, desc="OSM Buildings patches", leave=False)):
        lbl_raw = _load_label_tif(RAW_DIR / "osm_buildings", pid)
        if lbl_raw is None:
            continue

        binary = np.full_like(lbl_raw, fill_value=-1, dtype=np.int32)
        binary[lbl_raw == 1] = 1      # Building
        binary[lbl_raw == 0] = 0      # Non-building

        for j, month in enumerate(months):
            emb_map = emb_array[i, j]
            lbl_rs = _resample_label(binary, H, W)
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
        print("  No valid data")
        return {"error": "no data"}

    X = np.concatenate(all_emb)
    y = np.concatenate(all_lbl)
    print(f"  Total samples: {len(X):,}")
    print(f"  Class distribution: 0={ (y==0).sum() }, 1={ (y==1).sum() }")

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
    f1 = f1_score(y[split_idx[n_train:]], y_pred, average="binary", zero_division=0)

    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  F1 (binary): {f1:.4f}")

    out_path = OUT_DIR / "osm_buildings_linear_probe.pkl"
    joblib.dump({
        "scaler": scaler,
        "model": lr,
        "classes": [0, 1],
        "class_names": ["Non-building", "Building"],
        "colors": [(100, 100, 100), (250, 0, 0)],
    }, out_path)
    print(f"  Saved: {out_path}")

    return {
        "task": "osm_buildings",
        "n_samples": int(len(X)),
        "n_classes": 2,
        "balanced_accuracy": float(bacc),
        "f1_score": float(f1),
        "f1_average": "binary",
    }


def main():
    print("=" * 60)
    print("V5 下游任务 Linear Probe 训练")
    print(f"Embedding: {EMB_DIR}")
    print("=" * 60)

    patch_ids, months, emb_array = _load_all_embeddings()
    results = []

    # Task 1: WorldCover
    # 先扫描实际存在的类
    print("\n[Scan] Checking WorldCover class distribution...")
    present_classes = set()
    for pid in patch_ids[:50]:  # 采样前50个
        lbl = _load_label_tif(RAW_DIR / "worldcover", pid)
        if lbl is not None:
            present_classes.update(lbl.flatten())
    present_classes = {c for c in present_classes if c in WORLDCOVER_CLASSES}
    print(f"  Present classes: {sorted(present_classes)}")

    wc_class_map = {c: WORLDCOVER_CLASSES[c] for c in sorted(present_classes)}
    wc_color_map = {c: WORLDCOVER_COLORS[c] for c in sorted(present_classes)}
    results.append(_train_task(
        "WorldCover", "worldcover", wc_class_map, wc_color_map,
        patch_ids, months, emb_array,
    ))

    # Task 2: Dynamic World
    print("\n[Scan] Checking Dynamic World class distribution...")
    present_dw = set()
    for pid in patch_ids[:50]:
        lbl = _load_label_tif(RAW_DIR / "dynamic_world", pid)
        if lbl is not None:
            present_dw.update(lbl.flatten())
    present_dw = {c for c in present_dw if c in DYNAMIC_WORLD_CLASSES}
    print(f"  Present classes: {sorted(present_dw)}")

    dw_class_map = {c: DYNAMIC_WORLD_CLASSES[c] for c in sorted(present_dw)}
    dw_color_map = {c: DYNAMIC_WORLD_COLORS[c] for c in sorted(present_dw)}
    results.append(_train_task(
        "Dynamic World", "dynamic_world", dw_class_map, dw_color_map,
        patch_ids, months, emb_array,
    ))

    # Task 3: JRC Water
    results.append(_train_task(
        "JRC Water", "jrc_water", {0: "Non-water", 1: "Water"},
        {0: (180, 180, 180), 1: (0, 100, 200)},
        patch_ids, months, emb_array,
        is_binary=True, jrc_mode=True,
    ))

    # Task 4: OSM Buildings
    results.append(_train_osm_buildings(patch_ids, months, emb_array))

    # 保存 metrics
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll done. Metrics saved to {metrics_path}")

    # 打印汇总表
    print("\n" + "=" * 60)
    print("V5 下游任务汇总")
    print("=" * 60)
    print(f"{'Task':<20} {'Samples':>10} {'Classes':>8} {'BAcc':>8} {'F1':>8}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            print(f"{r['task']:<20} {r['n_samples']:>10,} {r['n_classes']:>8} {r['balanced_accuracy']:>8.4f} {r['f1_score']:>8.4f}")


if __name__ == "__main__":
    main()
