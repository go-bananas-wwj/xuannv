from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from . import backbone

LABEL_MAPPING = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
NUM_CLASSES = len(LABEL_MAPPING)
ESA_CODES = list(LABEL_MAPPING.keys())


def _load_worldcover_label(
    patch_id: str, label_dir: Path, H: int, W: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    patch_label_dir = label_dir / patch_id / "worldcover"
    if not patch_label_dir.exists():
        return None, None
    tifs = sorted([f for f in patch_label_dir.iterdir() if f.suffix.lower() == ".tif"])
    if not tifs:
        return None, None

    with rasterio.open(tifs[0]) as src:
        label = src.read(1)
        nodata = src.nodata

    if label.shape != (H, W):
        t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
        label = (
            F.interpolate(t, size=(H, W), mode="nearest")
            .squeeze()
            .numpy()
            .astype(label.dtype)
        )

    mapped = np.full_like(label, -1, dtype=np.int64)
    for code, idx in LABEL_MAPPING.items():
        mapped[label == code] = idx
    mask = mapped >= 0
    if nodata is not None:
        mask &= label != nodata
    return mapped, mask


def run_worldcover_knn(
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    k: int = 5,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, dataset, cfg = backbone.load_production_model(model_dir, device=device)
    patch_ids = [p for p in dataset.patches]
    if not patch_ids:
        raise ValueError("数据集中没有 patch")

    label_dir = Path(label_dir)
    if not label_dir.exists():
        H, W = getattr(cfg.model, "common_spatial_size", [64, 64])
        pred_classes = -np.ones((len(patch_ids), H, W), dtype=np.int64)
        np.savez_compressed(
            out_dir / "pred_worldcover.npz",
            patch_ids=np.array(patch_ids),
            pred_classes=pred_classes,
        )
        result = {
            "task": "worldcover",
            "k": k,
            "note": "未提供有效 WorldCover 标签，仅输出占位预测图。",
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        print(f"[worldcover_knn] {result['note']}")
        return result

    embeddings = backbone.extract_embeddings_for_patches(
        model, dataset, patch_ids, 2025, 6, device
    )
    valid_pids = [p for p in patch_ids if p in embeddings]
    if not valid_pids:
        raise ValueError("没有成功提取任何 embedding")

    emb0 = embeddings[valid_pids[0]]
    D, H, W = emb0.shape

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_pidx: list[np.ndarray] = []

    for pidx, pid in enumerate(valid_pids):
        label, mask = _load_worldcover_label(pid, label_dir, H, W)
        if label is None or mask.sum() == 0:
            continue
        emb = embeddings[pid]
        all_X.append(emb[:, mask].T)
        all_y.append(label[mask])
        all_pidx.append(np.full(mask.sum(), pidx))

    if not all_X:
        pred_classes = -np.ones((len(valid_pids), H, W), dtype=np.int64)
        np.savez_compressed(
            out_dir / "pred_worldcover.npz",
            patch_ids=np.array(valid_pids),
            pred_classes=pred_classes,
        )
        result = {
            "task": "worldcover",
            "k": k,
            "note": "未提供有效 WorldCover 标签，仅输出占位预测图。",
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        print(f"[worldcover_knn] {result['note']}")
        return result

    all_X = np.concatenate(all_X, 0)
    all_y = np.concatenate(all_y, 0)
    all_pidx = np.concatenate(all_pidx, 0)

    rng = np.random.RandomState(seed)
    n_test = max(1, int(len(valid_pids) * split_ratio))
    test_set = set(rng.choice(len(valid_pids), n_test, replace=False).tolist())
    train_mask = ~np.isin(all_pidx, list(test_set))

    X_tr, y_tr = all_X[train_mask], all_y[train_mask]
    X_te, y_te = all_X[~train_mask], all_y[~train_mask]

    if len(X_tr) == 0 or len(X_te) == 0:
        raise ValueError("训练集或测试集为空")

    clf = KNeighborsClassifier(
        n_neighbors=min(k, len(X_tr)), metric="euclidean", n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(NUM_CLASSES)))

    per_class: dict[str, dict] = {}
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        per_class[f"class_{c}"] = {
            "esa_code": ESA_CODES[c],
            "iou": float(tp / (tp + fp + fn + 1e-8)),
            "support": int(cm[c, :].sum()),
        }

    valid_ious = [v["iou"] for v in per_class.values() if v["support"] > 0]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    metrics = {
        "task": "worldcover",
        "k": k,
        "accuracy": float(acc),
        "mean_iou": miou,
        "num_train_pixels": int(len(X_tr)),
        "num_test_pixels": int(len(X_te)),
        "per_class": per_class,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )

    pred_maps: dict[str, np.ndarray] = {}
    for pidx in sorted(test_set):
        pid = valid_pids[pidx]
        emb = embeddings[pid]
        flat = emb.reshape(D, -1).T
        pred_flat = clf.predict(flat)
        pred_maps[pid] = pred_flat.reshape(H, W)

    if pred_maps:
        np.savez_compressed(
            out_dir / "pred_worldcover.npz",
            patch_ids=np.array(list(pred_maps.keys())),
            pred_classes=np.stack(list(pred_maps.values()), axis=0),
        )

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldCover kNN 推理")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument(
        "--label-dir",
        default="/workspace/xuannv/data_raw/haidian/scenes",
        help="包含 patch_id/worldcover/static.tif 的数据根目录；为空则仅输出占位图",
    )
    parser.add_argument("--output-dir", default="outputs/worldcover")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_worldcover_knn(
        model_dir=args.model_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        device=args.device,
        k=args.k,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
