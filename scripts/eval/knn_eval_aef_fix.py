#!/usr/bin/env python3
"""
对AEF官方嵌入做KNN下游评估（WorldCover）—— 修复版。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from scipy.ndimage import zoom


def aef_key_to_patch_dir(key: str) -> str:
    return f"patch_{int(key):06d}"


def load_label(patch_id: str, label_dir: str, data_root: Path) -> np.ndarray | None:
    patch_dir_name = aef_key_to_patch_dir(patch_id)
    patch_dir = data_root / label_dir / patch_dir_name
    if not patch_dir.exists():
        return None
    tifs = sorted([f for f in patch_dir.iterdir() if f.suffix.lower() == ".tif"])
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        return src.read(1)


def evaluate_knn(embeddings: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                 mapping: dict[int, int] | None = None, k: int = 5, sample_pixels: int = 5000) -> dict:
    all_emb, all_label = [], []
    for pid in embeddings:
        if pid not in labels:
            continue
        emb = embeddings[pid]
        label = labels[pid]
        if label.shape != (128, 128):
            label = zoom(label, (128 / label.shape[0], 128 / label.shape[1]), order=0)
        if mapping:
            mapped = np.full_like(label, -1, dtype=np.int64)
            for ko, kn in mapping.items():
                mapped[label == ko] = kn
            label = mapped
        emb_flat = emb.reshape(64, -1).T
        label_flat = label.reshape(-1)
        valid = label_flat >= 0
        all_emb.append(emb_flat[valid])
        all_label.append(label_flat[valid])

    all_emb = np.concatenate(all_emb, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    if len(all_emb) > sample_pixels * 10:
        indices = np.random.choice(len(all_emb), sample_pixels * 10, replace=False)
        all_emb = all_emb[indices]
        all_label = all_label[indices]

    n = len(all_emb)
    n_train = n // 2
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train = all_emb[train_idx] / (np.linalg.norm(all_emb[train_idx], axis=1, keepdims=True) + 1e-8)
    y_train = all_label[train_idx]
    X_test = all_emb[test_idx] / (np.linalg.norm(all_emb[test_idx], axis=1, keepdims=True) + 1e-8)
    y_test = all_label[test_idx]

    print(f"  KNN: train={len(X_train)}, test={len(X_test)}, k={k}")
    batch_size = 1000
    predictions = []
    for i in tqdm(range(0, len(X_test), batch_size), desc="KNN predict"):
        batch = X_test[i:i+batch_size]
        sim = batch @ X_train.T
        topk_idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
        for idx in topk_idx:
            labels_k = y_train[idx]
            unique, counts = np.unique(labels_k, return_counts=True)
            predictions.append(int(unique[np.argmax(counts)]))

    predictions = np.array(predictions)
    acc = float(accuracy_score(y_test, predictions))

    classes = [int(c) for c in sorted(np.unique(np.concatenate([y_test, predictions])))]
    ious = []
    for c in classes:
        pred_c = (predictions == c)
        true_c = (y_test == c)
        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        if union > 0:
            ious.append(float(intersection / union))
    miou = float(np.mean(ious)) if ious else 0.0

    cm = confusion_matrix(y_test, predictions, labels=classes).tolist()

    return {"accuracy": acc, "miou": miou, "classes": classes, "confusion_matrix": cm, "n_samples": int(len(y_test))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="/workspace/outputs/aef_official_embeddings/aef_haidian_2025.npz")
    parser.add_argument("--data-root", default="/workspace/raw/phase2_heilongjiang/haidian")
    parser.add_argument("--output", default="/workspace/outputs/aef_official_embeddings/knn_aef_haidian.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample-pixels", type=int, default=10000)
    args = parser.parse_args()

    print("Loading AEF embeddings...")
    data = np.load(args.embedding)
    embeddings = {k: data[k] for k in data.files if not k.startswith('_')}
    print(f"Loaded {len(embeddings)} patches")

    data_root = Path(args.data_root)

    print("\n=== WorldCover ===")
    labels = {pid: load_label(pid, "worldcover", data_root) for pid in embeddings}
    labels = {k: v for k, v in labels.items() if v is not None}
    mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
    result = evaluate_knn(embeddings, labels, mapping=mapping, k=args.k, sample_pixels=args.sample_pixels)
    print(f"  Acc: {result['accuracy']:.4f}, mIoU: {result['miou']:.4f}")
    print(f"  Classes: {result['classes']}")

    output = {"worldcover": result}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
