#!/usr/bin/env python3
"""OlmoEarth embedding KNN 下游评估
（哈尔滨新区：WorldCover / JRC Water / Dynamic World）

与 knn_eval_aef_harbin.py 评估协议完全对齐，修复了 hardcode 64 维的问题。
embedding 维度由实际 shape[0] 决定；label resize 到 embedding 的空间分辨率。
"""
from __future__ import annotations

import json, argparse
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from scipy.ndimage import zoom


def key_to_patch_dir(key: str) -> str:
    return f"patch_{int(key):06d}"


def load_label(key: str, label_dir: str, data_root: Path) -> np.ndarray | None:
    patch_dir = data_root / label_dir / key_to_patch_dir(key)
    if not patch_dir.exists():
        return None
    tifs = sorted([f for f in patch_dir.iterdir() if f.suffix.lower() == ".tif"])
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        return src.read(1)


def evaluate_knn(embeddings: dict, labels: dict, mapping=None, k=5, sample_pixels=10000):
    all_emb, all_label = [], []
    for pid, emb in embeddings.items():
        if pid not in labels:
            continue
        label = labels[pid]
        D, eH, eW = emb.shape   # D=768, eH=eW=32 (原始token分辨率)
        # label resize 到 embedding 的空间分辨率
        if label.shape != (eH, eW):
            label = zoom(label, (eH / label.shape[0], eW / label.shape[1]), order=0)
        if mapping:
            mapped = np.full_like(label, -1, dtype=np.int64)
            for ko, kn in mapping.items():
                mapped[label == ko] = kn
            label = mapped
        emb_flat = emb.reshape(D, -1).T   # (eH*eW, D)
        label_flat = label.reshape(-1)
        valid = label_flat >= 0
        all_emb.append(emb_flat[valid])
        all_label.append(label_flat[valid])

    all_emb   = np.concatenate(all_emb,   axis=0)
    all_label = np.concatenate(all_label, axis=0)
    print(f"  总有效像素: {len(all_emb)}")

    if len(all_emb) > sample_pixels * 10:
        idx = np.random.choice(len(all_emb), sample_pixels * 10, replace=False)
        all_emb   = all_emb[idx]
        all_label = all_label[idx]

    n       = len(all_emb)
    n_train = n // 2
    perm    = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train = all_emb[train_idx]
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    y_train = all_label[train_idx]
    X_test  = all_emb[test_idx]
    X_test  = X_test  / (np.linalg.norm(X_test,  axis=1, keepdims=True) + 1e-8)
    y_test  = all_label[test_idx]

    print(f"  KNN: train={len(X_train)}, test={len(X_test)}, k={k}, D={all_emb.shape[1]}")

    batch_size  = 500
    predictions = []
    for i in tqdm(range(0, len(X_test), batch_size), desc="KNN predict"):
        batch     = X_test[i:i+batch_size]
        sim       = batch @ X_train.T
        topk_idx  = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
        for row in topk_idx:
            labels_k = y_train[row]
            unique, counts = np.unique(labels_k, return_counts=True)
            predictions.append(int(unique[np.argmax(counts)]))

    predictions = np.array(predictions)
    acc     = float(accuracy_score(y_test, predictions))
    classes = [int(c) for c in sorted(np.unique(np.concatenate([y_test, predictions])))]
    ious    = []
    for c in classes:
        inter = np.logical_and(predictions == c, y_test == c).sum()
        union = np.logical_or( predictions == c, y_test == c).sum()
        if union > 0:
            ious.append(float(inter / union))
    miou = float(np.mean(ious)) if ious else 0.0
    cm   = confusion_matrix(y_test, predictions, labels=classes).tolist()
    return {"accuracy": acc, "miou": miou, "classes": classes,
            "confusion_matrix": cm, "n_samples": int(len(y_test))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", default="/workspace/outputs/olmoearth_harbin/eval/olmoearth_harbin_emb32.npz")
    p.add_argument("--data-root", default="/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered")
    p.add_argument("--output",    default="/workspace/outputs/olmoearth_harbin/eval/knn_olmoearth.json")
    p.add_argument("--k",         type=int, default=5)
    p.add_argument("--sample",    type=int, default=10000)
    args = p.parse_args()

    data_root = Path(args.data_root)
    print(f"加载 OlmoEarth embedding: {args.embedding}")
    data = np.load(args.embedding)
    embeddings = {k: data[k] for k in data.files}
    print(f"  Loaded {len(embeddings)} patches, shape={next(iter(embeddings.values())).shape}")

    results = {}

    # WorldCover
    print("\n=== WorldCover ===")
    labels = {pid: load_label(pid, "worldcover", data_root) for pid in embeddings}
    labels = {k: v for k, v in labels.items() if v is not None}
    print(f"  Labels found: {len(labels)}/{len(embeddings)}")
    if labels:
        mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
        r = evaluate_knn(embeddings, labels, mapping=mapping, k=args.k, sample_pixels=args.sample)
        print(f"  Acc={r['accuracy']:.4f}  mIoU={r['miou']:.4f}  classes={r['classes']}")
        results["worldcover"] = r

    # JRC Water
    print("\n=== JRC Water ===")
    labels = {pid: load_label(pid, "jrc_water", data_root) for pid in embeddings}
    labels = {k: v for k, v in labels.items() if v is not None}
    print(f"  Labels found: {len(labels)}/{len(embeddings)}")
    if labels:
        r = evaluate_knn(embeddings, labels, mapping=None, k=args.k, sample_pixels=args.sample)
        print(f"  Acc={r['accuracy']:.4f}  mIoU={r['miou']:.4f}  classes={r['classes']}")
        results["jrc_water"] = r

    # Dynamic World
    print("\n=== Dynamic World ===")
    labels = {pid: load_label(pid, "dynamic_world", data_root) for pid in embeddings}
    labels = {k: v for k, v in labels.items() if v is not None}
    print(f"  Labels found: {len(labels)}/{len(embeddings)}")
    if labels:
        r = evaluate_knn(embeddings, labels, mapping=None, k=args.k, sample_pixels=args.sample)
        print(f"  Acc={r['accuracy']:.4f}  mIoU={r['miou']:.4f}  classes={r['classes']}")
        results["dynamic_world"] = r

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果保存: {args.output}")


if __name__ == "__main__":
    main()
