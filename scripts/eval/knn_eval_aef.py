#!/usr/bin/env python3
"""
对AEF官方嵌入做KNN下游评估（WorldCover / JRC Water / Dynamic World）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from tqdm import tqdm


def load_aef_embeddings(npz_path: str) -> dict[str, np.ndarray]:
    """加载AEF嵌入，返回 {patch_id: (64, 128, 128)}。"""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files if not k.startswith('_')}


def aef_key_to_patch_dir(key: str) -> str:
    """AEF key '0' -> 'patch_000000'。"""
    return f"patch_{int(key):06d}"


def load_label(patch_id: str, label_dir: str, fname: str, data_root: Path) -> np.ndarray | None:
    """加载标签TIFF。"""
    patch_dir_name = aef_key_to_patch_dir(patch_id)
    patch_dir = data_root / label_dir / patch_dir_name
    if not patch_dir.exists():
        return None
    
    # 自动查找tif文件
    tifs = sorted([f for f in patch_dir.iterdir() if f.suffix.lower() == ".tif"])
    if not tifs:
        return None
    
    with rasterio.open(tifs[0]) as src:
        return src.read(1)


def evaluate_knn(
    embeddings: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    mapping: dict[int, int] | None = None,
    k: int = 5,
    sample_pixels: int = 5000,
) -> dict:
    """KNN评估。"""
    # 收集所有像素
    all_emb = []
    all_label = []
    
    for pid in embeddings:
        if pid not in labels:
            continue
        emb = embeddings[pid]  # (64, 128, 128)
        label = labels[pid]  # (128, 128)
        
        # 下采样标签到128x128（如果不同）
        if label.shape != (128, 128):
            from scipy.ndimage import zoom
            zoom_y = 128 / label.shape[0]
            zoom_x = 128 / label.shape[1]
            label = zoom(label, (zoom_y, zoom_x), order=0)
        
        # 应用mapping
        if mapping:
            mapped = np.full_like(label, -1, dtype=np.int64)
            for k_old, k_new in mapping.items():
                mapped[label == k_old] = k_new
            label = mapped
        
        # 展平
        emb_flat = emb.reshape(64, -1).T  # (128*128, 64)
        label_flat = label.reshape(-1)
        
        # 过滤无效值
        valid = label_flat >= 0
        emb_flat = emb_flat[valid]
        label_flat = label_flat[valid]
        
        all_emb.append(emb_flat)
        all_label.append(label_flat)
    
    all_emb = np.concatenate(all_emb, axis=0)  # (N, 64)
    all_label = np.concatenate(all_label, axis=0)  # (N,)
    
    # 采样（避免内存爆炸）
    if len(all_emb) > sample_pixels * 10:
        indices = np.random.choice(len(all_emb), sample_pixels * 10, replace=False)
        all_emb = all_emb[indices]
        all_label = all_label[indices]
    
    # KNN（简化版：用cosine similarity）
    # 随机划分train/test
    n = len(all_emb)
    n_train = n // 2
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    X_train = all_emb[train_idx]
    y_train = all_label[train_idx]
    X_test = all_emb[test_idx]
    y_test = all_label[test_idx]
    
    # 归一化（AEF已经在球面上）
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
    
    # KNN with cosine similarity
    print(f"  KNN: train={len(X_train)}, test={len(X_test)}, k={k}")
    
    batch_size = 1000
    predictions = []
    
    for i in tqdm(range(0, len(X_test), batch_size), desc="KNN predict"):
        batch = X_test[i:i+batch_size]
        sim = batch @ X_train.T  # cosine similarity
        topk_idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
        
        for j, idx in enumerate(topk_idx):
            labels_k = y_train[idx]
            # 多数投票
            unique, counts = np.unique(labels_k, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
    
    predictions = np.array(predictions)
    
    acc = accuracy_score(y_test, predictions)
    
    # mIoU
    classes = sorted(np.unique(np.concatenate([y_test, predictions])))
    ious = []
    for c in classes:
        pred_c = (predictions == c)
        true_c = (y_test == c)
        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        if union > 0:
            ious.append(intersection / union)
    
    miou = np.mean(ious) if ious else 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=classes)
    
    return {
        "accuracy": acc,
        "miou": miou,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="/workspace/xuannv/outputs/aef_official_embeddings/aef_haidian_2025.npz")
    parser.add_argument("--data-root", default="/workspace/xuannv/data_raw/phase2_heilongjiang/haidian")
    parser.add_argument("--output", default="/workspace/xuannv/outputs/aef_official_embeddings/knn_aef_haidian.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample-pixels", type=int, default=5000)
    args = parser.parse_args()
    
    print("Loading AEF embeddings...")
    embeddings = load_aef_embeddings(args.embedding)
    print(f"Loaded {len(embeddings)} patches")
    
    data_root = Path(args.data_root)
    
    # WorldCover
    print("\n=== WorldCover ===")
    labels = {}
    for pid in embeddings:
        label = load_label(pid, "worldcover", "", data_root)
        if label is not None:
            labels[pid] = label
    
    if labels:
        mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
        result = evaluate_knn(embeddings, labels, mapping=mapping, k=args.k, sample_pixels=args.sample_pixels)
        print(f"  Acc: {result['accuracy']:.4f}, mIoU: {result['miou']:.4f}")
        print(f"  Classes: {result['classes']}")
    else:
        print("  No labels found")
        result = None
    
    # JRC Water
    print("\n=== JRC Water ===")
    labels = {}
    for pid in embeddings:
        label = load_label(pid, "jrc_water", "", data_root)
        if label is not None:
            labels[pid] = label
    
    if labels:
        result_jrc = evaluate_knn(embeddings, labels, mapping=None, k=args.k, sample_pixels=args.sample_pixels)
        print(f"  Acc: {result_jrc['accuracy']:.4f}, mIoU: {result_jrc['miou']:.4f}")
        print(f"  Classes: {result_jrc['classes']}")
    else:
        print("  No labels found")
        result_jrc = None
    
    # Dynamic World
    print("\n=== Dynamic World ===")
    labels = {}
    for pid in embeddings:
        label = load_label(pid, "dynamic_world", "", data_root)
        if label is not None:
            labels[pid] = label
    
    if labels:
        result_dw = evaluate_knn(embeddings, labels, mapping=None, k=args.k, sample_pixels=args.sample_pixels)
        print(f"  Acc: {result_dw['accuracy']:.4f}, mIoU: {result_dw['miou']:.4f}")
        print(f"  Classes: {result_dw['classes']}")
    else:
        print("  No labels found")
        result_dw = None
    
    # 保存结果
    output = {
        "worldcover": result,
        "jrc_water": result_jrc,
        "dynamic_world": result_dw,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
