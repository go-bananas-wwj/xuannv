#!/usr/bin/env python3
"""KNN 下游评估 — 快速版: 随机 80/20 split + PyTorch NPU."""
import sys, os, argparse, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import rasterio
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_ROOT = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered"
DOWNSTREAM_TASKS = [
    ("worldcover", "worldcover", "static.tif", 10),
    ("jrc_water", "jrc_water", "static.tif", 2),
    ("dynamic_world", "dynamic_world", "2025Q2.tif", 9),
]

def load_label(patch_id, label_dir, fname):
    path = Path(DATA_ROOT) / label_dir / patch_id / fname
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        label = src.read(1)
        nodata = src.nodata
    return label, nodata

def knn_predict_npu(X_train, y_train, X_test, k, device):
    """PyTorch KNN on NPU."""
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    batch_size = 32
    preds = []
    for i in range(0, len(X_test), batch_size):
        batch = torch.from_numpy(X_test[i:i+batch_size]).to(device)
        dist = torch.cdist(batch, X_train_t)
        _, idx = dist.topk(min(k, len(X_train)), largest=False, dim=1)
        neighbor_labels = y_train_t[idx]
        for j in range(neighbor_labels.shape[0]):
            vals, counts = torch.unique(neighbor_labels[j], return_counts=True)
            preds.append(vals[counts.argmax()].item())
    return np.array(preds)

def knn_eval_fast(embeddings, labels_dict, k, device, seed=42):
    """随机 80/20 split (按 patch)，确保每个 patch 都有样本在 train/test 中."""
    patch_ids = sorted(embeddings.keys())
    rng = np.random.RandomState(seed)
    
    # 收集所有有效像素
    all_X = []
    all_y = []
    all_patch_idx = []
    for p_idx, pid in enumerate(patch_ids):
        emb = embeddings[pid]
        label = labels_dict.get(pid)
        if label is None or emb.shape[1:] != label.shape:
            continue
        mask = label >= 0
        if mask.sum() == 0:
            continue
        all_X.append(emb[:, mask].T)
        all_y.append(label[mask])
        all_patch_idx.append(np.full(mask.sum(), p_idx))
    
    if len(all_X) == 0:
        return None
    
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_patch_idx = np.concatenate(all_patch_idx)
    
    # 按 patch 分层 80/20 split
    n_patches = len(patch_ids)
    n_train_patches = max(1, int(n_patches * 0.8))
    train_pids = rng.choice(patch_ids, n_train_patches, replace=False)
    test_pids = [p for p in patch_ids if p not in train_pids]
    
    train_mask = np.isin(all_patch_idx, [patch_ids.index(p) for p in train_pids])
    test_mask = ~train_mask
    
    X_train = all_X[train_mask]
    y_train = all_y[train_mask]
    X_test = all_X[test_mask]
    y_test = all_y[test_mask]
    
    # Subsample train if too large
    if len(X_train) > 50000:
        idx = rng.choice(len(X_train), 50000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    y_pred = knn_predict_npu(X_train, y_train, X_test, k, device)
    
    acc = accuracy_score(y_test, y_pred)
    all_labels = sorted(set(y_test) | set(y_pred))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    per_class_iou = {}
    for c in all_labels:
        i = label_to_idx[c]
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        if tp + fp + fn > 0:
            per_class_iou[int(c)] = float(tp / (tp + fp + fn + 1e-8))
    
    return {
        "accuracy": float(acc),
        "mean_iou": float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0,
        "per_class_iou": per_class_iou,
        "num_samples": int(len(y_test)),
        "num_train": int(len(y_train)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    
    exp_dir = Path(f"/workspace/outputs/xuannv_round1/{args.experiment}")
    emb_path = exp_dir / "downstream_knn" / "embeddings.npy"
    if not emb_path.exists():
        print(f"No embeddings: {emb_path}"); return
    
    embeddings = np.load(emb_path, allow_pickle=True).item()
    print(f"Loaded {len(embeddings)} embeddings")
    
    device = args.device
    if device.startswith("npu"):
        import torch_npu
        torch_npu.npu.set_device(int(device.split(":")[-1]))
    
    results = {}
    for task_name, label_dir, fname, num_classes in DOWNSTREAM_TASKS:
        print(f"\nTask: {task_name}")
        labels_dict = {}
        for pid in embeddings:
            label, nodata = load_label(pid, label_dir, fname)
            if label is None: continue
            if nodata is not None:
                label = label.astype(np.int32)
                label[label == nodata] = -1
            emb_h, emb_w = embeddings[pid].shape[1:]
            if label.shape != (emb_h, emb_w):
                from scipy.ndimage import zoom
                label = zoom(label.astype(float), (emb_h/label.shape[0], emb_w/label.shape[1]), order=0).astype(np.int32)
            labels_dict[pid] = label
        common = sorted(set(embeddings) & set(labels_dict))
        print(f"  Common: {len(common)}")
        emb_c = {pid: embeddings[pid] for pid in common}
        lbl_c = {pid: labels_dict[pid] for pid in common}
        task_res = {}
        for k in [5, 20]:
            print(f"  K={k}...", end=" ", flush=True)
            m = knn_eval_fast(emb_c, lbl_c, k, device)
            if m:
                print(f"Acc={m['accuracy']:.4f}, mIoU={m['mean_iou']:.4f}")
                task_res[f"k{k}"] = m
            else:
                print("Failed")
        results[task_name] = task_res
    
    out_path = exp_dir / "downstream_knn" / "results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
