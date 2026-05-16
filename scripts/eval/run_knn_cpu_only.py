#!/usr/bin/env python3
"""Step 2: 纯 CPU KNN 下游评估（读取预提取的 embedding）."""
import sys, os, argparse, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import rasterio
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
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

def knn_eval(embeddings, labels_dict, k=5):
    patch_ids = sorted(embeddings.keys())
    all_preds, all_trues = [], []
    for test_pid in patch_ids:
        train_pids = [p for p in patch_ids if p != test_pid]
        X_train, y_train = [], []
        for pid in train_pids:
            emb = embeddings[pid]
            label = labels_dict[pid]
            if emb.shape[1:] != label.shape:
                continue
            mask = label >= 0
            if mask.sum() == 0:
                continue
            X_train.append(emb[:, mask].T)
            y_train.append(label[mask])
        if len(X_train) == 0:
            continue
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        if len(X_train) > 100000:
            idx = np.random.choice(len(X_train), 100000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
        test_emb = embeddings[test_pid]
        test_label = labels_dict[test_pid]
        if test_emb.shape[1:] != test_label.shape:
            continue
        mask = test_label >= 0
        if mask.sum() == 0:
            continue
        X_test = test_emb[:, mask].T
        y_test = test_label[mask]
        knn = KNeighborsClassifier(n_neighbors=min(k, len(X_train)), algorithm='auto', n_jobs=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        all_preds.append(y_pred)
        all_trues.append(y_test)
    if len(all_preds) == 0:
        return None
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    acc = accuracy_score(all_trues, all_preds)
    cm = confusion_matrix(all_trues, all_preds)
    per_class_iou = {}
    for c in np.unique(all_trues):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        per_class_iou[int(c)] = float(tp / (tp + fp + fn + 1e-8))
    return {
        "accuracy": float(acc),
        "mean_iou": float(np.mean(list(per_class_iou.values()))),
        "per_class_iou": per_class_iou,
        "num_samples": int(len(all_trues)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    
    exp_dir = Path(f"/workspace/outputs/xuannv_round1/{args.experiment}")
    emb_path = exp_dir / "downstream_knn" / "embeddings.npy"
    pid_path = exp_dir / "downstream_knn" / "patch_ids.json"
    if not emb_path.exists():
        print(f"No embeddings found: {emb_path}"); return
    
    embeddings = np.load(emb_path, allow_pickle=True).item()
    print(f"Loaded {len(embeddings)} embeddings")
    
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
            m = knn_eval(emb_c, lbl_c, k)
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
