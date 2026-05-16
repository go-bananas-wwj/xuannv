#!/usr/bin/env python3
"""冻结 backbone + KNN 下游评估."""
import sys, argparse, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import rasterio
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.utils.checkpoint import load_checkpoint

DATA_ROOT = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered"

DOWNSTREAM_TASKS = [
    ("worldcover", "worldcover", "static.tif", 10, False),
    ("jrc_water", "jrc_water", "static.tif", 2, False),
    ("dynamic_world", "dynamic_world", "2025Q2.tif", 9, True),
]

def load_label(patch_id, label_dir, fname):
    path = Path(DATA_ROOT) / label_dir / patch_id / fname
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        label = src.read(1)
        nodata = src.nodata
    return label, nodata

def extract_embeddings(model, dataset, device, batch_size=4):
    """提取所有样本的 embedding map，按 patch_id 聚合（取平均）."""
    model.eval()
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    patch_embs = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            output = model(
                source_frames=batch["source_frames"].to(device),
                source_timestamps_ms=batch["source_timestamps_ms"].to(device),
                source_frame_mask=batch["source_frame_mask"].to(device),
                source_input_mask=batch["source_input_mask"].to(device),
                source_type_ids=batch["source_type_ids"].to(device),
                valid_start_ms=batch["valid_start_ms"].to(device),
                valid_end_ms=batch["valid_end_ms"].to(device),
                target_relative_time=batch.get("target_relative_time"),
                target_metadata=batch.get("target_metadata"),
            )
            emb_map = output.embedding_map.cpu().numpy()  # [B, D, H, W]
            for i, pid in enumerate(batch["patch_id"]):
                patch_embs[pid].append(emb_map[i])
    
    # Average over months
    result = {}
    for pid, embs in patch_embs.items():
        result[pid] = np.mean(embs, axis=0)  # [D, H, W]
    return result

def knn_eval(embeddings, labels_dict, k=5):
    patch_ids = sorted(embeddings.keys())
    all_preds = []
    all_trues = []
    
    for test_pid in patch_ids:
        train_pids = [p for p in patch_ids if p != test_pid]
        
        X_train = []
        y_train = []
        for pid in train_pids:
            emb = embeddings[pid]  # [D, H, W]
            label = labels_dict[pid]
            if emb.shape[1:] != label.shape:
                continue
            D, H, W = emb.shape
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
        
        knn = KNeighborsClassifier(n_neighbors=min(k, len(X_train)), algorithm='auto', n_jobs=4)
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
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        per_class_iou[int(c)] = float(iou)
    mean_iou = np.mean(list(per_class_iou.values()))
    
    return {
        "accuracy": float(acc),
        "mean_iou": float(mean_iou),
        "per_class_iou": per_class_iou,
        "num_samples": int(len(all_trues)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--k-values", default="5,20")
    args = parser.parse_args()
    
    exp_dir = Path(f"/workspace/outputs/xuannv_round1/{args.experiment}")
    ckpt_path = list(exp_dir.glob("epoch_best_*.pt"))
    if not ckpt_path:
        print(f"No checkpoint found in {exp_dir}")
        return
    ckpt_path = ckpt_path[0]
    
    config_path = Path("/workspace/xuannv/configs") / f"{args.experiment}.yaml"
    print(f"Exp: {args.experiment}, ckpt: {ckpt_path}, cfg: {config_path}")
    
    cfg = load_config(str(config_path))
    cfg.experiment.name = args.experiment
    cfg.data.preload = False
    
    device = args.device
    if device.startswith("npu"):
        dev_idx = int(device.split(":")[-1])
        import torch_npu
        torch_npu.npu.set_device(dev_idx)
    
    model = AEFModel(cfg)
    state = load_checkpoint(str(ckpt_path))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    dataset = HarbinPatchDataset(cfg)
    print(f"Dataset: {len(dataset)} samples, {len(dataset.patches)} patches")
    
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataset, device, batch_size=4)
    print(f"Extracted {len(embeddings)} patch embeddings")
    
    del model
    if device.startswith("npu"):
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()
    
    results = {}
    k_values = [int(k) for k in args.k_values.split(",")]
    
    for task_name, label_dir, fname, num_classes, _ in DOWNSTREAM_TASKS:
        print(f"\nTask: {task_name}")
        labels_dict = {}
        for pid in embeddings.keys():
            label, nodata = load_label(pid, label_dir, fname)
            if label is None:
                continue
            if nodata is not None:
                label = label.astype(np.int32)
                label[label == nodata] = -1
            # Resize label to match embedding size
            emb_h, emb_w = embeddings[pid].shape[1:]
            if label.shape != (emb_h, emb_w):
                from scipy.ndimage import zoom
                zoom_y = emb_h / label.shape[0]
                zoom_x = emb_w / label.shape[1]
                label = zoom(label.astype(float), (zoom_y, zoom_x), order=0).astype(np.int32)
            labels_dict[pid] = label
        
        common = sorted(set(embeddings.keys()) & set(labels_dict.keys()))
        print(f"  Common patches: {len(common)}")
        emb_c = {pid: embeddings[pid] for pid in common}
        lbl_c = {pid: labels_dict[pid] for pid in common}
        
        task_results = {}
        for k in k_values:
            print(f"  K={k}...", end=" ", flush=True)
            m = knn_eval(emb_c, lbl_c, k=k)
            if m:
                print(f"Acc={m['accuracy']:.4f}, mIoU={m['mean_iou']:.4f}")
                task_results[f"k{k}"] = m
            else:
                print("Failed")
        results[task_name] = task_results
    
    out_dir = exp_dir / "downstream_knn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
