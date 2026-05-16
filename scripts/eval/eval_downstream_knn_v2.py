#!/usr/bin/env python3
"""冻结 backbone + KNN 下游评估 (v2 — skip_decoder + CPU单线程KNN)."""
import sys, os, argparse, json, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.utils.checkpoint import load_checkpoint

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

def extract_embeddings(model, dataset, device, cfg):
    """逐 patch 提取 embedding，skip decoder."""
    embeddings = {}
    for i, pid in enumerate(dataset.patches):
        idx = dataset.patches.index(pid)
        item = dataset[idx]
        with torch.no_grad():
            out = model(
                source_frames=item["source_frames"].unsqueeze(0).to(device),
                source_timestamps_ms=item["source_timestamps_ms"].unsqueeze(0).to(device),
                source_frame_mask=item["source_frame_mask"].unsqueeze(0).to(device),
                source_input_mask=item["source_input_mask"].unsqueeze(0).to(device),
                source_type_ids=item["source_type_ids"].unsqueeze(0).to(device),
                valid_start_ms=item["valid_start_ms"].unsqueeze(0).to(device),
                valid_end_ms=item["valid_end_ms"].unsqueeze(0).to(device),
                target_relative_time=torch.zeros(1, cfg.data.num_target_sources, device=device),
                target_metadata=torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=device),
                skip_decoder=True,
            )
            emb = F.normalize(out.embedding_map, p=2, dim=1)
        embeddings[pid] = emb.squeeze(0).cpu().numpy()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dataset.patches)} done")
    return embeddings

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
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    
    exp_dir = Path(f"/workspace/outputs/xuannv_round1/{args.experiment}")
    ckpt_path = list(exp_dir.glob("epoch_best_*.pt"))
    if not ckpt_path:
        print(f"No checkpoint in {exp_dir}"); return
    ckpt_path = ckpt_path[0]
    config_path = Path("/workspace/xuannv/configs") / f"{args.experiment}.yaml"
    print(f"Exp: {args.experiment}, ckpt: {ckpt_path}")
    
    cfg = load_config(str(config_path))
    cfg.experiment.name = args.experiment
    cfg.data.preload = False
    
    device = args.device
    if device.startswith("npu"):
        import torch_npu
        torch_npu.npu.set_device(int(device.split(":")[-1]))
    
    model = AEFModel(cfg)
    model.load_state_dict(load_checkpoint(str(ckpt_path)))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    dataset = HarbinPatchDataset(cfg)
    print(f"Dataset: {len(dataset.patches)} patches")
    
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataset, device, cfg)
    print(f"Extracted {len(embeddings)} embeddings")
    
    # 彻底释放 NPU
    del model
    if device.startswith("npu"):
        torch.npu.synchronize()
        torch.npu.empty_cache()
    
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
    
    out_dir = exp_dir / "downstream_knn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
