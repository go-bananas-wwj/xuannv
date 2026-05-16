#!/usr/bin/env python3
"""Round 8 综合下游任务评估 V2 — 修复 patch 遍历 bug.

评估任务:
1. WorldCover 语义分割 (11类)
2. Dynamic World 语义分割 (9类)
3. JRC Water 二值分割
4. OSM Buildings 二值分割

用法:
    python comprehensive_downstream_eval_v2.py \
        --config configs/round8_single_exp1.yaml \
        --checkpoint /workspace/outputs/round8_single_exp1/epoch_19.pt \
        --device npu:0 \
        --output /workspace/outputs/round8_single_exp1/downstream_results.json
"""
import sys, json, time, argparse
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score, roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=3)
    return parser.parse_args()


def load_backbone(config_path, checkpoint_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    cfg = load_config(config_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, cfg


def extract_embedding_for_patch(model, dataset, pidx, device):
    """提取指定月度样本索引的 embedding map."""
    batch = dataset[pidx]
    patch_id = batch.get("patch_id", dataset.monthly_samples[pidx][0])

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            source_frames=batch_dev["source_frames"],
            source_timestamps_ms=batch_dev["source_timestamps_ms"],
            source_frame_mask=batch_dev["source_frame_mask"],
            source_input_mask=batch_dev["source_input_mask"],
            source_type_ids=batch_dev["source_type_ids"],
            valid_start_ms=batch_dev["valid_start_ms"],
            valid_end_ms=batch_dev["valid_end_ms"],
            target_relative_time=batch_dev["target_relative_time"],
            target_metadata=batch_dev["target_metadata"],
        )
        emb_map = out.embedding_map[0].cpu()  # [D, H, W]
    return emb_map, patch_id


def load_label_direct(patch_id, label_type):
    """直接读取标签文件."""
    import rasterio
    from src.data.transforms import WC_CLASS_MAP

    if label_type == "worldcover":
        paths = [
            f"/workspace/raw/harbin_scenes/harbin/worldcover/{patch_id}/static.tif",
            f"/workspace/raw/phase1_harbin/harbin/worldcover/{patch_id}/static.tif",
        ]
        for path in paths:
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                if data is not None and data.size > 0:
                    mapped = np.full_like(data, -1, dtype=np.int64)
                    for val, idx in WC_CLASS_MAP.items():
                        mapped[data == val] = idx
                    return mapped
            except Exception:
                continue
        return None

    elif label_type == "dynamic_world":
        paths = [
            f"/workspace/raw/harbin_scenes/harbin/dynamic_world/{patch_id}/static.tif",
            f"/workspace/raw/phase1_harbin/harbin/dynamic_world/{patch_id}/static.tif",
        ]
        for path in paths:
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                if data is not None and data.size > 0:
                    return data.astype(np.int64)
            except Exception:
                continue
        return None

    elif label_type == "jrc_water":
        paths = [
            f"/workspace/raw/harbin_scenes/harbin/jrc_water/{patch_id}/static.tif",
            f"/workspace/raw/phase1_harbin/harbin/jrc_water/{patch_id}/static.tif",
        ]
        for path in paths:
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                if data is not None and data.size > 0:
                    return data.astype(np.int64)
            except Exception:
                continue
        return None

    elif label_type == "osm_buildings":
        path = f"/workspace/raw/harbin_scenes/osm_buildings/{patch_id}/static.tif"
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
            if data is not None and data.size > 0:
                return (data > 0).astype(np.int64)
        except Exception:
            pass
        return None

    return None


def resize_label(label, target_h, target_w):
    label_t = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(label_t, size=(target_h, target_w), mode='nearest')[0, 0]
    return resized.numpy().astype(np.int64)


def prepare_data(emb_maps, labels, num_classes):
    X_list, y_list = [], []
    for emb, label in zip(emb_maps, labels):
        D, H, W = emb.shape
        emb_flat = emb.reshape(D, -1).T
        label_flat = label.reshape(-1)
        valid_mask = (label_flat >= 0) & (label_flat < num_classes)
        if valid_mask.sum() == 0:
            continue
        X_list.append(emb_flat[valid_mask])
        y_list.append(label_flat[valid_mask])
    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)


def evaluate_semantic(X, y, n_folds=3):
    present_classes = np.unique(y)
    n_classes = len(present_classes)
    if n_classes < 2:
        return {"error": f"only {n_classes} class"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bacc_scores, f1m_scores, f1w_scores, miou_scores = [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        max_train = 30000
        if len(y_train) > max_train:
            indices = np.random.choice(len(y_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        clf = LogisticRegression(max_iter=300, multi_class='multinomial', solver='lbfgs', n_jobs=4, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro', labels=present_classes, zero_division=0)
        f1w = f1_score(y_test, y_pred, average='weighted', labels=present_classes, zero_division=0)

        ious = []
        for c in present_classes:
            inter = ((y_pred == c) & (y_test == c)).sum()
            union = ((y_pred == c) | (y_test == c)).sum()
            ious.append(inter / max(union, 1))
        miou = np.mean(ious)

        bacc_scores.append(bacc)
        f1m_scores.append(f1m)
        f1w_scores.append(f1w)
        miou_scores.append(miou)

    return {
        "balanced_accuracy": float(np.mean(bacc_scores)),
        "f1_macro": float(np.mean(f1m_scores)),
        "f1_weighted": float(np.mean(f1w_scores)),
        "miou": float(np.mean(miou_scores)),
        "n_classes": int(n_classes),
        "n_pixels": int(len(y)),
    }


def evaluate_binary(X, y, n_folds=3):
    if len(np.unique(y)) < 2:
        return {"error": "only 1 class"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bacc_scores, f1_scores, iou_scores, auc_scores = [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        max_train = 30000
        if len(y_train) > max_train:
            indices = np.random.choice(len(y_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        clf = LogisticRegression(max_iter=300, n_jobs=4, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        inter = ((y_pred == 1) & (y_test == 1)).sum()
        union = ((y_pred == 1) | (y_test == 1)).sum()
        iou = inter / max(union, 1)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        bacc_scores.append(bacc)
        f1_scores.append(f1)
        iou_scores.append(iou)
        auc_scores.append(auc)

    return {
        "balanced_accuracy": float(np.mean(bacc_scores)),
        "f1": float(np.mean(f1_scores)),
        "iou": float(np.mean(iou_scores)),
        "auc": float(np.mean(auc_scores)),
        "n_pixels": int(len(y)),
    }


def main():
    args = parse_args()
    print("=" * 70)
    print(f"  Round 8 综合下游任务评估 V2")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 70)

    device = args.device
    if device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(device)

    print("\n[1/4] 加载模型...")
    model, dataset, cfg = load_backbone(args.config, args.checkpoint, device)

    # 为每个 patch 找到第一个月度样本索引
    patch_to_idx = {}
    for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
        if pid not in patch_to_idx:
            patch_to_idx[pid] = idx

    all_patches = [p for p in dataset.patches if p in patch_to_idx]
    print(f"  Patches: {len(all_patches)}")

    # 提取 embedding
    print("\n[2/4] 提取 embedding maps...")
    emb_maps = []
    patch_ids = []
    for i, pid in enumerate(all_patches):
        pidx = patch_to_idx[pid]
        try:
            emb, _ = extract_embedding_for_patch(model, dataset, pidx, device)
            emb_maps.append(emb.numpy())
            patch_ids.append(pid)
        except Exception as e:
            print(f"  跳过 {pid}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_patches)}...")
    print(f"  完成: {len(emb_maps)} patches")

    if not emb_maps:
        print("错误: 没有成功提取任何 embedding")
        return

    D, H, W = emb_maps[0].shape
    results = {}

    # Task 1: WorldCover
    print("\n[3/4] WorldCover 语义分割...")
    wc_labels = [load_label_direct(pid, "worldcover") for pid in patch_ids]
    valid = [(e, l) for e, l in zip(emb_maps, wc_labels) if l is not None]
    if valid:
        X, y = prepare_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 11)
        if X is not None:
            results["worldcover"] = evaluate_semantic(X, y, args.folds)
            print(f"  mIoU={results['worldcover']['miou']:.4f}, BAcc={results['worldcover']['balanced_accuracy']:.4f}")
        else:
            results["worldcover"] = {"error": "no valid data"}
    else:
        results["worldcover"] = {"error": "no labels"}

    # Task 2: Dynamic World
    print("\n      Dynamic World 语义分割...")
    dw_labels = [load_label_direct(pid, "dynamic_world") for pid in patch_ids]
    valid = [(e, l) for e, l in zip(emb_maps, dw_labels) if l is not None]
    if valid:
        X, y = prepare_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 9)
        if X is not None:
            results["dynamic_world"] = evaluate_semantic(X, y, args.folds)
            print(f"  mIoU={results['dynamic_world']['miou']:.4f}, BAcc={results['dynamic_world']['balanced_accuracy']:.4f}")
        else:
            results["dynamic_world"] = {"error": "no valid data"}
    else:
        results["dynamic_world"] = {"error": "no labels"}

    # Task 3: JRC Water
    print("\n      JRC Water 二值分割...")
    jrc_labels = [load_label_direct(pid, "jrc_water") for pid in patch_ids]
    valid = [(e, l) for e, l in zip(emb_maps, jrc_labels) if l is not None]
    if valid:
        X, y = prepare_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 2)
        if X is not None:
            y = (y > 0).astype(np.int64)
            results["jrc_water"] = evaluate_binary(X, y, args.folds)
            print(f"  IoU={results['jrc_water']['iou']:.4f}, F1={results['jrc_water']['f1']:.4f}, AUC={results['jrc_water']['auc']:.4f}")
        else:
            results["jrc_water"] = {"error": "no valid data"}
    else:
        results["jrc_water"] = {"error": "no labels"}

    # Task 4: OSM Buildings
    print("\n      OSM Buildings 二值分割...")
    osm_labels = [load_label_direct(pid, "osm_buildings") for pid in patch_ids]
    valid = [(e, l) for e, l in zip(emb_maps, osm_labels) if l is not None]
    if valid:
        X, y = prepare_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 2)
        if X is not None:
            results["osm_buildings"] = evaluate_binary(X, y, args.folds)
            print(f"  IoU={results['osm_buildings']['iou']:.4f}, F1={results['osm_buildings']['f1']:.4f}, AUC={results['osm_buildings']['auc']:.4f}")
        else:
            results["osm_buildings"] = {"error": "no valid data"}
    else:
        results["osm_buildings"] = {"error": "no labels"}

    # 保存
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
