#!/usr/bin/env python3
"""Round 8 综合下游任务评估 — Linear Probe on frozen embeddings.

评估任务:
1. WorldCover 语义分割 (11类)
2. Dynamic World 语义分割 (9类)
3. JRC Water 二值分割
4. OSM Buildings 二值分割

用法:
    python comprehensive_downstream_eval.py \
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
    parser.add_argument("--folds", type=int, default=5)
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


def extract_embedding_map(model, dataset, pidx, device):
    """提取单个 patch 的 embedding map."""
    batch = dataset[pidx]
    patch_id = batch.get("patch_id", dataset.patches[pidx])

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


def load_label(dataset, patch_id, label_type):
    """加载指定类型的标签（像素级）."""
    import rasterio
    from src.data.transforms import WC_CLASS_MAP

    if label_type == "worldcover":
        source_name = "worldcover"
    elif label_type == "dynamic_world":
        source_name = "dynamic_world"
    elif label_type == "jrc_water":
        source_name = "jrc_water"
    elif label_type == "osm_buildings":
        path = f"/workspace/raw/harbin_scenes/osm_buildings/{patch_id}/static.tif"
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                if data is None or data.size == 0:
                    return None
                return (data > 0).astype(np.int64)
        except Exception:
            return None
        return None
    else:
        return None

    # 使用 dataset 的路径解析来找到源目录
    try:
        src_dir = dataset._resolve_source_dir(source_name, patch_id)
        if src_dir is None:
            return None
        tif_files = list(src_dir.glob("*.tif"))
        if not tif_files:
            return None

        with rasterio.open(tif_files[0]) as src:
            data = src.read(1)
        if data is None or data.size == 0:
            return None

        if label_type == "worldcover":
            mapped = np.full_like(data, -1, dtype=np.int64)
            for val, idx in WC_CLASS_MAP.items():
                mapped[data == val] = idx
            return mapped
        elif label_type == "dynamic_world":
            return data.astype(np.int64)
        elif label_type == "jrc_water":
            return data.astype(np.int64)
    except Exception:
        return None
    return None


def resize_label(label, target_h, target_w):
    """将标签下采样到 embedding 尺寸."""
    label_t = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(label_t, size=(target_h, target_w), mode='nearest')[0, 0]
    return resized.numpy().astype(np.int64)


def prepare_data(emb_maps, labels, num_classes):
    """展平 embedding 和标签，准备 sklearn 输入."""
    X_list, y_list = [], []
    for emb, label in zip(emb_maps, labels):
        D, H, W = emb.shape
        emb_flat = emb.reshape(D, -1).T  # [H*W, D]
        label_flat = label.reshape(-1)    # [H*W]

        # 过滤无效标签 (-1, 255, 0背景等)
        valid_mask = (label_flat >= 0) & (label_flat < num_classes)
        if valid_mask.sum() == 0:
            continue

        X_list.append(emb_flat[valid_mask])
        y_list.append(label_flat[valid_mask])

    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)


def evaluate_semantic_task(X, y, task_name, n_folds=5):
    """使用 K-Fold CV 评估语义分割任务."""
    present_classes = np.unique(y)
    n_classes = len(present_classes)

    if n_classes < 2:
        return {"error": f"only {n_classes} class present"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    bacc_scores = []
    f1_macro_scores = []
    f1_weighted_scores = []
    miou_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 采样训练数据（避免太多样本）
        max_train = 50000
        if len(y_train) > max_train:
            indices = np.random.choice(len(y_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        clf = LogisticRegression(
            max_iter=500,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=4,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', labels=present_classes, zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', labels=present_classes, zero_division=0)

        # mIoU
        ious = []
        for c in present_classes:
            inter = ((y_pred == c) & (y_test == c)).sum()
            union = ((y_pred == c) | (y_test == c)).sum()
            iou = inter / max(union, 1)
            ious.append(iou)
        miou = np.mean(ious)

        bacc_scores.append(bacc)
        f1_macro_scores.append(f1_macro)
        f1_weighted_scores.append(f1_weighted)
        miou_scores.append(miou)

    return {
        "balanced_accuracy": float(np.mean(bacc_scores)),
        "f1_macro": float(np.mean(f1_macro_scores)),
        "f1_weighted": float(np.mean(f1_weighted_scores)),
        "miou": float(np.mean(miou_scores)),
        "bacc_std": float(np.std(bacc_scores)),
        "f1_macro_std": float(np.std(f1_macro_scores)),
        "miou_std": float(np.std(miou_scores)),
        "n_classes": int(n_classes),
        "n_pixels": int(len(y)),
    }


def evaluate_binary_task(X, y, task_name, n_folds=5):
    """使用 K-Fold CV 评估二值分割任务."""
    present_classes = np.unique(y)
    if len(present_classes) < 2:
        return {"error": f"only {len(present_classes)} class present"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    bacc_scores = []
    f1_scores_list = []
    iou_scores = []
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        max_train = 50000
        if len(y_train) > max_train:
            indices = np.random.choice(len(y_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        clf = LogisticRegression(max_iter=500, n_jobs=4, random_state=42)
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
        f1_scores_list.append(f1)
        iou_scores.append(iou)
        auc_scores.append(auc)

    return {
        "balanced_accuracy": float(np.mean(bacc_scores)),
        "f1": float(np.mean(f1_scores_list)),
        "iou": float(np.mean(iou_scores)),
        "auc": float(np.mean(auc_scores)),
        "bacc_std": float(np.std(bacc_scores)),
        "f1_std": float(np.std(f1_scores_list)),
        "n_pixels": int(len(y)),
    }


def main():
    args = parse_args()
    print("=" * 70)
    print(f"  Round 8 综合下游任务评估")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 70)

    device = args.device
    if device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(device)

    # 加载模型
    print("\n[1/5] 加载模型...")
    model, dataset, cfg = load_backbone(args.config, args.checkpoint, device)
    print(f"  Dataset: {len(dataset)} patches")

    # 提取所有 patch 的 embedding
    print("\n[2/5] 提取 embedding maps...")
    emb_maps = []
    patch_ids = []
    all_patches = dataset.patches
    for pidx, pid in enumerate(all_patches):
        try:
            item_idx = dataset.patches.index(pid)
            emb, _ = extract_embedding_map(model, dataset, item_idx, device)
            emb_maps.append(emb.numpy())
            patch_ids.append(pid)
        except Exception as e:
            print(f"  跳过 {pid}: {e}")
            continue
        if (pidx + 1) % 50 == 0:
            print(f"  {pidx+1}/{len(all_patches)}...")
    print(f"  完成: {len(emb_maps)} patches")

    results = {}
    D, H, W = emb_maps[0].shape

    # ── Task 1: WorldCover ──
    print("\n[3/5] WorldCover 语义分割...")
    wc_labels = []
    for pid in patch_ids:
        label = load_label(dataset, pid, "worldcover")
        if label is not None:
            label = resize_label(label, H, W)
        wc_labels.append(label)

    valid_pairs = [(e, l) for e, l in zip(emb_maps, wc_labels) if l is not None]
    if valid_pairs:
        X, y = prepare_data([p[0] for p in valid_pairs], [p[1] for p in valid_pairs], 11)
        if X is not None:
            results["worldcover"] = evaluate_semantic_task(X, y, "WorldCover", args.folds)
            print(f"  结果: mIoU={results['worldcover']['miou']:.4f}, BAcc={results['worldcover']['balanced_accuracy']:.4f}")
        else:
            results["worldcover"] = {"error": "no valid data"}
    else:
        results["worldcover"] = {"error": "no labels found"}

    # ── Task 2: Dynamic World ──
    print("\n[4/5] Dynamic World 语义分割...")
    dw_labels = []
    for pid in patch_ids:
        label = load_label(dataset, pid, "dynamic_world")
        if label is not None:
            label = resize_label(label, H, W)
        dw_labels.append(label)

    valid_pairs = [(e, l) for e, l in zip(emb_maps, dw_labels) if l is not None]
    if valid_pairs:
        X, y = prepare_data([p[0] for p in valid_pairs], [p[1] for p in valid_pairs], 9)
        if X is not None:
            results["dynamic_world"] = evaluate_semantic_task(X, y, "DynamicWorld", args.folds)
            print(f"  结果: mIoU={results['dynamic_world']['miou']:.4f}, BAcc={results['dynamic_world']['balanced_accuracy']:.4f}")
        else:
            results["dynamic_world"] = {"error": "no valid data"}
    else:
        results["dynamic_world"] = {"error": "no labels found"}

    # ── Task 3: JRC Water ──
    print("\n[5/5] JRC Water 二值分割...")
    jrc_labels = []
    for pid in patch_ids:
        label = load_label(dataset, pid, "jrc_water")
        if label is not None:
            label = resize_label(label, H, W)
        jrc_labels.append(label)

    valid_pairs = [(e, l) for e, l in zip(emb_maps, jrc_labels) if l is not None]
    if valid_pairs:
        X, y = prepare_data([p[0] for p in valid_pairs], [p[1] for p in valid_pairs], 2)
        if X is not None:
            # 二值化: >0 = water
            y = (y > 0).astype(np.int64)
            results["jrc_water"] = evaluate_binary_task(X, y, "JRC_Water", args.folds)
            print(f"  结果: IoU={results['jrc_water']['iou']:.4f}, F1={results['jrc_water']['f1']:.4f}, AUC={results['jrc_water']['auc']:.4f}")
        else:
            results["jrc_water"] = {"error": "no valid data"}
    else:
        results["jrc_water"] = {"error": "no labels found"}

    # ── Task 4: OSM Buildings ──
    print("\n[ bonus ] OSM Buildings 二值分割...")
    osm_labels = []
    for pid in patch_ids:
        label = load_label(dataset, pid, "osm_buildings")
        if label is not None:
            label = resize_label(label, H, W)
        osm_labels.append(label)

    valid_pairs = [(e, l) for e, l in zip(emb_maps, osm_labels) if l is not None]
    if valid_pairs:
        X, y = prepare_data([p[0] for p in valid_pairs], [p[1] for p in valid_pairs], 2)
        if X is not None:
            results["osm_buildings"] = evaluate_binary_task(X, y, "OSM_Buildings", args.folds)
            print(f"  结果: IoU={results['osm_buildings']['iou']:.4f}, F1={results['osm_buildings']['f1']:.4f}, AUC={results['osm_buildings']['auc']:.4f}")
        else:
            results["osm_buildings"] = {"error": "no valid data"}
    else:
        results["osm_buildings"] = {"error": "no labels found"}

    # 保存结果
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {args.output}")

    print("\n" + "=" * 70)
    print("  评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
