#!/usr/bin/env python3
"""V7 下游头对比实验：在 frozen embedding 上训练 MLP/KNN/LR，测试变化检测 AUC.

对比基线: Bare AUC (cosine distance)
下游方法:
  - LR on |diff|
  - MLP on |diff|
  - KNN on |diff|
  - LR on concat(e1, e2)
  - LR on CD-head-style features

使用 5-fold stratified CV 评估（样本量小，CV 更稳定）。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch_npu  # 注册 NPU 后端
import torch.nn.functional as F
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, "/workspace/xuannv")

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
CONFIG_PATH = "/workspace/xuannv/configs/xuannv_v7.yaml"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # ⚠️ 已过时，标注为2025年变化
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)   # ⚠️ 已过时，标注为2025年变化


def load_model(ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset


def extract_embedding(model, dataset, patch_id, window, device):
    """提取单个 patch 在指定时间窗口的 embedding."""
    try:
        item = dataset[dataset.patches.index(patch_id)]
    except ValueError:
        return None

    item["valid_start_ms"] = torch.tensor([window[0]], dtype=torch.float64)
    item["valid_end_ms"] = torch.tensor([window[1]], dtype=torch.float64)

    mask = (item["source_timestamps_ms"] >= window[0]) & (item["source_timestamps_ms"] <= window[1])
    item["source_frame_mask"] = mask

    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

    with torch.no_grad():
        out = model(
            source_frames=batch["source_frames"],
            source_timestamps_ms=batch["source_timestamps_ms"],
            source_frame_mask=batch["source_frame_mask"],
            source_input_mask=batch["source_input_mask"],
            source_type_ids=batch["source_type_ids"],
            valid_start_ms=batch["valid_start_ms"],
            valid_end_ms=batch["valid_end_ms"],
            target_relative_time=batch["target_relative_time"],
            target_metadata=batch["target_metadata"],
        )
        emb = out.embedding[0]  # [D]
        emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy()


def load_annotations():
    """加载所有标注变化图斑，返回 patch_id -> list of geometries."""
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                all_changes.append({
                    "geometry": geom,
                    "patch_id": row.get("patch_id", None),
                })
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

    patch_changes = {}
    for change in all_changes:
        if change["patch_id"]:
            pid = change["patch_id"]
            patch_changes.setdefault(pid, []).append(change["geometry"])
        else:
            pt = change["geometry"].centroid
            for pid, bounds in patch_bounds.items():
                if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                    patch_changes.setdefault(pid, []).append(change["geometry"])
                    break

    annotated_patches = [pid for pid, geoms in patch_changes.items() if len(geoms) > 0]
    return annotated_patches, patch_changes


def extract_embeddings_fast(model, dataset, patch_ids, device, desc=""):
    """批量提取 embedding，带进度打印。"""
    embs_before, embs_after, valid_pids = [], [], []
    total = len(patch_ids)
    for i, pid in enumerate(patch_ids):
        if i % 20 == 0:
            print(f"  {desc} {i}/{total} ...", flush=True)
        emb_b = extract_embedding(model, dataset, pid, BEFORE_WINDOW, device)
        emb_a = extract_embedding(model, dataset, pid, AFTER_WINDOW, device)
        if emb_b is None or emb_a is None:
            continue
        # NaN check
        if np.isnan(emb_b).any() or np.isnan(emb_a).any():
            print(f"    NaN embedding: {pid}, skip", flush=True)
            continue
        embs_before.append(emb_b)
        embs_after.append(emb_a)
        valid_pids.append(pid)
    print(f"  {desc} 完成: {len(valid_pids)}/{total} 成功", flush=True)
    return np.stack(embs_before), np.stack(embs_after), valid_pids


def compute_features(embs_before, embs_after, feat_type="diff"):
    """根据 feat_type 生成特征矩阵."""
    if feat_type == "diff":
        return np.abs(embs_before - embs_after)  # [N, D]
    elif feat_type == "diff_sq":
        return (embs_before - embs_after) ** 2
    elif feat_type == "concat":
        return np.concatenate([embs_before, embs_after], axis=1)  # [N, 2D]
    elif feat_type == "cdhead":
        diff = embs_before - embs_after
        mul = embs_before * embs_after
        return np.concatenate([np.abs(diff), mul, embs_before, embs_after], axis=1)  # [N, 4D]
    elif feat_type == "cosine":
        cos_sim = np.sum(embs_before * embs_after, axis=1, keepdims=True)
        return 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown feat_type: {feat_type}")


def eval_method(name, X, y, n_splits=5):
    """用 5-fold stratified CV 评估 AUC."""
    if len(set(y)) < 2:
        return {"auc": 0.5, "fold_aucs": [], "note": "only one class"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if name.startswith("LR"):
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        elif name.startswith("MLP"):
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42, early_stopping=True)
        elif name.startswith("KNN"):
            clf = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown method: {name}")

        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]

        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)

    return {"auc": float(np.mean(fold_aucs)), "std": float(np.std(fold_aucs)), "fold_aucs": fold_aucs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--device", type=str, default="npu:7", help="设备")
    parser.add_argument("--n_splits", type=int, default=5, help="CV fold 数")
    parser.add_argument("--n_neg", type=int, default=300, help="随机抽样的负样本 patch 数")
    args = parser.parse_args()

    device = args.device
    ckpt_path = args.checkpoint

    print("=" * 70)
    print("  V7 下游头对比实验: MLP / KNN / LR on Frozen Embedding")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {device}")
    print("=" * 70)

    # 加载模型
    print("\n[1/5] 加载模型...")
    model, dataset = load_model(ckpt_path, device)
    print(f"模型加载完成，Dataset: {len(dataset)} patches")

    # 加载标注
    print("\n[2/5] 加载标注数据...")
    annotated_patches, patch_changes = load_annotations()
    print(f"有标注(正例候选) patch 数: {len(annotated_patches)}")

    # 准备 patch 列表: 正例 + 随机负例
    import random
    random.seed(42)
    neg_candidates = [pid for pid in dataset.patches if pid not in patch_changes]
    n_neg = min(args.n_neg, len(neg_candidates))
    neg_patches = random.sample(neg_candidates, n_neg) if n_neg > 0 else []
    all_eval_patches = annotated_patches + neg_patches
    print(f"评估样本: {len(annotated_patches)} 正例候选 + {len(neg_patches)} 负例 = {len(all_eval_patches)} 总计")

    # 提取 embedding
    print("\n[3/5] 提取 embedding...")
    embs_before, embs_after, pids = extract_embeddings_fast(
        model, dataset, all_eval_patches, device, desc="提取"
    )

    # 构造标签
    y = np.array([1 if len(patch_changes.get(pid, [])) > 0 else 0 for pid in pids], dtype=int)
    print(f"\n最终样本: N={len(y)}, 正例={y.sum()}, 负例={len(y)-y.sum()}")

    if len(set(y)) < 2:
        print("错误: 只有一类样本，无法计算 AUC")
        return

    # ──────────────────────────────────────────
    # 基线: Bare Cosine Distance
    # ──────────────────────────────────────────
    print("\n[4/5] 评估各方法...")
    cos_sim = np.sum(embs_before * embs_after, axis=1)
    bare_scores = 1.0 - cos_sim

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    bare_fold_aucs = []
    for train_idx, test_idx in skf.split(embs_before, y):
        try:
            auc = roc_auc_score(y[test_idx], bare_scores[test_idx])
        except ValueError:
            auc = 0.5
        bare_fold_aucs.append(auc)

    results = []
    results.append({
        "method": "Bare (cosine dist)",
        "feat": "-",
        "auc_mean": float(np.mean(bare_fold_aucs)),
        "auc_std": float(np.std(bare_fold_aucs)),
        "fold_aucs": bare_fold_aucs,
    })

    # ──────────────────────────────────────────
    # 下游头实验
    # ──────────────────────────────────────────
    experiments = [
        ("LR", "diff", "|e1-e2|"),
        ("MLP", "diff", "|e1-e2|"),
        ("KNN", "diff", "|e1-e2|"),
        ("LR", "diff_sq", "(e1-e2)^2"),
        ("LR", "concat", "concat(e1,e2)"),
        ("LR", "cdhead", "[abs,mult,e1,e2]"),
        ("LR", "cosine", "1-cos_sim"),
    ]

    for method, feat_type, feat_desc in experiments:
        X = compute_features(embs_before, embs_after, feat_type)
        r = eval_method(method, X, y, args.n_splits)
        results.append({
            "method": f"{method} ({feat_desc})",
            "feat": feat_type,
            "auc_mean": r["auc"],
            "auc_std": r.get("std", 0.0),
            "fold_aucs": r["fold_aucs"],
        })

    # ──────────────────────────────────────────
    # 打印结果
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  结果汇总 (5-fold Stratified CV, N={len(y)} patches)")
    print("=" * 70)
    print(f"{'Method':<35} {'AUC':>8} {'Std':>8} {'Fold AUCs'}")
    print("-" * 70)
    for r in results:
        fold_str = ", ".join(f"{a:.3f}" for a in r["fold_aucs"])
        print(f"{r['method']:<35} {r['auc_mean']:>8.4f} {r['auc_std']:>8.4f}  [{fold_str}]")
    print("=" * 70)

    # 保存结果
    out_path = Path(ckpt_path).parent / f"downstream_head_results_{Path(ckpt_path).stem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
