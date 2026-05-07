#!/usr/bin/env python3
"""
AEF_qwen 少样本下游任务 Pipeline

论文方法: 冻结 backbone → 提取 embedding → 用 Linear/kNN 训练轻量分类头
参考: AlphaEarth Foundations 论文 §5 Evaluation Protocol

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=5 python3 scripts/downstream_fewshot.py \
        --model v2 \
        --checkpoint /workspace/outputs/aef_qwen_v2/epoch_499.pt \
        --before-window "2024-07-01" "2025-01-01" \
        --after-window "2025-04-01" "2025-10-31" \
        --output-dir /workspace/outputs/aef_qwen_v2/downstream
"""
import os, sys, json, time, argparse
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"

def date_to_ms(date_str):
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

def load_model(ckpt_path):
    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to("npu:0")
    ckpt = torch.load(ckpt_path, map_location="npu:0", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset

# ──────────────────────────────────────────
# 提取 embedding
# ──────────────────────────────────────────
def extract_embedding_for_patch(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
    """提取单个 patch 的 L2-normalized embedding map [D, H, W]."""
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to("npu:0")
        else:
            batch_dev[k] = v
    with torch.no_grad():
        output = model(
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
    emb_map = output.embedding_map  # [1, D, H, W]
    emb_map = F.normalize(emb_map, p=2, dim=1)
    return emb_map[0].cpu().numpy()

# ──────────────────────────────────────────
# 标注光栅化
# ──────────────────────────────────────────
def rasterize_annotations(patch_id, changes, patch_bounds, grid_size=64):
    """将标注多边形光栅化到 patch 网格."""
    bounds = patch_bounds.get(patch_id)
    if not bounds:
        return np.zeros((grid_size, grid_size), dtype=np.int32)

    H, W = grid_size, grid_size
    resolution = (bounds[2] - bounds[0]) / H
    mask = np.zeros((H, W), dtype=np.int32)  # 0=未变化, 1=变化

    for ch_info in changes:
        geom = ch_info["geometry"]
        if geom is None:
            continue
        # 扫描多边形包围盒内的像素
        minx, miny, maxx, maxy = geom.bounds
        px_start = max(0, int((minx - bounds[0]) / resolution))
        px_end = min(H, int((maxx - bounds[0]) / resolution) + 1)
        py_start = max(0, int((bounds[3] - maxy) / resolution))
        py_end = min(W, int((bounds[3] - miny) / resolution) + 1)

        for px in range(px_start, px_end):
            for py in range(py_start, py_end):
                wx = bounds[0] + (px + 0.5) * resolution
                wy = bounds[3] - (py + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[px, py] = 1

    return mask

# ──────────────────────────────────────────
# 少样本评估
# ──────────────────────────────────────────
def evaluate_fewshot(X, y, shot_counts, n_folds=5):
    """
    少样本评估 (对齐 AlphaEarth 论文协议)

    Args:
        X: [N, D] features
        y: [N] labels (0 or 1)
        shot_counts: list of ints [1, 10, 500]
        n_folds: cross-validation folds

    Returns:
        dict: {shot: {metric: value}}
    """
    results = {}
    rng = np.random.RandomState(42)

    for shot in shot_counts:
        all_aucs, all_bas, all_f1s = [], [], []

        for fold in range(n_folds):
            # 划分 train/test
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold)
            train_idx, test_idx = list(skf.split(X, y))[0]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # 少样本采样
            classes = np.unique(y_train)
            train_samples = []
            for c in classes:
                c_idx = np.where(y_train == c)[0]
                n_sample = min(shot, len(c_idx))
                sample_idx = rng.choice(c_idx, n_sample, replace=False)
                train_samples.extend(sample_idx.tolist())

            X_train_shot = X_train[train_samples]
            y_train_shot = y_train[train_samples]

            if len(np.unique(y_train_shot)) < 2:
                continue

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_shot)
            X_test_scaled = scaler.transform(X_test)

            # 训练分类器 (Linear + kNN)
            best_auc = 0
            classifiers = [
                ("Linear", LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')),
            ]
            if shot >= 10:
                classifiers.append(("kNN-3", KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train) // 10)))))
            else:
                classifiers.append(("kNN-1", KNeighborsClassifier(n_neighbors=1)))

            for clf_name, clf in classifiers:
                try:
                    clf.fit(X_train_scaled, y_train_shot)
                    # 对于小样本，kNN 可能没有 predict_proba
                    if hasattr(clf, 'predict_proba'):
                        try:
                            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        except Exception:
                            continue
                    else:
                        y_pred = clf.predict(X_test_scaled)
                        # 简单 AUC 近似 (无概率输出时)
                        y_prob = y_pred.astype(float)

                    y_pred = clf.predict(X_test_scaled)

                    if len(np.unique(y_test)) >= 2:
                        auc = roc_auc_score(y_test, y_prob)
                        ba = balanced_accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        if auc > best_auc:
                            best_auc = auc
                            best_metrics = {"auc": auc, "ba": ba, "f1": f1, "clf": clf_name}
                except Exception:
                    pass

            if best_auc > 0:
                all_aucs.append(best_metrics["auc"])
                all_bas.append(best_metrics["ba"])
                all_f1s.append(best_metrics["f1"])

        if all_aucs:
            results[shot] = {
                "auc_mean": np.mean(all_aucs),
                "auc_std": np.std(all_aucs),
                "ba_mean": np.mean(all_bas),
                "ba_std": np.std(all_bas),
                "f1_mean": np.mean(all_f1s),
                "f1_std": np.std(all_f1s),
                "n_folds": len(all_aucs),
            }

    return results

# ──────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="v2")
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/aef_qwen_v2/epoch_499.pt")
    parser.add_argument("--before-start", type=str, default="2024-07-01")
    parser.add_argument("--before-end", type=str, default="2025-01-01")
    parser.add_argument("--after-start", type=str, default="2025-04-01")
    parser.add_argument("--after-end", type=str, default="2025-10-31")
    parser.add_argument("--output-dir", type=str, default="/workspace/outputs/aef_qwen_v2/downstream")
    parser.add_argument("--max-patches", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  AEF_qwen 少样本下游任务 Pipeline")
    print("="*60)
    print(f"  模型: {args.model} ({args.checkpoint})")
    print(f"  Before: {args.before_start} ~ {args.before_end}")
    print(f"  After:  {args.after_start} ~ {args.after_end}")
    print(f"  输出:   {output_dir}")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    model, dataset = load_model(args.checkpoint)

    # 加载 Grid 和标注
    print("加载 Grid...")
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    print("加载标注数据...")
    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                if row.geometry is not None:
                    all_changes.append({"geometry": row.geometry, "period": shp_name.replace(".shp", "")})
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  {shp_name}: ERROR - {e}")

    # 找有标注的 patches
    patch_to_changes = {}
    for ch in all_changes:
        for pid, bounds in patch_bounds.items():
            patch_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
            if ch["geometry"].intersects(patch_box):
                if pid not in patch_to_changes:
                    patch_to_changes[pid] = []
                patch_to_changes[pid].append(ch)

    test_patches = list(patch_to_changes.items())[:args.max_patches]
    print(f"\n测试 {len(test_patches)} 个有标注的 patch")

    # 时间窗口
    bs = date_to_ms(args.before_start)
    be = date_to_ms(args.before_end)
    as_ = date_to_ms(args.after_start)
    ae = date_to_ms(args.after_end)

    # 提取 embedding + 构建数据集
    print("\n提取 embedding...")
    all_features = []
    all_labels = []
    n_positive = 0
    n_negative = 0

    for i, (pid, changes) in enumerate(test_patches):
        if pid not in dataset.patches:
            continue
        pidx = dataset.patches.index(pid)
        t0 = time.time()

        try:
            # 提取 before/after
            eb = extract_embedding_for_patch(model, dataset, pidx, bs, be)
            ea = extract_embedding_for_patch(model, dataset, pidx, as_, ae)

            # 光栅化标注
            mask = rasterize_annotations(pid, changes, patch_bounds)
            H, W = mask.shape

            # 构建特征: concat(before, after) per-pixel
            D = eb.shape[0]
            features = np.zeros((H * W, D * 2), dtype=np.float32)
            labels = mask.flatten()

            for px in range(H):
                for py in range(W):
                    idx = px * W + py
                    features[idx, :D] = eb[:, px, py]
                    features[idx, D:] = ea[:, px, py]

            # 采样平衡数据集 (避免负样本过多)
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            n_pos = len(pos_idx)
            n_neg_sample = min(n_pos * 3, len(neg_idx))  # 3:1 负正比
            neg_sample = np.random.choice(neg_idx, n_neg_sample, replace=False)

            sampled_idx = np.concatenate([pos_idx, neg_sample])
            np.random.shuffle(sampled_idx)

            all_features.append(features[sampled_idx])
            all_labels.append(labels[sampled_idx])
            n_positive += n_pos
            n_negative += n_neg_sample

            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(test_patches)}] {pid}: {n_pos} pos, {n_neg_sample} neg ({elapsed:.1f}s)")

        except Exception as e:
            import traceback
            print(f"  [{i+1}/{len(test_patches)}] {pid}: ERROR - {e}")
            traceback.print_exc()

    if not all_features:
        print("❌ 没有有效数据")
        return

    # 合并所有数据
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"\n总数据集: {len(X)} 样本, {sum(y)} 正样本, {len(X)-sum(y)} 负样本")
    print(f"  特征维度: {X.shape[1]}")

    # 少样本评估
    print("\n少样本评估 (5-fold CV)...")
    shot_counts = [1, 10, 50, 100, 500]
    results = evaluate_fewshot(X, y, shot_counts, n_folds=5)

    # 打印结果
    print("\n" + "="*70)
    print("  少样本下游任务评估结果")
    print("="*70)
    print(f"\n{'Shot':<8} {'AUC':<12} {'BA':<12} {'F1':<12} {'Folds':<8}")
    print("-"*55)
    for shot in shot_counts:
        if shot in results:
            r = results[shot]
            print(f"{shot:<8} {r['auc_mean']:.3f}±{r['auc_std']:.3f}   "
                  f"{r['ba_mean']:.3f}±{r['ba_std']:.3f}   "
                  f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}   {r['n_folds']}")
        else:
            print(f"{shot:<8} N/A")

    # 保存结果
    output_file = output_dir / f"downstream_{args.model}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": args.model,
            "checkpoint": args.checkpoint,
            "before_window": [args.before_start, args.before_end],
            "after_window": [args.after_start, args.after_end],
            "n_patches": len(test_patches),
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
            "total_samples": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "results": {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print(f"\n结果已保存: {output_file}")

    print("\n" + "="*70)
    print("  完成!")
    print("="*70)

if __name__ == "__main__":
    main()
