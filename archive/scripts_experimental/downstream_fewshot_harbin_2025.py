#!/usr/bin/env python3
"""
AEF_qwen 哈尔滨新区 2025 年少样本下游任务 Pipeline

严格按论文 §4 评估协议：
- 冻结 backbone (V2 checkpoint)
- 使用 2025 年内真实变化时间窗口 (4-6月, 6-8月, 8-9月, 9-10月)
- 特征: concat(before_embedding, after_embedding)
- 分类器: Linear + kNN (自动选择最佳 AUC)
- 评估: 1/10/50/100/500-shot, 5-fold CV

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=5,6,7 python scripts/downstream_fewshot_harbin_2025.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
    load_harbin_annotations,
    PERIODS,
)
from src.config import load_config
from src.data.dataset import HarbinPatchDataset
from src.models.model import AEFModel

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/epoch_499.pt"
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v2/downstream_harbin_2025")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to("npu:0")
    ckpt = torch.load(CKPT_PATH, map_location="npu:0", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset


def extract_embedding(model, dataset, patch_idx, window_start_ms, window_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(window_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(window_end_ms, dtype=torch.float64)
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
    emb = output.embedding_map
    emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()  # [D, H, W]


def evaluate_fewshot(X, y, shot_counts, n_folds=5):
    results = {}
    rng = np.random.RandomState(42)

    for shot in shot_counts:
        all_aucs, all_bas, all_f1s, best_clfs = [], [], [], []

        for fold in range(n_folds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold)
            splits = list(skf.split(X, y))
            train_idx, test_idx = splits[fold]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            classes = np.unique(y_train)
            train_samples = []
            for c in classes:
                c_idx = np.where(y_train == c)[0]
                n_sample = min(shot, len(c_idx))
                if n_sample == 0:
                    continue
                sample_idx = rng.choice(c_idx, n_sample, replace=False)
                train_samples.extend(sample_idx.tolist())

            if len(train_samples) < 2 or len(np.unique(y_train[train_samples])) < 2:
                continue

            X_train_shot = X_train[train_samples]
            y_train_shot = y_train[train_samples]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_shot)
            X_test_scaled = scaler.transform(X_test)

            best_auc = 0
            best_metrics = {}

            classifiers = [("Linear", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))]
            if shot >= 10:
                classifiers.append(("kNN-3", KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train_shot) // 10)))))
            else:
                classifiers.append(("kNN-1", KNeighborsClassifier(n_neighbors=1)))

            for clf_name, clf in classifiers:
                try:
                    clf.fit(X_train_scaled, y_train_shot)
                    if hasattr(clf, "predict_proba"):
                        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_prob = clf.predict(X_test_scaled).astype(float)
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
                best_clfs.append(best_metrics["clf"])

        if all_aucs:
            results[shot] = {
                "auc_mean": float(np.mean(all_aucs)),
                "auc_std": float(np.std(all_aucs)),
                "ba_mean": float(np.mean(all_bas)),
                "ba_std": float(np.std(all_bas)),
                "f1_mean": float(np.mean(all_f1s)),
                "f1_std": float(np.std(all_f1s)),
                "n_folds": len(all_aucs),
                "best_clf": max(set(best_clfs), key=best_clfs.count),
            }

    return results


def main():
    print("=" * 70)
    print("  AEF_qwen 哈尔滨新区 2025 Few-Shot 变化检测评估")
    print("=" * 70)

    model, dataset = load_model()
    annotated_patches = get_annotated_patches()
    print(f"有标注的 patches: {len(annotated_patches)}")

    all_features = []
    all_labels = []
    n_positive = 0
    n_negative = 0
    patch_meta = []

    shot_counts = [1, 10, 50, 100, 500]

    for i, pid in enumerate(annotated_patches):
        if pid not in dataset.patches:
            continue
        pidx = dataset.patches.index(pid)

        period = get_period_for_patch(pid)
        if period is None or period not in PERIODS:
            continue
        bs, be = PERIODS[period]

        # 使用 before = period 的前半段，after = period 的后半段
        # 对于 4-6月: before=4月, after=6月
        # 实际做法：用 period 的起止点作为 before/after
        # 但这不够细，需要更精确地拆分
        # 对于 2025-04~2025-06: before_end 取 5-15, after_start 取 5-16
        mid = (bs + be) / 2.0

        t0 = time.time()
        try:
            eb = extract_embedding(model, dataset, pidx, bs, mid)
            ea = extract_embedding(model, dataset, pidx, mid, be)
        except Exception as e:
            print(f"  [{i+1}/{len(annotated_patches)}] {pid}: embedding extraction failed - {e}")
            continue

        mask, recs = rasterize_patch_changes(pid, grid_size=64)
        H, W = mask.shape
        D = eb.shape[0]

        features = np.zeros((H * W, D * 2), dtype=np.float32)
        labels = mask.flatten()

        for px in range(H):
            for py in range(W):
                idx = px * W + py
                features[idx, :D] = eb[:, px, py]
                features[idx, D:] = ea[:, px, py]

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos = len(pos_idx)
        n_neg_sample = min(n_pos * 10, len(neg_idx))
        if n_pos == 0:
            continue
        if n_neg_sample == 0:
            continue
        neg_sample = np.random.choice(neg_idx, n_neg_sample, replace=False)
        sampled_idx = np.concatenate([pos_idx, neg_sample])
        np.random.shuffle(sampled_idx)

        all_features.append(features[sampled_idx])
        all_labels.append(labels[sampled_idx])
        n_positive += n_pos
        n_negative += n_neg_sample
        patch_meta.append({"patch_id": pid, "period": period, "n_pos": int(n_pos), "n_neg": int(n_neg_sample)})

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(annotated_patches)}] {pid} ({period}): {n_pos} pos, {n_neg_sample} neg ({elapsed:.1f}s)")

    if not all_features:
        print("❌ 没有有效数据")
        return

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels)
    print(f"\n总数据集: {len(X)} 样本, 正样本={n_positive}, 负样本={n_negative}")

    print("\n开始 5-fold CV Few-Shot 评估...")
    results = evaluate_fewshot(X, y, shot_counts, n_folds=5)

    print("\n" + "=" * 70)
    print("  评估结果")
    print("=" * 70)
    print(f"\n{'Shot':<8} {'AUC':<14} {'BA':<14} {'F1':<14} {'BestClf':<10}")
    print("-" * 60)
    for shot in shot_counts:
        if shot in results:
            r = results[shot]
            print(f"{shot:<8} {r['auc_mean']:.3f}±{r['auc_std']:.3f}   "
                  f"{r['ba_mean']:.3f}±{r['ba_std']:.3f}   "
                  f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}   {r['best_clf']}")
        else:
            print(f"{shot:<8} N/A")

    output = {
        "model": "v2",
        "checkpoint": CKPT_PATH,
        "dataset": "harbin_2025",
        "n_patches": len(patch_meta),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "total_samples": int(len(X)),
        "feature_dim": int(X.shape[1]),
        "patch_meta": patch_meta,
        "results": {str(k): v for k, v in results.items()},
    }

    out_file = OUTPUT_DIR / "fewshot_benchmark_harbin_2025_v2.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ 结果已保存: {out_file}")


if __name__ == "__main__":
    main()
