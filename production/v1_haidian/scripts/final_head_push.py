#!/usr/bin/env python3
"""在 phase1 best 微调 embedding 上训练一个更强的下游 MLP head，目标 F1>=0.6."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from sklearn.metrics import f1_score, jaccard_score, roc_auc_score, precision_recall_curve
from xuannv_v1 import haidian_tasks
from xuannv_v1.haidian_heads import PixelMLPHeadV3


def main() -> int:
    cache_path = Path("outputs/eval_e20best_tuned_v2/.cache/embeddings_best.npz")
    label_dir = Path("/workspace/xuannv/haidian_label/labeljson")
    task_name = "shigongjiandu"
    device = "npu:1"

    data = np.load(cache_path, allow_pickle=False)
    pids = [str(p) for p in data["patch_ids"]]
    emb_dec = {pid: data["emb_dec"][i] for i, pid in enumerate(pids)}
    emb_apr = {pid: data["emb_apr"][i] for i, pid in enumerate(pids)}

    rng = np.random.RandomState(42)
    train_pids, test_pids, reason = haidian_tasks._stratified_split(
        pids, label_dir, task_name, rng
    )
    if reason:
        print("split failed", reason)
        return 1

    target_size = 128
    train_before, train_after, train_labels, test_before, test_after, test_labels = [], [], [], [], [], []
    for pid in pids:
        if pid not in emb_apr or pid not in emb_dec:
            continue
        json_path = label_dir / f"{pid}_20260430_rgb_uint8.json"
        masks = haidian_tasks.load_label_json(json_path, image_size=(427, 427))
        mask = haidian_tasks._merged_mask(masks, task_name)
        label = haidian_tasks.resize_mask(mask, target_size)
        before = emb_dec[pid]
        after = emb_apr[pid]
        if pid in train_pids:
            train_before.append(before)
            train_after.append(after)
            train_labels.append(label)
        else:
            test_before.append(before)
            test_after.append(after)
            test_labels.append(label)

    X_train, y_train = haidian_tasks._build_upsampled_pixel_features(
        train_before, train_after, train_labels, target_size
    )
    X_test, y_test = haidian_tasks._build_upsampled_pixel_features(
        test_before, test_after, test_labels, target_size
    )

    print(f"train: {len(y_train)} pos={y_train.sum()} test: {len(y_test)} pos={y_test.sum()}")

    clf = PixelMLPHeadV3(
        input_dim=int(X_train.shape[1]),
        hidden=(512, 256),
        dropout=0.3,
        lr=1e-3,
        epochs=200,
        batch_size=8192,
        device=device,
        patience=30,
        pos_sample_weight=50.0,
        num_train_samples=200000,
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)
    print("threshold", clf.threshold)
    print("AUC", roc_auc_score(y_test, prob))
    print("F1@th", f1_score(y_test, prob > clf.threshold), "IoU@th", jaccard_score(y_test, prob > clf.threshold))
    print("F1@0.5", f1_score(y_test, prob > 0.5), "IoU@0.5", jaccard_score(y_test, prob > 0.5))
    # test-set oracle threshold for reference
    prec, rec, thr = precision_recall_curve(y_test, prob)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_idx = int(np.argmax(f1s))
    print("oracle thr", thr[best_idx], "F1", f1s[best_idx])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
