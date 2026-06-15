#!/usr/bin/env python3
"""快速试验多种下游头 + 阈值调优，使用已缓存的 embedding.

不修改 haidian_tasks.py，独立脚本便于快速迭代。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from xuannv_v1 import haidian_tasks
from xuannv_v1.haidian_heads import PixelMLPHeadV2


def _best_threshold_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    """在 prob 上扫描阈值，选择训练集 F1 最大的阈值."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, prob)
    # thr 长度比 prec/rec 少 1
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    best_idx = int(np.argmax(f1s))
    return float(thr[best_idx])


def _metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob > thr).astype(np.uint8)
    return {
        "threshold": float(thr),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "iou": float(jaccard_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "pos_ratio": float(y_true.mean()),
    }


def _run_hgbc(X_train, y_train, X_test, y_test):
    print("[exp] HistGradientBoostingClassifier ...")
    clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]
    thr = _best_threshold_f1(y_train, prob_train)
    return _metrics(y_test, prob_test, thr)


def _run_lr(X_train, y_train, X_test, y_test):
    print("[exp] LogisticRegression ...")
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]
    thr = _best_threshold_f1(y_train, prob_train)
    return _metrics(y_test, prob_test, thr)


def _run_mlp_sklearn(X_train, y_train, X_test, y_test):
    print("[exp] MLPClassifier (sklearn) ...")
    pos = y_train.sum()
    neg = len(y_train) - pos
    sample_weight = np.where(y_train == 1, neg / max(pos, 1), pos / max(neg, 1))
    sample_weight = sample_weight / sample_weight.mean()
    sample_weight = np.clip(sample_weight, a_min=None, a_max=1000.0)
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]
    thr = _best_threshold_f1(y_train, prob_train)
    return _metrics(y_test, prob_test, thr)


def _run_mlp_torch_v2(X_train, y_train, X_test, y_test, device: str):
    print("[exp] PixelMLPHeadV2 + threshold tuning ...")
    clf = PixelMLPHeadV2(
        input_dim=int(X_train.shape[1]),
        hidden=(64, 64),
        dropout=0.3,
        device=device,
    )
    clf.fit(X_train, y_train)
    prob_train = clf.predict_proba(X_train)
    prob_test = clf.predict_proba(X_test)
    thr = _best_threshold_f1(y_train, prob_train)
    return _metrics(y_test, prob_test, thr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/head_ablation/.cache/embeddings.npz")
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    data = np.load(cache_path, allow_pickle=False)
    pids = [str(p) for p in data["patch_ids"]]
    emb_dec_arr = data["emb_dec"]
    emb_apr_arr = data["emb_apr"]
    emb_dec = {pid: emb_dec_arr[i] for i, pid in enumerate(pids)}
    emb_apr = {pid: emb_apr_arr[i] for i, pid in enumerate(pids)}

    label_dir = Path(args.label_dir)
    task_name = "shigongjiandu"
    rng = np.random.RandomState(args.seed)
    train_pids, test_pids, reason = haidian_tasks._stratified_split(
        pids, label_dir, task_name, rng
    )
    if reason:
        print("[error] split failed:", reason)
        return 1

    X_train, y_train, X_test, y_test = [], [], [], []
    for pid in pids:
        if pid not in emb_apr or pid not in emb_dec:
            continue
        emb = np.concatenate([emb_apr[pid], emb_dec[pid]], axis=0)  # bitemporal
        D, H, W = emb.shape
        json_path = label_dir / f"{pid}_20260430_rgb_uint8.json"
        masks = haidian_tasks.load_label_json(json_path, image_size=(427, 427))
        mask = haidian_tasks._merged_mask(masks, task_name)
        label = haidian_tasks.resize_mask(mask, H).flatten()
        X = emb.reshape(D, -1).T.astype(np.float32)
        if pid in train_pids:
            X_train.append(X)
            y_train.append(label)
        else:
            X_test.append(X)
            y_test.append(label)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    print(f"[exp] train pixels: {len(y_train)} pos={y_train.sum()} test pixels: {len(y_test)} pos={y_test.sum()}")

    results = {}
    results["hgbc"] = _run_hgbc(X_train, y_train, X_test, y_test)
    results["lr_tuned"] = _run_lr(X_train, y_train, X_test, y_test)
    results["mlp_sklearn_tuned"] = _run_mlp_sklearn(X_train, y_train, X_test, y_test)
    if args.device != "cpu":
        results["mlp_torch_v2_tuned"] = _run_mlp_torch_v2(X_train, y_train, X_test, y_test, args.device)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
