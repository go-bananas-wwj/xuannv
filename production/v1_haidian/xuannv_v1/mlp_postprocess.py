#!/usr/bin/env python3
"""MLP 预测结果后处理：阈值优化、形态学去噪、小连通域过滤.

用于把 MLP 的“概率图”转换成更干净的汇报级预测 mask。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects


def _compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def find_best_threshold(
    prob_maps: np.ndarray,
    label_maps: np.ndarray,
    thresholds: list[float] | None = None,
    metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    """在验证/测试集上搜索最优阈值（按指定指标）。

    Args:
        prob_maps: [N, H, W]
        label_maps: [N, H, W]
        thresholds: 候选阈值列表，默认 0.1~0.9
        metric: "f1" | "iou" | "precision"

    Returns:
        (best_thr, best_metrics)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    best_thr = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    for thr in thresholds:
        preds = (prob_maps > thr).astype(np.uint8)
        scores = []
        metrics_sum: dict[str, float] = {"precision": 0, "recall": 0, "f1": 0, "iou": 0}
        for p, t in zip(preds, label_maps):
            m = _compute_metrics(p, t)
            scores.append(m[metric])
            for k in metrics_sum:
                metrics_sum[k] += m[k]
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_thr = thr
            n = len(preds)
            best_metrics = {k: v / n for k, v in metrics_sum.items()}
            best_metrics["threshold"] = best_thr
            best_metrics["target_metric"] = metric
            best_metrics["target_score"] = mean_score
    return best_thr, best_metrics


def postprocess_prob(
    prob: np.ndarray,
    threshold: float,
    min_area: int = 8,
    opening_radius: int = 1,
    closing_radius: int = 1,
) -> np.ndarray:
    """对单张概率图做后处理，返回二值 mask.

    步骤：
        1. 阈值化
        2. 开运算去噪点
        3. 闭运算连接断裂
        4. 删除小连通域
    """
    pred = (prob >= threshold).astype(bool)

    if opening_radius > 0:
        pred = ndimage.binary_opening(pred, iterations=opening_radius)
    if closing_radius > 0:
        pred = ndimage.binary_closing(pred, iterations=closing_radius)

    if min_area > 0:
        pred = remove_small_objects(pred, min_size=min_area)

    return pred.astype(np.uint8)


def postprocess_batch(
    prob_maps: np.ndarray,
    threshold: float,
    min_area: int = 8,
    opening_radius: int = 1,
    closing_radius: int = 1,
) -> np.ndarray:
    """对批量概率图做后处理。"""
    out = np.zeros_like(prob_maps, dtype=np.uint8)
    for i, prob in enumerate(prob_maps):
        out[i] = postprocess_prob(
            prob, threshold, min_area, opening_radius, closing_radius
        )
    return out
