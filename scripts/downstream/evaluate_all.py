#!/usr/bin/env python3
"""
全面评估脚本 — 评估所有下游任务性能。

用法:
  python scripts/downstream/evaluate_all.py \
    --embedding-dir /path/to/embeddings \
    --heads-dir /path/to/trained/heads \
    --output-dir /path/to/eval/results \
    --device npu:0
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from src.downstream.heads import SegmentationHead, ClassificationHead, ChangeDetectionHeadSimple
from src.models.heads import ChangeDetectionHeadV2


# ============ 评估指标计算 ============

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """计算IoU"""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / (union + 1e-8))

def compute_metrics_binary(pred: np.ndarray, target: np.ndarray, scores: np.ndarray | None = None) -> dict:
    """计算二分类任务的完整指标"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # 基础指标
    tp = np.logical_and(pred_flat, target_flat).sum()
    fp = np.logical_and(pred_flat, ~target_flat).sum()
    fn = np.logical_and(~pred_flat, target_flat).sum()
    tn = np.logical_and(~pred_flat, ~target_flat).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    result = {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # AUC (如果有scores)
    if scores is not None and len(np.unique(target_flat)) > 1:
        try:
            auc = roc_auc_score(target_flat, scores.flatten())
            result["auc"] = float(auc)
        except Exception:
            pass

    return result


def compute_metrics_multiclass(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """计算多分类任务的指标"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Pixel accuracy
    acc = (pred_flat == target_flat).mean()

    # Per-class IoU
    ious = []
    f1s = []
    for c in range(num_classes):
        pred_c = (pred_flat == c)
        target_c = (target_flat == c)
        tp = np.logical_and(pred_c, target_c).sum()
        fp = np.logical_and(pred_c, ~target_c).sum()
        fn = np.logical_and(~pred_c, target_c).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        ious.append(iou)
        f1s.append(f1)

    return {
        "pixel_accuracy": float(acc),
        "mean_iou": float(np.mean(ious)),
        "mean_f1": float(np.mean(f1s)),
        "per_class_iou": [float(x) for x in ious],
        "per_class_f1": [float(x) for x in f1s],
    }


# ============ 任务评估 ============

def evaluate_change_detection(
    embedding_dir: Path,
    mask_dir: Path,
    head_ckpt: Path,
    period: str,
    before_month: str,
    after_month: str,
    device: torch.device,
    embedding_dim: int = 128,
) -> dict:
    """评估变化检测"""
    print(f"\n{'='*60}")
    print(f"评估变化检测: {period}")
    print(f"{'='*60}")

    # 加载head
    head = ChangeDetectionHeadV2(embedding_dim=embedding_dim, hidden_dim=64).to(device)
    ckpt = torch.load(head_ckpt, map_location=str(device))
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    # 加载数据
    mask_files = sorted((mask_dir / period).glob("*.npy"))
    all_metrics = []

    with torch.no_grad():
        for mask_file in tqdm(mask_files, desc=f"  {period}"):
            pid = mask_file.stem
            emb_b = np.load(embedding_dir / f"{pid}_{before_month}.npy")
            emb_a = np.load(embedding_dir / f"{pid}_{after_month}.npy")
            mask = np.load(mask_file)

            emb_b = torch.from_numpy(emb_b).unsqueeze(0).float().to(device)
            emb_a = torch.from_numpy(emb_a).unsqueeze(0).float().to(device)

            logits = head(emb_b, emb_a).squeeze().cpu().numpy()
            scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
            pred = (scores > 0.5).astype(np.uint8)

            metrics = compute_metrics_binary(pred, mask, scores)
            metrics["patch_id"] = pid
            all_metrics.append(metrics)

    # 汇总
    avg_metrics = {
        "iou": np.mean([m["iou"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "f1": np.mean([m["f1"] for m in all_metrics]),
    }
    if "auc" in all_metrics[0]:
        avg_metrics["auc"] = np.mean([m["auc"] for m in all_metrics if "auc" in m])

    print(f"  平均 IoU: {avg_metrics['iou']:.4f}")
    print(f"  平均 F1: {avg_metrics['f1']:.4f}")
    if "auc" in avg_metrics:
        print(f"  平均 AUC: {avg_metrics['auc']:.4f}")

    return {"period": period, "avg": avg_metrics, "per_patch": all_metrics}


def evaluate_segmentation(
    embedding_dir: Path,
    label_dir: Path,
    head_ckpt: Path,
    task_name: str,
    month: str,
    device: torch.device,
    embedding_dim: int = 128,
) -> dict:
    """评估分割任务（水体/建筑物）"""
    print(f"\n{'='*60}")
    print(f"评估分割任务: {task_name}")
    print(f"{'='*60}")

    head = SegmentationHead(embedding_dim=embedding_dim, hidden_dim=64).to(device)
    ckpt = torch.load(head_ckpt, map_location=str(device))
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    label_files = sorted((label_dir).glob("*.npy"))
    all_metrics = []

    with torch.no_grad():
        for label_file in tqdm(label_files, desc=f"  {task_name}"):
            pid = label_file.stem
            emb = np.load(embedding_dir / f"{pid}_{month}.npy")
            label = np.load(label_file)

            emb = torch.from_numpy(emb).unsqueeze(0).float().to(device)
            logits = head(emb).squeeze().cpu().numpy()
            scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
            pred = (scores > 0.5).astype(np.uint8)

            metrics = compute_metrics_binary(pred, label, scores)
            metrics["patch_id"] = pid
            all_metrics.append(metrics)

    avg_metrics = {
        "iou": np.mean([m["iou"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "f1": np.mean([m["f1"] for m in all_metrics]),
    }
    if "auc" in all_metrics[0]:
        avg_metrics["auc"] = np.mean([m["auc"] for m in all_metrics if "auc" in m])

    print(f"  平均 IoU: {avg_metrics['iou']:.4f}")
    print(f"  平均 F1: {avg_metrics['f1']:.4f}")

    return {"task": task_name, "avg": avg_metrics, "per_patch": all_metrics}


def evaluate_classification(
    embedding_dir: Path,
    label_dir: Path,
    head_ckpt: Path,
    task_name: str,
    month: str,
    num_classes: int,
    device: torch.device,
    embedding_dim: int = 128,
) -> dict:
    """评估分类任务（土地利用）"""
    print(f"\n{'='*60}")
    print(f"评估分类任务: {task_name}")
    print(f"{'='*60}")

    head = ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=64).to(device)
    ckpt = torch.load(head_ckpt, map_location=str(device))
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    label_files = sorted((label_dir).glob("*.npy"))
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for label_file in tqdm(label_files, desc=f"  {task_name}"):
            pid = label_file.stem
            emb = np.load(embedding_dir / f"{pid}_{month}.npy")
            label = np.load(label_file)

            emb = torch.from_numpy(emb).unsqueeze(0).float().to(device)
            logits = head(emb).cpu().numpy()
            pred = logits.argmax(axis=1)[0]

            all_preds.append(pred)
            all_targets.append(label)

    all_preds = np.stack(all_preds)
    all_targets = np.stack(all_targets)

    metrics = compute_metrics_multiclass(all_preds, all_targets, num_classes)

    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Mean F1: {metrics['mean_f1']:.4f}")

    return {"task": task_name, "metrics": metrics}


# ============ 主函数 ============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--heads-dir", required=True)
    parser.add_argument("--mask-dir", default="/workspace/xuannv/data/change_masks")
    parser.add_argument("--label-dir", default="/workspace/xuannv/data/labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    embedding_dir = Path(args.embedding_dir)
    heads_dir = Path(args.heads_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. 变化检测 (4个时间段)
    period_map = {
        "june": ("2024-04", "2024-06"),
        "aug": ("2024-06", "2024-08"),
        "september": ("2024-08", "2024-09"),
        "october": ("2024-09", "2024-10"),
    }

    results["change_detection"] = {}
    for period, (before, after) in period_map.items():
        head_ckpt = heads_dir / f"cd_{period}_best.pt"
        if head_ckpt.exists():
            results["change_detection"][period] = evaluate_change_detection(
                embedding_dir, Path(args.mask_dir), head_ckpt, period, before, after,
                device, args.embedding_dim
            )

    # 2. 水体检测
    head_ckpt = heads_dir / "water_detection_best.pt"
    if head_ckpt.exists():
        results["water_detection"] = evaluate_segmentation(
            embedding_dir, Path(args.label_dir) / "water", head_ckpt,
            "water_detection", "2024-06", device, args.embedding_dim
        )

    # 3. 建筑物分割
    head_ckpt = heads_dir / "building_segmentation_best.pt"
    if head_ckpt.exists():
        results["building_segmentation"] = evaluate_segmentation(
            embedding_dir, Path(args.label_dir) / "building", head_ckpt,
            "building_segmentation", "2024-06", device, args.embedding_dim
        )

    # 4. 土地利用分割
    head_ckpt = heads_dir / "landuse_segmentation_best.pt"
    if head_ckpt.exists():
        results["landuse_segmentation"] = evaluate_classification(
            embedding_dir, Path(args.label_dir) / "landuse", head_ckpt,
            "landuse_segmentation", "2024-06", 7, device, args.embedding_dim
        )

    # 保存结果
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"评估完成！结果保存到: {output_dir / 'evaluation_results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
