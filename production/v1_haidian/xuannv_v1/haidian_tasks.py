from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from . import backbone
from .haidian_heads import PixelMLPHead, PixelMLPHeadV2, UNetHead

LABEL_NORMALIZE = {
    "gongdi": "gongdi",
    "jiazhudongdi": "jianzhudongdi",
    "jianzhudongdi": "jianzhudongdi",
    "weijian": "weijian",
    "nongyongdi": "nongyongdi",
    "chachu": "chaichu",
    "chaichu": "chaichu",
    "daolubianhuo": "daolubianhua",
    "daolubianhua": "daolubianhua",
}

CLASS_NAMES = [
    "gongdi",
    "jianzhudongdi",
    "weijian",
    "nongyongdi",
    "chaichu",
    "daolubianhua",
]

CLASS_NAMES_CN = {
    "gongdi": "施工工地",
    "jianzhudongdi": "建筑用地",
    "weijian": "疑似违建",
    "nongyongdi": "农用地变化",
    "chaichu": "建筑消失",
    "daolubianhua": "施工道路",
}

# 合并任务：将多个细分类别合并成一个下游任务进行训练/评估
MERGED_TASKS = {
    "shigongjiandu": ["gongdi", "jianzhudongdi", "weijian", "daolubianhua"],
}


def _merged_mask(masks: dict[str, np.ndarray], task_name: str) -> np.ndarray:
    """获取某个任务（含合并任务）对应的二值 mask."""
    sources = MERGED_TASKS.get(task_name, [task_name])
    # 用第一个存在的 mask 初始化尺寸
    first = next((masks[s] for s in sources if s in masks), None)
    if first is None:
        raise ValueError(f"任务 {task_name} 找不到任何源 mask")
    merged = np.zeros_like(first)
    for src in sources:
        if src in masks:
            merged |= masks[src]
    return merged


def load_label_json(
    json_path: Path, image_size: tuple[int, int] = (427, 427)
) -> dict[str, np.ndarray]:
    with open(json_path) as f:
        data = json.load(f)

    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if h is None or w is None:
        h, w = image_size
    h = int(h)
    w = int(w)

    masks: dict[str, np.ndarray] = {
        name: np.zeros((h, w), dtype=np.uint8) for name in CLASS_NAMES
    }

    for shape in data.get("shapes", []):
        raw_label = shape.get("label", "").strip().lower()
        norm_label = LABEL_NORMALIZE.get(raw_label)
        if norm_label is None or norm_label not in masks:
            continue
        pts = [(int(p[0]), int(p[1])) for p in shape["points"]]
        if len(pts) < 3:
            continue
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
        masks[norm_label] |= np.array(img, dtype=np.uint8)

    return masks


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def discover_labeled_patches(label_dir: Path) -> list[str]:
    pids: set[str] = set()
    for f in label_dir.glob("*.json"):
        m = re.search(r"(patch_\d+)", f.name)
        if m:
            pids.add(m.group(1))
    return sorted(pids)


def _extract_embeddings(
    model: Any,
    dataset: Any,
    patch_ids: list[str],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    emb_dec = backbone.extract_embeddings_for_patches(
        model, dataset, patch_ids, 2025, 12, device
    )
    emb_apr = backbone.extract_embeddings_for_patches(
        model, dataset, patch_ids, 2026, 4, device
    )
    return emb_dec, emb_apr


def _stratified_split(
    patch_ids: list[str],
    label_dir: Path,
    task_name: str,
    rng: np.random.RandomState,
    train_ratio: float = 0.8,
) -> tuple[set[str] | None, list[str] | None, str | None]:
    """按每个 patch 是否包含任务正例像素进行分层 80/20 划分（支持合并任务）。"""
    pos_pids: list[str] = []
    neg_pids: list[str] = []
    for pid in patch_ids:
        masks = load_label_json(label_dir / f"{pid}_20260430_rgb_uint8.json")
        task_mask = _merged_mask(masks, task_name)
        if task_mask.any():
            pos_pids.append(pid)
        else:
            neg_pids.append(pid)

    n_pos = len(pos_pids)
    n_neg = len(neg_pids)
    if n_pos == 0:
        return None, None, "无正例 patch"

    rng.shuffle(pos_pids)
    rng.shuffle(neg_pids)

    if n_pos == 1:
        train_pos = pos_pids
        test_pos: list[str] = []
    else:
        n_train_pos = max(1, int(n_pos * train_ratio))
        n_train_pos = min(n_train_pos, n_pos - 1)
        train_pos = pos_pids[:n_train_pos]
        test_pos = pos_pids[n_train_pos:]

    n_train_neg = max(0, int(n_neg * train_ratio))
    train_neg = neg_pids[:n_train_neg]
    test_neg = neg_pids[n_train_neg:]

    if not test_pos and not test_neg:
        return None, None, "无法划分训练集和测试集"

    train_pids = set(train_pos + train_neg)
    test_pids = test_pos + test_neg
    return train_pids, test_pids, None


def run_task(
    task_name: str,
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    mode: str = "bitemporal",
    classifier: str = "linear",
    seed: int = 42,
    patch_ids: list[str] | None = None,
    model: Any = None,
    dataset: Any = None,
    emb_dec: dict[str, np.ndarray] | None = None,
    emb_apr: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if task_name not in CLASS_NAMES and task_name not in MERGED_TASKS:
        raise ValueError(
            f"未知任务: {task_name}，可选: {CLASS_NAMES} 或 {list(MERGED_TASKS.keys())}"
        )
    if mode not in ("single", "bitemporal"):
        raise ValueError(f"未知 mode: {mode}")
    if classifier not in ("linear", "mlp", "mlp_torch", "mlp_torch_v2", "unet"):
        raise ValueError(f"未知 classifier: {classifier}")

    out_dir = Path(output_dir) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if model is None or dataset is None:
        model, dataset, _ = backbone.load_production_model(model_dir, device=device)
    label_dir = Path(label_dir)

    candidate_pids = patch_ids if patch_ids else discover_labeled_patches(label_dir)
    valid_pids = [
        p
        for p in candidate_pids
        if (label_dir / f"{p}_20260430_rgb_uint8.json").exists()
    ]

    if len(valid_pids) < 2:
        result = {"skipped": True, "reason": "带标注的 patch 不足 2 个"}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    rng = np.random.RandomState(seed)
    train_pids, test_pids, split_reason = _stratified_split(
        valid_pids, label_dir, task_name, rng
    )
    if split_reason:
        result = {"skipped": True, "reason": split_reason}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    if emb_dec is None or emb_apr is None:
        emb_dec, emb_apr = _extract_embeddings(model, dataset, valid_pids, device)

    # 空间特征/标签，用于支持 U-Net 等空间头
    train_features_spatial: list[np.ndarray] = []
    train_labels_spatial: list[np.ndarray] = []
    test_features_spatial: list[np.ndarray] = []
    test_labels_spatial: list[np.ndarray] = []
    test_patch_ids: list[str] = []
    test_shapes: set[tuple[int, int]] = set()

    for pid in valid_pids:
        if pid not in emb_apr:
            continue
        emb = emb_apr[pid]
        if mode == "bitemporal":
            if pid not in emb_dec:
                continue
            emb = np.concatenate([emb, emb_dec[pid]], axis=0)

        D, H, W = emb.shape
        json_path = label_dir / f"{pid}_20260430_rgb_uint8.json"
        masks = load_label_json(json_path, image_size=(427, 427))
        task_mask = _merged_mask(masks, task_name)
        label_mask = resize_mask(task_mask, H)

        if pid in train_pids:
            train_features_spatial.append(emb)
            train_labels_spatial.append(label_mask)
        else:
            test_features_spatial.append(emb)
            test_labels_spatial.append(label_mask)
            test_patch_ids.append(pid)
            test_shapes.add((H, W))

    if not train_features_spatial or not test_features_spatial:
        result = {"skipped": True, "reason": "训练集或测试集为空"}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    if len(test_shapes) != 1:
        raise ValueError(
            f"test patch embeddings have inconsistent spatial sizes: {test_shapes}"
        )
    H, W = next(iter(test_shapes))

    # 按 head 类型构造训练/测试数据
    if classifier in ("linear", "mlp", "mlp_torch", "mlp_torch_v2"):
        X_train = np.concatenate(
            [f.reshape(f.shape[0], -1).T for f in train_features_spatial], 0
        )
        y_train = np.concatenate([l.flatten() for l in train_labels_spatial], 0)
        X_test = np.concatenate(
            [f.reshape(f.shape[0], -1).T for f in test_features_spatial], 0
        )
        y_test = np.concatenate([l.flatten() for l in test_labels_spatial], 0)
    elif classifier == "unet":
        X_train = np.stack(train_features_spatial, axis=0).astype(np.float32)
        y_train = np.stack(train_labels_spatial, axis=0).astype(np.float32)
        X_test = np.stack(test_features_spatial, axis=0).astype(np.float32)
        y_test = np.concatenate([l.flatten() for l in test_labels_spatial], 0)
    else:
        raise ValueError(f"未知 classifier: {classifier}")

    pos_ratio = y_train.mean()
    if pos_ratio < 1e-6:
        result = {"skipped": True, "reason": "训练集无正例"}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    # 训练下游头
    if classifier == "linear":
        clf = LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=seed
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
    elif classifier == "mlp":
        sample_weight = np.where(
            y_train == 1, 1.0 / max(pos_ratio, 1e-6), 1.0 / (1 - pos_ratio)
        )
        sample_weight = sample_weight / sample_weight.mean()
        sample_weight = np.clip(sample_weight, a_min=None, a_max=1000.0)
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=200,
            random_state=seed,
            early_stopping=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train, sample_weight=sample_weight)
        prob = clf.predict_proba(X_test)[:, 1]
    elif classifier == "mlp_torch":
        clf = PixelMLPHead(
            input_dim=int(X_train.shape[1]),
            hidden=(128,),
            device=device,
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)
    elif classifier == "mlp_torch_v2":
        clf = PixelMLPHeadV2(
            input_dim=int(X_train.shape[1]),
            hidden=(64, 64),
            dropout=0.3,
            device=device,
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)
    elif classifier == "unet":
        clf = UNetHead(
            in_channels=int(X_train.shape[1]),
            base_channels=32,
            device=device,
        )
        clf.fit(X_train, y_train)
        prob_spatial = clf.predict_proba(X_test)  # [N, H, W]
        prob = prob_spatial.reshape(-1)

    y_pred = (prob > 0.5).astype(np.uint8)

    metrics: dict[str, Any] = {
        "task": task_name,
        "mode": mode,
        "classifier": classifier,
        "n_train_patches": len(train_pids),
        "n_test_patches": len(test_pids),
        "pos_ratio": float(pos_ratio),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "iou": float(jaccard_score(y_test, y_pred, zero_division=0)),
    }
    if prob is not None and len(np.unique(y_test)) == 2:
        metrics["auc"] = float(roc_auc_score(y_test, prob))
    elif len(np.unique(y_test)) < 2:
        metrics["auc"] = None
        metrics["auc_note"] = "test set has only one class"
    else:
        metrics["auc"] = 0.0

    if prob is not None and H is not None and W is not None:
        prob_map = np.zeros((len(test_patch_ids), H, W), dtype=np.float32)
        label_map = np.zeros((len(test_patch_ids), H, W), dtype=np.uint8)
        offset = 0
        for idx in range(len(test_patch_ids)):
            n_pix = H * W
            prob_map[idx] = prob[offset : offset + n_pix].reshape(H, W)
            label_map[idx] = y_test[offset : offset + n_pix].reshape(H, W)
            offset += n_pix
        np.savez_compressed(
            out_dir / "pred.npz",
            patch_ids=np.array(test_patch_ids),
            prob_map=prob_map,
            label_map=label_map,
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    return metrics


def run_all_tasks(
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    mode: str = "bitemporal",
    classifier: str = "linear",
    patch_ids: list[str] | None = None,
    model: Any = None,
    dataset: Any = None,
    emb_dec: dict[str, np.ndarray] | None = None,
    emb_apr: dict[str, np.ndarray] | None = None,
    tasks: list[str] | None = None,
) -> dict[str, dict]:
    if model is None or dataset is None:
        model, dataset, _ = backbone.load_production_model(model_dir, device=device)
    task_list = tasks if tasks is not None else CLASS_NAMES
    summary: dict[str, dict] = {}
    for task in task_list:
        task_cn = CLASS_NAMES_CN.get(task, task)
        print(f"\n[run_all_tasks] 开始任务: {task} ({task_cn})")
        summary[task] = run_task(
            task,
            model_dir,
            label_dir,
            output_dir,
            device,
            mode,
            classifier,
            model=model,
            dataset=dataset,
            patch_ids=patch_ids,
            emb_dec=emb_dec,
            emb_apr=emb_apr,
        )
    (Path(output_dir) / "metrics_all.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="海淀 6 任务推理")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    parser.add_argument("--output-dir", default="outputs/haidian")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", default="bitemporal", choices=["single", "bitemporal"])
    parser.add_argument(
        "--classifier",
        default="linear",
        choices=["linear", "mlp", "mlp_torch", "mlp_torch_v2", "unet"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_tasks(
        model_dir=args.model_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        device=args.device,
        mode=args.mode,
        classifier=args.classifier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
