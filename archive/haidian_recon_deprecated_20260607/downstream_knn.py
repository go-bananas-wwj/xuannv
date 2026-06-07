"""HRE 下游验证 — 像素级 kNN 分类 (WorldCover)."""
from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import rasterio
import torch
import torch_npu
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from haidian_recon.config import Config
from haidian_recon.data.dataset import HaidianReconDataset, collate_fn
from haidian_recon.models.hre_model import HREModel


def read_worldcover_label(patch_dir: str, target_size: int = 128) -> np.ndarray | None:
    """读取 WorldCover 标签并 resize 到 target_size."""
    wc_dir = os.path.join(patch_dir, "worldcover")
    if not os.path.isdir(wc_dir):
        return None
    tiffs = [f for f in os.listdir(wc_dir) if f.endswith(".tif")]
    if not tiffs:
        return None
    tiff_path = os.path.join(wc_dir, tiffs[0])
    try:
        with rasterio.open(tiff_path) as src:
            arr = src.read(1).astype(np.int64)
            # 简单裁剪到 target_size（假设 worldcover >= target_size）
            h, w = arr.shape
            if h > target_size:
                arr = arr[:target_size, :]
            if w > target_size:
                arr = arr[:, :target_size]
            return arr
    except Exception:
        return None


def compute_miou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """计算 mIoU、总体准确率、每类 IoU."""
    ious = []
    class_iou = {}
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            class_iou[c] = float(iou)
        else:
            class_iou[c] = None  # 该类别未出现

    miou = np.mean(ious) if ious else 0.0
    acc = (pred == target).sum() / pred.size
    return {
        "miou": float(miou),
        "accuracy": float(acc),
        "class_iou": class_iou,
        "n_classes_present": len(ious),
    }


def extract_embeddings_and_labels(
    checkpoint_path: str,
    data_root: str,
    stats_dir: str,
    planet_root: str,
    cache_dir: str,
    split: str = "val",
    num_patches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """提取 embedding 和 WorldCover 标签."""
    device = torch.device("npu:0")

    cfg = Config()
    source_channels = {s["name"]: s["channels"] for s in cfg.data.sources}

    model = HREModel(
        source_channels=source_channels,
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        use_gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()

    dataset = HaidianReconDataset(
        data_root=data_root,
        planet_root=planet_root,
        stats_dir=stats_dir,
        split=split,
        image_size=cfg.data.image_size,
        source_names=list(source_channels.keys()),
        cache_dir=cache_dir,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=False)

    embeddings = []
    labels = []
    patch_names = []

    with torch.no_grad():
        for batch in loader:
            # 检查是否有至少一个源的数据有效
            has_valid = False
            for key in batch:
                if key.endswith("_valid") and key != "aef_embedding_valid":
                    if batch[key][0].item():
                        has_valid = True
                        break
            if not has_valid:
                continue

            patch_name = batch["patch_id"][0]
            patch_dir = os.path.join(data_root, patch_name)
            wc_label = read_worldcover_label(patch_dir, target_size=cfg.data.image_size)
            if wc_label is None:
                continue

            # 统计众数作为 patch 级标签
            # 但这里我们保留像素级标签，后续每个像素独立预测
            # 实际上同一 patch 的所有像素共享同一个 embedding
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            output = model(batch_device, mask_info=None)
            emb = output["embedding"][0].cpu().numpy()  # [D]

            embeddings.append(emb)
            labels.append(wc_label)
            patch_names.append(patch_name)

            if num_patches and len(embeddings) >= num_patches:
                break

    return np.stack(embeddings), np.array(labels), patch_names


def patch_level_knn(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    """Patch 级 kNN：每个 patch 的 embedding 预测该 patch 的众数标签."""
    # 取每个 patch 的众数标签
    patch_labels = []
    for lbl in labels:
        unique, counts = np.unique(lbl, return_counts=True)
        patch_labels.append(unique[np.argmax(counts)])
    patch_labels = np.array(patch_labels)

    unique_classes = np.unique(patch_labels)
    print(f"Unique classes in patch labels: {unique_classes}")

    # 简单 80/20 分割
    n = len(embeddings)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    knn = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)), metric="cosine")
    knn.fit(embeddings[train_idx], patch_labels[train_idx])
    pred = knn.predict(embeddings[test_idx])

    # 计算指标
    num_classes = int(patch_labels.max()) + 1
    results = compute_miou(pred, patch_labels[test_idx], num_classes)
    results["k"] = k
    results["n_train"] = len(train_idx)
    results["n_test"] = len(test_idx)
    return results


def pixel_level_knn(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    """像素级 kNN：每个像素独立预测（同一 patch 内所有像素共享 embedding）."""
    # 构建像素级数据集
    pixel_embs = []
    pixel_labels = []
    for emb, lbl in zip(embeddings, labels):
        h, w = lbl.shape
        pixel_embs.append(np.repeat(emb.reshape(1, -1), h * w, axis=0))
        pixel_labels.append(lbl.flatten())

    pixel_embs = np.concatenate(pixel_embs, axis=0)
    pixel_labels = np.concatenate(pixel_labels, axis=0)

    # 采样加速（像素太多）
    n_pixels = len(pixel_embs)
    if n_pixels > 50000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_pixels, size=50000, replace=False)
        pixel_embs = pixel_embs[sample_idx]
        pixel_labels = pixel_labels[sample_idx]
        n_pixels = 50000

    # 80/20 分割
    indices = np.arange(n_pixels)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    split = int(0.8 * n_pixels)
    train_idx, test_idx = indices[:split], indices[split:]

    unique_classes = np.unique(pixel_labels)
    print(f"Unique classes in pixel labels: {unique_classes}")

    knn = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)), metric="cosine")
    knn.fit(pixel_embs[train_idx], pixel_labels[train_idx])
    pred = knn.predict(pixel_embs[test_idx])

    num_classes = int(pixel_labels.max()) + 1
    results = compute_miou(pred, pixel_labels[test_idx], num_classes)
    results["k"] = k
    results["n_train"] = len(train_idx)
    results["n_test"] = len(test_idx)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data_raw/haidian/scenes")
    parser.add_argument("--planet-root", type=str, default="data_raw/beijing/planetscene")
    parser.add_argument("--stats-dir", type=str, default="statistics/haidian")
    parser.add_argument("--cache-dir", type=str, default="haidian_recon/.cache")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num-patches", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/hre_eval/knn_results.json")
    args = parser.parse_args()

    print(f"Extracting embeddings from {args.checkpoint}...")
    embeddings, labels, patch_names = extract_embeddings_and_labels(
        args.checkpoint,
        args.data_root,
        args.stats_dir,
        args.planet_root,
        args.cache_dir,
        split=args.split,
        num_patches=args.num_patches,
    )
    print(f"Extracted {len(embeddings)} patches")

    print(f"\n=== Patch-level kNN (k={args.k}) ===")
    patch_results = patch_level_knn(embeddings, labels, k=args.k)
    print(f"Patch-level mIoU: {patch_results['miou']:.4f}, Acc: {patch_results['accuracy']:.4f}")

    print(f"\n=== Pixel-level kNN (k={args.k}) ===")
    pixel_results = pixel_level_knn(embeddings, labels, k=args.k)
    print(f"Pixel-level mIoU: {pixel_results['miou']:.4f}, Acc: {pixel_results['accuracy']:.4f}")

    results = {
        "checkpoint": args.checkpoint,
        "k": args.k,
        "n_patches": len(embeddings),
        "patch_level": patch_results,
        "pixel_level": pixel_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
