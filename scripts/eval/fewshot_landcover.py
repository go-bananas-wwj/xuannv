"""Round7 Few-Shot 土地利用分类评估.

冻结 backbone，用 K 个 patch 的 WorldCover 标签训练 PixelConvHead，测试其余 patch.

用法:
    python scripts/eval/fewshot_landcover.py \
        --experiment 2 \
        --k-shots 5,20,50,100 \
        --pixels-per-patch 500 \
        --device npu:0
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score
import warnings
warnings.filterwarnings('ignore')


def find_experiment_files(exp_id: int):
    config_candidates = glob.glob(f"/workspace/xuannv/configs/round7_8gpu/exp{exp_id}_*.yaml")
    if not config_candidates:
        raise ValueError(f"Config not found for exp{exp_id}")
    config_path = config_candidates[0]

    dirs = glob.glob(f"/workspace/xuannv/outputs/round7_exp{exp_id}_*")
    if not dirs:
        raise ValueError(f"Output dir not found for exp{exp_id}")
    exp_dir = dirs[0]

    checkpoints = glob.glob(f"{exp_dir}/epoch_*.pt") + glob.glob(f"{exp_dir}/epoch_best_*.pt")
    if not checkpoints:
        raise ValueError(f"Checkpoint not found for exp{exp_id}")
    checkpoint_path = max(checkpoints, key=lambda p: Path(p).stat().st_mtime)

    return config_path, checkpoint_path


def load_backbone(config_path: str, checkpoint_path: str, device: str):
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


def extract_embedding_and_label(model, dataset, cfg, pidx, device):
    """提取单个 patch 的 embedding map 和 WorldCover 标签."""
    batch = dataset[pidx]
    patch_id = batch["patch_id"]

    # Embedding
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

    # WorldCover label
    from src.data.transforms import WORLDCOVER_CLASSES
    label = dataset._get_worldcover_label(patch_id)  # [H, W] uint8
    if label is None:
        return None

    # 下采样到 embedding 尺寸 (假设 H=64, W=64)
    label_tensor = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    emb_h, emb_w = emb_map.shape[1], emb_map.shape[2]
    label_resized = F.interpolate(label_tensor, size=(emb_h, emb_w), mode='nearest')[0, 0].long()  # [H,W]

    return emb_map, label_resized, patch_id


class PixelConvHead(nn.Module):
    """轻量 PixelConvHead for land cover."""
    def __init__(self, in_channels=128, num_classes=11, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.out = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out(x)


def sample_pixels(emb_map, label_map, n_pixels=500):
    """从 patch 中随机采样像素作为训练样本."""
    D, H, W = emb_map.shape
    emb_flat = emb_map.reshape(D, -1).T  # [H*W, D]
    label_flat = label_map.reshape(-1)    # [H*W]

    # 过滤无效标签 (0 = no-data 或背景)
    valid_mask = label_flat > 0
    if valid_mask.sum() == 0:
        return None, None

    valid_emb = emb_flat[valid_mask]
    valid_label = label_flat[valid_mask]

    n = min(n_pixels, len(valid_label))
    indices = torch.randperm(len(valid_label))[:n]
    return valid_emb[indices], valid_label[indices]


def train_landcover_head(head, train_emb, train_label, device, epochs=30, lr=1e-3, num_classes=11):
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    # 计算类别权重
    class_counts = torch.bincount(train_label, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    for epoch in range(epochs):
        head.train()
        emb = train_emb.to(device).unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # [1, D, N, 1]
        label = train_label.to(device).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

        # 需要把 emb reshape 成 [1, D, H, W] 形式，这里用 1xN
        # 简化：直接训练一个线性分类器（而非 conv）
        # 因为像素已经展平了，conv 不适用
        pass

    # 实际上，对于采样像素，用 MLP 更合适
    return head


def evaluate_landcover_head(head, test_data, device, num_classes=11):
    """评估 head 在测试 patch 上的性能."""
    head.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for emb_map, label_map in test_data:
            emb = emb_map.unsqueeze(0).to(device)  # [1, D, H, W]
            logits = head(emb)  # [1, C, H, W]
            pred = logits.argmax(dim=1).cpu()[0]  # [H, W]

            all_preds.extend(pred.reshape(-1).tolist())
            all_labels.extend(label_map.reshape(-1).tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 过滤无效标签
    valid_mask = all_labels > 0
    preds_valid = all_preds[valid_mask]
    labels_valid = all_labels[valid_mask]

    if len(labels_valid) == 0:
        return {"error": "no valid labels"}

    # 只保留实际出现的类别
    present_classes = np.unique(labels_valid)
    preds_mapped = preds_valid
    labels_mapped = labels_valid

    bacc = balanced_accuracy_score(labels_mapped, preds_mapped)
    f1_macro = f1_score(labels_mapped, preds_mapped, average='macro', labels=present_classes, zero_division=0)
    f1_weighted = f1_score(labels_mapped, preds_mapped, average='weighted', labels=present_classes, zero_division=0)

    # Per-class IoU
    ious = []
    for c in present_classes:
        inter = ((preds_mapped == c) & (labels_mapped == c)).sum()
        union = ((preds_mapped == c) | (labels_mapped == c)).sum()
        iou = inter / max(union, 1)
        ious.append(iou)
    miou = np.mean(ious)

    return {
        "balanced_accuracy": float(bacc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "miou": float(miou),
        "n_pixels": int(len(labels_valid)),
        "n_classes": int(len(present_classes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--k-shots", type=str, default="5,20,50,100")
    parser.add_argument("--pixels-per-patch", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_shots = [int(x.strip()) for x in args.k_shots.split(",")]

    print("=" * 70)
    print(f" Few-Shot 土地利用分类 — exp{args.experiment}")
    print("=" * 70)

    # 查找文件
    print("\n[1/3] 查找实验文件...")
    config_path, checkpoint_path = find_experiment_files(args.experiment)
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")

    # 加载 backbone
    print("\n[2/3] 加载 backbone 并提取数据...")
    model, dataset, cfg = load_backbone(config_path, checkpoint_path, args.device)
    print(f"  Dataset: {len(dataset)} patches")

    # 提取所有 patch 的 embedding + label
    all_data = []
    for pidx in range(len(dataset.patches)):
        result = extract_embedding_and_label(model, dataset, cfg, pidx, args.device)
        if result is not None:
            emb_map, label_map, pid = result
            all_data.append((emb_map, label_map, pid))

    print(f"  有效 patch: {len(all_data)}")
    if len(all_data) < max(k_shots) + 10:
        print("  有效 patch 太少")
        return

    # 运行 few-shot 实验
    print(f"\n[3/3] Few-Shot 实验...")
    k_results = []
    for k in k_shots:
        print(f"\n--- K={k} ---")
        split_results = []
        for split in range(args.n_splits):
            indices = list(range(len(all_data)))
            np.random.shuffle(indices)
            train_idx = indices[:k]
            test_idx = indices[k:]

            # 采样训练像素
            train_emb_list = []
            train_label_list = []
            for i in train_idx:
                emb_map, label_map, _ = all_data[i]
                emb_samples, label_samples = sample_pixels(emb_map, label_map, args.pixels_per_patch)
                if emb_samples is not None:
                    train_emb_list.append(emb_samples)
                    train_label_list.append(label_samples)

            if len(train_emb_list) == 0:
                continue

            train_emb = torch.cat(train_emb_list, dim=0)  # [N, D]
            train_label = torch.cat(train_label_list, dim=0)  # [N]

            # 训练线性分类器（对采样像素，conv head 不适用）
            # 用简单的 Linear + BN + ReLU
            from src.models.downstream_heads import PixelMLPHead
            head = PixelMLPHead(cfg.model.embedding_dim, num_classes=11, hidden_dim=128)
            head = head.to(args.device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

            # 将像素 reshape 为 [1, D, N, 1] 以适配 conv head
            # 或者直接用线性层... 这里简化用 MLP
            # PixelMLPHead 期望 [B, D, H, W]，但我们的数据是 [N, D] 采样像素
            # 重新设计：直接用 sklearn LogisticRegression（线性 probe）
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500, class_weight='balanced', multi_class='multinomial')
            clf.fit(train_emb.numpy(), train_label.numpy())

            # 评估
            test_data = [all_data[i][:2] for i in test_idx]
            all_preds = []
            all_labels = []
            for emb_map, label_map in test_data:
                D, H, W = emb_map.shape
                emb_flat = emb_map.reshape(D, -1).T.numpy()  # [H*W, D]
                label_flat = label_map.reshape(-1).numpy()
                preds = clf.predict(emb_flat)
                all_preds.extend(preds.tolist())
                all_labels.extend(label_flat.tolist())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            valid_mask = all_labels > 0
            preds_valid = all_preds[valid_mask]
            labels_valid = all_labels[valid_mask]

            if len(labels_valid) == 0:
                continue

            present_classes = np.unique(labels_valid)
            bacc = balanced_accuracy_score(labels_valid, preds_valid)
            f1_macro = f1_score(labels_valid, preds_valid, average='macro', labels=present_classes, zero_division=0)
            f1_weighted = f1_score(labels_valid, preds_valid, average='weighted', labels=present_classes, zero_division=0)

            ious = []
            for c in present_classes:
                inter = ((preds_valid == c) & (labels_valid == c)).sum()
                union = ((preds_valid == c) | (labels_valid == c)).sum()
                iou = inter / max(union, 1)
                ious.append(iou)
            miou = np.mean(ious)

            split_results.append({
                "balanced_accuracy": float(bacc),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
                "miou": float(miou),
                "n_train_pixels": int(len(train_label)),
                "n_test_pixels": int(len(labels_valid)),
            })

        if split_results:
            k_results.append({
                "k_shot": k,
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in split_results])),
                "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in split_results])),
                "f1_macro_mean": float(np.mean([r["f1_macro"] for r in split_results])),
                "f1_macro_std": float(np.std([r["f1_macro"] for r in split_results])),
                "miou_mean": float(np.mean([r["miou"] for r in split_results])),
                "miou_std": float(np.std([r["miou"] for r in split_results])),
                "splits": split_results,
            })
            print(f"  BAcc={k_results[-1]['balanced_accuracy_mean']:.4f}±{k_results[-1]['balanced_accuracy_std']:.4f} "
                  f"F1={k_results[-1]['f1_macro_mean']:.4f}±{k_results[-1]['f1_macro_std']:.4f} "
                  f"mIoU={k_results[-1]['miou_mean']:.4f}±{k_results[-1]['miou_std']:.4f}")

    # 保存
    output_data = {
        "experiment": args.experiment,
        "checkpoint": checkpoint_path,
        "config": config_path,
        "k_shots": k_shots,
        "pixels_per_patch": args.pixels_per_patch,
        "n_splits": args.n_splits,
        "results": k_results,
    }

    if args.output is None:
        args.output = f"/workspace/xuannv/outputs/round7_downstream_eval/exp{args.experiment}_landcover_fewshot.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n📄 结果保存: {args.output}")


if __name__ == "__main__":
    main()
