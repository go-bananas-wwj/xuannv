"""基于预计算 Embedding 的 Few-Shot 下游分类评估.

从已提取的 patch_embeddings.npz 加载 embedding，用少量像素训练分类器。
支持 WorldCover / JRC Water / Dynamic World。

用法:
    python fewshot_downstream_from_embedding.py \
        --embedding-file /path/to/patch_embeddings.npz \
        --k-pixels 100,1000,10000,100000 \
        --n-splits 3 \
        --device npu:0
"""
from __future__ import annotations

import sys, os, json, argparse, time, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu  # 必须导入以注册 npu 设备
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from pathlib import Path
from sklearn.metrics import accuracy_score

DATA_ROOT = Path("/workspace/xuannv/data_raw/harbin/scenes")
EVAL_DIR = Path("/workspace/xuannv/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation")

TASKS = [
    ("worldcover", "worldcover", "static.tif", 10),
    ("jrc_water", "jrc_water", "static.tif", 2),
    ("dynamic_world", "dynamic_world", "2025Q2.tif", 9),
]

LABEL_MAPPINGS = {
    "worldcover": {10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 6},
}


class TinyMLP(nn.Module):
    """轻量 MLP 分类器."""
    def __init__(self, in_dim, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_label(patch_id, label_dir, fname):
    path = DATA_ROOT / label_dir / patch_id / fname
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        label = src.read(1)
        nodata = src.nodata
    return label, nodata


def resize_label(label, target_h, target_w):
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(label_t, size=(target_h, target_w), mode="nearest")
    return resized.squeeze().numpy().astype(label.dtype)


def prepare_fewshot_data(spatial_maps, patch_ids, label_dir, label_file, task_name, num_classes, k_pixels):
    """准备 Few-Shot 数据：从所有 patch 中采样 K 个像素."""
    N, _, D, H, W = spatial_maps.shape
    month_idx = 5  # June

    all_embs = []
    all_labels = []

    for i, pid in enumerate(patch_ids):
        pid_str = str(pid)
        label, nodata = load_label(pid_str, label_dir, label_file)
        if label is None:
            continue

        emb = spatial_maps[i, month_idx]  # [D, H, W]

        if label.shape != (H, W):
            label = resize_label(label, H, W)

        # 标签映射
        if task_name in LABEL_MAPPINGS:
            mapping = LABEL_MAPPINGS[task_name]
            label = np.vectorize(lambda x: mapping.get(x, x))(label)

        # flatten
        emb_flat = emb.reshape(D, -1).T  # [H*W, D]
        label_flat = label.flatten()

        # 过滤无效像素
        if nodata is not None:
            mask = label_flat != nodata
        else:
            mask = label_flat >= 0
        
        # 通用过滤：确保 label 在有效范围内
        mask = mask & (label_flat >= 0) & (label_flat < num_classes)

        if task_name == "dynamic_world":
            mask = mask & (label_flat > 0) & (label_flat <= num_classes)

        if task_name == "worldcover":
            mask = mask & np.isin(label_flat, list(mapping.values()))

        all_embs.append(emb_flat[mask])
        all_labels.append(label_flat[mask])

    if len(all_embs) == 0:
        return None, None, None, None

    X = np.concatenate(all_embs, axis=0)  # [M, D]
    y = np.concatenate(all_labels, axis=0)

    # L2 归一化 embedding
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # 随机采样 K 个像素作为训练集
    M = len(X_norm)
    if k_pixels >= M:
        k_pixels = max(1, M // 2)

    indices = np.random.permutation(M)
    train_idx = indices[:k_pixels]
    test_idx = indices[k_pixels:]

    X_train = torch.from_numpy(X_norm[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    X_test = torch.from_numpy(X_norm[test_idx]).float()
    y_test = torch.from_numpy(y[test_idx]).long()

    return X_train, y_train, X_test, y_test


def train_mlp(mlp, X_train, y_train, X_test, y_test, num_classes, device, epochs=30, lr=1e-3, batch_size=1024):
    """训练 MLP."""
    mlp = mlp.to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    criterion = nn.CrossEntropyLoss()

    n_train = len(X_train)
    for epoch in range(epochs):
        mlp.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            logits = mlp(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

    # 评估
    mlp.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            logits = mlp(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_test[i:i+batch_size].numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def run_fewshot_task(spatial_maps, patch_ids, task_name, label_dir, label_file, num_classes, k_pixels_list, n_splits, device):
    """运行一个下游任务的 Few-Shot 实验."""
    D = spatial_maps.shape[2]
    results = []

    for k in k_pixels_list:
        split_accs = []
        for split in range(n_splits):
            data = prepare_fewshot_data(
                spatial_maps, patch_ids, label_dir, label_file,
                task_name, num_classes, k
            )
            if data[0] is None:
                continue

            X_train, y_train, X_test, y_test = data
            mlp = TinyMLP(D, num_classes, hidden=min(256, D * 4))
            acc = train_mlp(mlp, X_train, y_train, X_test, y_test, num_classes, device)
            split_accs.append(acc)

        if split_accs:
            results.append({
                "k_pixels": k,
                "accuracy_mean": float(np.mean(split_accs)),
                "accuracy_std": float(np.std(split_accs)),
                "n_splits": len(split_accs),
            })
            print(f"  K={k:>7}: Acc={np.mean(split_accs):.4f}±{np.std(split_accs):.4f} (n={len(split_accs)})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-file", type=str, required=True)
    parser.add_argument("--k-pixels", type=str, default="100,1000,10000,100000")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_pixels_list = [int(x.strip()) for x in args.k_pixels.split(",")]

    print("=" * 70)
    print(" Few-Shot 下游分类评估 (基于预计算 Embedding)")
    print("=" * 70)

    # 加载 embedding
    print(f"\n[1/2] 加载 embedding...")
    data = np.load(args.embedding_file)
    spatial_maps = data["spatial_maps"]
    patch_ids = data["patch_ids"]
    print(f"  形状: {spatial_maps.shape}, patches={len(patch_ids)}")

    # 运行所有任务
    print(f"\n[2/2] Few-Shot 实验...")
    all_results = {}

    for task_name, label_dir, label_file, num_classes in TASKS:
        print(f"\n--- {task_name} ({num_classes} classes) ---")
        results = run_fewshot_task(
            spatial_maps, patch_ids, task_name, label_dir, label_file,
            num_classes, k_pixels_list, args.n_splits, args.device
        )
        all_results[task_name] = results

    # 保存结果
    output_data = {
        "k_pixels": k_pixels_list,
        "n_splits": args.n_splits,
        "results": all_results,
    }

    if args.output is None:
        args.output = str(EVAL_DIR / "fewshot_downstream.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n📄 结果保存: {args.output}")

    # 汇总表
    print(f"\n{'='*70}")
    print(" 汇总")
    print(f"{'='*70}")
    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        for r in results:
            print(f"  K={r['k_pixels']:>7}: Acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
