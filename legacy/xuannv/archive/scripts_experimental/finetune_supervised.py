#!/usr/bin/env python3
"""
AEF_qwen 有监督微调脚本 — 用 105 个光学标注多边形微调模型的时间敏感度

核心思想:
- 冻结 backbone (保留空间表征能力)
- 微调最后几层 + temporal head
- 用标注的 before/after 像素对做有监督训练
- 目标: 让模型学会区分变化区域和未变化区域

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=5 python3 scripts/finetune_supervised.py \
        --checkpoint /workspace/outputs/aef_qwen_v2/epoch_499.pt \
        --epochs 50 \
        --lr 1e-6 \
        --output-dir /workspace/outputs/aef_qwen_v2_finetune
"""
import os, sys, json, time, argparse
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"

# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)    # 2024Q3-2025Q4

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
    model = AEFModel(cfg).to("cuda:0")
    ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg

# ──────────────────────────────────────────
# 提取 embedding
# ──────────────────────────────────────────
def extract_embeddings_for_patches(model, dataset, patch_ids):
    """提取所有 patch 的 before/after embedding maps."""
    before_maps = {}
    after_maps = {}

    for i, pid in enumerate(patch_ids):
        if pid not in dataset.patches:
            continue
        pidx = dataset.patches.index(pid)
        t0 = time.time()

        # Before
        batch = dataset[pidx]
        batch["valid_start_ms"] = torch.tensor(BEFORE_WINDOW[0], dtype=torch.float64)
        batch["valid_end_ms"] = torch.tensor(BEFORE_WINDOW[1], dtype=torch.float64)
        batch_dev = {k: v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
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
        before_maps[pid] = F.normalize(out.embedding_map, p=2, dim=1).cpu().numpy()[0]

        # After
        batch["valid_start_ms"] = torch.tensor(AFTER_WINDOW[0], dtype=torch.float64)
        batch["valid_end_ms"] = torch.tensor(AFTER_WINDOW[1], dtype=torch.float64)
        batch_dev = {k: v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
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
        after_maps[pid] = F.normalize(out.embedding_map, p=2, dim=1).cpu().numpy()[0]

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(patch_ids)}] {pid}: before+after ({elapsed:.1f}s)")

    return before_maps, after_maps

# ──────────────────────────────────────────
# 构建训练数据集
# ──────────────────────────────────────────
def build_training_dataset(before_maps, after_maps, patch_to_changes, patch_bounds,
                           n_positive_per_patch=200, n_negative_per_patch=600):
    """
    构建有监督微调数据集

    正样本: 标注多边形内的像素 (变化区域)
    负样本: 随机采样的非变化区域像素

    返回:
        features: [N, D*2] concat(before, after)
        labels: [N] 0/1
    """
    all_features = []
    all_labels = []

    for pid in before_maps:
        before = before_maps[pid]  # [D, H, W]
        after = after_maps[pid]    # [D, H, W]
        D, H, W = before.shape

        # 构建变化掩码
        mask = np.zeros((H, W), dtype=np.int32)
        changes = patch_to_changes.get(pid, [])

        bounds = patch_bounds.get(pid)
        if not bounds:
            continue
        resolution = (bounds[2] - bounds[0]) / H

        for ch_info in changes:
            geom = ch_info["geometry"]
            if geom is None:
                continue
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

        # 采样
        pos_idx = np.where(mask.flatten() == 1)[0]
        neg_idx = np.where(mask.flatten() == 0)[0]

        n_pos = min(n_positive_per_patch, len(pos_idx))
        n_neg = min(n_negative_per_patch, len(neg_idx))

        if n_pos == 0 or n_neg == 0:
            continue

        rng = np.random.RandomState(42 + hash(pid) % 1000)
        pos_sample = rng.choice(pos_idx, n_pos, replace=False)
        neg_sample = rng.choice(neg_idx, n_neg, replace=False)

        for idx in pos_sample:
            px, py = divmod(idx, W)
            feat = np.concatenate([before[:, px, py], after[:, px, py]])
            all_features.append(feat)
            all_labels.append(1)

        for idx in neg_sample:
            px, py = divmod(idx, W)
            feat = np.concatenate([before[:, px, py], after[:, px, py]])
            all_features.append(feat)
            all_labels.append(0)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    return X, y

# ──────────────────────────────────────────
# 训练
# ──────────────────────────────────────────
class TemporalHead(nn.Module):
    """轻量时间敏感度头 — 学习从 concat(before, after) 预测变化."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


def train_model(model, X_train, y_train, X_val, y_val, lr=1e-6, epochs=50, output_dir=None):
    """
    有监督微调:
    - 冻结 backbone
    - 只训练 temporal head
    """
    device = "cuda:0"
    head = TemporalHead(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(device))  # 3:1 权重

    # 转换数据
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_auc = 0
    best_state = None

    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val AUC':<10} {'Val BA':<10} {'Best AUC':<10}")
    print("-"*65)

    for epoch in range(epochs):
        # Train
        head.train()
        optimizer.zero_grad()
        logits = head(X_train_t)
        train_loss = criterion(logits, y_train_t)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        # Val
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_probs > 0.5).astype(float)
            val_auc = roc_auc_score(y_val, val_probs)
            val_ba = balanced_accuracy_score(y_val, val_pred)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<6} {train_loss.item():<12.4f} {val_loss.item():<12.4f} "
                  f"{val_auc:<10.4f} {val_ba:<10.4f} {best_auc:<10.4f}")

    # Save
    if output_dir and best_state:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / "temporal_head_best.pt")
        print(f"\n最佳模型已保存: {output_dir / 'temporal_head_best.pt'}")

    return best_state, best_auc

# ──────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/aef_qwen_v2/epoch_499.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default="/workspace/outputs/aef_qwen_v2_finetune")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  AEF_qwen 有监督微调 — 用标注数据提升时间敏感度")
    print("="*60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  输出: {output_dir}")
    print("="*60)

    # 加载模型
    print("\n加载 V2 模型...")
    model, cfg = load_model(args.checkpoint)
    model.eval()
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    # 加载 Grid 和标注
    print("加载 Grid 和标注...")
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

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

    test_patches = [pid for pid in patch_to_changes.keys() if pid in dataset.patches]
    print(f"\n{len(test_patches)} 个有标注的 patch")

    # 提取 embedding
    print("\n提取 before/after embedding...")
    before_maps, after_maps = extract_embeddings_for_patches(model, dataset, test_patches)

    # 构建训练数据
    print("\n构建训练数据集...")
    X, y = build_training_dataset(before_maps, after_maps, patch_to_changes, patch_bounds,
                                  n_positive_per_patch=300, n_negative_per_patch=900)
    print(f"  总样本: {len(X)}, 正样本: {int(y.sum())}, 负样本: {int(len(y)-y.sum())}")
    print(f"  特征维度: {X.shape[1]}")

    # 划分 train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # 训练
    print(f"\n开始微调 (lr={args.lr}, epochs={args.epochs})...")
    best_state, best_auc = train_model(model, X_train, y_train, X_val, y_val,
                                        lr=args.lr, epochs=args.epochs, output_dir=output_dir)

    # 保存结果
    result = {
        "model": "v2_finetune",
        "checkpoint": args.checkpoint,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_val_auc": float(best_auc),
    }
    with open(output_dir / "finetune_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  微调完成! 最佳验证 AUC: {best_auc:.4f}")
    print(f"{'='*60}")

    # 与 V2 baseline 对比
    print(f"\n  V2 baseline (500-shot): 0.677")
    print(f"  V2+finetune (有监督):    {best_auc:.4f}")
    print(f"  改善: {'✅' if best_auc > 0.677 else '❌'}")

if __name__ == "__main__":
    main()
