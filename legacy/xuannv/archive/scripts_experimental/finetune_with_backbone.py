#!/usr/bin/env python3
"""
AEF_qwen 有监督对比微调 — 用标注数据教会模型编码时间状态

核心问题:
  当前模型的 before/after embedding 几乎完全相同 (cos_sim=0.97)
  因为模型学到的是"时间不变的平均表示"

  论文的做法:
  模型对不同的 valid_period 产生不同的 embedding，编码该时间段的土地状态
  变化区域的 before/after embedding 应该不同
  未变化区域的 before/after embedding 应该相同

解决方案: 有监督三元组损失 + 部分解冻 backbone
  正样本对: 未变化区域的 (before, after) → 应该相似
  负样本对: 变化区域的 (before, after) → 应该不同

  Loss = mean(unchanged: cos_sim(before, after)) - mean(changed: cos_sim(before, after))
  目标: 最大化这个差值 → 未变化区域相似度高, 变化区域相似度低

  关键改进: 解冻 bottleneck 和最后 2 个 STP block
  让模型学会对不同时间窗口产生不同 embedding

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=5 python3 scripts/finetune_with_backbone.py \
        --checkpoint /workspace/outputs/aef_qwen_v2/epoch_499.pt \
        --epochs 20 \
        --lr 1e-6 \
        --backbone-lr 1e-7 \
        --output-dir /workspace/outputs/aef_qwen_v2_finetune_backbone
"""
import os, sys, json, time, argparse
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def extract_embedding_for_patch(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
    """提取单个 patch 的 L2-normalized embedding map [D, H, W]."""
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to("cuda:0")
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
    emb_map = output.embedding_map  # [1, D, H, W]
    emb_map = F.normalize(emb_map, p=2, dim=1)
    return emb_map[0].cpu().numpy()

# ──────────────────────────────────────────
# 构建训练数据
# ──────────────────────────────────────────
def build_supervised_pairs(before_maps, after_maps, patch_to_changes, patch_bounds,
                           n_unchanged=300, n_changed=300):
    """
    构建有监督对比学习数据集

    返回:
      emb_before: [N, D]
      emb_after: [N, D]
      labels: [N] 0=未变化, 1=变化
    """
    all_before = []
    all_after = []
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
        changed_idx = np.where(mask.flatten() == 1)[0]
        unchanged_idx = np.where(mask.flatten() == 0)[0]

        n_ch = min(n_changed, len(changed_idx))
        n_un = min(n_unchanged, len(unchanged_idx))

        if n_ch == 0 or n_un == 0:
            continue

        rng = np.random.RandomState(42 + hash(pid) % 1000)
        ch_sample = rng.choice(changed_idx, n_ch, replace=False)
        un_sample = rng.choice(unchanged_idx, n_un, replace=False)

        for idx in ch_sample:
            px, py = divmod(idx, W)
            all_before.append(before[:, px, py])
            all_after.append(after[:, px, py])
            all_labels.append(1)  # 变化

        for idx in un_sample:
            px, py = divmod(idx, W)
            all_before.append(before[:, px, py])
            all_after.append(after[:, px, py])
            all_labels.append(0)  # 未变化

    emb_before = np.array(all_before, dtype=np.float32)
    emb_after = np.array(all_after, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    return emb_before, emb_after, labels


class ChangeDetectionHead(nn.Module):
    """变化检测头: 从 before/after embedding 对预测变化."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + 1, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, before, after):
        cos_sim = F.cosine_similarity(before, after, dim=-1).unsqueeze(-1)
        abs_diff = torch.abs(before - after)
        features = torch.cat([abs_diff, cos_sim], dim=-1)
        return self.head(features).squeeze(-1)


def train_with_partial_backbone(model, emb_before, emb_after, labels,
                                 dataset, patch_to_changes, patch_bounds,
                                 lr=1e-6, backbone_lr=1e-7, epochs=20,
                                 output_dir=None):
    """
    有监督微调 + 部分解冻 backbone:
    - 冻结 sensor_encoder_bank 和前 6 个 STP block
    - 解冻最后 2 个 STP block + bottleneck + classification_head
    - 训练 change detection head
    """
    device = "cuda:0"

    # 创建 head
    head = ChangeDetectionHead(emb_before.shape[1]).to(device)

    # 冻结大部分 backbone
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后 2 个 STP block
    if hasattr(model, 'stp_blocks'):
        for block in model.stp_blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
    # 解冻 bottleneck
    for param in model.bottleneck.parameters():
        param.requires_grad = True
    # 解冻 classification heads
    for param in model.classification_head.parameters():
        param.requires_grad = True
    for param in model.aux_cls_head.parameters():
        param.requires_grad = True
    for param in model.bottleneck_cls_head.parameters():
        param.requires_grad = True

    # 计算可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"\n  可训练参数: backbone={trainable_params:,}, head={head_params:,}")

    # 优化器: 分别设置 backbone 和 head 的学习率
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    head_params_list = list(head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params_list, 'lr': lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 权重: 处理正负样本不平衡
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - labels.sum())
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1), 1.0)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 划分 train/val
    idx = np.arange(len(labels))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    X_before_train = torch.FloatTensor(emb_before[train_idx]).to(device)
    X_after_train = torch.FloatTensor(emb_after[train_idx]).to(device)
    y_train = torch.FloatTensor(labels[train_idx]).to(device)
    X_before_val = torch.FloatTensor(emb_before[val_idx]).to(device)
    X_after_val = torch.FloatTensor(emb_after[val_idx]).to(device)
    y_val = torch.FloatTensor(labels[val_idx]).to(device)

    best_auc = 0
    best_state = None

    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val AUC':<10} {'Val BA':<10} {'Best AUC':<10}")
    print("-"*65)

    for epoch in range(epochs):
        # Train
        model.train()
        head.train()
        optimizer.zero_grad()
        logits = head(X_before_train, X_after_train)
        train_loss = criterion(logits, y_train)
        train_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + head_params_list,
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        head.eval()
        with torch.no_grad():
            val_logits = head(X_before_val, X_after_val)
            val_loss = criterion(val_logits, y_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_probs > 0.5).astype(float)
            val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs) if len(np.unique(y_val.cpu().numpy())) >= 2 else 0.5
            val_ba = balanced_accuracy_score(y_val.cpu().numpy(), val_pred)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
                'backbone': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            }

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<6} {train_loss.item():<12.4f} {val_loss.item():<12.4f} "
                  f"{val_auc:<10.4f} {val_ba:<10.4f} {best_auc:<10.4f}")

    # Save
    if output_dir and best_state:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / "finetuned_model.pt")
        print(f"\n最佳模型已保存: {output_dir / 'finetuned_model.pt'}")

    return best_state, best_auc


# ──────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/aef_qwen_v2/epoch_499.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=str, default="/workspace/outputs/aef_qwen_v2_finetune_backbone")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  AEF_qwen 有监督微调 (部分解冻 backbone)")
    print("="*60)

    # 加载模型
    print("\n加载 V2 模型...")
    model, cfg = load_model(args.checkpoint)
    model.eval()
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    # 加载标注
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
        except Exception as e:
            pass

    patch_to_changes = {}
    for ch in all_changes:
        for pid, bounds in patch_bounds.items():
            patch_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
            if ch["geometry"].intersects(patch_box):
                if pid not in patch_to_changes:
                    patch_to_changes[pid] = []
                patch_to_changes[pid].append(ch)

    test_patches = [pid for pid in patch_to_changes.keys() if pid in dataset.patches]
    print(f"  {len(test_patches)} 个有标注的 patch")

    # 提取 embedding
    print("\n提取 before/after embedding...")
    before_maps = {}
    after_maps = {}
    for i, pid in enumerate(test_patches):
        pidx = dataset.patches.index(pid)
        t0 = time.time()
        before_maps[pid] = extract_embedding_for_patch(model, dataset, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
        after_maps[pid] = extract_embedding_for_patch(model, dataset, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1])
        print(f"  [{i+1}/{len(test_patches)}] {pid} ({time.time()-t0:.1f}s)")

    # 构建有监督对比数据集
    print("\n构建有监督对比数据集...")
    emb_before, emb_after, labels = build_supervised_pairs(
        before_maps, after_maps, patch_to_changes, patch_bounds,
        n_unchanged=300, n_changed=300
    )
    print(f"  总样本: {len(labels)}, 变化: {int(labels.sum())}, 未变化: {int(len(labels)-labels.sum())}")

    # 训练
    print(f"\n开始微调 (lr={args.lr}, backbone_lr={args.backbone_lr}, epochs={args.epochs})...")
    best_state, best_auc = train_with_partial_backbone(
        model, emb_before, emb_after, labels,
        dataset, patch_to_changes, patch_bounds,
        lr=args.lr, backbone_lr=args.backbone_lr,
        epochs=args.epochs, output_dir=output_dir
    )

    # 保存结果
    with open(output_dir / "result.json", "w") as f:
        json.dump({
            "best_val_auc": float(best_auc),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  微调完成! 最佳验证 AUC: {best_auc:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
