#!/usr/bin/env python3
"""
AEF_qwen 像素级有监督微调 — 用 105 个标注多边形提升 embedding 时间敏感度

核心思想:
  论文中模型对不同的 valid_period 产生不同的 embedding，编码该时间段的土地状态。
  我们的模型对不同的 valid_period 产生几乎相同的 embedding (cos_sim=0.97)。

  改进方法:
  - 对标注为"变化"的区域: 推远 before/after embedding
  - 对标注为"未变化"的区域: 拉近 before/after embedding
  
  损失:
    L_change = mean(cos_sim(before, after) for changed pixels)  # 应该 → -1
    L_unchanged = mean(-cos_sim(before, after) for unchanged pixels)  # 应该 → -1
    L = L_change + L_unchanged

  训练时:
    - 对每个 patch，提取 before/after embedding
    - 根据标注生成像素级变化掩码
    - 计算像素级的时间对比损失
    - 微调 backbone (部分解冻)

用法:
    CUDA_VISIBLE_DEVICES=5 python3 scripts/finetune_pixel_level.py \
        --checkpoint /workspace/outputs/aef_qwen_v2/epoch_499.pt \
        --epochs 30 \
        --lr 5e-7 \
        --backbone-lr 5e-8 \
        --output-dir /workspace/outputs/aef_qwen_v2_pixel_ft
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
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
    model = AEFModel(cfg).to("npu:0")
    ckpt = torch.load(ckpt_path, map_location="npu:0", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg

def extract_embeddings(model, dataset, patch_idx, training=False):
    """提取 patch 的 before/after embedding."""
    batch = dataset[patch_idx]
    before_map = None
    after_map = None
    
    for ws, we in [(BEFORE_WINDOW[0], BEFORE_WINDOW[1]), (AFTER_WINDOW[0], AFTER_WINDOW[1])]:
        batch["valid_start_ms"] = torch.tensor(ws, dtype=torch.float64)
        batch["valid_end_ms"] = torch.tensor(we, dtype=torch.float64)
        batch_dev = {k: v.unsqueeze(0).to("npu:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
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
        emb = F.normalize(output.embedding_map, p=2, dim=1)
        if before_map is None:
            before_map = emb
        else:
            after_map = emb
    
    return before_map[0], after_map[0]  # [D, H, W]

# ──────────────────────────────────────────
# 训练
# ──────────────────────────────────────────
def train_pixel_level(model, dataset, patch_to_changes, patch_bounds,
                       lr=5e-7, backbone_lr=5e-8, epochs=30, output_dir=None):
    """
    像素级有监督微调:
    - 冻结前 80% backbone
    - 微调最后几个 block
    - 用标注的像素级变化信息做对比学习
    """
    device = "npu:0"
    
    # 冻结大部分 backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻 bottleneck + 最后 2 个 STP block + classification heads
    for name, param in model.named_parameters():
        if any(x in name for x in ['bottleneck', 'classification_head', 'aux_cls_head', 'bottleneck_cls_head']):
            param.requires_grad = True
        # 尝试解冻最后的 STP blocks
        if 'stp_blocks' in name and any(f'stp_blocks.{i}' in name for i in range(6, 8)):
            param.requires_grad = True
    
    # 计算可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  可训练参数: {trainable_params:,} ({trainable_params/(trainable_params+frozen_params)*100:.1f}%)")
    print(f"  冻结参数: {frozen_params:,}")
    
    # 优化器
    trainable_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': trainable_list, 'lr': backbone_lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数
    # L = mean(cos_sim for changed) - mean(cos_sim for unchanged)
    # 目标: changed 区域的 cos_sim → -1, unchanged 区域的 cos_sim → 1
    criterion_change = nn.MSELoss()  # 推远 changed 区域的 embedding
    
    best_auc = 0
    best_state = None
    
    test_patches = [pid for pid in patch_to_changes.keys() if pid in dataset.patches]
    
    print(f"\n{'Epoch':<6} {'Change Loss':<14} {'Unchanged Loss':<16} {'Total Loss':<12} {'Time':<8}")
    print("-"*60)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        total_loss = 0
        n_patches = 0
        t0 = time.time()
        
        # 对每个有标注的 patch
        for pid in test_patches:
            pidx = dataset.patches.index(pid)
            
            # 提取 before/after embedding
            before_map, after_map = extract_embeddings(model, dataset, pidx)
            D, H, W = before_map.shape
            
            # 构建变化掩码
            bounds = patch_bounds.get(pid)
            if not bounds:
                continue
            resolution = (bounds[2] - bounds[0]) / H
            mask = torch.zeros((H, W), dtype=torch.bool, device=device)
            
            for ch_info in patch_to_changes.get(pid, []):
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
                            mask[px, py] = True
            
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(
                before_map.reshape(D, -1), 
                after_map.reshape(D, -1), 
                dim=0
            ).reshape(H, W)
            
            # Changed 区域: 应该 cos_sim → -1 (不同)
            changed_cos = cos_sim[mask]
            loss_change = ((changed_cos + 1.0) ** 2).mean()  # target: -1
            
            # Unchanged 区域: 应该 cos_sim → 1 (相同)
            unchanged_cos = cos_sim[~mask]
            loss_unchanged = ((unchanged_cos - 1.0) ** 2).mean()  # target: 1
            
            # 总损失 (加权)
            loss = 0.5 * loss_change + 0.5 * loss_unchanged
            total_loss += loss.item()
            n_patches += 1
            
            loss.backward()
        
        if n_patches == 0:
            continue
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        
        optimizer.step()
        scheduler.step()
        
        elapsed = time.time() - t0
        avg_loss = total_loss / n_patches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<6} {avg_loss:<14.4f} {'':<16} {'':<12} {elapsed:<8.1f}s")
        
        # 保存最佳模型 (简化版: 每 10 个 epoch 评估)
        if (epoch + 1) % 10 == 0:
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                ckpt_path = output_dir_path / f"epoch_{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                }, ckpt_path)
                print(f"  ✓ Saved checkpoint: {ckpt_path}")
    
    return model

# ──────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/aef_qwen_v2/epoch_499.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--backbone-lr", type=float, default=5e-8)
    parser.add_argument("--output-dir", type=str, default="/workspace/outputs/aef_qwen_v2_pixel_ft")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("  AEF_qwen 像素级有监督微调")
    print("="*60)
    
    # 加载模型
    print("\n加载 V2 模型...")
    model, cfg = load_model(args.checkpoint)
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
                    all_changes.append({"geometry": row.geometry})
        except:
            pass
    
    patch_to_changes = {}
    for ch in all_changes:
        for pid, bounds in patch_bounds.items():
            if ch["geometry"].intersects(box(bounds[0], bounds[1], bounds[2], bounds[3])):
                if pid not in patch_to_changes:
                    patch_to_changes[pid] = []
                patch_to_changes[pid].append(ch)
    
    test_patches = [pid for pid in patch_to_changes.keys() if pid in dataset.patches]
    print(f"  {len(test_patches)} 个有标注的 patch")
    
    # 训练
    print(f"\n开始像素级微调 (lr={args.lr}, backbone_lr={args.backbone_lr}, epochs={args.epochs})...")
    model = train_pixel_level(
        model, dataset, patch_to_changes, patch_bounds,
        lr=args.lr, backbone_lr=args.backbone_lr,
        epochs=args.epochs, output_dir=output_dir
    )
    
    # 保存最终模型
    final_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_path)
    print(f"\n最终模型已保存: {final_path}")
    
    print(f"\n{'='*60}")
    print("  像素级微调完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
