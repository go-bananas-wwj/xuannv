#!/usr/bin/env python3
"""V8 Clean CD Head V3 训练 — 从预提取的 embedding 加载，跳过提取阶段.

用法:
    cd /workspace/xuannv
    python scripts/eval/train_cd_head_v8_v3.py \
        --config configs/xuannv_v8_clean.yaml \
        --precomputed /workspace/outputs/xuannv_backbone_v8_clean/precomputed_embeddings.pt \
        --device npu:0 --epochs 200 --lr 5e-4 --folds 5
"""
import sys, json, time, argparse
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"


def rasterize_annotations(changes, bounds, H=64, W=64):
    resolution = (bounds[2] - bounds[0]) / W
    resolution_y = (bounds[3] - bounds[1]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


def augment_embedding(eb, ea, mask):
    """数据增强: 随机水平/垂直翻转."""
    if np.random.rand() > 0.5:
        eb = torch.flip(eb, dims=[2])
        ea = torch.flip(ea, dims=[2])
        if mask is not None:
            mask = torch.flip(mask, dims=[1])
    if np.random.rand() > 0.5:
        eb = torch.flip(eb, dims=[1])
        ea = torch.flip(ea, dims=[1])
        if mask is not None:
            mask = torch.flip(mask, dims=[0])
    return eb, ea, mask


def focal_dice_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal BCE + Dice Loss."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    focal_loss = focal.mean()

    pred_sig = torch.sigmoid(pred)
    pred_flat = pred_sig.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    dice_loss = 1.0 - dice.mean()

    return focal_loss + dice_loss


def train_fold(cd_head, train_data, val_data, device, epochs=200, lr=5e-4):
    cd_head = cd_head.to(device)
    optimizer = torch.optim.AdamW(cd_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0
    patience = 30

    for epoch in range(epochs):
        cd_head.train()
        train_loss = 0.0
        np.random.shuffle(train_data)
        for item in train_data:
            eb = item["eb"].unsqueeze(0).to(device)
            ea = item["ea"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).unsqueeze(0).to(device)

            eb, ea, mask = augment_embedding(eb, ea, mask)
            pred = cd_head(eb, ea)
            loss = focal_dice_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cd_head.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        cd_head.eval()
        all_preds = []
        all_masks = []
        with torch.no_grad():
            for item in val_data:
                eb = item["eb"].unsqueeze(0).to(device)
                ea = item["ea"].unsqueeze(0).to(device)
                mask = item["mask"]
                pred = cd_head(eb, ea)
                pred_prob = torch.sigmoid(pred).cpu().numpy().flatten()
                all_preds.extend(pred_prob.tolist())
                all_masks.extend(mask.flatten().tolist())

        preds = np.array(all_preds)
        masks = np.array(all_masks)

        if len(np.unique(masks)) > 1:
            val_auc = roc_auc_score(masks, preds)
        else:
            val_auc = 0.0

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in cd_head.state_dict().items()}
            best_epoch = epoch

        if epoch - best_epoch > patience:
            print(f"  Early stopping at epoch {epoch+1} (best at {best_epoch+1})")
            break

        if (epoch + 1) % 20 == 0 or improved:
            flag = " ***" if improved else ""
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_data):.4f} val_auc={val_auc:.4f} best={best_val_auc:.4f} (E{best_epoch+1}){flag}")

    return best_val_auc, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xuannv_v8_clean.yaml")
    parser.add_argument("--precomputed", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  V8 Clean CD Head V3 训练 — 预提取 embedding + Focal Loss")
    print("=" * 70)
    print(f"  Precomputed: {args.precomputed}")
    print(f"  Device:      {args.device}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Folds:       {args.folds}")
    print("=" * 70)

    device = args.device
    torch.npu.set_device(device)

    # ── 加载预提取的 embedding ──
    print("\n[1/2] 加载预提取的 embedding...")
    precomputed = torch.load(args.precomputed, map_location='cpu')
    print(f"  预提取 patch 数: {len(precomputed)}")

    # ── 加载标注 ──
    print("\n[2/2] 加载标注...")
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
                geom = row.geometry
                if geom is None: continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                all_changes.append({"geometry": geom, "patch_id": row.get("patch_id", None)})
        except: pass

    patch_changes = {}
    for change in all_changes:
        if change["patch_id"]:
            patch_changes.setdefault(change["patch_id"], []).append(change["geometry"])
        else:
            pt = change["geometry"].centroid
            for pid, bounds in patch_bounds.items():
                if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                    patch_changes.setdefault(pid, []).append(change["geometry"])
                    break

    # 构建有标注 patch 的数据
    labeled_data = []
    for pid, emb_data in precomputed.items():
        if pid not in patch_changes:
            continue
        mask = rasterize_annotations(patch_changes[pid], patch_bounds[pid])
        labeled_data.append({
            "pid": pid,
            "eb": emb_data["eb"],
            "ea": emb_data["ea"],
            "mask": torch.from_numpy(mask).float(),
        })

    print(f"  有标注且有 embedding 的 patch: {len(labeled_data)}")

    # ── 5-fold CV ──
    print(f"\n5-fold CV 训练 CD Head V3...")
    from src.config import load_config
    cfg = load_config(args.config)
    from src.models.heads import ChangeDetectionHeadV3

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(labeled_data)):
        print(f"\n--- Fold {fold_idx+1}/{args.folds} ---")
        train_data = [labeled_data[i] for i in train_idx]
        val_data = [labeled_data[i] for i in val_idx]
        print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        cd_head = ChangeDetectionHeadV3(embedding_dim=cfg.model.embedding_dim, hidden_dim=64, dropout=0.3)
        n_params = sum(p.numel() for p in cd_head.parameters())
        print(f"  CD Head V3 params: {n_params:,}")

        best_auc, best_state = train_fold(cd_head, train_data, val_data, device, epochs=args.epochs, lr=args.lr)
        print(f"  Fold {fold_idx+1} Best Val AUC: {best_auc:.4f}")
        fold_results.append(best_auc)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("  5-fold CV 结果汇总 (CD Head V3)")
    print("=" * 70)
    for i, auc in enumerate(fold_results):
        print(f"  Fold {i+1}: AUC = {auc:.4f}")
    print(f"\n  Mean AUC:   {np.mean(fold_results):.4f}")
    print(f"  Std AUC:    {np.std(fold_results):.4f}")
    print(f"  Min/Max:    {np.min(fold_results):.4f} / {np.max(fold_results):.4f}")
    print("=" * 70)

    print("\n对比:")
    print(f"  CD Head V1 (BCE+Dice):          0.5649")
    print(f"  CD Head V3 (Focal+增强+V3+ES):  {np.mean(fold_results):.4f}")
    print(f"  提升: {np.mean(fold_results) - 0.5649:+.4f}")


if __name__ == "__main__":
    main()
