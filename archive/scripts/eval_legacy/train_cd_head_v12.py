#!/usr/bin/env python3
"""V12 CD Head 训练 — 修复索引 + 正确2025年period窗口.

用法:
    python scripts/eval/train_cd_head_v12.py \
        --config configs/xuannv_v12_clean.yaml \
        --checkpoint /workspace/outputs/exp_v2_A_skipL2_50ep_0520/epoch_best_epoch48.pt \
        --device npu:0 \
        --use-pre-norm
"""
import sys, json, time, argparse, os
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
try:
    torch_npu.npu.set_device("npu:0")
except:
    pass
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

# 2025年各月份时间窗口 (ms) — 与 validate_v12_bare.py 一致
MONTH_WINDOWS_2025 = {
    4:  (1743436800000, 1746028799000),   # 2025-04
    6:  (1748707200000, 1751299199000),   # 2025-06
    8:  (1753977600000, 1756655999000),   # 2025-08
    9:  (1756656000000, 1759247999000),   # 2025-09
    10: (1759248000000, 1761926399000),   # 2025-10
}

# Shapefile → 月份对映射
PERIOD_MONTHS = {
    "june":        (4, 6),
    "aug":         (6, 8),
    "September":   (8, 9),
    "October":     (9, 10),
}


def load_backbone(cfg_path, ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.inference.engine import extract_embedding_for_month

    cfg = load_config(cfg_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    cfg.data.preload = True  # 预加载到内存，加速提取
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, extract_embedding_for_month, cfg


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
    """随机水平/垂直翻转."""
    if np.random.rand() > 0.5:
        eb = torch.flip(eb, dims=[2])
        ea = torch.flip(ea, dims=[2])
        mask = torch.flip(mask, dims=[1])
    if np.random.rand() > 0.5:
        eb = torch.flip(eb, dims=[1])
        ea = torch.flip(ea, dims=[1])
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


def extract_all_data(model, dataset, extract_fn, device, patch_changes, patch_bounds, use_pre_norm=False):
    """为每个有标注的 patch 和 period 提取 before/after embedding + mask."""
    data = []
    skipped = 0
    total_items = sum(len(v) for v in patch_changes.values())
    processed = 0
    for pid, period_changes in patch_changes.items():
        if pid not in patch_bounds:
            continue
        bounds = patch_bounds[pid]

        # 按 period 分组
        for period, geoms in period_changes.items():
            processed += 1
            before_m, after_m = PERIOD_MONTHS[period]
            try:
                eb = extract_fn(model, dataset, pid, 2025, before_m, device, normalize=True, use_pre_norm=use_pre_norm)
                ea = extract_fn(model, dataset, pid, 2025, after_m, device, normalize=True, use_pre_norm=use_pre_norm)
                mask = rasterize_annotations(geoms, bounds)
                n_changed = int(mask.sum())
                if n_changed < 5:
                    skipped += 1
                    continue
                data.append({
                    "pid": pid,
                    "period": period,
                    "eb": torch.from_numpy(eb).float(),
                    "ea": torch.from_numpy(ea).float(),
                    "mask": torch.from_numpy(mask).float(),
                })
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  [Skip] {pid} {period}: {e}")
            if processed % 10 == 0 or processed == total_items:
                print(f"  ... {processed}/{total_items} periods extracted, {len(data)} valid, {skipped} skipped")
    print(f"  成功提取 {len(data)} 个样本，跳过 {skipped} 个")
    return data


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

        # Val
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

        improved = False
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in cd_head.state_dict().items()}
            best_epoch = epoch
            improved = True

        if epoch - best_epoch > patience:
            print(f"  Early stopping at epoch {epoch+1} (best at {best_epoch+1})")
            break

        if (epoch + 1) % 20 == 0 or improved:
            flag = " ***" if improved else ""
            print(f"  Epoch {epoch+1:3d}: loss={train_loss/len(train_data):.4f} val_auc={val_auc:.4f} best={best_val_auc:.4f} (E{best_epoch+1}){flag}")

    return best_val_auc, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xuannv_v12_clean.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-pre-norm", action="store_true", help="使用pre_norm embedding")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64, help="CD Head hidden dim")
    parser.add_argument("--dropout", type=float, default=0.3, help="CD Head dropout")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    embed_type = "pre-norm" if args.use_pre_norm else "L2-norm"
    print("=" * 70)
    print(f"  V12 CD Head 训练 — {embed_type}")
    print("=" * 70)
    print(f"  Backbone:   {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Folds:      {args.folds}")
    print("=" * 70)

    device = args.device
    try:
        torch.npu.set_device(device)
    except:
        pass

    # ── 加载 backbone ──
    print("\n[1/4] 加载冻结的 backbone...")
    model, dataset, extract_fn, cfg = load_backbone(args.config, args.checkpoint, device)
    print(f"  Dataset: {len(dataset.patches)} patches, {len(dataset.monthly_samples)} monthly samples")
    print(f"  Embedding dim: {cfg.model.embedding_dim}")

    # ── 加载 Grid 和标注 ──
    print("\n[2/4] 加载标注数据...")
    with open(GRID_PATH) as f:
        grid_data = json.load(f)
    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    patch_changes = {}  # pid -> {period: [geoms]}
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        period = shp_name.replace(".shp", "")
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                pid = row.get("patch_id", None)
                if not pid:
                    # 通过空间匹配
                    pt = geom.centroid
                    for candidate_pid, bounds in patch_bounds.items():
                        if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                            pid = candidate_pid
                            break
                if pid:
                    patch_changes.setdefault(pid, {}).setdefault(period, []).append(geom)
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

    # 过滤只在数据集中的 patch
    valid_patch_changes = {pid: v for pid, v in patch_changes.items() if pid in dataset.patches}
    print(f"  有效标注 patch: {len(valid_patch_changes)}")

    # ── 提取 embedding ──
    print(f"\n[3/4] 提取 {embed_type} embedding...")
    start = time.time()
    all_data = extract_all_data(model, dataset, extract_fn, device, valid_patch_changes, patch_bounds, use_pre_norm=args.use_pre_norm)
    print(f"  耗时 {time.time()-start:.1f}s，总样本 {len(all_data)}")

    if len(all_data) < 10:
        print("[ERROR] 样本太少，无法训练")
        return

    # 统计
    n_changed_pixels = sum(int(d["mask"].sum().item()) for d in all_data)
    n_total_pixels = sum(d["mask"].numel() for d in all_data)
    print(f"  变化像素比例: {n_changed_pixels}/{n_total_pixels} = {100*n_changed_pixels/n_total_pixels:.2f}%")

    # ── 5-fold CV ──
    print(f"\n[4/4] {args.folds}-fold CV 训练 CD Head V3...")
    from src.models.heads import ChangeDetectionHeadV3

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n--- Fold {fold_idx+1}/{args.folds} ---")
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        cd_head = ChangeDetectionHeadV3(embedding_dim=cfg.model.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        n_params = sum(p.numel() for p in cd_head.parameters())
        print(f"  CD Head V3 (h={args.hidden_dim}) params: {n_params:,}")

        best_auc, best_state = train_fold(cd_head, train_data, val_data, device, epochs=args.epochs, lr=args.lr)
        print(f"  Fold {fold_idx+1} Best Val AUC: {best_auc:.4f}")
        fold_results.append({"auc": best_auc, "state": best_state})

    # ── 汇总 ──
    aucs = [r["auc"] for r in fold_results]
    print("\n" + "=" * 70)
    print(f"  {args.folds}-fold CV 结果汇总 ({embed_type})")
    print("=" * 70)
    for i, auc in enumerate(aucs):
        print(f"  Fold {i+1}: AUC = {auc:.4f}")
    print(f"\n  Mean AUC:   {np.mean(aucs):.4f}")
    print(f"  Std AUC:    {np.std(aucs):.4f}")
    print(f"  Min/Max:    {np.min(aucs):.4f} / {np.max(aucs):.4f}")
    print("=" * 70)

    # 保存最佳 fold 的 checkpoint
    best_fold = int(np.argmax(aucs))
    best_state = fold_results[best_fold]["state"]

    if args.output is None:
        suffix = "_prenorm" if args.use_pre_norm else ""
        exp_name = os.path.basename(os.path.dirname(args.checkpoint))
        args.output = os.path.join(os.path.dirname(args.checkpoint), f"cd_head_v12_best{suffix}.pt")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "cd_head": best_state,
        "config": {
            "embedding_dim": cfg.model.embedding_dim,
            "hidden_dim": 64,
            "dropout": 0.3,
        },
        "fold_results": [{"fold": i, "auc": float(a)} for i, a in enumerate(aucs)],
        "mean_auc": float(np.mean(aucs)),
        "backbone_checkpoint": args.checkpoint,
        "embed_type": embed_type,
    }, args.output)
    print(f"\n  最佳 CD Head 已保存: {args.output} (Fold {best_fold+1}, AUC={aucs[best_fold]:.4f})")


if __name__ == "__main__":
    main()
