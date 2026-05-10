#!/usr/bin/env python3
"""V8 Clean CD Head 训练 — 冻结 backbone，5-fold CV.

用法:
    cd /workspace/xuannv
    python scripts/eval/train_cd_head_v8.py \
        --config configs/xuannv_v8_clean.yaml \
        --checkpoint /workspace/outputs/xuannv_backbone_v8_clean/epoch_best_epoch223.pt \
        --device npu:0
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
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)


def load_backbone(cfg_path, ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.inference.engine import extract_embedding_map

    cfg = load_config(cfg_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, extract_embedding_map, cfg


def compute_change_map(emb_before, emb_after):
    D, H, W = emb_before.shape
    fb = emb_before.reshape(D, -1)
    fa = emb_after.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    return ((1.0 - cos_sim) / 2.0).reshape(H, W)


def rasterize_annotations(changes, bounds, H=64, W=64):
    resolution = (bounds[2] - bounds[0]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class SimpleCDHead(nn.Module):
    """轻量 2-layer CD Head."""
    def __init__(self, embedding_dim=64, hidden_dim=64):
        super().__init__()
        in_dim = embedding_dim * 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, emb_before, emb_after):
        diff = emb_before - emb_after
        feat = torch.cat([
            torch.abs(diff),
            emb_before * emb_after,
            emb_before,
            emb_after,
        ], dim=1)
        x = self.conv1(feat)
        x = self.conv2(x)
        return self.out(x)


def extract_all_data(model, dataset, extract_fn, device, patch_infos):
    """提取所有 patch 的 before/after embedding + mask."""
    data = []
    for pid, pidx, bounds, changes in patch_infos:
        try:
            eb = extract_fn(model, dataset, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1], device, normalize=True)
            ea = extract_fn(model, dataset, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1], device, normalize=True)
            mask = rasterize_annotations(changes, bounds)
            data.append({
                "pid": pid,
                "eb": torch.from_numpy(eb).float(),
                "ea": torch.from_numpy(ea).float(),
                "mask": torch.from_numpy(mask).float(),
            })
        except Exception as e:
            print(f"  跳过 {pid}: {e}")
    return data


def train_fold(cd_head, train_data, val_data, device, epochs=100, lr=1e-3):
    cd_head = cd_head.to(device)
    optimizer = torch.optim.AdamW(cd_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    best_val_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        cd_head.train()
        train_loss = 0.0
        np.random.shuffle(train_data)
        for item in train_data:
            eb = item["eb"].unsqueeze(0).to(device)
            ea = item["ea"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

            pred = cd_head(eb, ea)
            loss_bce = F.binary_cross_entropy_with_logits(pred, mask)
            loss_dice = dice_loss(pred, mask)
            loss = loss_bce + loss_dice

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

        # 过滤无效样本
        if len(np.unique(masks)) > 1:
            val_auc = roc_auc_score(masks, preds)
        else:
            val_auc = 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in cd_head.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_data):.4f} val_auc={val_auc:.4f} best={best_val_auc:.4f}")

    return best_val_auc, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xuannv_v8_clean.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  V8 Clean CD Head 训练 — 冻结 Backbone + 5-fold CV")
    print("=" * 70)
    print(f"  Backbone:   {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Folds:      {args.folds}")
    print("=" * 70)

    device = args.device
    torch.npu.set_device(device)

    # ── 加载 backbone ──
    print("\n[1/4] 加载冻结的 backbone...")
    model, dataset, extract_fn, cfg = load_backbone(args.config, args.checkpoint, device)
    print(f"  Dataset: {len(dataset)} patches")

    # ── 加载标注 ──
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
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

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

    patch_infos = []
    for pid, changes in patch_changes.items():
        if pid not in dataset.patches:
            continue
        patch_infos.append((pid, dataset.patches.index(pid), patch_bounds[pid], changes))

    print(f"  有效 patch: {len(patch_infos)}")

    # ── 提取所有 embedding ──
    print(f"\n[3/4] 提取所有 patch 的 embedding...")
    start = time.time()
    all_data = extract_all_data(model, dataset, extract_fn, device, patch_infos)
    print(f"  成功提取 {len(all_data)} 个 patch，耗时 {time.time()-start:.1f}s")

    # ── 5-fold CV ──
    print(f"\n[4/4] {args.folds}-fold CV 训练 CD Head...")
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n--- Fold {fold_idx+1}/{args.folds} ---")
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        cd_head = SimpleCDHead(embedding_dim=cfg.model.embedding_dim, hidden_dim=64)
        n_params = sum(p.numel() for p in cd_head.parameters())
        print(f"  CD Head params: {n_params:,}")

        best_auc, best_state = train_fold(cd_head, train_data, val_data, device, epochs=args.epochs, lr=args.lr)
        print(f"  Fold {fold_idx+1} Best Val AUC: {best_auc:.4f}")
        fold_results.append(best_auc)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("  5-fold CV 结果汇总")
    print("=" * 70)
    for i, auc in enumerate(fold_results):
        print(f"  Fold {i+1}: AUC = {auc:.4f}")
    print(f"\n  Mean AUC:   {np.mean(fold_results):.4f}")
    print(f"  Std AUC:    {np.std(fold_results):.4f}")
    print(f"  Min/Max:    {np.min(fold_results):.4f} / {np.max(fold_results):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
