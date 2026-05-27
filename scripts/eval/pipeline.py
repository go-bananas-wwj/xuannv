#!/usr/bin/env python3
"""标准化评估流水线 — 一站式评估入口。

Phase 1: 预计算 Embedding (before/after)
Phase 2: Embedding 质量分析 (RankMe, Uniformity, ActiveDims)
Phase 3: 变化检测 Few-Shot 评估 (Bare AUC + CD Head)
Phase 4: 语义分割 Few-Shot 评估 (WorldCover + Dynamic World)
Phase 5: 二值分割 Few-Shot 评估 (JRC Water + OSM Buildings)

用法:
    python scripts/eval/full_evaluation_pipeline.py \
        --config configs/round8_single_exp1.yaml \
        --checkpoint /workspace/outputs/round8_single_exp1/epoch_19.pt \
        --output /workspace/outputs/round8_single_exp1/eval_results.json \
        --device npu:0
"""
from __future__ import annotations

import sys, json, time, argparse, warnings, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score, roc_auc_score
from sklearn.model_selection import KFold
import rasterio

warnings.filterwarnings('ignore')

# ── 固定配置 ──
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # ⚠️ 已过时，标注为2025年变化
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)   # ⚠️ 已过时，标注为2025年变化
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

# ── 类别映射 ──
from src.data.transforms import WC_CLASS_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="训练配置 YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--device", default="npu:0", help="NPU 设备")
    parser.add_argument("--precomputed", default=None, help="预计算 embedding 路径 (.pt)")
    parser.add_argument("--skip-embedding", action="store_true", help="跳过 embedding 提取（已有预计算文件）")
    parser.add_argument("--cd-epochs", type=int, default=50, help="CD Head 训练 epoch")
    parser.add_argument("--sem-folds", type=int, default=3, help="语义分割 K-Fold 数")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: 加载模型 + 预计算 Embedding
# ═══════════════════════════════════════════════════════════════════════

def load_backbone(config_path: str, checkpoint_path: str, device: str):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.inference.engine import extract_embedding_map

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
    return model, dataset, extract_embedding_map, cfg


def precompute_embeddings(model, dataset, extract_fn, device, patch_ids, output_path):
    """预计算所有 patch 的 before/after embedding，保存为 .pt."""
    print("\n[Phase 1] 预计算 Embedding maps...")
    results = {}
    t0 = time.time()
    for i, pid in enumerate(patch_ids):
        try:
            idx = dataset.patches.index(pid)
            eb = extract_fn(model, dataset, idx, BEFORE_WINDOW[0], BEFORE_WINDOW[1], device, normalize=True)
            ea = extract_fn(model, dataset, idx, AFTER_WINDOW[0], AFTER_WINDOW[1], device, normalize=True)
            results[pid] = {
                "eb": torch.from_numpy(eb).float(),
                "ea": torch.from_numpy(ea).float(),
            }
        except Exception as e:
            print(f"  跳过 {pid}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(patch_ids)}... ({time.time()-t0:.1f}s)")
    print(f"  完成: {len(results)} patches ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(results, output_path)
    print(f"  已保存: {output_path}")
    return results


def load_precomputed(path):
    print(f"\n[Phase 1] 加载预计算 Embedding: {path}")
    data = torch.load(path, weights_only=False)
    print(f"  加载: {len(data)} patches")
    return data


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Embedding 质量分析
# ═══════════════════════════════════════════════════════════════════════

def compute_rankme(embeddings):
    """RankMe: SVD 特征值熵."""
    N, D = embeddings.shape
    if N < 2 or D < 2:
        return 0.0
    _, s, _ = np.linalg.svd(embeddings, full_matrices=False)
    s_norm = s / (s.sum() + 1e-8)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    return float(entropy / np.log(min(N, D)))


def compute_stable_rank(embeddings):
    """Stable Rank: 奇异值平方和 / 最大奇异值平方."""
    _, s, _ = np.linalg.svd(embeddings, full_matrices=False)
    return float((s ** 2).sum() / (s[0] ** 2 + 1e-8))


def compute_active_dims(embeddings, thresholds=[0.05, 0.10, 0.15, 0.20]):
    """计算各阈值下的活跃维度数."""
    stds = embeddings.std(axis=0)
    max_std = stds.max()
    return {f"active_dims_t{t}": int((stds >= t * max_std).sum()) for t in thresholds}


def compute_uniformity_metrics(eb_list, ea_list):
    """计算 uniformity 相关指标 (基于 global mean embedding)."""
    from src.training.losses import raw_uniformity_loss, directional_uniformity_loss

    # 对每个 patch 做 global mean pooling，然后堆叠成 [N, D]
    global_embs = []
    for emb in eb_list + ea_list:
        global_embs.append(emb.mean(axis=(1, 2)))
    all_e = torch.from_numpy(np.stack(global_embs)).float()

    # L2 normalize
    all_e_norm = F.normalize(all_e, p=2, dim=1)

    raw_unif = raw_uniformity_loss(all_e_norm).item()
    dir_unif = directional_uniformity_loss(all_e).item()

    # 计算 mean pairwise cosine similarity
    cos_sim_matrix = all_e_norm @ all_e_norm.T
    mean_cos_sim = cos_sim_matrix.mean().item()

    return {
        "raw_uniformity": raw_unif,
        "directional_uniformity": dir_unif,
        "mean_cosine_similarity": mean_cos_sim,
    }


def compute_temporal_discriminability(eb_list, ea_list):
    """计算时间可区分性: 同 patch 前后距离 vs 跨 patch 距离."""
    temporal_dists = []
    for eb, ea in zip(eb_list, ea_list):
        # global mean pooling
        eb_vec = eb.mean(axis=(1, 2))
        ea_vec = ea.mean(axis=(1, 2))
        # cosine distance
        cos_sim = np.dot(eb_vec, ea_vec) / (np.linalg.norm(eb_vec) * np.linalg.norm(ea_vec) + 1e-8)
        temporal_dists.append((1 - cos_sim) / 2.0)

    # 跨 patch 距离
    cross_dists = []
    n = len(eb_list)
    for i in range(min(n, 100)):
        for j in range(i + 1, min(i + 10, n)):
            e1 = eb_list[i].mean(axis=(1, 2))
            e2 = eb_list[j].mean(axis=(1, 2))
            cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            cross_dists.append((1 - cos_sim) / 2.0)

    td_score = np.mean(temporal_dists) / (np.mean(cross_dists) + 1e-8)
    return {
        "temporal_discriminability": float(td_score),
        "mean_temporal_distance": float(np.mean(temporal_dists)),
        "mean_cross_distance": float(np.mean(cross_dists)),
    }


def evaluate_embedding_quality(precomputed):
    print("\n[Phase 2] Embedding 质量分析...")
    t0 = time.time()

    pids = sorted(precomputed.keys())
    eb_list = [precomputed[pid]["eb"].numpy() for pid in pids]
    ea_list = [precomputed[pid]["ea"].numpy() for pid in pids]

    # 全局 embedding 矩阵 [N, D] (mean pool over spatial)
    global_embs = []
    for eb, ea in zip(eb_list, ea_list):
        global_embs.append(eb.mean(axis=(1, 2)))
        global_embs.append(ea.mean(axis=(1, 2)))
    embs_matrix = np.stack(global_embs)

    results = {
        "n_patches": len(pids),
        "embedding_dim": eb_list[0].shape[0],
        "spatial_size": eb_list[0].shape[1:],
        "rankme": compute_rankme(embs_matrix),
        "stable_rank": compute_stable_rank(embs_matrix),
    }
    results.update(compute_active_dims(embs_matrix))
    results.update(compute_uniformity_metrics(eb_list, ea_list))
    results.update(compute_temporal_discriminability(eb_list, ea_list))

    print(f"  RankMe: {results['rankme']:.4f}")
    print(f"  StableRank: {results['stable_rank']:.2f}")
    print(f"  RawUnif: {results['raw_uniformity']:.4f}")
    print(f"  TemporalDisc: {results['temporal_discriminability']:.4f}")
    print(f"  ({time.time()-t0:.1f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: 变化检测评估
# ═══════════════════════════════════════════════════════════════════════

def load_annotations():
    """加载变化检测标注."""
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
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                if row.geometry is not None:
                    all_changes.append(row.geometry)
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

    # 按 patch 分组
    patch_changes = {}
    for geom in all_changes:
        pt = geom.centroid
        for pid, bounds in patch_bounds.items():
            if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                patch_changes.setdefault(pid, []).append(geom)
                break

    return patch_bounds, patch_changes


def rasterize_changes(changes, bounds, H=64, W=64):
    """将变化图斑光栅化为 [H, W] mask."""
    resolution = (bounds[2] - bounds[0]) / W
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


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


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(pred.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def train_cd_head_on_data(head, train_data, device, epochs=50, lr=1e-3):
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    for epoch in range(epochs):
        head.train()
        np.random.shuffle(train_data)
        for item in train_data:
            eb = item["eb"].unsqueeze(0).to(device)
            ea = item["ea"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).unsqueeze(0).to(device)
            pred = head(eb, ea)
            loss = F.binary_cross_entropy_with_logits(pred, mask) + dice_loss(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return head


def evaluate_change_detection(precomputed, patch_bounds, patch_changes, device, cd_epochs=50):
    print("\n[Phase 3] 变化检测评估...")
    t0 = time.time()

    results = {"bare_auc": {}, "cdhead_auc": {}}

    # 准备有标注的 patch 数据
    test_patches = []
    for pid, changes in patch_changes.items():
        if pid in precomputed and len(changes) > 0:
            test_patches.append(pid)

    print(f"  有标注的 patches: {len(test_patches)}")
    if len(test_patches) < 5:
        print("  标注 patch 太少，跳过变化检测")
        return results

    # ── 3a: Bare AUC ──
    bare_aucs = []
    for pid in test_patches:
        eb = precomputed[pid]["eb"].numpy()
        ea = precomputed[pid]["ea"].numpy()

        # cosine distance map
        D, H, W = eb.shape
        fb = eb.reshape(D, -1)
        fa = ea.reshape(D, -1)
        fb = fb / np.maximum(np.linalg.norm(fb, axis=0, keepdims=True), 1e-8)
        fa = fa / np.maximum(np.linalg.norm(fa, axis=0, keepdims=True), 1e-8)
        cos_sim = np.sum(fb * fa, axis=0)
        cd = ((1.0 - cos_sim) / 2.0).reshape(H, W)

        # 光栅化标注
        bounds = patch_bounds[pid]
        mask = rasterize_changes(patch_changes[pid], bounds, H, W)

        flat_cd = cd.flatten()
        flat_mask = mask.flatten()
        if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
            auc = roc_auc_score(flat_mask, flat_cd)
            bare_aucs.append(auc)

    if bare_aucs:
        results["bare_auc"] = {
            "mean": float(np.mean(bare_aucs)),
            "median": float(np.median(bare_aucs)),
            "std": float(np.std(bare_aucs)),
            "min": float(np.min(bare_aucs)),
            "max": float(np.max(bare_aucs)),
            "n_patches": len(bare_aucs),
        }
        print(f"  Bare AUC: {results['bare_auc']['mean']:.4f} (±{results['bare_auc']['std']:.3f})")

    # ── 3b: Few-Shot CD Head ──
    # 准备数据
    data_list = []
    for pid in test_patches:
        data_list.append({
            "pid": pid,
            "eb": precomputed[pid]["eb"],
            "ea": precomputed[pid]["ea"],
            "mask": torch.from_numpy(rasterize_changes(patch_changes[pid], patch_bounds[pid], 64, 64)).float(),
        })

    for k_shot in [1, 5, 10, 20]:
        if k_shot >= len(data_list):
            continue
        k_aucs = []
        n_splits = min(5, len(data_list) // max(k_shot, 1))
        for split in range(n_splits):
            np.random.seed(42 + split)
            indices = np.random.permutation(len(data_list))
            train_idx = indices[:k_shot]
            test_idx = indices[k_shot:]

            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]

            emb_dim = data_list[0]["eb"].shape[0]
            head = SimpleCDHead(embedding_dim=emb_dim, hidden_dim=64)
            head = train_cd_head_on_data(head, train_data, device, epochs=cd_epochs)

            # 评估
            head.eval()
            all_preds, all_masks = [], []
            with torch.no_grad():
                for item in test_data:
                    eb = item["eb"].unsqueeze(0).to(device)
                    ea = item["ea"].unsqueeze(0).to(device)
                    pred = torch.sigmoid(head(eb, ea)).cpu().numpy().flatten()
                    all_preds.extend(pred.tolist())
                    all_masks.extend(item["mask"].flatten().tolist())

            preds = np.array(all_preds)
            masks = np.array(all_masks)
            if len(np.unique(masks)) > 1:
                auc = roc_auc_score(masks, preds)
                k_aucs.append(auc)

        if k_aucs:
            results["cdhead_auc"][f"k{k_shot}"] = {
                "mean": float(np.mean(k_aucs)),
                "std": float(np.std(k_aucs)),
                "n_splits": len(k_aucs),
            }
            print(f"  CDHead K={k_shot}: {results['cdhead_auc'][f'k{k_shot}']['mean']:.4f}")

    print(f"  ({time.time()-t0:.1f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: 语义分割 + 二值分割评估
# ═══════════════════════════════════════════════════════════════════════

def load_label_direct(patch_id, label_type):
    """直接读取标签文件."""
    if label_type == "worldcover":
        paths = [
            f"/workspace/raw/harbin_scenes/worldcover/{patch_id}/static.tif",
            f"/workspace/raw/harbin/worldcover/{patch_id}/static.tif",
        ]
        for path in paths:
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                if data is not None and data.size > 0:
                    mapped = np.full_like(data, -1, dtype=np.int64)
                    for val, idx in WC_CLASS_MAP.items():
                        mapped[data == val] = idx
                    return mapped
            except Exception:
                continue
        return None

    elif label_type == "dynamic_world":
        # Dynamic World 是时序数据，使用第一个可用季度文件作为静态标签
        base = f"/workspace/raw/harbin/dynamic_world/{patch_id}"
        try:
            files = sorted([f for f in os.listdir(base) if f.endswith('.tif')])
            if files:
                with rasterio.open(os.path.join(base, files[0])) as src:
                    data = src.read(1)
                if data is not None and data.size > 0:
                    # Dynamic World 0=NoData，1-8=有效类别
                    data = data.astype(np.int64)
                    data[data == 0] = -1  # 标记为无效
                    return data
        except Exception:
            pass
        return None

    elif label_type == "jrc_water":
        paths = [
            f"/workspace/raw/harbin_scenes/jrc_water/{patch_id}/static.tif",
            f"/workspace/raw/harbin/jrc_water/{patch_id}/static.tif",
        ]
        for path in paths:
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                if data is not None and data.size > 0:
                    data = data.astype(np.int64)
                    data[data == -128] = -1  # 标记 nodata 为无效
                    return data
            except Exception:
                continue
        return None

    elif label_type == "osm_buildings":
        path = f"/workspace/raw/harbin_scenes/osm_buildings/{patch_id}/static.tif"
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
            if data is not None and data.size > 0:
                return (data > 0).astype(np.int64)
        except Exception:
            pass
        return None

    return None


def resize_label(label, target_h, target_w):
    label_t = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(label_t, size=(target_h, target_w), mode='nearest')[0, 0]
    return resized.numpy().astype(np.int64)


def prepare_semantic_data(emb_maps, labels, num_classes):
    X_list, y_list = [], []
    for emb, label in zip(emb_maps, labels):
        D, H, W = emb.shape
        emb_flat = emb.reshape(D, -1).T
        label_flat = label.reshape(-1)
        valid_mask = (label_flat >= 0) & (label_flat < num_classes)
        if valid_mask.sum() == 0:
            continue
        X_list.append(emb_flat[valid_mask])
        y_list.append(label_flat[valid_mask])
    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)


def evaluate_semantic_task(X, y, n_folds=3):
    present_classes = np.unique(y)
    n_classes = len(present_classes)
    if n_classes < 2:
        return {"error": f"only {n_classes} class"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bacc_scores, f1m_scores, f1w_scores, miou_scores = [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        max_train = 30000
        if len(y_train) > max_train:
            indices = np.random.choice(len(y_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        clf = LogisticRegression(max_iter=300, multi_class='multinomial', solver='lbfgs', n_jobs=4, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro', labels=present_classes, zero_division=0)
        f1w = f1_score(y_test, y_pred, average='weighted', labels=present_classes, zero_division=0)

        ious = []
        for c in present_classes:
            inter = ((y_pred == c) & (y_test == c)).sum()
            union = ((y_pred == c) | (y_test == c)).sum()
            ious.append(inter / max(union, 1))
        miou = np.mean(ious)

        bacc_scores.append(bacc)
        f1m_scores.append(f1m)
        f1w_scores.append(f1w)
        miou_scores.append(miou)

    return {
        "balanced_accuracy": float(np.mean(bacc_scores)),
        "f1_macro": float(np.mean(f1m_scores)),
        "f1_weighted": float(np.mean(f1w_scores)),
        "miou": float(np.mean(miou_scores)),
        "n_classes": int(n_classes),
        "n_pixels": int(len(y)),
    }


def evaluate_binary_task(X, y, n_folds=3):
    if len(np.unique(y)) < 2:
        return {"error": "only 1 class"}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bacc_scores, f1_scores, iou_scores, auc_scores = [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        max_train = 30000
        if len(y_train) > max_train:
            indices = np.random.choice(len(y_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        clf = LogisticRegression(max_iter=300, n_jobs=4, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        inter = ((y_pred == 1) & (y_test == 1)).sum()
        union = ((y_pred == 1) | (y_test == 1)).sum()
        iou = inter / max(union, 1)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        bacc_scores.append(bacc)
        f1_scores.append(f1)
        iou_scores.append(iou)
        auc_scores.append(auc)

    return {
        "balanced_accuracy": float(np.mean(bacc_scores)),
        "f1": float(np.mean(f1_scores)),
        "iou": float(np.mean(iou_scores)),
        "auc": float(np.mean(auc_scores)),
        "n_pixels": int(len(y)),
    }


def evaluate_downstream_tasks(precomputed, n_folds=3):
    print("\n[Phase 4/5] 下游任务评估...")
    t0 = time.time()

    pids = sorted(precomputed.keys())
    emb_maps = [precomputed[pid]["eb"].numpy() for pid in pids]
    D, H, W = emb_maps[0].shape
    results = {}

    # WorldCover
    print("  WorldCover...")
    wc_labels = [load_label_direct(pid, "worldcover") for pid in pids]
    valid = [(e, l) for e, l in zip(emb_maps, wc_labels) if l is not None]
    if valid:
        X, y = prepare_semantic_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 11)
        if X is not None:
            results["worldcover"] = evaluate_semantic_task(X, y, n_folds)
            print(f"    mIoU={results['worldcover']['miou']:.4f}, BAcc={results['worldcover']['balanced_accuracy']:.4f}")
        else:
            results["worldcover"] = {"error": "no valid data"}
    else:
        results["worldcover"] = {"error": "no labels"}

    # Dynamic World
    print("  Dynamic World...")
    dw_labels = [load_label_direct(pid, "dynamic_world") for pid in pids]
    valid = [(e, l) for e, l in zip(emb_maps, dw_labels) if l is not None]
    if valid:
        X, y = prepare_semantic_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 9)
        if X is not None:
            results["dynamic_world"] = evaluate_semantic_task(X, y, n_folds)
            print(f"    mIoU={results['dynamic_world']['miou']:.4f}, BAcc={results['dynamic_world']['balanced_accuracy']:.4f}")
        else:
            results["dynamic_world"] = {"error": "no valid data"}
    else:
        results["dynamic_world"] = {"error": "no labels"}

    # JRC Water
    print("  JRC Water...")
    jrc_labels = [load_label_direct(pid, "jrc_water") for pid in pids]
    valid = [(e, l) for e, l in zip(emb_maps, jrc_labels) if l is not None]
    if valid:
        X, y = prepare_semantic_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 2)
        if X is not None:
            y = (y > 0).astype(np.int64)
            results["jrc_water"] = evaluate_binary_task(X, y, n_folds)
            print(f"    IoU={results['jrc_water']['iou']:.4f}, F1={results['jrc_water']['f1']:.4f}, AUC={results['jrc_water']['auc']:.4f}")
        else:
            results["jrc_water"] = {"error": "no valid data"}
    else:
        results["jrc_water"] = {"error": "no labels"}

    # OSM Buildings
    print("  OSM Buildings...")
    osm_labels = [load_label_direct(pid, "osm_buildings") for pid in pids]
    valid = [(e, l) for e, l in zip(emb_maps, osm_labels) if l is not None]
    if valid:
        X, y = prepare_semantic_data([p[0] for p in valid], [resize_label(p[1], H, W) for p in valid], 2)
        if X is not None:
            results["osm_buildings"] = evaluate_binary_task(X, y, n_folds)
            print(f"    IoU={results['osm_buildings']['iou']:.4f}, F1={results['osm_buildings']['f1']:.4f}, AUC={results['osm_buildings']['auc']:.4f}")
        else:
            results["osm_buildings"] = {"error": "no valid data"}
    else:
        results["osm_buildings"] = {"error": "no labels"}

    print(f"  ({time.time()-t0:.1f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    print("=" * 70)
    print(f"  Round 8+ 标准化评估流水线")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    device = args.device
    if device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(device)

    # ── Phase 1: 获取 Embedding ──
    if args.precomputed and os.path.exists(args.precomputed):
        precomputed = load_precomputed(args.precomputed)
    elif args.skip_embedding:
        print("错误: --skip-embedding 需要配合 --precomputed")
        return
    else:
        model, dataset, extract_fn, cfg = load_backbone(args.config, args.checkpoint, device)
        patch_ids = dataset.patches
        precomputed_path = args.precomputed or os.path.join(
            os.path.dirname(args.checkpoint), "precomputed_embeddings.pt"
        )
        precomputed = precompute_embeddings(model, dataset, extract_fn, device, patch_ids, precomputed_path)

    pids = sorted(precomputed.keys())
    print(f"\n  总 patches: {len(pids)}")

    # ── Phase 2: Embedding 质量 ──
    quality_results = evaluate_embedding_quality(precomputed)

    # ── Phase 3: 变化检测 ──
    patch_bounds, patch_changes = load_annotations()
    cd_results = evaluate_change_detection(precomputed, patch_bounds, patch_changes, device, args.cd_epochs)

    # ── Phase 4/5: 下游任务 ──
    downstream_results = evaluate_downstream_tasks(precomputed, args.sem_folds)

    # ── 汇总 ──
    final_results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "evaluated_at": datetime.now().isoformat(),
        "n_patches": len(pids),
        "embedding": quality_results,
        "change_detection": cd_results,
        "downstream": downstream_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"  评估完成! 结果已保存: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
