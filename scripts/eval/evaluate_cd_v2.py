#!/usr/bin/env python
"""变化检测 AUC 评估 v2 — 使用预计算 embedding。

评估方法:
1. Cosine Distance (patch-level baseline)
2. Linear Discriminator (LogisticRegression on concat(before, after))

用法:
    python evaluate_cd_v2.py \
        --embedding-file /path/to/patch_embeddings.npz \
        --output-dir /path/to/evaluation/change_detection \
        --annot-dir /workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import box
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/xuannv")

ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")

PERIODS = {
    "apr_jun": {"before": 4, "after": 6, "shp": "june.shp", "n_annot": 38},
    "jun_aug": {"before": 6, "after": 8, "shp": "aug.shp", "n_annot": 18},
    "aug_sep": {"before": 8, "after": 9, "shp": "September.shp", "n_annot": 25},
    "sep_oct": {"before": 9, "after": 10, "shp": "October.shp", "n_annot": 24},
}


def load_patch_bounds():
    """加载 patch 边界."""
    gdf = gpd.read_file(GRID_PATH)
    patch_bounds = {}
    for _, row in gdf.iterrows():
        pid = row.get("sample_id") or row.get("patch_id") or row.get("id")
        if pid is None:
            continue
        coords = list(row.geometry.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return patch_bounds


def load_annotations(period_info, patch_bounds):
    """加载某个时期的变化标注，返回 {patch_id: has_change(bool)}."""
    shp_path = ANNOT_DIR / period_info["shp"]
    if not shp_path.exists():
        print(f"  Warning: {shp_path} not found")
        return {}
    
    gdf = gpd.read_file(shp_path)
    # CRS fix (from AGENTS.md bug fix)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    
    # 找出哪些 patch 包含变化标注
    changed_pids = set()
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        for pid, bounds in patch_bounds.items():
            if box(*bounds).intersects(geom):
                changed_pids.add(pid)
                break
    
    return changed_pids


def evaluate_period(global_mean, patch_ids, period_info, changed_pids, patch_bounds):
    """评估单个时期.
    
    Returns:
        dict with cosine_auc, lr_auc, n_positive, n_negative
    """
    # embedding 月份从 2025-04 开始，index offset = 4
    before_month = period_info["before"] - 4
    after_month = period_info["after"] - 4
    
    # 构建特征和标签
    X_cosine = []   # cosine distance
    X_lr = []       # concat(before, after)
    y = []
    pid_list = []
    
    for p_idx, pid in enumerate(patch_ids):
        pid_str = str(pid)
        # 处理多区域 patch_id 格式（如 harbin_patch_000000 -> patch_000000）
        local_pid = pid_str.split('_', 1)[1] if '_' in pid_str and not pid_str.startswith('patch_') else pid_str
        if local_pid not in patch_bounds:
            continue
        
        emb_before = global_mean[p_idx, before_month]  # [D]
        emb_after = global_mean[p_idx, after_month]    # [D]
        
        # Cosine distance
        cos_sim = np.dot(emb_before, emb_after) / (np.linalg.norm(emb_before) * np.linalg.norm(emb_after) + 1e-8)
        cos_dist = 1.0 - cos_sim
        
        # LR feature
        feat = np.concatenate([emb_before, emb_after])  # [2D]
        
        label = 1 if local_pid in changed_pids else 0
        
        X_cosine.append(cos_dist)
        X_lr.append(feat)
        y.append(label)
        pid_list.append(pid_str)
    
    X_cosine = np.array(X_cosine)
    X_lr = np.array(X_lr)
    y = np.array(y)
    
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        print(f"  警告: {period_info} 正样本={n_pos}, 负样本={n_neg}, 无法计算 AUC")
        return {"cosine_auc": None, "lr_auc": None, "n_positive": int(n_pos), "n_negative": int(n_neg)}
    
    # Cosine AUC
    cosine_auc = roc_auc_score(y, X_cosine)
    
    # Linear Discriminator AUC (5-fold CV)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_scores = []
    
    for train_idx, test_idx in skf.split(X_lr, y):
        X_tr, X_te = X_lr[train_idx], X_lr[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        lr_scores.append(roc_auc_score(y_te, proba))
    
    lr_auc = float(np.mean(lr_scores))
    
    # 训练全量 LR 模型保存
    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_full.fit(X_lr, y)
    
    return {
        "cosine_auc": float(cosine_auc),
        "lr_auc": float(lr_auc),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "lr_model": clf_full,
        "pids": pid_list,
        "labels": y.tolist(),
        "cosine_scores": X_cosine.tolist(),
    }


def main():
    global ANNOT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--annot-dir", type=str, default=str(ANNOT_DIR))
    args = p.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR = Path(args.annot_dir)
    
    # 加载 embedding
    print("[CD] 加载 embedding...")
    data = np.load(args.embedding_file)
    if "global_mean" in data:
        global_mean = data["global_mean"]  # [N, M, D]
    else:
        # 从 spatial_maps 计算 global_mean
        spatial_maps = data["spatial_maps"]  # [N, M, D, H, W]
        global_mean = spatial_maps.mean(axis=(3, 4))  # [N, M, D]
        print(f"      从 spatial_maps 计算 global_mean")
    patch_ids = data["patch_ids"]      # [N]
    print(f"      形状: {global_mean.shape}")
    
    # 加载 patch 边界
    print("[CD] 加载 patch 边界...")
    patch_bounds = load_patch_bounds()
    print(f"      {len(patch_bounds)} 个 patch")
    
    # 评估每个时期
    all_results = {}
    weighted_auc_cosine = 0.0
    weighted_auc_lr = 0.0
    total_n = 0
    
    for period_name, period_info in PERIODS.items():
        print(f"[CD] 评估时期: {period_name} ({period_info['shp']})...")
        changed_pids = load_annotations(period_info, patch_bounds)
        print(f"      变化 patch: {len(changed_pids)} 个")
        
        result = evaluate_period(global_mean, patch_ids, period_info, changed_pids, patch_bounds)
        all_results[period_name] = {
            "shp": period_info["shp"],
            "before_month": period_info["before"],
            "after_month": period_info["after"],
            "n_positive": result["n_positive"],
            "n_negative": result["n_negative"],
            "cosine_auc": result.get("cosine_auc"),
            "lr_auc": result.get("lr_auc"),
        }
        
        if result["cosine_auc"] is not None:
            n_total = result["n_positive"] + result["n_negative"]
            weighted_auc_cosine += result["cosine_auc"] * n_total
            weighted_auc_lr += result["lr_auc"] * n_total
            total_n += n_total
            
            # 保存 LR 模型
            import pickle
            with open(output_dir / f"linear_discriminator_{period_name}.pkl", "wb") as f:
                pickle.dump(result["lr_model"], f)
        
        print(f"      Cosine AUC={result.get('cosine_auc', 'N/A')}, LR AUC={result.get('lr_auc', 'N/A')}")
    
    # 加权平均
    summary = {
        "weighted_cosine_auc": float(weighted_auc_cosine / total_n) if total_n > 0 else None,
        "weighted_lr_auc": float(weighted_auc_lr / total_n) if total_n > 0 else None,
        "total_patches_evaluated": total_n,
        "periods": all_results,
    }
    
    with open(output_dir / "auc_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n[CD] 完成!")
    print(f"  加权平均 Cosine AUC: {summary['weighted_cosine_auc']:.4f}" if summary['weighted_cosine_auc'] else "  N/A")
    print(f"  加权平均 LR AUC: {summary['weighted_lr_auc']:.4f}" if summary['weighted_lr_auc'] else "  N/A")


if __name__ == "__main__":
    main()
