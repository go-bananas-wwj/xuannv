#!/usr/bin/env python3
"""V5 Level 1: Backbone bare AUC on 20 annotated patches."""
import sys, json, time
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v5_mixed_scale.yaml"
CKPT_V5 = "/workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
DEVICE = "npu:0"  # GPU 6

# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)   # 2024Q3-Q4

print("="*60)
print("  V5 Level 1: Backbone Bare AUC (20 annotated patches)")
print(f"  Checkpoint: {CKPT_V5}")
print("="*60)

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

def load_model(ckpt_path):
    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset

print("\n加载 V5 模型...")
model_v5, dataset_v5 = load_model(CKPT_V5)

# ──────────────────────────────────────────
# 加载 Grid 和标注
# ──────────────────────────────────────────
print("\n加载 Grid...")
with open(GRID_PATH) as f:
    grid_data = json.load(f)

patch_bounds = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

print("加载标注数据...")
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

print(f"  总计: {len(all_changes)} polygons")

# ──────────────────────────────────────────
# 提取 embedding 并验证
# ──────────────────────────────────────────
def extract_emb(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(DEVICE)
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

# 找有标注的 patches
patch_to_changes = {}
for ch in all_changes:
    geom = ch["geometry"]
    for pid, bounds in patch_bounds.items():
        patch_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        if geom.intersects(patch_box):
            if pid not in patch_to_changes:
                patch_to_changes[pid] = []
            patch_to_changes[pid].append(ch)

test_patches = sorted(patch_to_changes.items(), 
                      key=lambda x: sum(c["geometry"].area for c in x[1]), 
                      reverse=True)[:20]

print(f"\n测试 {len(test_patches)} 个有标注的 patch")

# 验证
results_v5 = []
patch_details = []

for i, (pid, changes) in enumerate(test_patches):
    if pid not in dataset_v5.patches:
        continue
    
    pidx = dataset_v5.patches.index(pid)
    print(f"  [{i+1}/{len(test_patches)}] {pid}... ", end="", flush=True)
    t0 = time.time()
    
    try:
        eb_v5 = extract_emb(model_v5, dataset_v5, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
        ea_v5 = extract_emb(model_v5, dataset_v5, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1])
        cd_v5 = compute_change_map(eb_v5, ea_v5)
        
        # 光栅化标注
        bounds = patch_bounds[pid]
        H, W = 64, 64
        resolution = (bounds[2] - bounds[0]) / H
        change_mask = np.zeros((H, W), dtype=np.float32)
        
        for ch_info in changes:
            geom = ch_info["geometry"]
            for row in range(H):
                for col in range(W):
                    wx = bounds[0] + (col + 0.5) * resolution
                    wy = bounds[3] - (row + 0.5) * resolution
                    if geom.contains(Point(wx, wy)):
                        change_mask[row, col] = 1.0
        
        # AUC
        flat_v5 = cd_v5.flatten()
        flat_mask = change_mask.flatten()
        
        if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
            auc_v5 = roc_auc_score(flat_mask, flat_v5)
        else:
            auc_v5 = None
        
        elapsed = time.time() - t0
        
        # 统计
        n_changed = int(flat_mask.sum())
        dist_mean = float(cd_v5.mean())
        dist_changed = float(cd_v5[change_mask > 0].mean()) if n_changed > 0 else 0.0
        dist_unchanged = float(cd_v5[change_mask == 0].mean())
        
        print(f"AUC={auc_v5:.3f}, dist_mean={dist_mean:.4f}, "
              f"changed={dist_changed:.4f}, unchg={dist_unchanged:.4f} ({elapsed:.1f}s)")
        
        if auc_v5 is not None:
            results_v5.append(auc_v5)
            patch_details.append({
                "patch_id": pid,
                "auc": float(auc_v5),
                "dist_mean": dist_mean,
                "dist_changed": dist_changed,
                "dist_unchanged": dist_unchanged,
                "n_changed_pixels": n_changed,
                "n_total_pixels": H * W,
            })
            
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

# ──────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────
print("\n" + "="*60)
print("  V5 Level 1 结果汇总")
print("="*60)

if results_v5:
    print(f"\nV5 Backbone Bare: {len(results_v5)} patches")
    print(f"  AUC mean  = {np.mean(results_v5):.3f}")
    print(f"  AUC median= {np.median(results_v5):.3f}")
    print(f"  AUC std   = {np.std(results_v5):.3f}")
    print(f"  AUC min   = {np.min(results_v5):.3f}")
    print(f"  AUC max   = {np.max(results_v5):.3f}")
    print(f"  AUC > 0.5: {sum(1 for a in results_v5 if a > 0.5)}/{len(results_v5)}")
    print(f"  AUC > 0.6: {sum(1 for a in results_v5 if a > 0.6)}/{len(results_v5)}")
    print(f"  AUC > 0.7: {sum(1 for a in results_v5 if a > 0.7)}/{len(results_v5)}")
    
    print(f"\n  对比 V2 baseline (backbone bare AUC ≈ 0.49):")
    delta = np.mean(results_v5) - 0.49
    print(f"  V5 → V2 变化: {np.mean(results_v5):.3f} vs 0.49 ({'↑+' if delta > 0 else '↓'}{abs(delta):.3f})")

# 保存详细结果
import os
os.makedirs("/workspace/outputs/aef_qwen_v5_mixed_scale/eval", exist_ok=True)
out_path = "/workspace/outputs/aef_qwen_v5_mixed_scale/eval/level1_bare_auc.json"
import json as _json
with open(out_path, "w") as f:
    _json.dump({
        "model": "V5_mixed_scale_epoch161",
        "n_patches": len(results_v5),
        "auc_mean": float(np.mean(results_v5)) if results_v5 else None,
        "auc_median": float(np.median(results_v5)) if results_v5 else None,
        "auc_std": float(np.std(results_v5)) if results_v5 else None,
        "auc_min": float(np.min(results_v5)) if results_v5 else None,
        "auc_max": float(np.max(results_v5)) if results_v5 else None,
        "patch_details": patch_details,
    }, f, indent=2)
print(f"\n  结果已保存: {out_path}")

print("\n" + "="*60)
print("Level 1 验证完成")
print("="*60)
