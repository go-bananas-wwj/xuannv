#!/usr/bin/env python3
"""Compare Finetune vs From Scratch on change detection AUC."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

CKPT_FT = "/workspace/outputs/aef_qwen_v2_hr_finetune/epoch_19.pt"
CKPT_FS = "/workspace/outputs/aef_qwen_v2_hr_from_scratch/epoch_39.pt"

CONFIG_FT = "/workspace/xuannv/configs/qwen_v2_hr_finetune.yaml"
CONFIG_FS = "/workspace/xuannv/configs/qwen_v2_hr_from_scratch.yaml"

# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)    # 2024Q3-2025Q4

print("="*60)
print("  变化检测对比验证: Finetune vs From Scratch")
print("="*60)

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

def load_model(ckpt_path, config_path):
    cfg = load_config(config_path)
    model = AEFModel(cfg).to("npu:0")
    ckpt = torch.load(ckpt_path, map_location="npu:0", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset

print("\n加载 Finetune 模型...")
model_ft, ds_ft = load_model(CKPT_FT, CONFIG_FT)
print("加载 From Scratch 模型...")
model_fs, ds_fs = load_model(CKPT_FS, CONFIG_FS)

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
# 提取 embedding
# ──────────────────────────────────────────
def extract_emb(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to("npu:0")
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
    emb_map = output.embedding_map
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

results_ft = []
results_fs = []

for i, (pid, changes) in enumerate(test_patches):
    if pid not in ds_ft.patches or pid not in ds_fs.patches:
        continue
    
    pidx_ft = ds_ft.patches.index(pid)
    pidx_fs = ds_fs.patches.index(pid)
    print(f"  [{i+1}/{len(test_patches)}] {pid}... ", end="", flush=True)
    t0 = time.time()
    
    try:
        eb_ft = extract_emb(model_ft, ds_ft, pidx_ft, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
        ea_ft = extract_emb(model_ft, ds_ft, pidx_ft, AFTER_WINDOW[0], AFTER_WINDOW[1])
        cd_ft = compute_change_map(eb_ft, ea_ft)
        
        eb_fs = extract_emb(model_fs, ds_fs, pidx_fs, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
        ea_fs = extract_emb(model_fs, ds_fs, pidx_fs, AFTER_WINDOW[0], AFTER_WINDOW[1])
        cd_fs = compute_change_map(eb_fs, ea_fs)
        
        # 光栅化标注
        bounds = patch_bounds[pid]
        H, W = 64, 64
        resolution = (bounds[2] - bounds[0]) / H
        change_mask = np.zeros((H, W), dtype=np.float32)
        
        for ch_info in changes:
            geom = ch_info["geometry"]
            for px in range(H):
                for py in range(W):
                    wx = bounds[0] + (px + 0.5) * resolution
                    wy = bounds[3] - (py + 0.5) * resolution
                    if geom.contains(Point(wx, wy)):
                        change_mask[px, py] = 1.0
        
        flat_ft = cd_ft.flatten()
        flat_fs = cd_fs.flatten()
        flat_mask = change_mask.flatten()
        
        if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
            auc_ft = roc_auc_score(flat_mask, flat_ft)
            auc_fs = roc_auc_score(flat_mask, flat_fs)
        else:
            auc_ft = auc_fs = None
        
        elapsed = time.time() - t0
        print(f"FT AUC={'%.3f' % auc_ft if auc_ft else 'N/A'}, FS AUC={'%.3f' % auc_fs if auc_fs else 'N/A'}, "
              f"dist_ft={cd_ft.mean():.4f}, dist_fs={cd_fs.mean():.4f} ({elapsed:.1f}s)")
        
        if auc_ft is not None:
            results_ft.append(auc_ft)
        if auc_fs is not None:
            results_fs.append(auc_fs)
            
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

# ──────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────
print("\n" + "="*60)
print("  结果汇总")
print("="*60)

if results_ft:
    print(f"\nFinetune (epoch_59): {len(results_ft)} patches, AUC mean={np.mean(results_ft):.3f}, "
          f"median={np.median(results_ft):.3f}, std={np.std(results_ft):.3f}")
    print(f"  AUC > 0.6: {sum(1 for a in results_ft if a > 0.6)}/{len(results_ft)}")
    print(f"  AUC > 0.7: {sum(1 for a in results_ft if a > 0.7)}/{len(results_ft)}")

if results_fs:
    print(f"\nFrom Scratch (epoch_39): {len(results_fs)} patches, AUC mean={np.mean(results_fs):.3f}, "
          f"median={np.median(results_fs):.3f}, std={np.std(results_fs):.3f}")
    print(f"  AUC > 0.6: {sum(1 for a in results_fs if a > 0.6)}/{len(results_fs)}")
    print(f"  AUC > 0.7: {sum(1 for a in results_fs if a > 0.7)}/{len(results_fs)}")

if results_ft and results_fs:
    print(f"\nFinetune vs From Scratch: AUC {np.mean(results_ft):.3f} vs {np.mean(results_fs):.3f} "
          f"({'Finetune better' if np.mean(results_ft) > np.mean(results_fs) else 'From Scratch better'})")
    ft_better = sum(1 for ft, fs in zip(results_ft, results_fs) if ft > fs)
    print(f"  Finetune 优于 From Scratch 的 patch: {ft_better}/{len(results_fs)}")

print("\n" + "="*60)
print("验证完成")
print("="*60)
