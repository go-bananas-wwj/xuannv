#!/usr/bin/env python3
"""V4 Official 变化检测验证 — 在光学标注上计算 AUC."""
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
CONFIG_PATH = "configs/qwen_v4_official.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v4_official/epoch_best_epoch231.pt"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

# 时间窗口 (与 V2 对齐)
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)    # 2024Q3-2025Q4

print("="*60)
print("  V4 Official 变化检测验证")
print("="*60)
print(f"  Checkpoint: {CKPT_PATH}")
print("="*60)

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
from src.inference.engine import load_backbone, extract_embedding_map

device = torch.device("npu:0")
model, dataset, cfg = load_backbone(CONFIG_PATH, CKPT_PATH, device=device)

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


def compute_change_map(emb_before, emb_after):
    """计算 cosine distance change map."""
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
results_v4 = []

for i, (pid, changes) in enumerate(test_patches):
    if pid not in dataset.patches:
        continue

    pidx = dataset.patches.index(pid)
    print(f"  [{i+1}/{len(test_patches)}] {pid}... ", end="", flush=True)
    t0 = time.time()

    try:
        eb = extract_embedding_map(model, dataset, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1], device, normalize=True)
        ea = extract_embedding_map(model, dataset, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1], device, normalize=True)
        cd_v4 = compute_change_map(eb, ea)

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
        flat_v4 = cd_v4.flatten()
        flat_mask = change_mask.flatten()

        if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
            auc_v4 = roc_auc_score(flat_mask, flat_v4)
        else:
            auc_v4 = None

        elapsed = time.time() - t0
        print(f"AUC={'%.3f' % auc_v4 if auc_v4 else 'N/A'}, "
              f"dist_mean={cd_v4.mean():.4f}, dist_std={cd_v4.std():.4f} ({elapsed:.1f}s)")

        if auc_v4 is not None:
            results_v4.append(auc_v4)

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

if results_v4:
    print(f"\nV4 Official: {len(results_v4)} patches, AUC mean={np.mean(results_v4):.3f}, "
          f"median={np.median(results_v4):.3f}, std={np.std(results_v4):.3f}")
    print(f"  AUC > 0.6: {sum(1 for a in results_v4 if a > 0.6)}/{len(results_v4)}")
    print(f"  AUC > 0.7: {sum(1 for a in results_v4 if a > 0.7)}/{len(results_v4)}")
    print(f"  AUC > 0.8: {sum(1 for a in results_v4 if a > 0.8)}/{len(results_v4)}")
else:
    print("\n无有效结果")

print("\n" + "="*60)
print("验证完成")
print("="*60)
