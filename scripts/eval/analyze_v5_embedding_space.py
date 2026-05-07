#!/usr/bin/env python3
"""V5 Embedding 空间诊断分析 — 跨年度大窗口."""
import sys, json
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v5_mixed_scale.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt"
DEVICE = "npu:0"

# 时间窗口
BEFORE_WINDOW = (1672531200000.0, 1703980800000.0)  # 2023全年
AFTER_WINDOW = (1704067200000.0, 1735603200000.0)   # 2024全年

print("="*60)
print("  V5 Embedding 空间诊断分析")
print(f"  Checkpoint: {CKPT_PATH}")
print("="*60)

# 加载模型
print("\n加载 V5 模型...")
cfg = load_config(CONFIG_PATH)
model = AEFModel(cfg).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

cfg.data.preload = False
dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

# 加载 Grid
with open("/workspace/index/harbin/grid/harbin_grid.geojson") as f:
    grid_data = json.load(f)

patch_bounds = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

# 加载标注
print("加载标注数据...")
from shapely.geometry import box, Point
import geopandas as gpd

all_changes = []
for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
    try:
        gdf = gpd.read_file(f"/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件/{shp_name}")
        if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        for _, row in gdf.iterrows():
            if row.geometry is not None:
                all_changes.append({"geometry": row.geometry, "period": shp_name.replace(".shp", "")})
    except Exception as e:
        pass

print(f"  总计: {len(all_changes)} polygons")

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

# 提取 embedding
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
    emb_map = output.embedding_map
    emb_map = F.normalize(emb_map, p=2, dim=1)
    return emb_map[0].cpu().numpy()

all_embeddings_before = []
all_embeddings_after = []
all_pids = []
all_masks = []

for i, (pid, changes) in enumerate(test_patches):
    if pid not in dataset.patches:
        continue
    pidx = dataset.patches.index(pid)
    try:
        eb = extract_emb(model, dataset, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
        ea = extract_emb(model, dataset, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1])

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

        all_embeddings_before.append(eb)
        all_embeddings_after.append(ea)
        all_pids.append(pid)
        all_masks.append(change_mask)
    except Exception as e:
        print(f"  Skip {pid}: {e}")

print(f"\n成功提取 {len(all_pids)} 个 patch 的 embedding")

# ──────────────────────────────────────────
# 诊断指标计算
# ──────────────────────────────────────────
print("\n" + "="*60)
print("  诊断指标")
print("="*60)

# 1. 计算所有 patch 的 uniformity (基于 vector-level embedding)
vectors = []
for eb, ea in zip(all_embeddings_before, all_embeddings_after):
    v_b = eb.reshape(eb.shape[0], -1).mean(axis=1)
    v_a = ea.reshape(ea.shape[0], -1).mean(axis=1)
    vectors.append(v_b)
    vectors.append(v_a)

vectors = np.stack(vectors, axis=0)  # [2N, D]
vectors_t = torch.from_numpy(vectors).float()

# Uniformity (参考 raw_uniformity_loss)
pairwise_dist = torch.cdist(vectors_t, vectors_t, p=2)
uniformity = torch.log(torch.mean(torch.exp(-2 * pairwise_dist))).item()
print(f"\n1. Uniformity (vector-level): {uniformity:.4f}")
print(f"   参考: V4 基线 ≈ -3.04, 理想值 <-3.5")

# 2. Mean Pairwise Cosine Distance
cos_sims = []
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        cos_sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8)
        cos_sims.append(cos_sim)
mean_cos_sim = np.mean(cos_sims)
mean_cos_dist = (1 - mean_cos_sim) / 2
print(f"\n2. Mean Pairwise Cosine Similarity: {mean_cos_sim:.4f}")
print(f"   Mean Pairwise Cosine Distance: {mean_cos_dist:.4f}")
print(f"   参考: V4 基线 distance ≈ 0.50")

# 3. Changed vs Unchanged Distance 分布
changed_dists = []
unchanged_dists = []
auc_per_patch = []

for eb, ea, mask in zip(all_embeddings_before, all_embeddings_after, all_masks):
    D, H, W = eb.shape
    fb = eb.reshape(D, -1)
    fa = ea.reshape(D, -1)
    fb = fb / np.maximum(np.linalg.norm(fb, axis=0, keepdims=True), 1e-8)
    fa = fa / np.maximum(np.linalg.norm(fa, axis=0, keepdims=True), 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    dist = ((1.0 - cos_sim) / 2.0).reshape(H, W)

    changed_dists.extend(dist[mask > 0].tolist())
    unchanged_dists.extend(dist[mask == 0].tolist())

    flat_mask = mask.flatten()
    flat_dist = dist.flatten()
    if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
        auc = roc_auc_score(flat_mask, flat_dist)
        auc_per_patch.append(auc)

changed_dists = np.array(changed_dists)
unchanged_dists = np.array(unchanged_dists)

print(f"\n3. Distance Distribution:")
print(f"   Changed   pixels: mean={changed_dists.mean():.4f}, std={changed_dists.std():.4f}, n={len(changed_dists)}")
print(f"   Unchanged pixels: mean={unchanged_dists.mean():.4f}, std={unchanged_dists.std():.4f}, n={len(unchanged_dists)}")
print(f"   Separation (mean diff): {changed_dists.mean() - unchanged_dists.mean():.4f}")
print(f"   参考: 若 separation > 0.05，说明 embedding 有一定判别力")

if auc_per_patch:
    print(f"\n4. Per-Patch AUC (from distance):")
    print(f"   Mean AUC: {np.mean(auc_per_patch):.4f}")
    print(f"   Median AUC: {np.median(auc_per_patch):.4f}")

# 5. 可视化直方图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 距离分布直方图
axes[0].hist(unchanged_dists, bins=50, alpha=0.7, label='Unchanged', density=True, color='blue')
axes[0].hist(changed_dists, bins=50, alpha=0.7, label='Changed', density=True, color='red')
axes[0].set_xlabel('Cosine Distance')
axes[0].set_ylabel('Density')
axes[0].set_title(f'V5 Distance Distribution\nSeparation={changed_dists.mean()-unchanged_dists.mean():.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# AUC 分布
if auc_per_patch:
    axes[1].hist(auc_per_patch, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(auc_per_patch), color='red', linestyle='--', label=f'Mean={np.mean(auc_per_patch):.3f}')
    axes[1].set_xlabel('AUC')
    axes[1].set_ylabel('Count')
    axes[1].set_title('V5 Per-Patch AUC Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_dir = "/workspace/outputs/aef_qwen_v5_mixed_scale/eval"
import os
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/v5_embedding_distance_hist.png", dpi=150)
print(f"\n  图表已保存: {out_dir}/v5_embedding_distance_hist.png")

# 保存诊断报告
diag = {
    "model": "V5_mixed_scale_epoch161",
    "uniformity": uniformity,
    "mean_pairwise_cosine_similarity": float(mean_cos_sim),
    "mean_pairwise_cosine_distance": float(mean_cos_dist),
    "changed_distance_mean": float(changed_dists.mean()),
    "changed_distance_std": float(changed_dists.std()),
    "unchanged_distance_mean": float(unchanged_dists.mean()),
    "unchanged_distance_std": float(unchanged_dists.std()),
    "separation": float(changed_dists.mean() - unchanged_dists.mean()),
    "per_patch_auc_mean": float(np.mean(auc_per_patch)) if auc_per_patch else None,
    "per_patch_auc_median": float(np.median(auc_per_patch)) if auc_per_patch else None,
    "n_patches": len(all_pids),
    "n_changed_pixels": int(len(changed_dists)),
    "n_unchanged_pixels": int(len(unchanged_dists)),
}

with open(f"{out_dir}/embedding_space_diagnosis.json", "w") as f:
    json.dump(diag, f, indent=2)
print(f"  诊断报告已保存: {out_dir}/embedding_space_diagnosis.json")

print("\n" + "="*60)
print("Embedding 空间诊断完成")
print("="*60)
