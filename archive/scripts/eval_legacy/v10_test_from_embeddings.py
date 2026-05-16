#!/usr/bin/env python3
"""
从预提取的 embedding 运行 V10 测试.

测试:
1. 变化检测 Bare AUC
2. 嵌入质量指标 (RankMe, Stable Rank, Temporal Discriminability)
"""
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

print("="*60)
print("  V10 Embedding 测试")
print("="*60)

# 加载 embeddings
emb_path = "/workspace/outputs/v10_eval_e100/embeddings.npz"
print(f"\n加载 embeddings from {emb_path}...")
data = np.load(emb_path)

embeddings = {}
for key in data.files:
    arr = data[key]  # [3, D] — full, before, after
    embeddings[key] = {
        'full': arr[0],
        'before': arr[1],
        'after': arr[2],
    }

print(f"  共 {len(embeddings)} patches, embedding dim={arr.shape[1]}")

# =============================================================================
# 1. 变化检测 Bare AUC
# =============================================================================
print("\n[1/2] 变化检测 Bare AUC...")

import geopandas as gpd
import glob, os

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

grid = gpd.read_file(GRID_PATH)

# 加载标注
shp_files = glob.glob(os.path.join(ANNOT_DIR, "*.shp"))
all_labels = []
for shp_path in shp_files:
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            continue
        label_type = 1 if '新增建筑' in shp_path or '新增建设' in shp_path else 0
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            center = geom.centroid if hasattr(geom, 'centroid') else geom
            all_labels.append({
                'geometry': center,
                'label': label_type,
                'source': os.path.basename(shp_path),
            })
    except:
        pass

labels_gdf = gpd.GeoDataFrame(all_labels, crs="EPSG:4326")

scores = []
labels = []

for _, row in labels_gdf.iterrows():
    pt = row.geometry
    if pt is None:
        continue
    
    matched = None
    for _, g_row in grid.iterrows():
        if g_row.geometry.contains(pt):
            matched = g_row
            break
    
    if matched is None:
        continue
    
    patch_id = matched.get('patch_id', f"patch_{matched.name:06d}")
    if patch_id not in embeddings:
        continue
    
    emb_b = embeddings[patch_id].get('before')
    emb_a = embeddings[patch_id].get('after')
    
    # 如果 before/after 是全零（窗口无数据），跳过
    if emb_b is None or emb_a is None or np.allclose(emb_b, 0) or np.allclose(emb_a, 0):
        continue
    
    sim = np.dot(emb_b, emb_a)
    dist = 1.0 - sim
    
    scores.append(dist)
    labels.append(row['label'])

auc = None
if len(scores) >= 10:
    auc = roc_auc_score(labels, scores)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Bare AUC = {auc:.4f} (n={len(scores)}, pos={n_pos}, neg={n_neg})")
else:
    print(f"  样本不足: {len(scores)}")

# =============================================================================
# 2. 嵌入质量指标
# =============================================================================
print("\n[2/2] 嵌入质量指标...")

# RankMe
full_embs = []
for pid in sorted(embeddings.keys()):
    emb = embeddings[pid].get('full')
    if emb is not None:
        full_embs.append(emb)

embs_matrix = np.stack(full_embs)
N, D = embs_matrix.shape

# SVD
_, s, _ = np.linalg.svd(embs_matrix, full_matrices=False)
s_norm = s / (s.sum() + 1e-8)
entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
rankme = entropy / np.log(min(N, D))

# Stable Rank
stable_rank = (s ** 2).sum() / (s[0] ** 2 + 1e-8)

# Temporal Discriminability
temporal_dists = []
for pid in embeddings:
    emb_b = embeddings[pid].get('before')
    emb_a = embeddings[pid].get('after')
    if emb_b is not None and emb_a is not None and not np.allclose(emb_b, 0) and not np.allclose(emb_a, 0):
        dist = 1.0 - np.dot(emb_b, emb_a)
        temporal_dists.append(dist)

cross_dists = []
patch_ids = list(embeddings.keys())
np.random.seed(42)
for _ in range(min(500, len(patch_ids) * (len(patch_ids) - 1) // 2)):
    i, j = np.random.choice(len(patch_ids), 2, replace=False)
    emb_i = embeddings[patch_ids[i]].get('full')
    emb_j = embeddings[patch_ids[j]].get('full')
    if emb_i is not None and emb_j is not None:
        dist = 1.0 - np.dot(emb_i, emb_j)
        cross_dists.append(dist)

td_score = np.mean(temporal_dists) / (np.mean(cross_dists) + 1e-8) if len(cross_dists) > 0 else 0

print(f"  RankMe: {rankme:.4f} (理想值接近 1.0)")
print(f"  Stable Rank: {stable_rank:.2f} / D={D} (理想值接近 {D})")
print(f"  Temporal Discriminability: {td_score:.4f} (越高越好)")
print(f"  Temporal mean distance: {np.mean(temporal_dists):.4f}")
print(f"  Cross-patch mean distance: {np.mean(cross_dists):.4f}")

# =============================================================================
# 保存结果
# =============================================================================
results = {
    'change_detection': {'auc': float(auc), 'n_samples': len(scores)} if auc else None,
    'embedding_quality': {
        'rankme': float(rankme),
        'stable_rank': float(stable_rank),
        'temporal_discriminability': float(td_score),
        'n_patches': N,
        'embedding_dim': D,
    }
}

out_path = "/workspace/outputs/v10_eval_e100/test_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  测试完成！结果保存至: {out_path}")
print(f"{'='*60}")

# 汇总
print("\n=== V10 E100 测试汇总 ===")
if auc:
    status = "✅ 良好" if auc > 0.65 else ("✅ 及格" if auc > 0.55 else "⚠️ 需改进")
    print(f"变化检测 Bare AUC: {auc:.4f} {status}")
print(f"RankMe: {rankme:.4f} {'✅' if rankme > 0.8 else '⚠️'}")
print(f"Stable Rank: {stable_rank:.2f} / {D}")
print(f"Temporal Discriminability: {td_score:.4f}")
