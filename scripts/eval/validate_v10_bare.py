#!/usr/bin/env python3
"""V10 Bare AUC 验证 — 在 105 个光学标注上测试 embedding 时间敏感性."""
import sys, json, time
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from rasterio.transform import from_bounds
from shapely.geometry import box, Point
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
CONFIG_PATH = "configs/xuannv_v10_temporal.yaml"
CKPT_PATH = "/workspace/outputs/xuannv_backbone_v10_temporal/epoch_best_epoch97.pt"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)

print("="*60)
print("  V10 Bare AUC 验证")
print(f"  Checkpoint: {CKPT_PATH}")
print("="*60)

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

cfg = load_config(CONFIG_PATH)
cfg.data.preload = False
cfg.data.manifest_path = "/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered"

model = AEFModel(cfg).to("npu:0")
ckpt = torch.load(CKPT_PATH, map_location="npu:0", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

print(f"模型加载完成，Dataset: {len(dataset)} patches")

# ──────────────────────────────────────────
# 加载 Grid 和标注
# ──────────────────────────────────────────
grid = gpd.read_file(GRID_PATH)
patch_id_to_idx = {f"patch_{i:06d}": i for i in range(len(grid))}

# 加载所有标注
import glob, os
shp_files = glob.glob(os.path.join(ANNOT_DIR, "*.shp"))
all_labels = []
for shp_path in shp_files:
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty or 'geometry' not in gdf.columns:
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
    except Exception as e:
        pass

labels_gdf = gpd.GeoDataFrame(all_labels, crs="EPSG:4326")
print(f"标注总数: {len(labels_gdf)}")

# ──────────────────────────────────────────
# 提取 embedding
# ──────────────────────────────────────────
@torch.no_grad()
def extract_embedding(patch_idx, window):
    """提取单个 patch 在指定时间窗口的 embedding."""
    try:
        sample = dataset[patch_idx]
        source_frames = sample["source_frames"].unsqueeze(0).to("npu:0")
        source_timestamps_ms = sample["source_timestamps_ms"].unsqueeze(0).to("npu:0")
        source_frame_mask = sample["source_frame_mask"].unsqueeze(0).to("npu:0")
        source_input_mask = sample["source_input_mask"].unsqueeze(0).to("npu:0")
        source_type_ids = sample["source_type_ids"].unsqueeze(0).to("npu:0")

        # 筛选窗口内的帧
        valid_start, valid_end = window
        B, S, T = source_frame_mask.shape
        frame_mask = source_frame_mask.clone()
        for b in range(B):
            for s in range(S):
                for t in range(T):
                    ts = source_timestamps_ms[b, s, t].item()
                    if ts < valid_start or ts > valid_end:
                        frame_mask[b, s, t] = False

        # 如果没有帧在窗口内，返回 None
        if not frame_mask.any():
            return None

        # 前向传播
        out = model(
            source_frames=source_frames,
            source_timestamps_ms=source_timestamps_ms,
            source_frame_mask=frame_mask,
            source_input_mask=source_input_mask,
            source_type_ids=source_type_ids,
            valid_start_ms=torch.tensor([valid_start], device="npu:0"),
            valid_end_ms=torch.tensor([valid_end], device="npu:0"),
            target_relative_time=torch.zeros(1, 1, device="npu:0"),
            target_metadata=torch.zeros(1, cfg.data.metadata_dim, device="npu:0"),
        )

        emb = out.embedding[0]  # [D]
        return F.normalize(emb, dim=0).cpu().numpy()
    except Exception as e:
        return None

# ──────────────────────────────────────────
# 计算每个 patch 的 before/after embedding
# ──────────────────────────────────────────
patch_embs = {}
pbar_every = 50

for patch_id in sorted(patch_id_to_idx.keys()):
    idx = patch_id_to_idx[patch_id]
    if idx >= len(dataset):
        continue

    emb_before = extract_embedding(idx, BEFORE_WINDOW)
    emb_after = extract_embedding(idx, AFTER_WINDOW)

    if emb_before is not None and emb_after is not None:
        patch_embs[patch_id] = {
            'before': emb_before,
            'after': emb_after,
        }

    if len(patch_embs) % pbar_every == 0:
        print(f"  已处理 {len(patch_embs)} patches...")

print(f"成功提取 {len(patch_embs)} patches 的 embedding")

# ──────────────────────────────────────────
# 计算 AUC
# ──────────────────────────────────────────
scores = []
labels = []

for _, row in labels_gdf.iterrows():
    pt = row.geometry
    if pt is None:
        continue

    # 找到包含该点的 patch
    matched = None
    for _, g_row in grid.iterrows():
        if g_row.geometry.contains(pt):
            matched = g_row
            break

    if matched is None:
        continue

    patch_id = matched.get('patch_id', f"patch_{matched.name:06d}")
    if patch_id not in patch_embs:
        continue

    emb_b = patch_embs[patch_id]['before']
    emb_a = patch_embs[patch_id]['after']

    # Cosine distance = 1 - cosine_similarity
    sim = np.dot(emb_b, emb_a)
    dist = 1.0 - sim

    scores.append(dist)
    labels.append(row['label'])

if len(scores) < 10:
    print(f"警告: 有效样本数太少 ({len(scores)})，无法计算 AUC")
else:
    auc = roc_auc_score(labels, scores)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"\n{'='*60}")
    print(f"  Bare AUC = {auc:.4f}")
    print(f"  样本数: {len(scores)} (正例={n_pos}, 负例={n_neg})")
    print(f"{'='*60}")
    print(f"\n解读:")
    print(f"  AUC > 0.7: 时间敏感性及格")
    print(f"  AUC > 0.8: 良好")
    print(f"  AUC > 0.85: 优秀")
    if auc < 0.55:
        print(f"  ⚠️ AUC={auc:.4f} 接近随机，embedding 可能时间盲")
    elif auc < 0.65:
        print(f"  ⚠️ AUC={auc:.4f} 偏弱，需要继续训练或架构调整")
    elif auc < 0.75:
        print(f"  ✅ AUC={auc:.4f} 及格，有基本时间敏感性")
    else:
        print(f"  ✅ AUC={auc:.4f} 良好/优秀")
