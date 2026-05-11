#!/usr/bin/env python3
"""V12 Bare AUC: 无CD Head，直接用cosine distance评估embedding时间敏感性."""
import sys, json, time, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box
from sklearn.metrics import roc_auc_score

# ──────────────────────────────────────────
# 配置（命令行参数覆盖）
# ──────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/xuannv_v12_clean.yaml")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--device", type=str, default="npu:0")
parser.add_argument("--annot-dir", type=str, default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
parser.add_argument("--grid", type=str, default="/workspace/index/harbin/grid/harbin_grid.geojson")
args = parser.parse_args()

CONFIG_PATH = args.config
CKPT_PATH = args.checkpoint
DEVICE = args.device
ANNOT_DIR = args.annot_dir
GRID_PATH = args.grid

# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)   # 2024Q3-Q4

print("="*60)
print("  V12 Bare AUC Validation")
print(f"  Config: {CONFIG_PATH}")
print(f"  Checkpoint: {CKPT_PATH}")
print(f"  Device: {DEVICE}")
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
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, cfg

print("\n加载 V12 模型...")
model, dataset, cfg = load_model(CKPT_PATH)

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
    except Exception as e:
        print(f"  Warning: could not load {shp_name}: {e}")

print(f"  共 {len(all_changes)} 个变化标注")

# ──────────────────────────────────────────
# 提取 embedding
# ──────────────────────────────────────────
@torch.no_grad()
def extract_embedding(model, dataset, patch_id, valid_start, valid_end):
    """提取单个patch在指定时间窗口的embedding."""
    idx = dataset.patches.index(patch_id)
    item = dataset[idx]

    source_frames = item["source_frames"].unsqueeze(0).to(DEVICE)
    source_timestamps_ms = item["source_timestamps_ms"].unsqueeze(0).to(DEVICE)
    source_frame_mask = item["source_frame_mask"].unsqueeze(0).to(DEVICE)
    source_input_mask = item["source_input_mask"].unsqueeze(0).to(DEVICE)
    source_type_ids = item["source_type_ids"].unsqueeze(0).to(DEVICE)
    valid_start_t = torch.tensor([valid_start], dtype=torch.int64, device=DEVICE)
    valid_end_t = torch.tensor([valid_end], dtype=torch.int64, device=DEVICE)
    target_relative_time = torch.zeros(1, cfg.data.num_target_sources, device=DEVICE)
    target_metadata = torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=DEVICE)

    out = model(
        source_frames=source_frames,
        source_timestamps_ms=source_timestamps_ms,
        source_frame_mask=source_frame_mask,
        source_input_mask=source_input_mask,
        source_type_ids=source_type_ids,
        valid_start_ms=valid_start_t,
        valid_end_ms=valid_end_t,
        target_relative_time=target_relative_time,
        target_metadata=target_metadata,
        skip_decoder=True,
    )
    emb = out.embedding_map  # [1, D, H, W]
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()  # [D, H, W]

# 找出有标注的patch
annotated_pids = set()
for change in all_changes:
    geom = change["geometry"]
    for pid, bounds in patch_bounds.items():
        if box(*bounds).intersects(geom):
            annotated_pids.add(pid)
            break

print(f"\n带标注的patch: {len(annotated_pids)} 个")

# ──────────────────────────────────────────
# 计算 Bare AUC
# ──────────────────────────────────────────
all_scores = []
all_labels = []
all_changed_means = []
all_unchanged_means = []

for pid in sorted(annotated_pids):
    try:
        emb_before = extract_embedding(model, dataset, pid, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
        emb_after = extract_embedding(model, dataset, pid, AFTER_WINDOW[0], AFTER_WINDOW[1])
    except Exception as e:
        print(f"  [Skip] {pid}: {e}")
        continue

    D, H, W = emb_before.shape
    changed_mask = np.zeros((H, W), dtype=bool)

    for change in all_changes:
        geom = change["geometry"]
        bounds = patch_bounds.get(pid)
        if bounds is None:
            continue
        minx, miny, maxx, maxy = bounds
        if not box(minx, miny, maxx, maxy).intersects(geom):
            continue

        for y in range(H):
            for x in range(W):
                px = minx + (x + 0.5) / W * (maxx - minx)
                py = maxy - (y + 0.5) / H * (maxy - miny)
                pt = box(px, py, px, py)
                if geom.contains(pt) or geom.intersects(pt):
                    changed_mask[y, x] = True

    # 像素级 cosine distance
    cos_map = np.sum(emb_before * emb_after, axis=0)  # [H, W]
    dist_map = 1.0 - cos_map  # [H, W]

    labels_flat = changed_mask.flatten()
    scores_flat = dist_map.flatten()

    if labels_flat.sum() == 0 or labels_flat.sum() == len(labels_flat):
        continue

    all_scores.extend(scores_flat.tolist())
    all_labels.extend(labels_flat.tolist())

    changed_mean = float(dist_map[changed_mask].mean()) if changed_mask.any() else 0.0
    unchanged_mean = float(dist_map[~changed_mask].mean()) if (~changed_mask).any() else 0.0
    all_changed_means.append(changed_mean)
    all_unchanged_means.append(unchanged_mean)

# ──────────────────────────────────────────
# 结果输出
# ──────────────────────────────────────────
if len(all_labels) == 0:
    print("\n[ERROR] 没有有效的评估样本")
    sys.exit(1)

auc = roc_auc_score(all_labels, all_scores)
changed_mean = np.mean(all_changed_means)
unchanged_mean = np.mean(all_unchanged_means)
separation = changed_mean - unchanged_mean

print("\n" + "="*60)
print(f"  Bare AUC: {auc:.4f}")
print(f"  Changed mean dist:   {changed_mean:.4f}")
print(f"  Unchanged mean dist: {unchanged_mean:.4f}")
print(f"  Separation:          {separation:.4f}")
print("="*60)

# Uniformity 诊断
if len(all_scores) > 10:
    emb_all = np.array(all_scores)
    print(f"\n  Score stats: min={emb_all.min():.4f} max={emb_all.max():.4f} "
          f"mean={emb_all.mean():.4f} std={emb_all.std():.4f}")
