#!/usr/bin/env python3
"""AEF Bare AUC: 多时期对评估 (4月→6月, 6月→8月, 8月→9月, 9月→10月).

用法:
    python scripts/eval/validate_aef_bare.py --config configs/aef_baseline.yaml --checkpoint /workspace/outputs/aef_baseline/epoch_20.pt
"""
import sys, json, time, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch_npu
except ImportError:
    pass
import geopandas as gpd
from shapely.geometry import box
from sklearn.metrics import roc_auc_score
import calendar

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径")
parser.add_argument("--device", type=str, default="npu:0")
parser.add_argument("--annot-dir", type=str, default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
parser.add_argument("--grid", type=str, default="/workspace/index/harbin/grid/harbin_grid.geojson")
args = parser.parse_args()

DEVICE = args.device
ANNOT_DIR = args.annot_dir
GRID_PATH = args.grid

# 时期定义: (名称, before窗口, after窗口, shapefile)
PERIODS = [
    ("apr→jun", (1743436800000, 1746057599000), (1748736000000, 1751327999000), "june.shp"),
    ("jun→aug", (1748736000000, 1751327999000), (1754006400000, 1756655999000), "aug.shp"),
    ("aug→sept", (1754006400000, 1756655999000), (1756771200000, 1759247999000), "September.shp"),
    ("sept→oct", (1756771200000, 1759247999000), (1759449600000, 1761926399000), "October.shp"),
]

print("="*60)
print("  AEF Bare AUC Validation (Multi-Period)")
print(f"  Config: {args.config}")
print(f"  Checkpoint: {args.checkpoint}")
print(f"  Device: {DEVICE}")
print("="*60)

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

cfg = load_config(args.config)
model = AEFModel(cfg).to(DEVICE)
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
cfg.data.preload = False
dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

# ──────────────────────────────────────────
# 加载 Grid
# ──────────────────────────────────────────
with open(GRID_PATH) as f:
    grid_data = json.load(f)

patch_bounds = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

# ──────────────────────────────────────────
# 提取 embedding 函数
# ──────────────────────────────────────────
@torch.no_grad()
def extract_embedding(model, patch_id, valid_start, valid_end):
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
    emb = out.embedding_map
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()

# ──────────────────────────────────────────
# 逐时期评估
# ──────────────────────────────────────────
all_period_aucs = {}
all_period_separations = {}

for period_name, before_win, after_win, shp_name in PERIODS:
    print(f"\n{'='*60}")
    print(f"  Period: {period_name}")
    print(f"  Before: {before_win}")
    print(f"  After:  {after_win}")
    print(f"  Shapefile: {shp_name}")
    print(f"{'='*60}")

    # 加载当前时期的变化标注
    try:
        gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
        if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        changes = []
        for _, row in gdf.iterrows():
            if row.geometry is not None:
                changes.append({"geometry": row.geometry})
        print(f"  标注数: {len(changes)}")
    except Exception as e:
        print(f"  [Skip] 无法加载 {shp_name}: {e}")
        continue

    # 找出有标注的 patch
    annotated_pids = set()
    for change in changes:
        for pid, bounds in patch_bounds.items():
            if box(*bounds).intersects(change["geometry"]):
                annotated_pids.add(pid)
                break
    print(f"  带标注的 patch: {len(annotated_pids)} 个")

    all_scores = []
    all_labels = []
    all_changed_means = []
    all_unchanged_means = []

    for pid in sorted(annotated_pids):
        if pid not in dataset.patches:
            continue
        try:
            emb_before = extract_embedding(model, pid, before_win[0], before_win[1])
            emb_after = extract_embedding(model, pid, after_win[0], after_win[1])
        except Exception as e:
            print(f"  [Skip] {pid}: {e}")
            continue

        D, H, W = emb_before.shape
        changed_mask = np.zeros((H, W), dtype=bool)

        for change in changes:
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
        cos_map = np.sum(emb_before * emb_after, axis=0)
        dist_map = 1.0 - cos_map

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

    if len(all_labels) == 0:
        print(f"  [ERROR] 没有有效的评估样本")
        all_period_aucs[period_name] = None
        continue

    auc = roc_auc_score(all_labels, all_scores)
    changed_mean = np.mean(all_changed_means)
    unchanged_mean = np.mean(all_unchanged_means)
    separation = changed_mean - unchanged_mean

    all_period_aucs[period_name] = auc
    all_period_separations[period_name] = separation

    print(f"  AUC: {auc:.4f}")
    print(f"  Changed mean dist:   {changed_mean:.4f}")
    print(f"  Unchanged mean dist: {unchanged_mean:.4f}")
    print(f"  Separation:          {separation:.4f}")

# ──────────────────────────────────────────
# 汇总结果
# ──────────────────────────────────────────
valid_aucs = {k: v for k, v in all_period_aucs.items() if v is not None}
if valid_aucs:
    mean_auc = np.mean(list(valid_aucs.values()))
    mean_separation = np.mean([v for k, v in all_period_separations.items() if k in valid_aucs])

    print("\n" + "="*60)
    print("  汇总结果")
    print("="*60)
    for name, auc in valid_aucs.items():
        print(f"  {name}: AUC={auc:.4f}")
    print(f"\n  平均 AUC:        {mean_auc:.4f}")
    print(f"  平均 Separation: {mean_separation:.4f}")
    print("="*60)
else:
    print("\n[ERROR] 所有时期评估失败")
    sys.exit(1)
