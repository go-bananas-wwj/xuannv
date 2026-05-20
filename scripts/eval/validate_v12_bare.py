#!/usr/bin/env python3
"""V12 Bare AUC: 无CD Head，直接用cosine distance评估embedding时间敏感性.

变化检测标注对应2025年具体月份对：
  - june.shp: 2025-04 → 2025-06
  - aug.shp: 2025-06 → 2025-08
  - September.shp: 2025-08 → 2025-09
  - October.shp: 2025-09 → 2025-10
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

# 2025年各月份对时间窗口（毫秒，UTC）
PERIODS = {
    "june": {
        "before": (1743436800000, 1746028799000),   # 2025-04
        "after":  (1748707200000, 1751299199000),   # 2025-06
    },
    "aug": {
        "before": (1748707200000, 1751299199000),   # 2025-06
        "after":  (1753977600000, 1756655999000),   # 2025-08
    },
    "September": {
        "before": (1753977600000, 1756655999000),   # 2025-08
        "after":  (1756656000000, 1759247999000),   # 2025-09
    },
    "October": {
        "before": (1756656000000, 1759247999000),   # 2025-09
        "after":  (1759248000000, 1761926399000),   # 2025-10
    },
}

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
# 按 period 分组存储变化标注
changes_by_period: dict[str, list] = {p: [] for p in PERIODS}
for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
    period = shp_name.replace(".shp", "")
    try:
        gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
        # CRS fix (from AGENTS.md)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        for _, row in gdf.iterrows():
            if row.geometry is not None:
                changes_by_period[period].append({"geometry": row.geometry, "period": period})
    except Exception as e:
        print(f"  Warning: could not load {shp_name}: {e}")

total_changes = sum(len(v) for v in changes_by_period.values())
print(f"  共 {total_changes} 个变化标注")
for period, changes in changes_by_period.items():
    print(f"    {period}: {len(changes)} 个")

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

# ──────────────────────────────────────────
# 计算 Bare AUC（按 period 分别计算 + 全局汇总）
# ──────────────────────────────────────────
all_scores = []
all_labels = []
all_changed_means = []
all_unchanged_means = []
period_results = {}

for period, period_info in PERIODS.items():
    changes = changes_by_period.get(period, [])
    if not changes:
        continue

    # 找出该 period 有标注的 patch
    annotated_pids = set()
    for change in changes:
        geom = change["geometry"]
        for pid, bounds in patch_bounds.items():
            if box(*bounds).intersects(geom):
                annotated_pids.add(pid)
                break

    period_scores = []
    period_labels = []
    period_changed_means = []
    period_unchanged_means = []

    for pid in sorted(annotated_pids):
        try:
            emb_before = extract_embedding(model, dataset, pid, period_info["before"][0], period_info["before"][1])
            emb_after = extract_embedding(model, dataset, pid, period_info["after"][0], period_info["after"][1])
        except Exception as e:
            print(f"  [Skip] {pid} ({period}): {e}")
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
        cos_map = np.sum(emb_before * emb_after, axis=0)  # [H, W]
        dist_map = 1.0 - cos_map  # [H, W]

        labels_flat = changed_mask.flatten()
        scores_flat = dist_map.flatten()

        if labels_flat.sum() == 0 or labels_flat.sum() == len(labels_flat):
            continue

        period_scores.extend(scores_flat.tolist())
        period_labels.extend(labels_flat.tolist())
        all_scores.extend(scores_flat.tolist())
        all_labels.extend(labels_flat.tolist())

        changed_mean = float(dist_map[changed_mask].mean()) if changed_mask.any() else 0.0
        unchanged_mean = float(dist_map[~changed_mask].mean()) if (~changed_mask).any() else 0.0
        period_changed_means.append(changed_mean)
        period_unchanged_means.append(unchanged_mean)
        all_changed_means.append(changed_mean)
        all_unchanged_means.append(unchanged_mean)

    # 该 period 的 AUC
    if len(period_labels) > 0 and sum(period_labels) > 0 and sum(period_labels) < len(period_labels):
        period_auc = roc_auc_score(period_labels, period_scores)
        period_changed = np.mean(period_changed_means) if period_changed_means else 0.0
        period_unchanged = np.mean(period_unchanged_means) if period_unchanged_means else 0.0
        period_results[period] = {
            "auc": float(period_auc),
            "changed_mean": float(period_changed),
            "unchanged_mean": float(period_unchanged),
            "separation": float(period_changed - period_unchanged),
            "n_samples": len(period_labels),
            "n_positive": int(sum(period_labels)),
        }

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
print("  Bare AUC 结果")
print("="*60)
print(f"\n  全局 AUC: {auc:.4f}")
print(f"  Changed mean dist:   {changed_mean:.4f}")
print(f"  Unchanged mean dist: {unchanged_mean:.4f}")
print(f"  Separation:          {separation:.4f}")

print("\n  分时期 AUC:")
for period, res in period_results.items():
    print(f"    {period:12s}: AUC={res['auc']:.4f}  separation={res['separation']:.4f}  "
          f"n={res['n_samples']} pos={res['n_positive']}")
print("="*60)

# Uniformity 诊断
if len(all_scores) > 10:
    emb_all = np.array(all_scores)
    print(f"\n  Score stats: min={emb_all.min():.4f} max={emb_all.max():.4f} "
          f"mean={emb_all.mean():.4f} std={emb_all.std():.4f}")

# 保存 JSON
output = {
    "global": {
        "auc": float(auc),
        "changed_mean": float(changed_mean),
        "unchanged_mean": float(unchanged_mean),
        "separation": float(separation),
    },
    "periods": period_results,
}
output_path = os.path.join(os.path.dirname(CKPT_PATH), "bare_auc.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  结果已保存: {output_path}")
