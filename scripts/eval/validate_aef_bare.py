#!/usr/bin/env python3
"""AEF Bare AUC: 多时期对评估 + embedding 保存 + 可视化.

用法:
    python scripts/eval/validate_aef_bare.py --config configs/aef_baseline.yaml \
        --checkpoint /workspace/outputs/aef_baseline/epoch_best_epoch20.pt \
        --output-dir /workspace/outputs/aef_baseline/eval
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
from sklearn.metrics import roc_auc_score, roc_curve
import calendar
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--device", type=str, default="npu:0")
parser.add_argument("--annot-dir", type=str, default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
parser.add_argument("--grid", type=str, default="/workspace/index/harbin/grid/harbin_grid.geojson")
parser.add_argument("--output-dir", type=str, required=True, help="输出目录: embeddings/ + distance_maps/ + results.json")
parser.add_argument("--save-embeddings", action="store_true", default=True, help="保存 embedding (默认开启)")
parser.add_argument("--min-changed-pixels", type=int, default=10, help="最小变化像素数")
parser.add_argument("--no-viz", action="store_true", help="跳过可视化")
args = parser.parse_args()

DEVICE = args.device
ANNOT_DIR = args.annot_dir
GRID_PATH = args.grid
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = OUTPUT_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)
DIST_DIR = OUTPUT_DIR / "distance_maps"
DIST_DIR.mkdir(exist_ok=True)

PERIODS = [
    ("apr→jun", (1743436800000, 1746057599000), (1748736000000, 1751327999000), "june.shp"),
    ("jun→aug", (1748736000000, 1751327999000), (1754006400000, 1756655999000), "aug.shp"),
    ("aug→sept", (1754006400000, 1756655999000), (1756771200000, 1759247999000), "September.shp"),
    ("sept→oct", (1756771200000, 1759247999000), (1759449600000, 1761926399000), "October.shp"),
]

print("="*60)
print(f"  AEF Bare AUC Validation")
print(f"  Config: {args.config}")
print(f"  Checkpoint: {args.checkpoint}")
print(f"  Device: {DEVICE}")
print(f"  Output: {OUTPUT_DIR}")
print("="*60)

# ──────────────────────────────────────────
# 加载模型和数据
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
# 提取 embedding
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
# 阶段 1: 提取并保存 embedding
# ──────────────────────────────────────────
all_period_results = {}

for period_name, before_win, after_win, shp_name in PERIODS:
    print(f"\n{'='*60}")
    print(f"  Period: {period_name}")
    print(f"  Shapefile: {shp_name}")
    print(f"{'='*60}")

    # 加载标注
    try:
        gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        changes = [{"geometry": row.geometry} for _, row in gdf.iterrows() if row.geometry is not None]
        print(f"  标注数: {len(changes)}")
    except Exception as e:
        print(f"  [Skip] 无法加载 {shp_name}: {e}")
        continue

    # 找出所有相交的 patch
    annotated_pids = set()
    for change in changes:
        for pid, bounds in patch_bounds.items():
            if box(*bounds).intersects(change["geometry"]):
                annotated_pids.add(pid)
    print(f"  带标注的 patch: {len(annotated_pids)} 个")

    period_emb_dir = EMB_DIR / period_name.replace("→", "_")
    period_emb_dir.mkdir(exist_ok=True)

    patch_data = []

    for pid in sorted(annotated_pids):
        if pid not in dataset.patches:
            continue

        # 保存路径
        before_path = period_emb_dir / f"{pid}_before.npy"
        after_path = period_emb_dir / f"{pid}_after.npy"
        mask_path = period_emb_dir / f"{pid}_mask.npy"

        # 提取或加载 embedding
        if before_path.exists() and after_path.exists() and mask_path.exists():
            emb_before = np.load(before_path)
            emb_after = np.load(after_path)
            changed_mask = np.load(mask_path)
        else:
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

            if args.save_embeddings:
                np.save(before_path, emb_before)
                np.save(after_path, emb_after)
                np.save(mask_path, changed_mask)

        patch_data.append({
            "pid": pid,
            "emb_before": emb_before,
            "emb_after": emb_after,
            "changed_mask": changed_mask,
        })

    all_period_results[period_name] = {
        "changes": changes,
        "annotated_pids": list(annotated_pids),
        "patch_data": patch_data,
    }

    print(f"  成功加载/提取: {len(patch_data)} 个 patch")

# 释放模型，后续纯 CPU 计算
print("\n[Phase 1] Embedding 提取完成，释放模型...")
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ──────────────────────────────────────────
# 阶段 2: 计算 AUC 和指标
# ──────────────────────────────────────────
results = {
    "experiment": cfg.experiment.name,
    "checkpoint": str(args.checkpoint),
    "periods": {},
}

for period_name, pdata in all_period_results.items():
    print(f"\n{'='*60}")
    print(f"  Computing: {period_name}")
    print(f"{'='*60}")

    all_scores = []
    all_labels = []
    all_changed_means = []
    all_unchanged_means = []
    patch_aucs = []
    patch_changed_ratios = []
    valid_patch_count = 0

    for p in pdata["patch_data"]:
        emb_before = p["emb_before"]
        emb_after = p["emb_after"]
        changed_mask = p["changed_mask"]
        pid = p["pid"]

        cos_map = np.sum(emb_before * emb_after, axis=0)
        dist_map = 1.0 - cos_map

        labels_flat = changed_mask.flatten()
        scores_flat = dist_map.flatten()
        n_changed = labels_flat.sum()

        if n_changed == 0 or n_changed == len(labels_flat):
            continue
        if n_changed < args.min_changed_pixels:
            print(f"  [Skip] {pid}: only {n_changed} changed pixels (<{args.min_changed_pixels})")
            continue

        all_scores.extend(scores_flat.tolist())
        all_labels.extend(labels_flat.tolist())

        changed_mean = float(dist_map[changed_mask].mean()) if changed_mask.any() else 0.0
        unchanged_mean = float(dist_map[~changed_mask].mean()) if (~changed_mask).any() else 0.0
        all_changed_means.append(changed_mean)
        all_unchanged_means.append(unchanged_mean)

        # Per-patch AUC
        try:
            p_auc = roc_auc_score(labels_flat, scores_flat)
            patch_aucs.append({"patch": pid, "auc": float(p_auc), "n_changed": int(n_changed)})
        except ValueError:
            pass

        patch_changed_ratios.append(float(n_changed / len(labels_flat)))
        valid_patch_count += 1

    if len(all_labels) == 0:
        print(f"  [ERROR] 没有有效的评估样本")
        results["periods"][period_name] = {"error": "no valid samples"}
        continue

    auc = roc_auc_score(all_labels, all_scores)
    changed_mean = np.mean(all_changed_means)
    unchanged_mean = np.mean(all_unchanged_means)
    separation = changed_mean - unchanged_mean

    # 额外指标
    patch_auc_values = [p["auc"] for p in patch_aucs]
    patch_auc_median = np.median(patch_auc_values) if patch_auc_values else 0.0
    patch_auc_min = np.min(patch_auc_values) if patch_auc_values else 0.0
    patch_auc_max = np.max(patch_auc_values) if patch_auc_values else 0.0
    patch_auc_std = np.std(patch_auc_values) if patch_auc_values else 0.0
    mean_changed_ratio = np.mean(patch_changed_ratios) if patch_changed_ratios else 0.0

    # Distance map 可视化
    if not args.no_viz and pdata["patch_data"]:
        _make_viz(pdata["patch_data"], period_name, DIST_DIR)

    results["periods"][period_name] = {
        "auc": float(auc),
        "changed_mean_dist": float(changed_mean),
        "unchanged_mean_dist": float(unchanged_mean),
        "separation": float(separation),
        "valid_patches": valid_patch_count,
        "total_patches": len(pdata["patch_data"]),
        "mean_changed_ratio": float(mean_changed_ratio),
        "per_patch_auc": {
            "median": float(patch_auc_median),
            "min": float(patch_auc_min),
            "max": float(patch_auc_max),
            "std": float(patch_auc_std),
            "details": patch_aucs,
        },
    }

    print(f"  AUC: {auc:.4f}")
    print(f"  Separation: {separation:.4f}")
    print(f"  Valid patches: {valid_patch_count}/{len(pdata['patch_data'])}")
    print(f"  Patch AUC: median={patch_auc_median:.4f} min={patch_auc_min:.4f} max={patch_auc_max:.4f} std={patch_auc_std:.4f}")
    print(f"  Mean changed ratio: {mean_changed_ratio:.4f}")

# 汇总
valid_periods = {k: v for k, v in results["periods"].items() if "error" not in v}
if valid_periods:
    aucs = [v["auc"] for v in valid_periods.values()]
    separations = [v["separation"] for v in valid_periods.values()]
    results["summary"] = {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_separation": float(np.mean(separations)),
        "n_periods": len(valid_periods),
    }

    print("\n" + "="*60)
    print("  汇总结果")
    print("="*60)
    for name, r in valid_periods.items():
        print(f"  {name}: AUC={r['auc']:.4f} sep={r['separation']:.4f}")
    print(f"\n  平均 AUC: {results['summary']['mean_auc']:.4f} ± {results['summary']['std_auc']:.4f}")
    print(f"  平均 Separation: {results['summary']['mean_separation']:.4f}")
    print("="*60)
else:
    print("\n[ERROR] 所有时期评估失败")
    sys.exit(1)

# 保存 JSON
json_path = OUTPUT_DIR / "results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] {json_path}")


# ──────────────────────────────────────────
# 可视化辅助
# ──────────────────────────────────────────
def _make_viz(patch_data, period_name, dist_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = min(len(patch_data), 4)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, p in enumerate(patch_data[:n]):
        emb_before = p["emb_before"]
        emb_after = p["emb_after"]
        changed_mask = p["changed_mask"]
        pid = p["pid"]

        cos_map = np.sum(emb_before * emb_after, axis=0)
        dist_map = 1.0 - cos_map

        vmin, vmax = dist_map.min(), dist_map.max()

        axes[i, 0].imshow(dist_map, vmin=vmin, vmax=vmax, cmap='hot')
        axes[i, 0].set_title(f'{pid} Distance')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(changed_mask, cmap='Greens')
        axes[i, 1].set_title(f'{pid} GT Mask')
        axes[i, 1].axis('off')

        overlay = plt.cm.hot((dist_map - vmin) / (vmax - vmin + 1e-8))
        overlay[changed_mask] = [0, 1, 0, 0.5]
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'{pid} Overlay')
        axes[i, 2].axis('off')

    plt.tight_layout()
    fig.savefig(dist_dir / f'{period_name.replace("→", "_")}_distance_maps.png', dpi=150)
    plt.close(fig)
    print(f"  [Viz] Saved {period_name} distance maps")

