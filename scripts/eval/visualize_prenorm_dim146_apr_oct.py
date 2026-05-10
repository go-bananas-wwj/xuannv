#!/usr/bin/env python3
"""
可视化 V5 Pre-Norm Embedding（无 L2 Normalization）
patch_000146: 2025-04 vs 2025-10，每个维度单独一张图 + 变化区域边界
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import box, Point
import torch
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──
PATCH_ID = "patch_000146"
PERIOD_BEFORE = "2025-04"
PERIOD_AFTER = "2025-10"
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v5_mixed_scale.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt"
SHP_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
S2_DIR = Path(f"/workspace/raw/harbin_scenes/s2/{PATCH_ID}")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval/dim146_prenorm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0"

# 时间窗口 (ms epoch)
MONTH_TO_MS = {
    "2025-04": (1711929600000.0, 1714521600000.0),
    "2025-06": (1717200000000.0, 1719792000000.0),
    "2025-08": (1722470400000.0, 1725148800000.0),
    "2025-09": (1725148800000.0, 1727740800000.0),
    "2025-10": (1727740800000.0, 1730419200000.0),
}

# ── 1. 从 S2 TIFF 读取 Patch Bounds ──
def get_patch_bounds_from_s2():
    tif_files = sorted(S2_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIFF found in {S2_DIR}")
    with rasterio.open(tif_files[0]) as src:
        bounds = src.bounds
        crs = src.crs
    return bounds, crs

bounds, crs = get_patch_bounds_from_s2()
print(f"[{PATCH_ID}] Bounds: {bounds}, CRS: {crs}")

# ── 2. 栅格化变化标注（合并 4-6, 6-8, 8-9, 9-10）──
def rasterize_annotations(bounds, grid_size=64):
    minx, miny, maxx, maxy = bounds
    res_x = (maxx - minx) / grid_size
    res_y = (maxy - miny) / grid_size
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    optical_shps = {
        "june.shp": "2025-04~2025-06",
        "aug.shp": "2025-06~2025-08",
        "September.shp": "2025-08~2025-09",
        "October.shp": "2025-09~2025-10",
    }

    for shp_name, period in optical_shps.items():
        shp_path = SHP_DIR / shp_name
        if not shp_path.exists():
            continue
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
        except Exception as e:
            print(f"  Skip {shp_name}: {e}")
            continue

        patch_box = box(minx, miny, maxx, maxy)
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or not geom.intersects(patch_box):
                continue
            gminx, gminy, gmaxx, gmaxy = geom.bounds
            px_start = max(0, int((gminx - minx) / res_x))
            px_end = min(grid_size, int((gmaxx - minx) / res_x) + 1)
            py_start = max(0, int((maxy - gmaxy) / res_y))
            py_end = min(grid_size, int((maxy - gminy) / res_y) + 1)

            for px in range(px_start, px_end):
                for py in range(py_start, py_end):
                    wx = minx + (px + 0.5) * res_x
                    wy = maxy - (py + 0.5) * res_y
                    if geom.contains(Point(wx, wy)):
                        mask[py, px] = 1.0

    return mask

print("栅格化标注...")
change_mask = rasterize_annotations(bounds)
n_changed = int(change_mask.sum())
n_total = change_mask.size
print(f"  Changed pixels: {n_changed}/{n_total} ({100*n_changed/n_total:.1f}%)")

# ── 3. 加载模型 ──
print("加载 V5 模型...")
import sys
sys.path.insert(0, "/workspace/xuannv")
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

cfg = load_config(CONFIG_PATH)
# ★ 修复路径: 直接指向实际数据目录
cfg.data.manifest_path = "/workspace/raw/harbin_scenes"
cfg.data.preload = False

model = AEFModel(cfg).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

assert PATCH_ID in dataset.patches, f"{PATCH_ID} not found in dataset"
pidx = dataset.patches.index(PATCH_ID)
print(f"  Patch index: {pidx}")

# ── 4. 提取 Pre-Norm Embedding ──
def extract_prenorm_emb(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
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
    pre_norm = output.pre_norm_map
    if pre_norm is None:
        raise ValueError("Model did not return pre_norm_map!")
    return pre_norm[0].cpu().numpy()  # [D, H, W]

print("提取 Pre-Norm Embedding...")
start_ms, end_ms = MONTH_TO_MS[PERIOD_BEFORE]
emb_before = extract_prenorm_emb(model, dataset, pidx, start_ms, end_ms)
start_ms, end_ms = MONTH_TO_MS[PERIOD_AFTER]
emb_after = extract_prenorm_emb(model, dataset, pidx, start_ms, end_ms)

D, H, W = emb_before.shape
print(f"  Pre-norm shape: {emb_before.shape}")
print(f"  Before: min={emb_before.min():.4f}, max={emb_before.max():.4f}, mean={emb_before.mean():.4f}")
print(f"  After:  min={emb_after.min():.4f}, max={emb_after.max():.4f}, mean={emb_after.mean():.4f}")

# ── 5. 逐维度变化分析 ──
delta = emb_after - emb_before  # [D, H, W]

# L2 norm 版本用于对比
def l2_normalize(x):
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    return x / np.maximum(norms, 1e-8)

emb_before_l2 = l2_normalize(emb_before)
emb_after_l2 = l2_normalize(emb_after)
delta_l2 = emb_after_l2 - emb_before_l2

def analyze_dimensions(delta_map, mask, label="pre-norm"):
    stats = []
    for d in range(D):
        dim_delta = delta_map[d]
        changed_mean = float(dim_delta[mask > 0].mean()) if n_changed > 0 else 0.0
        unchanged_mean = float(dim_delta[mask == 0].mean())
        diff = abs(changed_mean - unchanged_mean)
        stats.append({
            "dim": d,
            "changed_mean": changed_mean,
            "unchanged_mean": unchanged_mean,
            "diff": diff,
            "changed_std": float(dim_delta[mask > 0].std()) if n_changed > 0 else 0.0,
            "unchanged_std": float(dim_delta[mask == 0].std()),
        })
    stats.sort(key=lambda x: x["diff"], reverse=True)

    changed_abs_mean = np.mean([abs(s["changed_mean"]) for s in stats])
    unchanged_abs_mean = np.mean([abs(s["unchanged_mean"]) for s in stats])

    return stats, changed_abs_mean, unchanged_abs_mean

stats_pre, chg_pre, unchg_pre = analyze_dimensions(delta, change_mask, "pre-norm")
stats_l2, chg_l2, unchg_l2 = analyze_dimensions(delta_l2, change_mask, "l2-norm")

print(f"\n{'='*60}")
print("Pre-Norm vs L2-Norm 对比")
print(f"{'='*60}")
print(f"Pre-Norm:  变化区域 |Δ|={chg_pre:.6f}, 未变化 |Δ|={unchg_pre:.6f}, 比率={chg_pre/max(unchg_pre,1e-8):.2f}x")
print(f"L2-Norm:   变化区域 |Δ|={chg_l2:.6f}, 未变化 |Δ|={unchg_l2:.6f}, 比率={chg_l2/max(unchg_l2,1e-8):.2f}x")

print(f"\nPre-Norm Top 10:")
for s in stats_pre[:10]:
    sign = "+" if s["changed_mean"] > s["unchanged_mean"] else "-"
    print(f"  Dim {s['dim']:3d}: chg={s['changed_mean']:+.5f}, unchg={s['unchanged_mean']:+.5f}, diff={s['diff']:.5f} [{sign}]")

# ── 6. 生成汇总 heatmap (Top 16) ──
N_TOP = 16
top_dims = [s["dim"] for s in stats_pre[:N_TOP]]

colors = [(0.0, 0.2, 0.8), (0.85, 0.85, 0.85), (0.8, 0.1, 0.1)]
cmap = LinearSegmentedColormap.from_list("diverging", colors)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

from scipy import ndimage
edges = ndimage.binary_dilation(change_mask > 0) ^ (change_mask > 0)

for idx, d in enumerate(top_dims):
    ax = axes[idx]
    dim_delta = delta[d]
    vmax = np.percentile(np.abs(dim_delta), 95)
    vmin = -vmax
    im = ax.imshow(dim_delta, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.contour(edges, colors="lime", linewidths=0.8, levels=[0.5])

    s = stats_pre[idx]
    sign = "▲" if s["changed_mean"] > s["unchanged_mean"] else "▼"
    ax.set_title(f"Pre-Norm Dim {d}  {sign}\nchg={s['changed_mean']:+.4f}  unchg={s['unchanged_mean']:+.4f}",
                 fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"{PATCH_ID}  Pre-Norm Embedding: Top {N_TOP} Dimensions by Change Sensitivity\n"
             f"Red=Increase  Blue=Decrease  Green=Change boundary\n"
             f"Period: {PERIOD_BEFORE} → {PERIOD_AFTER}",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT_DIR / f"prenorm_dim_change_heatmap_{PATCH_ID}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved summary heatmap: {OUT_DIR / f'prenorm_dim_change_heatmap_{PATCH_ID}.png'}")

# 保存 delta 和 mask 供后续修复
np.save(OUT_DIR / ".delta.npy", delta)
np.save(OUT_DIR / ".mask.npy", change_mask)

# ── 7. ★ 每个维度单独一张图 (128 张) ──
print(f"\n生成 {D} 个维度的单独图片...")

# 为每个维度使用自适应 vmax（基于该维度变化的 95 分位数）
for d in range(D):
    dim_delta = delta[d]
    vmax = np.percentile(np.abs(dim_delta), 95)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(dim_delta, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.contour(edges, colors="lime", linewidths=1.0, levels=[0.5])

    s = stats_pre[d]
    rank = next((i for i, st in enumerate(stats_pre) if st["dim"] == d), D)
    sign = "▲" if s["changed_mean"] > s["unchanged_mean"] else "▼"

    ax.set_title(
        f"{PATCH_ID}  Pre-Norm Dim {d:03d}  {sign}\n"
        f"Δ(chg)={s['changed_mean']:+.4f}  Δ(unchg)={s['unchanged_mean']:+.4f}  |diff|={s['diff']:.4f}\n"
        f"Rank #{rank+1}/{D}  by change sensitivity",
        fontsize=10, fontweight="bold"
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(OUT_DIR / f"dim_{d:03d}_{PATCH_ID}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if (d + 1) % 32 == 0:
        print(f"  ... generated {d+1}/{D}")

print(f"  Saved {D} individual dimension images to {OUT_DIR}")

# ── 8. 对比箱线图 ──
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

def plot_box(ax, stats_list, title, delta_map, color_changed="#e74c3c", color_unchanged="#3498db"):
    changed_vals = []
    unchanged_vals = []
    dim_labels = []
    for s in stats_list[:40]:
        d = s["dim"]
        changed_vals.append(delta_map[d][change_mask > 0].flatten())
        unchanged_vals.append(delta_map[d][change_mask == 0].flatten())
        dim_labels.append(str(d))

    positions = np.arange(len(dim_labels)) * 2.0
    bp1 = ax.boxplot(changed_vals, positions=positions - 0.4, widths=0.7,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=color_changed, alpha=0.7),
                      medianprops=dict(color="white", linewidth=2))
    bp2 = ax.boxplot(unchanged_vals, positions=positions + 0.4, widths=0.7,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=color_unchanged, alpha=0.7),
                      medianprops=dict(color="white", linewidth=2))

    ax.set_xticks(positions)
    ax.set_xticklabels(dim_labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_ylabel("Δ Embedding (After - Before)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Changed", "Unchanged"], loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

plot_box(axes[0], stats_pre, f"Pre-Norm (no L2)\nRatio={chg_pre/max(unchg_pre,1e-8):.2f}x", delta)
plot_box(axes[1], stats_l2, f"L2-Normalized\nRatio={chg_l2/max(unchg_l2,1e-8):.2f}x", delta_l2)

plt.suptitle(f"{PATCH_ID}  Pre-Norm vs L2-Norm: Change-induced Embedding Shift", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / f"prenorm_vs_l2_boxplot_{PATCH_ID}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / f'prenorm_vs_l2_boxplot_{PATCH_ID}.png'}")

# ── 9. 保存报告 ──
report = {
    "patch_id": PATCH_ID,
    "period": f"{PERIOD_BEFORE}~{PERIOD_AFTER}",
    "pre_norm": {
        "changed_abs_mean": float(chg_pre),
        "unchanged_abs_mean": float(unchg_pre),
        "ratio": float(chg_pre / max(unchg_pre, 1e-8)),
        "top_20": stats_pre[:20],
    },
    "l2_norm": {
        "changed_abs_mean": float(chg_l2),
        "unchanged_abs_mean": float(unchg_l2),
        "ratio": float(chg_l2 / max(unchg_l2, 1e-8)),
        "top_20": stats_l2[:20],
    },
}

with open(OUT_DIR / f"prenorm_analysis_{PATCH_ID}.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved: {OUT_DIR / f'prenorm_analysis_{PATCH_ID}.json'}")

# ── 10. 结论 ──
print(f"\n{'='*60}")
print("结论")
print(f"{'='*60}")

if chg_pre > 1.5 * unchg_pre:
    print(f"✅ Pre-norm 空间变化信号显著更强 ({chg_pre/unchg_pre:.2f}x)")
    print("   说明 L2 normalization 确实压缩了变化信息!")
elif chg_pre > 1.2 * unchg_pre:
    print(f"⚠️ Pre-norm 空间略有改善 ({chg_pre/unchg_pre:.2f}x)，但不显著")
else:
    print(f"❌ Pre-norm 空间同样弱 ({chg_pre/unchg_pre:.2f}x)")
    print("   说明变化信息在 backbone 层面就没有被编码，不是 L2 norm 的问题!")

print(f"{'='*60}")
