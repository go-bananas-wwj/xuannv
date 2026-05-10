#!/usr/bin/env python3
"""
可视化 V5 Embedding 每个维度的变化强度.

目标: 检查变化信息是否被"隐藏"在某些维度中.
红色 = 该维度增加, 蓝色 = 该维度减少, 灰色 = 接近中性.
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
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──
PATCH_ID = "patch_000350"
PERIOD = ("2025-06", "2025-08")  # before, after
EMB_DIR = Path("/workspace/raw/xuannv_modelscope_upload/embeddings/v5_mixed_scale/monthly_embeddings_2025")
SHP_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
S2_DIR = Path(f"/workspace/raw/harbin_scenes/s2/{PATCH_ID}")
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. 从 S2 TIFF 读取 Patch Bounds ──
def get_patch_bounds_from_s2():
    tif_files = sorted(S2_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIFF found in {S2_DIR}")
    with rasterio.open(tif_files[0]) as src:
        bounds = src.bounds  # left, bottom, right, top
        crs = src.crs
    return bounds, crs

bounds, crs = get_patch_bounds_from_s2()
print(f"[{PATCH_ID}] Bounds: {bounds}, CRS: {crs}")

# ── 2. 栅格化变化标注 ──
def rasterize_annotations(bounds, grid_size=64):
    """将 SHP 标注栅格化为 [grid_size, grid_size] mask."""
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

            # 栅格化到 64x64
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

# ── 3. 加载 Embedding ──
def load_embedding(month):
    path = EMB_DIR / f"{PATCH_ID}_{month}.npy"
    emb = np.load(path)
    print(f"  Loaded {month}: shape={emb.shape}, dtype={emb.dtype}")
    return emb

print("加载 embedding...")
emb_before = load_embedding(PERIOD[0])
emb_after = load_embedding(PERIOD[1])

D, H, W = emb_before.shape
assert H == W == 64, f"Expected 64x64, got {H}x{W}"

# ── 4. 计算逐维度变化 ──
delta = emb_after - emb_before  # [D, H, W]

# 对每个维度，计算变化区域 vs 未变化区域的均值
dim_stats = []
for d in range(D):
    dim_delta = delta[d]  # [H, W]
    changed_mean = float(dim_delta[change_mask > 0].mean()) if n_changed > 0 else 0.0
    unchanged_mean = float(dim_delta[change_mask == 0].mean())
    diff = abs(changed_mean - unchanged_mean)
    dim_stats.append({
        "dim": d,
        "changed_mean": changed_mean,
        "unchanged_mean": unchanged_mean,
        "diff": diff,
        "changed_std": float(dim_delta[change_mask > 0].std()) if n_changed > 0 else 0.0,
        "unchanged_std": float(dim_delta[change_mask == 0].std()),
    })

# 按差异排序
dim_stats.sort(key=lambda x: x["diff"], reverse=True)

print("\nTop 20 变化最显著的维度:")
for i, s in enumerate(dim_stats[:20]):
    sign = "+" if s["changed_mean"] > s["unchanged_mean"] else "-"
    print(f"  Dim {s['dim']:3d}: changed={s['changed_mean']:+.5f}, unchg={s['unchanged_mean']:+.5f}, "
          f"diff={s['diff']:.5f} [{sign}]")

# ── 5. 可视化 Top 维度 ──
N_TOP = 16
top_dims = [s["dim"] for s in dim_stats[:N_TOP]]

# 自定义 diverging colormap: 蓝色(负) -> 灰色(0) -> 红色(正)
colors = [(0.0, 0.2, 0.8), (0.85, 0.85, 0.85), (0.8, 0.1, 0.1)]
cmap = LinearSegmentedColormap.from_list("diverging", colors)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for idx, d in enumerate(top_dims):
    ax = axes[idx]
    dim_delta = delta[d]

    # 确定该维度的显示范围（以数据本身的 95% 分位数为准）
    vmax = np.percentile(np.abs(dim_delta), 95)
    vmin = -vmax

    im = ax.imshow(dim_delta, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    # 叠加变化边界（半透明）
    from scipy import ndimage
    edges = ndimage.binary_dilation(change_mask > 0) ^ (change_mask > 0)
    ax.contour(edges, colors="lime", linewidths=0.8, levels=[0.5])

    s = dim_stats[idx]
    sign = "▲" if s["changed_mean"] > s["unchanged_mean"] else "▼"
    ax.set_title(f"Dim {d}  {sign}\nchg={s['changed_mean']:+.4f}  unchg={s['unchanged_mean']:+.4f}",
                 fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"{PATCH_ID}  Top {N_TOP} Embedding Dimensions by Change Sensitivity\n"
             f"Red=Increase  Blue=Decrease  Green contour=Change boundary\n"
             f"Period: {PERIOD[0]} → {PERIOD[1]}  |  Changed: {n_changed}/{n_total} pixels",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT_DIR / f"dim_change_heatmap_{PATCH_ID}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {OUT_DIR / f'dim_change_heatmap_{PATCH_ID}.png'}")

# ── 6. 箱线图: 所有维度的变化区域 vs 未变化区域 ──
fig, ax = plt.subplots(figsize=(20, 6))

changed_vals = []
unchanged_vals = []
dim_labels = []

# 只展示 top 40 维度
for s in dim_stats[:40]:
    d = s["dim"]
    dim_delta = delta[d]
    changed_vals.append(dim_delta[change_mask > 0].flatten())
    unchanged_vals.append(dim_delta[change_mask == 0].flatten())
    dim_labels.append(str(d))

# 绘制箱线图
positions = np.arange(len(dim_labels)) * 2.0
bp1 = ax.boxplot(changed_vals, positions=positions - 0.4, widths=0.7,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor="#e74c3c", alpha=0.7),
                  medianprops=dict(color="white", linewidth=2))
bp2 = ax.boxplot(unchanged_vals, positions=positions + 0.4, widths=0.7,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor="#3498db", alpha=0.7),
                  medianprops=dict(color="white", linewidth=2))

ax.set_xticks(positions)
ax.set_xticklabels(dim_labels, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", linestyle="--", alpha=0.3)
ax.set_xlabel("Embedding Dimension (sorted by sensitivity)", fontsize=12)
ax.set_ylabel("Δ Embedding (After - Before)", fontsize=12)
ax.set_title(f"{PATCH_ID}  Change-induced Embedding Shift by Dimension\n"
             f"Red=Changed pixels  Blue=Unchanged pixels", fontsize=13, fontweight="bold")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Changed", "Unchanged"], loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(OUT_DIR / f"dim_change_boxplot_{PATCH_ID}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / f'dim_change_boxplot_{PATCH_ID}.png'}")

# ── 7. 统计报告 ──
report = {
    "patch_id": PATCH_ID,
    "period": f"{PERIOD[0]}~{PERIOD[1]}",
    "embedding_dim": D,
    "grid_size": H,
    "n_changed_pixels": n_changed,
    "n_unchanged_pixels": n_total - n_changed,
    "top_20_dimensions": dim_stats[:20],
}

with open(OUT_DIR / f"dim_change_analysis_{PATCH_ID}.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved: {OUT_DIR / f'dim_change_analysis_{PATCH_ID}.json'}")

# ── 8. 关键发现 ──
print("\n" + "="*60)
print("关键发现")
print("="*60)

n_significant = sum(1 for s in dim_stats if s["diff"] > 0.001)
print(f"差异显著的维度 (diff > 0.001): {n_significant}/{D}")

# 变化区域是否有系统性偏移？
changed_means = [s["changed_mean"] for s in dim_stats]
unchanged_means = [s["unchanged_mean"] for s in dim_stats]

changed_abs_mean = np.mean(np.abs(changed_means))
unchanged_abs_mean = np.mean(np.abs(unchanged_means))
print(f"变化区域平均 |Δ|: {changed_abs_mean:.6f}")
print(f"未变化区域平均 |Δ|: {unchanged_abs_mean:.6f}")
print(f"比率: {changed_abs_mean / max(unchanged_abs_mean, 1e-8):.2f}x")

if changed_abs_mean < 1.5 * unchanged_abs_mean:
    print("\n⚠️ 结论: 变化区域和未变化区域的 embedding 偏移幅度相近!")
    print("   变化信息没有被'隐藏'在某些维度中——而是整体上几乎没有被编码。")
else:
    print("\n✅ 结论: 变化区域有明显的系统性偏移，变化信息蕴含在 embedding 中。")
    print("   但可能被 L2 norm 压缩，导致 raw cosine distance 无法捕捉。")

print("="*60)
