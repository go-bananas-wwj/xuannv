#!/usr/bin/env python3
"""修复 dim_xxx 单独图片的标题数据错误."""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import ndimage

PATCH_ID = "patch_000146"
OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval/dim146_prenorm")

# 加载已有分析数据
with open(OUT_DIR / f"prenorm_analysis_{PATCH_ID}.json") as f:
    report = json.load(f)

stats_pre = report["pre_norm"]["top_20"]
# 注意: JSON 中只有 top_20，我们需要全部 128 维度的 stats
# 从保存的 .npy 中重新计算

# 重新加载 delta 和 change_mask
delta = np.load(OUT_DIR / ".delta.npy") if (OUT_DIR / ".delta.npy").exists() else None
change_mask = np.load(OUT_DIR / ".mask.npy") if (OUT_DIR / ".mask.npy").exists() else None

if delta is None:
    print("ERROR: .delta.npy not found. Need to re-run full script.")
    exit(1)

D, H, W = delta.shape
n_changed = int(change_mask.sum())

# 重新计算全部维度 stats
stats_map = {}
for d in range(D):
    dim_delta = delta[d]
    changed_mean = float(dim_delta[change_mask > 0].mean()) if n_changed > 0 else 0.0
    unchanged_mean = float(dim_delta[change_mask == 0].mean())
    diff = abs(changed_mean - unchanged_mean)
    stats_map[d] = {
        "dim": d,
        "changed_mean": changed_mean,
        "unchanged_mean": unchanged_mean,
        "diff": diff,
    }

# 排序用于 rank
sorted_stats = sorted(stats_map.values(), key=lambda x: x["diff"], reverse=True)
rank_map = {s["dim"]: i for i, s in enumerate(sorted_stats)}

# 颜色
colors = [(0.0, 0.2, 0.8), (0.85, 0.85, 0.85), (0.8, 0.1, 0.1)]
cmap = LinearSegmentedColormap.from_list("diverging", colors)
edges = ndimage.binary_dilation(change_mask > 0) ^ (change_mask > 0)

# 重新生成每张图
print(f"Regenerating {D} dimension images...")
for d in range(D):
    dim_delta = delta[d]
    vmax = np.percentile(np.abs(dim_delta), 95)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(dim_delta, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.contour(edges, colors="lime", linewidths=1.0, levels=[0.5])

    s = stats_map[d]
    rank = rank_map[d]
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
        print(f"  ... done {d+1}/{D}")

print("Done!")
