#!/usr/bin/env python3
"""生成对比可视化: (1) 同period跨实验对比 (2) KNN结果条形图."""
import sys, warnings, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default="/workspace/outputs/xuannv_round1",
                    help="实验输出根目录")
args = parser.parse_args()

EXPERIMENTS = [
    "aef_baseline", "aef_high_consist", "aef_no_static", "aef_skip_l2",
    "aef_diff_recon", "aef_high_kappa", "aef_cyclic_unif", "aef_no_uniform",
]
PERIODS = ["apr_jun", "jun_aug", "aug_sept", "sept_oct"]
DATA_ROOT = Path(args.output_dir)
VIZ_DIR = DATA_ROOT / "comparison_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. KNN 结果条形图 ──────────────────────────
print("Generating KNN bar chart...")
tasks = ["worldcover", "jrc_water", "dynamic_world"]
task_labels = ["WorldCover (10类)", "JRC Water (2类)", "Dynamic World (9类)"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(EXPERIMENTS)))

for ax, task, tlabel in zip(axes, tasks, task_labels):
    accs = []
    for exp in EXPERIMENTS:
        path = DATA_ROOT / exp / "downstream_knn" / "results.json"
        if not path.exists():
            accs.append(0)
            continue
        with open(path) as f:
            d = json.load(f)
        acc = d.get(task, {}).get("k5", {}).get("accuracy", 0)
        accs.append(acc * 100)
    
    bars = ax.barh(range(len(EXPERIMENTS)), accs, color=colors)
    ax.set_yticks(range(len(EXPERIMENTS)))
    ax.set_yticklabels([e.replace("aef_", "") for e in EXPERIMENTS], fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_title(tlabel, fontsize=12)
    ax.set_xlim(0, max(accs) * 1.2 if max(accs) > 0 else 10)
    for i, v in enumerate(accs):
        ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=8)
    ax.invert_yaxis()

fig.suptitle("Round 1: KNN Downstream Evaluation (K=5, Patch-wise 80/20 Split)", fontsize=14)
plt.tight_layout()
out = VIZ_DIR / "knn_results_comparison.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")

# ── 2. 同 period 跨实验 embedding distance 对比 ──────────────────────────
print("\nGenerating period-wise experiment comparison...")
for period in PERIODS:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, exp in enumerate(EXPERIMENTS):
        img_path = DATA_ROOT / "embedding_distance_viz" / f"{exp}_{period}_emb_dist.png"
        if not img_path.exists():
            axes[idx].text(0.5, 0.5, "No image", ha='center', va='center')
            axes[idx].axis('off')
            continue
        img = plt.imread(str(img_path))
        axes[idx].imshow(img)
        axes[idx].set_title(exp.replace("aef_", ""), fontsize=10)
        axes[idx].axis('off')
    
    fig.suptitle(f"Embedding Cosine Distance Comparison — {period}", fontsize=14)
    plt.tight_layout()
    out = VIZ_DIR / f"comparison_{period}_all_experiments.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

# ── 3. 每个实验的 4-period 汇总 ──────────────────────────
print("\nGenerating per-experiment 4-period summaries...")
for exp in EXPERIMENTS:
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, period in enumerate(PERIODS):
        img_path = DATA_ROOT / "embedding_distance_viz" / f"{exp}_{period}_emb_dist.png"
        if not img_path.exists():
            axes[idx].text(0.5, 0.5, "No image", ha='center', va='center')
            axes[idx].axis('off')
            continue
        img = plt.imread(str(img_path))
        axes[idx].imshow(img)
        axes[idx].set_title(period.replace("_", " → "), fontsize=12)
        axes[idx].axis('off')
    
    fig.suptitle(f"{exp} — All Periods", fontsize=14)
    plt.tight_layout()
    out = DATA_ROOT / exp / "viz" / "all_periods_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

# ── 4. 生成一个大的总览网格 ──────────────────────────
print("\nGenerating master overview grid...")
fig, axes = plt.subplots(len(EXPERIMENTS), len(PERIODS), figsize=(20, 40))
for i, exp in enumerate(EXPERIMENTS):
    for j, period in enumerate(PERIODS):
        img_path = DATA_ROOT / "embedding_distance_viz" / f"{exp}_{period}_emb_dist.png"
        if not img_path.exists():
            axes[i, j].text(0.5, 0.5, "N/A", ha='center', va='center')
            axes[i, j].axis('off')
            continue
        img = plt.imread(str(img_path))
        axes[i, j].imshow(img)
        if i == 0:
            axes[i, j].set_title(period.replace("_", " → "), fontsize=10)
        if j == 0:
            axes[i, j].set_ylabel(exp.replace("aef_", ""), fontsize=10, rotation=0, ha='right', va='center')
        axes[i, j].axis('off')

fig.suptitle("Round 1: Embedding Distance Overview (All Experiments × All Periods)", fontsize=16)
plt.tight_layout()
out = VIZ_DIR / "master_overview_all.png"
fig.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")

print(f"\n{'='*60}")
print(f"All comparison visualizations saved to {VIZ_DIR}")
print(f"{'='*60}")
