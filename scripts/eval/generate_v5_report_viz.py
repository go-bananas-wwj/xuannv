#!/usr/bin/env python3
"""生成 V5 评估报告的可视化图表."""
import sys
import json
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval")
V4_BENCHMARK = Path("/workspace/outputs/aef_qwen_v4_official/eval/benchmark_full69_summary.json")
V5_BENCHMARK = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval/benchmark_full69_summary.json")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. V5 vs V4 AUC per-patch scatter
ax = axes[0, 0]
if V4_BENCHMARK.exists() and V5_BENCHMARK.exists():
    with open(V4_BENCHMARK) as f:
        v4_data = json.load(f)
    with open(V5_BENCHMARK) as f:
        v5_data = json.load(f)
    
    v4_aucs = {r["patch_id"]: r["auc"] for r in v4_data["records"]}
    v5_aucs = {r["patch_id"]: r["auc"] for r in v5_data["records"]}
    
    common_pids = sorted(set(v4_aucs.keys()) & set(v5_aucs.keys()))
    v4_vals = [v4_aucs[pid] for pid in common_pids]
    v5_vals = [v5_aucs[pid] for pid in common_pids]
    
    ax.scatter(v4_vals, v5_vals, alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('V4 CD Head AUC')
    ax.set_ylabel('V5 CD Head AUC')
    ax.set_title(f'V5 vs V4 CD Head AUC\n({len(common_pids)} common patches)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    better = sum(1 for v4, v5 in zip(v4_vals, v5_vals) if v5 > v4)
    ax.text(0.05, 0.95, f'V5 better: {better}/{len(common_pids)}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax.text(0.5, 0.5, 'Benchmark data not yet available', ha='center', va='center')
    ax.set_title('V5 vs V4 CD Head AUC')

# 2. V5 Level 1 Bare AUC bar chart
ax = axes[0, 1]
level1_path = EVAL_DIR / "level1_bare_auc.json"
if level1_path.exists():
    with open(level1_path) as f:
        l1 = json.load(f)
    pids = [d["patch_id"] for d in l1["patch_details"]]
    aucs = [d["auc"] for d in l1["patch_details"]]
    colors = ['red' if a < 0.5 else 'orange' if a < 0.6 else 'green' for a in aucs]
    ax.barh(range(len(pids)), aucs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels(pids, fontsize=6)
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='random=0.5')
    ax.set_xlabel('Bare AUC')
    ax.set_title(f'V5 Backbone Bare AUC\nMean={l1["auc_mean"]:.3f}')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'Level 1 data not available', ha='center', va='center')

# 3. Embedding space diagnosis radar / metrics
ax = axes[1, 0]
diag_path = EVAL_DIR / "embedding_space_diagnosis.json"
if diag_path.exists():
    with open(diag_path) as f:
        diag = json.load(f)
    
    metrics = ['Mean Cosine\nDistance', 'Changed\nDistance', 'Unchanged\nDistance', 'Separation']
    v5_vals = [
        diag["mean_pairwise_cosine_distance"],
        diag["changed_distance_mean"],
        diag["unchanged_distance_mean"],
        diag["separation"],
    ]
    # Normalize to 0-1 scale for visualization
    v5_vals_norm = [min(v * 2, 1.0) for v in v5_vals]  # scale up since values are small
    
    x = np.arange(len(metrics))
    width = 0.35
    bars = ax.bar(x, v5_vals, width, color='steelblue', alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('V5 Embedding Space Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='separation threshold=0.05')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text labels on bars
    for bar, val in zip(bars, v5_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
else:
    ax.text(0.5, 0.5, 'Diagnosis data not available', ha='center', va='center')

# 4. Summary comparison table / text
ax = axes[1, 1]
ax.axis('off')

summary_text = ""
if level1_path.exists():
    with open(level1_path) as f:
        l1 = json.load(f)
    summary_text += f"""
【V5 Backbone Bare AUC】
  Mean:    {l1['auc_mean']:.3f}
  Median:  {l1['auc_median']:.3f}
  >0.5:    {sum(1 for d in l1['patch_details'] if d['auc']>0.5)}/{l1['n_patches']}
  >0.6:    {sum(1 for d in l1['patch_details'] if d['auc']>0.6)}/{l1['n_patches']}
"""

if diag_path.exists():
    with open(diag_path) as f:
        diag = json.load(f)
    summary_text += f"""
【V5 Embedding Space】
  Uniformity:        {diag['uniformity']:.3f}
  Mean Cos Dist:     {diag['mean_pairwise_cosine_distance']:.4f}
  Changed Dist:      {diag['changed_distance_mean']:.4f}
  Unchanged Dist:    {diag['unchanged_distance_mean']:.4f}
  Separation:        {diag['separation']:.4f} (threshold: 0.05)
  Per-Patch AUC:     {diag['per_patch_auc_mean']:.3f}
"""

if V5_BENCHMARK.exists():
    with open(V5_BENCHMARK) as f:
        v5_b = json.load(f)
    summary_text += f"""
【V5 CD Head Benchmark】
  Head AUC:  {v5_b['head']['auc_mean']:.4f} (median: {v5_b['head']['auc_median']:.4f})
  Raw AUC:   {v5_b['raw']['auc_mean']:.4f} (median: {v5_b['raw']['auc_median']:.4f})
"""
    if V4_BENCHMARK.exists():
        with open(V4_BENCHMARK) as f:
            v4_b = json.load(f)
        summary_text += f"""
【V4 Baseline (参考)】
  Head AUC:  {v4_b['head']['auc_mean']:.4f} (median: {v4_b['head']['auc_median']:.4f})
  Raw AUC:   {v4_b['raw']['auc_mean']:.4f} (median: {v4_b['raw']['auc_median']:.4f})
"""

if not summary_text:
    summary_text = "Data not yet available."

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.suptitle('V5 Mixed Scale Evaluation Dashboard', fontsize=14)
plt.tight_layout()
out_path = EVAL_DIR / "v5_evaluation_dashboard.png"
plt.savefig(out_path, dpi=150)
print(f"Saved dashboard to {out_path}")
