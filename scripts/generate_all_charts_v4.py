"""生成汇报所需全部7张图表（英文版）"""
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 使用已安装的Noto CJK字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "/workspace/xuannv/references/charts"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Chart 1: Change Detection BA Comparison ==========
fig, ax = plt.subplots(figsize=(10, 6))
models = ['AEF\n(Google DeepMind)', 'XuanNv Base\n+ CD Head', 'XuanNv Base\n(Raw Embedding)']
bas = [0.713, 0.798, 0.500]
colors = ['#999999', '#E74C3C', '#F39C12']
bars = ax.bar(models, bas, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Balanced Accuracy (BA)', fontsize=14)
ax.set_title('Change Detection Accuracy: Unsupervised / Light-supervision', fontsize=16, fontweight='bold')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline (0.5)')
for bar, val in zip(bars, bas):
    height = bar.get_height()
    ax.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_cd_comparison_ba.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 1: Change Detection BA Comparison")

# ========== Chart 2: Competitor Radar Chart ==========
from math import pi
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['Compression\nEfficiency', 'Temporal\nSensitivity', 'Unsupervised\nChange Detection',
              'Multi-modal\nFusion', 'Retrieval\nTool', 'China Scene\nValidation']
aef_scores = [10, 5, 7, 9, 2, 3]
clay_scores = [3, 6, 5, 8, 9, 4]
xuannv_scores = [10, 9, 6, 9, 7, 10]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
for scores, label, color in [(aef_scores, 'AEF', '#999999'),
                              (clay_scores, 'Clay', '#3498DB'),
                              (xuannv_scores, 'XuanNv Base', '#E74C3C')]:
    values = scores + scores[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 10)
ax.set_title('Earth Embedding Model Capability Comparison', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_competitor_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 2: Competitor Radar Chart")

# ========== Chart 3: V5 BA/AUC Distribution ==========
with open('/workspace/outputs/aef_qwen_v5_mixed_scale/eval/benchmark_v5_final_summary.json') as f:
    data = json.load(f)
head_bas = [r['ba'] for r in data['records']]
head_aucs = [r['auc'] for r in data['records']]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(head_bas, bins=15, color='#E74C3C', edgecolor='black', alpha=0.8)
axes[0].axvline(x=0.798, color='darkred', linestyle='--', linewidth=2, label=f'Mean: 0.798')
axes[0].axvline(x=0.713, color='#999999', linestyle='--', linewidth=2, label='AEF: 0.713')
axes[0].set_xlabel('Balanced Accuracy', fontsize=12)
axes[0].set_ylabel('Number of Patches', fontsize=12)
axes[0].set_title('XuanNv V5 BA Distribution (69 patches)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].hist(head_aucs, bins=15, color='#3498DB', edgecolor='black', alpha=0.8)
axes[1].axvline(x=0.955, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: 0.955')
axes[1].set_xlabel('AUC', fontsize=12)
axes[1].set_ylabel('Number of Patches', fontsize=12)
axes[1].set_title('XuanNv V5 AUC Distribution (69 patches)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_v5_ba_auc_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 3: V5 BA/AUC Distribution")

# ========== Chart 4: Efficiency Comparison ==========
fig, ax = plt.subplots(figsize=(10, 6))
dimensions = ['Development\nCycle (mo)', 'Human\nEffort (ppl)', 'Storage\nCost (TB/yr)', 'Coverage\n(%)', 'Response\nSpeed (hr)']
traditional = [6, 8, 100, 30, 504]
xuannv = [2, 3, 20, 100, 24]
improvements = [t/x for t, x in zip(traditional, xuannv)]
colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB']
bars = ax.barh(dimensions, improvements, color=colors, edgecolor='black')
ax.set_xlabel('Improvement Factor (x)', fontsize=14)
ax.set_title('Efficiency Gain: XuanNv Base vs Traditional RS Analysis', fontsize=16, fontweight='bold')
for bar, val in zip(bars, improvements):
    width = bar.get_width()
    ax.annotate(f'{val:.1f}x', xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_efficiency_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 4: Efficiency Comparison")

# ========== Chart 5: Model Architecture Flow (NEW) ==========
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Define boxes
boxes = [
    (0.5, 3.5, 2.5, 1.5, 'Input\nS2 + S1 + Landsat\n(Time Series)', '#3498DB'),
    (4, 3.5, 2.5, 1.5, 'STP Encoder\nSpace + Time + Precision', '#9B59B6'),
    (7.5, 3.5, 2.5, 1.5, 'VMF Bottleneck\nSkip L2 (training)\nL2 + VMF (inference)', '#E74C3C'),
    (11, 3.5, 2.5, 1.5, 'Output\n64-dim Embedding\n64 bytes (int8)', '#2ECC71'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Arrows between boxes
arrow_style = dict(arrowstyle='->', color='black', lw=2)
ax.annotate('', xy=(4, 4.25), xytext=(3, 4.25), arrowprops=arrow_style)
ax.annotate('', xy=(7.5, 4.25), xytext=(6.5, 4.25), arrowprops=arrow_style)
ax.annotate('', xy=(11, 4.25), xytext=(10, 4.25), arrowprops=arrow_style)

# Training objectives below
obj_boxes = [
    (3.5, 1.0, 3.0, 1.2, 'Reconstruction Loss\n(7 target sources)', '#1ABC9C'),
    (7.0, 1.0, 3.0, 1.2, 'Anti-Collapse Loss\n(Raw Uniformity + Decorrelation\n+ Variance + Orthogonality)', '#E67E22'),
]
for x, y, w, h, text, color in obj_boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrows from encoder to objectives
ax.annotate('', xy=(5.0, 2.2), xytext=(5.25, 3.5), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=-0.2'))
ax.annotate('', xy=(8.5, 2.2), xytext=(8.25, 3.5), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.2'))

# Temporal contrastive label
ax.text(5.25, 0.3, 'Temporal Contrastive Loss: Two non-overlapping time windows → different embeddings',
        ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='#F39C12', alpha=0.3))

ax.set_title('XuanNv Base: Model Architecture & Training Objectives', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_model_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 5: Model Architecture Flow")

# ========== Chart 6: Compression Ladder (NEW) ==========
fig, ax = plt.subplots(figsize=(10, 7))
steps = ['Raw Satellite\nImagery\n(MajorTOM)', 'EarthEmbeddingExplorer\n(6 models, float32)', 'XuanNv Base\n(64-dim int8)']
sizes = [23e3, 3, 0.016]  # GB
sizes_gb = [23000, 3, 0.016]
colors = ['#2C3E50', '#3498DB', '#E74C3C']

# Draw ladder
for i, (step, size, color) in enumerate(zip(steps, sizes_gb, colors)):
    y = 2 - i * 0.7
    rect = plt.Rectangle((2, y), 6, 0.5, facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(5, y + 0.25, step, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    if size >= 1:
        label = f'{size:.0f} GB' if size < 1000 else f'{size/1000:.0f} TB'
    else:
        label = f'{size*1000:.0f} MB'
    ax.text(8.5, y + 0.25, label, ha='left', va='center', fontsize=14, fontweight='bold', color=color)
    if i < 2:
        ax.annotate('', xy=(5, y - 0.2), xytext=(5, y - 0.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ratio = sizes_gb[i] / sizes_gb[i+1]
        ax.text(5.3, y - 0.35, f'{ratio:,.0f}x smaller', ha='left', va='center', fontsize=10, color='gray')

ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 2.8)
ax.axis('off')
ax.set_title('Compression Ladder: From 23TB Raw Data to 16MB Embeddings', fontsize=16, fontweight='bold', pad=20)
# Bottom note
ax.text(5, -0.3, '23TB → 3GB → 16MB = ~1,500,000x compression.\nAny device can run offline.',
        ha='center', fontsize=12, style='italic', bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.8))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_compression_ladder.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 6: Compression Ladder")

# ========== Chart 7: Roadmap V6/V6.5/V7 (NEW) ==========
fig, ax = plt.subplots(figsize=(12, 5))
versions = ['V5\n(Current)', 'V6', 'V6.5', 'V7\n(Planned)']
x_pos = [1, 3, 5, 7]
model_improvements = [0.5, 0.6, 0.7, 0.75]  # Target raw BA
colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71']

bars = ax.bar(x_pos, model_improvements, color=colors, width=1.2, edgecolor='black', linewidth=1.5)
ax.axhline(y=0.713, color='#999999', linestyle='--', linewidth=2, label='AEF Baseline (BA=0.713)')
ax.set_ylim(0, 1.0)
ax.set_xticks(x_pos)
ax.set_xticklabels(versions, fontsize=12)
ax.set_ylabel('Target Raw Embedding BA', fontsize=13)
ax.set_title('XuanNv Base Roadmap: Closing the Gap with AEF', fontsize=16, fontweight='bold')

# Annotations
annotations = [
    'Mixed-scale dual window\n+ CD Head\nHarbin 695km²',
    'Pixel-level temporal loss\nEnhanced contrastive learning',
    'Gap-aware temporal sensing\nMore change samples',
    'Temporal Transformer\nSelf-supervised pretraining\nNational-scale data'
]
for x, y, ann, color in zip(x_pos, model_improvements, annotations, colors):
    ax.annotate(ann, xy=(x, y), xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

ax.legend(fontsize=11, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_roadmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Chart 7: Roadmap")

print(f"\nAll 7 charts saved to: {OUTPUT_DIR}/")
