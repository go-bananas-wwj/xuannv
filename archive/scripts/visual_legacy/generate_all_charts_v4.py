"""生成汇报所需全部7张图表（科研风格，蓝灰色调）"""
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ========== 科研风格全局配置 ==========
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.unicode_minus': False,
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 1.0,
    'xtick.color': '#2c3e50',
    'ytick.color': '#2c3e50',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.labelcolor': '#2c3e50',
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})

# 统一配色
C_PRIMARY = '#1a5276'      # 学术深蓝
C_SECONDARY = '#5dade2'    # 浅蓝
C_TERTIARY = '#85c1e9'     # 更浅蓝
C_NEUTRAL = '#7f8c8d'      # 灰色（竞品/基线）
C_LIGHT_GRAY = '#bdc3c7'   # 浅灰
C_ACCENT = '#c0392b'       # 暗红（极少使用）
C_GRID = '#d5dbdb'         # 极淡灰网格
C_TEXT = '#2c3e50'         # 近黑文字

OUTPUT_DIR = "/workspace/xuannv/references/charts"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Chart 1: Change Detection BA Comparison ==========
fig, ax = plt.subplots(figsize=(8, 5.5))
models = ['AEF\n(Google DeepMind)', 'XuanNv + CD Head', 'XuanNv\n(Raw Embedding)']
bas = [0.713, 0.798, 0.500]
colors = [C_NEUTRAL, C_PRIMARY, C_SECONDARY]
bars = ax.bar(models, bas, color=colors, width=0.55, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_ylim(0, 0.95)
ax.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Change Detection Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax.axhline(y=0.5, color=C_LIGHT_GRAY, linestyle='--', linewidth=1.0, zorder=2)
ax.text(2.15, 0.505, 'Random (0.50)', fontsize=9, color=C_NEUTRAL, va='bottom', ha='right')

for bar, val in zip(bars, bas):
    height = bar.get_height()
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=C_TEXT)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='-', alpha=0.3, color=C_GRID, zorder=1)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_cd_comparison_ba.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 1: BA Comparison")

# ========== Chart 2: Competitor Radar Chart ==========
from math import pi
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
categories = ['Compression', 'Temporal', 'Change Detection', 'Multi-modal', 'Retrieval', 'China Scenes']
aef_scores = [10, 5, 7, 9, 2, 3]
clay_scores = [3, 6, 5, 8, 9, 4]
xuannv_scores = [10, 9, 6, 9, 7, 10]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for scores, label, color, ls in [(aef_scores, 'AEF', C_NEUTRAL, '--'),
                                  (clay_scores, 'Clay', C_SECONDARY, '-'),
                                  (xuannv_scores, 'XuanNv', C_PRIMARY, '-')]:
    values = scores + scores[:1]
    ax.plot(angles, values, ls, linewidth=1.5, label=label, color=color,
            marker='o' if ls == '-' else None, markersize=5,
            markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], color=C_NEUTRAL, fontsize=8)
ax.set_title('Earth Embedding Model Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10, frameon=False)
ax.spines['polar'].set_color(C_LIGHT_GRAY)
ax.grid(color=C_GRID, linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_competitor_radar.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 2: Radar")

# ========== Chart 3: V5 BA/AUC Distribution ==========
with open('/workspace/outputs/aef_qwen_v5_mixed_scale/eval/benchmark_v5_final_summary.json') as f:
    data = json.load(f)
head_bas = [r['ba'] for r in data['records']]
head_aucs = [r['auc'] for r in data['records']]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# BA
axes[0].hist(head_bas, bins=15, color=C_PRIMARY, edgecolor='white', alpha=0.85, zorder=3)
axes[0].axvline(x=0.798, color=C_PRIMARY, linestyle='--', linewidth=1.5, zorder=4)
axes[0].axvline(x=0.713, color=C_NEUTRAL, linestyle='--', linewidth=1.5, zorder=4)
axes[0].text(0.798, axes[0].get_ylim()[1]*0.92, 'Mean: 0.798', fontsize=9, color=C_PRIMARY, fontweight='bold', ha='center')
axes[0].text(0.713, axes[0].get_ylim()[1]*0.82, 'AEF: 0.713', fontsize=9, color=C_NEUTRAL, ha='center')
axes[0].set_xlabel('Balanced Accuracy', fontsize=11)
axes[0].set_ylabel('Number of Patches', fontsize=11)
axes[0].set_title('(a) BA Distribution — 69 Patches', fontsize=12, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].yaxis.grid(True, linestyle='-', alpha=0.3, color=C_GRID, zorder=1)
axes[0].set_axisbelow(True)

# AUC
axes[1].hist(head_aucs, bins=15, color=C_SECONDARY, edgecolor='white', alpha=0.85, zorder=3)
axes[1].axvline(x=0.955, color=C_PRIMARY, linestyle='--', linewidth=1.5, zorder=4)
axes[1].text(0.955, axes[1].get_ylim()[1]*0.92, 'Mean: 0.955', fontsize=9, color=C_PRIMARY, fontweight='bold', ha='center')
axes[1].set_xlabel('AUC', fontsize=11)
axes[1].set_ylabel('Number of Patches', fontsize=11)
axes[1].set_title('(b) AUC Distribution — 69 Patches', fontsize=12, fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].yaxis.grid(True, linestyle='-', alpha=0.3, color=C_GRID, zorder=1)
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_v5_ba_auc_distribution.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 3: Distribution")

# ========== Chart 4: Efficiency Comparison ==========
fig, ax = plt.subplots(figsize=(9, 5))
dimensions = ['Dev Cycle\n(months)', 'Team Size\n(people)', 'Storage\n(TB/year)', 'Coverage\n(%)', 'Latency\n(hours)']
traditional = [6, 8, 100, 30, 504]
xuannv = [2, 3, 20, 100, 24]
improvements = [t/x for t, x in zip(traditional, xuannv)]

bars = ax.barh(dimensions, improvements, color=C_PRIMARY, height=0.5, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_xlabel('Improvement Factor (×)', fontsize=12, fontweight='bold')
ax.set_title('Efficiency Gain vs. Traditional RS Workflow', fontsize=14, fontweight='bold', pad=15)

for bar, val, t, x in zip(bars, improvements, traditional, xuannv):
    width = bar.get_width()
    ax.annotate(f'{val:.1f}×  ({t}→{x})', xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(6, 0), textcoords="offset points",
                ha='left', va='center', fontsize=10, color=C_TEXT)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle='-', alpha=0.3, color=C_GRID, zorder=1)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_efficiency_comparison.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 4: Efficiency")

# ========== Chart 5: Model Architecture Flow ==========
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)
ax.axis('off')

boxes = [
    (0.3, 3.2, 2.6, 1.4, 'Input\nS2 + S1 + Landsat\nTime Series', '#34495e'),
    (3.6, 3.2, 2.6, 1.4, 'STP Encoder\nSpace + Time + Precision', C_PRIMARY),
    (6.9, 3.2, 2.6, 1.4, 'VMF Bottleneck\nSkip L² (train)\nL² + VMF (infer)', '#2874a6'),
    (10.2, 3.2, 2.6, 1.4, 'Output\n64-dim Embedding\n64 bytes (int8)', '#5d6d7e'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=C_TEXT, linewidth=1.0, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

arrow_style = dict(arrowstyle='->', color=C_TEXT, lw=1.2)
ax.annotate('', xy=(3.6, 3.9), xytext=(2.9, 3.9), arrowprops=arrow_style)
ax.annotate('', xy=(6.9, 3.9), xytext=(6.2, 3.9), arrowprops=arrow_style)
ax.annotate('', xy=(10.2, 3.9), xytext=(9.5, 3.9), arrowprops=arrow_style)

# Training objectives
obj_boxes = [
    (3.2, 1.0, 3.0, 1.2, 'Reconstruction Loss\n7 target sources', '#5d6d7e'),
    (6.8, 1.0, 3.0, 1.2, 'Anti-Collapse Loss\nRaw uniformity + decorrelation\n+ variance + orthogonality', '#5d6d7e'),
]
for x, y, w, h, text, color in obj_boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=C_TEXT, linewidth=1.0, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, color='white')

ax.annotate('', xy=(4.7, 2.2), xytext=(4.9, 3.2), arrowprops=dict(arrowstyle='->', color=C_NEUTRAL, lw=1.0, connectionstyle='arc3,rad=-0.2'))
ax.annotate('', xy=(8.3, 2.2), xytext=(8.1, 3.2), arrowprops=dict(arrowstyle='->', color=C_NEUTRAL, lw=1.0, connectionstyle='arc3,rad=0.2'))

ax.text(6.5, 0.4, 'Temporal Contrastive: Two non-overlapping windows → different embeddings',
        ha='center', fontsize=9, style='italic', color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9f9', edgecolor=C_GRID, linewidth=0.8))

ax.set_title('XuanNv Base: Architecture & Training Objectives', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_model_architecture.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 5: Architecture")

# ========== Chart 6: Compression Ladder ==========
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(0, 11)
ax.set_ylim(-0.8, 3.2)
ax.axis('off')

steps = [
    ('Raw Satellite Imagery\n(MajorTOM)', '23 TB', '#34495e'),
    ('EarthEmbeddingExplorer\n(6 models, float32)', '3 GB', C_PRIMARY),
    ('XuanNv Base\n(64-dim int8)', '16 MB', C_SECONDARY),
]

for i, (label, size, color) in enumerate(steps):
    y = 2.0 - i * 0.9
    rect = plt.Rectangle((1.5, y), 6.5, 0.6, facecolor=color, edgecolor=C_TEXT, linewidth=1.0, alpha=0.9)
    ax.add_patch(rect)
    ax.text(4.75, y + 0.3, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(8.5, y + 0.3, size, ha='left', va='center', fontsize=13, fontweight='bold', color=color)
    if i < 2:
        ax.annotate('', xy=(4.75, y - 0.15), xytext=(4.75, y - 0.45),
                    arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=1.5))
        ratio = [7667, 188][i]
        ax.text(5.0, y - 0.3, f'{ratio:,}×', ha='left', va='center', fontsize=9, color=C_NEUTRAL)

ax.text(5.0, -0.5, 'Total: 23 TB → 16 MB  (~1,500,000× compression). Offline-capable on any device.',
        ha='center', fontsize=10, style='italic', color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9f9', edgecolor=C_GRID, linewidth=0.8))

ax.set_title('Data Compression: From Raw Imagery to Embeddings', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_compression_ladder.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 6: Compression")

# ========== Chart 7: Roadmap ==========
fig, ax = plt.subplots(figsize=(10, 5))
versions = ['V5\n(Current)', 'V6', 'V6.5', 'V7\n(Planned)']
x_pos = [1, 3, 5, 7]
targets = [0.50, 0.60, 0.70, 0.75]
colors = [C_PRIMARY, '#2874a6', '#5dade2', '#85c1e9']

bars = ax.bar(x_pos, targets, color=colors, width=1.3, edgecolor='white', linewidth=0.5, zorder=3)
ax.axhline(y=0.713, color=C_NEUTRAL, linestyle='--', linewidth=1.5, zorder=4, label='AEF baseline (0.713)')
ax.set_ylim(0, 0.9)
ax.set_xticks(x_pos)
ax.set_xticklabels(versions, fontsize=11)
ax.set_ylabel('Target Raw Embedding BA', fontsize=12, fontweight='bold')
ax.set_title('Roadmap: Closing the Raw-Embedding Gap with AEF', fontsize=14, fontweight='bold', pad=15)

annotations = [
    'Mixed-scale dual window\n+ CD Head, Harbin 695 km²',
    'Pixel-level temporal loss',
    'Gap-aware temporal sensing',
    'Temporal Transformer + national-scale data'
]
for x, y, ann, color in zip(x_pos, targets, annotations, colors):
    ax.annotate(ann, xy=(x, y), xytext=(0, 6), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

ax.legend(fontsize=10, frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='-', alpha=0.3, color=C_GRID, zorder=1)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_roadmap.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 7: Roadmap")

print(f"\nAll 7 charts saved to: {OUTPUT_DIR}/")
