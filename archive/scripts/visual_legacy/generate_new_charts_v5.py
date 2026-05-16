"""生成v5新增3张图表（科研风格，蓝灰色调）"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.unicode_minus': False,
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 1.0,
    'xtick.color': '#2c3e50',
    'ytick.color': '#2c3e50',
    'axes.labelcolor': '#2c3e50',
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})

C_PRIMARY = '#1a5276'
C_SECONDARY = '#5dade2'
C_TERTIARY = '#85c1e9'
C_NEUTRAL = '#7f8c8d'
C_LIGHT_GRAY = '#bdc3c7'
C_GRID = '#d5dbdb'
C_TEXT = '#2c3e50'
C_DARK = '#34495e'

OUTPUT_DIR = "/workspace/xuannv/references/charts"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Chart 8: 五大下游任务能力矩阵 ==========
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

tasks = [
    ("Change Detection", "PyTorch CD Head V3", "GPU", "✓"),
    ("WorldCover", "sklearn Linear Probe", "CPU", "✓"),
    ("Dynamic World", "sklearn Linear Probe", "CPU", "✓"),
    ("JRC Water", "sklearn Linear Probe", "CPU", "✓"),
    ("Building Extract", "sklearn Linear Probe", "CPU", "✓"),
]

# Header
headers = ["Task Head", "Technical Solution", "Compute", "Status"]
col_x = [0.3, 3.5, 7.2, 9.0]
header_colors = [C_DARK, C_DARK, C_DARK, C_DARK]

for x, h, c in zip(col_x, headers, header_colors):
    rect = plt.Rectangle((x, 5.0), [2.8, 3.3, 1.5, 1.5][col_x.index(x)], 0.7,
                          facecolor=c, edgecolor='white', linewidth=0.5, alpha=0.95)
    ax.add_patch(rect)
    ax.text(x + [2.8, 3.3, 1.5, 1.5][col_x.index(x)]/2, 5.35, h,
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Rows
row_colors = [C_PRIMARY, '#2874a6', '#3f88c5', '#5dade2', C_SECONDARY]
for i, (task, tech, comp, status) in enumerate(tasks):
    y = 4.1 - i * 0.85
    cells = [task, tech, comp, status]
    for x, cell, c in zip(col_x, cells, row_colors):
        w = [2.8, 3.3, 1.5, 1.5][col_x.index(x)]
        rect = plt.Rectangle((x, y), w, 0.75,
                              facecolor=c if col_x.index(x) == 0 else '#f8f9f9',
                              edgecolor=C_GRID, linewidth=0.8, alpha=0.9)
        ax.add_patch(rect)
        color = 'white' if col_x.index(x) == 0 else C_TEXT
        ax.text(x + w/2, y + 0.375, cell,
                ha='center', va='center', fontsize=10, fontweight='bold' if col_x.index(x)==0 else 'normal',
                color=color)

# Bottom note
ax.text(5.5, 0.3, '5 Task Heads · Single patch inference ~10 ms (CPU) · CD Head auto-selects freest GPU',
        ha='center', fontsize=9, style='italic', color=C_NEUTRAL,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9f9', edgecolor=C_GRID, linewidth=0.8))

ax.set_title('Downstream Task Capability Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_downstream_tasks.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 8: Downstream Tasks")

# ========== Chart 9: 交互式标注训练流程 ==========
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.5)
ax.axis('off')

steps = [
    (0.3, 2.0, 2.3, 1.2, 'Upload\nSatellite Image', C_DARK),
    (3.0, 2.0, 2.3, 1.2, 'SAM3\nAuto-Segment', C_PRIMARY),
    (5.7, 2.0, 2.3, 1.2, 'Interactive\nCorrection', '#2874a6'),
    (8.4, 2.0, 2.3, 1.2, 'Train Linear\nProbe', '#3f88c5'),
    (11.1, 2.0, 1.6, 1.2, 'Inference\n& Show', C_SECONDARY),
]

for x, y, w, h, text, color in steps:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=C_TEXT, linewidth=1.0, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrows
arrow_style = dict(arrowstyle='->', color=C_TEXT, lw=1.5)
for i in range(len(steps)-1):
    x1 = steps[i][0] + steps[i][2]
    x2 = steps[i+1][0]
    y_mid = steps[i][1] + steps[i][3]/2
    ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_style)

# Bottom labels
labels = [
    (1.45, 1.5, 'User uploads\nraw imagery'),
    (4.15, 1.5, 'SAM3 generates\nmask proposals'),
    (6.85, 1.5, 'User clicks to\nkeep / discard'),
    (9.55, 1.5, 'LogisticRegression\ntrains in seconds'),
    (11.9, 1.5, 'Result overlay\non map'),
]
for x, y, text in labels:
    ax.text(x, y, text, ha='center', va='top', fontsize=8, color=C_NEUTRAL, style='italic')

ax.text(6.5, 0.4, 'Zero-code custom training — no deep learning knowledge required · Full pipeline ~5 min',
        ha='center', fontsize=9, style='italic', color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9f9', edgecolor=C_GRID, linewidth=0.8))

ax.set_title('Interactive Annotation & Custom Training Pipeline', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_annotation_pipeline.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 9: Annotation Pipeline")

# ========== Chart 10: AI智能体交互示意 ==========
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis('off')

# Left panel: Input
rect_left = plt.Rectangle((0.3, 1.0), 4.5, 3.5, facecolor='#f8f9f9', edgecolor=C_GRID, linewidth=1.5)
ax.add_patch(rect_left)
ax.text(2.55, 4.2, 'Natural Language Input', ha='center', va='center', fontsize=11, fontweight='bold', color=C_PRIMARY)

# Input mock text
input_text = '"统计哈尔滨2025年4月到10月的\n建设用地变化面积，\n并生成对比报告。"'
ax.text(2.55, 3.0, input_text, ha='center', va='center', fontsize=9, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=C_SECONDARY, linewidth=1.5))

ax.text(2.55, 1.6, 'User prompt in Chinese', ha='center', va='center', fontsize=8, color=C_NEUTRAL, style='italic')

# Right panel: Output
rect_right = plt.Rectangle((6.2, 1.0), 4.5, 3.5, facecolor='#f8f9f9', edgecolor=C_GRID, linewidth=1.5)
ax.add_patch(rect_right)
ax.text(8.45, 4.2, 'AI-Generated Report', ha='center', va='center', fontsize=11, fontweight='bold', color=C_PRIMARY)

# Output mock
output_lines = [
    '## 哈尔滨建设用地变化分析',
    '',
    '| 区域 | 变化面积 (km²) | 变化率 |',
    '|------|---------------|--------|',
    '| 松北区 | 12.3 | +8.2% |',
    '| 道里区 | 5.7 | +3.1% |',
    '',
    '> 主要变化类型：农田→建设用地',
    '',
    '*报告由 DeepSeek AI 生成*'
]
for j, line in enumerate(output_lines):
    ax.text(8.45, 3.4 - j*0.22, line, ha='center', va='top', fontsize=7.5, color=C_TEXT,
            family='monospace')

# Center arrow
ax.annotate('', xy=(6.0, 2.75), xytext=(4.9, 2.75),
            arrowprops=dict(arrowstyle='->', color=C_PRIMARY, lw=2.5,
                           connectionstyle='arc3,rad=0'))
ax.text(5.45, 3.0, 'DeepSeek\nV4', ha='center', va='bottom', fontsize=8, fontweight='bold', color=C_PRIMARY)

# Bottom pipeline
pipeline_steps = ['flash\nparse', 'task\nexecute', 'pro\npolish']
pipeline_x = [1.5, 5.5, 9.5]
for x, step in zip(pipeline_x, pipeline_steps):
    circle = plt.Circle((x, 0.5), 0.35, facecolor=C_SECONDARY, edgecolor=C_TEXT, linewidth=1.0, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, 0.5, step, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

for i in range(len(pipeline_x)-1):
    ax.annotate('', xy=(pipeline_x[i+1]-0.4, 0.5), xytext=(pipeline_x[i]+0.4, 0.5),
                arrowprops=dict(arrowstyle='->', color=C_NEUTRAL, lw=1.2))

ax.text(5.5, -0.15, 'Pipeline: deepseek-v4-flash parses → task executor fetches data → deepseek-v4-pro polishes',
        ha='center', fontsize=8, color=C_NEUTRAL, style='italic')

ax.set_title('AI Agent: Natural Language → Remote Sensing Report', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_agent_report.png', bbox_inches='tight')
plt.close()
print("[OK] Chart 10: Agent Report")

print(f"\nAll 3 new charts saved to: {OUTPUT_DIR}/")
