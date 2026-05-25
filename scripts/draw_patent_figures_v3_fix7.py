#!/usr/bin/env python3
"""专利附图 v3 修复7 — 图1：去掉编码层横向箭头"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("/workspace/xuannv/docs/patent_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LW_MAIN = 1.0
LW_THIN = 0.6
FS_BOX = 10
FS_LAYER = 11

def _rect(ax, x, y, w, h, lw=LW_MAIN):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='black', linewidth=lw, zorder=2))

def _text(ax, x, y, s, fontsize=FS_BOX, ha='center', va='center', bold=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, color='black', fontweight='bold' if bold else 'normal', zorder=3, linespacing=1.05)

def _arrow(ax, x1, y1, x2, y2, lw=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='black', lw=lw))

def draw_fig1():
    fig, ax = plt.subplots(figsize=(11, 15))
    ax.set_xlim(0, 100); ax.set_ylim(2, 105); ax.axis('off'); ax.set_facecolor('white')

    layer_h, gap = 8.5, 7.5
    box_center_x, box_w = 56, 72

    layers = [
        ("输入层",       ["光学遥感", "SAR遥感", "多光谱遥感", "无人机", "激光雷达", "地面传感器"]),
        ("网格划分层",   ["离散格网系统\n(多源数据对齐)"]),
        ("编码层",       ["专用输入头", "时间编码器", "多路径时空\n特征提取单元"]),
        ("嵌入层",       ["VMF瓶颈层\n(训练保留原始幅度\n推理L2归一化+噪声)"]),
        ("解码层",       ["动态目标解码器", "静态目标解码器"]),
        ("损失层",       ["重建损失", "反坍缩损失", "一致性损失", "时序对比损失"]),
    ]

    centers = []
    box_centers = []
    for idx, (name, items) in enumerate(layers):
        cy = 98 - idx * (layer_h + gap)
        centers.append(cy)
        _text(ax, 2, cy, name, fontsize=FS_LAYER, ha='left', bold=True)
        n = len(items)
        if n == 1:
            _rect(ax, box_center_x - box_w/2, cy - layer_h/2, box_w, layer_h)
            _text(ax, box_center_x, cy, items[0], fontsize=FS_BOX)
            box_centers.append([box_center_x])
        else:
            item_w = (box_w - (n - 1) * 1.5) / n
            start_x = box_center_x - box_w/2
            xs = []
            for i, item in enumerate(items):
                x = start_x + i * (item_w + 1.5)
                _rect(ax, x, cy - layer_h/2, item_w, layer_h)
                _text(ax, x + item_w/2, cy, item, fontsize=FS_BOX - 1)
                xs.append(x + item_w/2)
            box_centers.append(xs)

    # 层间竖直箭头
    for i in range(len(centers) - 1):
        _arrow(ax, box_center_x, centers[i] - layer_h/2 - 0.5, box_center_x, centers[i+1] + layer_h/2 + 0.5, lw=1.0)

    # 解码层 → 损失层
    dec_xs, loss_xs = box_centers[4], box_centers[5]
    _arrow(ax, dec_xs[0], centers[4] - layer_h/2 - 0.5, (loss_xs[0]+loss_xs[1])/2, centers[5] + layer_h/2 + 0.5, lw=0.9)
    _arrow(ax, dec_xs[1], centers[4] - layer_h/2 - 0.5, (loss_xs[2]+loss_xs[3])/2, centers[5] + layer_h/2 + 0.5, lw=0.9)

    # 输出框
    out_y = centers[-1] - layer_h/2 - 5.5
    _rect(ax, box_center_x - 18, out_y - 2.5, 36, 4, lw=LW_THIN)
    _text(ax, box_center_x, out_y - 0.5, "通用嵌入向量", fontsize=FS_BOX)
    _arrow(ax, box_center_x, centers[-1] - layer_h/2 - 0.5, box_center_x, out_y + 2.5, lw=1.0)

    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=300, bbox_inches='tight', pad_inches=0.25, facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", dpi=300, bbox_inches='tight', pad_inches=0.25, facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1")

if __name__ == "__main__":
    draw_fig1()
