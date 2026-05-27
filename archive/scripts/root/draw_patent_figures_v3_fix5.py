#!/usr/bin/env python3
"""
专利附图 v3 修复5 — 图1：简化箭头、去多级网格、防底部贴边
"""
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
LW_ARROW = 1.0
FS_BOX = 10
FS_LAYER = 11


def _rect(ax, x, y, w, h, lw=LW_MAIN):
    rect = mpatches.Rectangle((x, y), w, h, fill=False,
                               edgecolor='black', linewidth=lw, zorder=2)
    ax.add_patch(rect)
    return rect


def _text(ax, x, y, s, fontsize=FS_BOX, ha='center', va='center', bold=False):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize,
            color='black', fontweight=weight, zorder=3, linespacing=1.1)


def _arrow(ax, x1, y1, x2, y2, lw=LW_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=lw))


def _setup(ax, xlim=(0, 100), ylim=(0, 100)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_facecolor('white')


def draw_fig1():
    fig, ax = plt.subplots(figsize=(11, 15))
    _setup(ax, xlim=(0, 100), ylim=(2, 105))

    layer_h = 10
    gap = 7
    left_label_x = 2
    box_center_x = 56
    box_w = 72

    layers = [
        ("输入层",       ["光学遥感", "SAR遥感", "多光谱遥感", "无人机", "激光雷达", "地面传感器"]),
        ("网格划分层",   ["离散格网系统\n(多源数据对齐)"]),  # 去掉多级网格单元
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
        _text(ax, left_label_x, cy, name, fontsize=FS_LAYER, ha='left', bold=True)

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

    # 层间竖直箭头：每层中心 → 下层中心（单根，清晰）
    for i in range(len(centers) - 1):
        y1 = centers[i] - layer_h/2 - 0.5
        y2 = centers[i+1] + layer_h/2 + 0.5
        _arrow(ax, box_center_x, y1, box_center_x, y2, lw=1.0)

    # 编码层内部横向箭头（画在框内中部偏下，不压出框外）
    enc_idx = 2
    enc_cy = centers[enc_idx]
    enc_xs = box_centers[enc_idx]
    arrow_y = enc_cy - 1.5  # 框内偏下位置
    for j in range(len(enc_xs) - 1):
        gap_x = (enc_xs[j+1] - enc_xs[j]) / 2
        _arrow(ax, enc_xs[j] + 1.0, arrow_y, enc_xs[j+1] - 1.0, arrow_y, lw=0.9)

    # 解码层 → 损失层：每解码框 → 对应损失框区域
    dec_idx = 4
    loss_idx = 5
    dec_xs = box_centers[dec_idx]
    loss_xs = box_centers[loss_idx]
    y1 = centers[dec_idx] - layer_h/2 - 0.5
    y2 = centers[loss_idx] + layer_h/2 + 0.5
    # 左解码框 → 损失层左半
    _arrow(ax, dec_xs[0], y1, (loss_xs[0]+loss_xs[1])/2, y2, lw=0.9)
    # 右解码框 → 损失层右半
    _arrow(ax, dec_xs[1], y1, (loss_xs[2]+loss_xs[3])/2, y2, lw=0.9)

    # 最底层输出框（下移，不贴边）
    out_y = centers[-1] - layer_h/2 - 5.0
    _rect(ax, box_center_x - 18, out_y - 2.5, 36, 4, lw=LW_THIN)
    _text(ax, box_center_x, out_y - 0.5, "通用嵌入向量", fontsize=FS_BOX)
    _arrow(ax, box_center_x, centers[-1] - layer_h/2 - 0.5,
           box_center_x, out_y + 2.5, lw=1.0)

    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=300,
                bbox_inches='tight', pad_inches=0.25,
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", dpi=300,
                bbox_inches='tight', pad_inches=0.25,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1")


if __name__ == "__main__":
    print("开始绘制专利附图 v3 修复5...")
    draw_fig1()
    print(f"\n完成！输出目录: {OUTPUT_DIR}")
