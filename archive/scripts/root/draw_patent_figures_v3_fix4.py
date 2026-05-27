#!/usr/bin/env python3
"""
专利附图 v3 修复4 — 图1：箭头全连、字放大、框矮间距大
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
FS_BOX = 10          # 框内字放大
FS_LABEL = 9
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
                arrowprops=dict(arrowstyle='->', color='black', lw=lw,
                                connectionstyle='arc3,rad=0'))


def _setup(ax, xlim=(0, 100), ylim=(0, 100)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_facecolor('white')


def _converge_arrows(ax, xs_from, y_from, x_to, y_to, lw=LW_ARROW):
    """多框汇聚到单点：每个框底部短竖线 → 水平汇聚线 → 箭头。"""
    if len(xs_from) == 1:
        _arrow(ax, xs_from[0], y_from, x_to, y_to, lw=lw)
        return
    y_mid = (y_from + y_to) / 2
    for x in xs_from:
        ax.plot([x, x], [y_from, y_mid], 'k-', linewidth=lw, zorder=1)
    ax.plot([min(xs_from), max(xs_from)], [y_mid, y_mid], 'k-', linewidth=lw, zorder=1)
    _arrow(ax, x_to, y_mid, x_to, y_to, lw=lw)


def _diverge_arrows(ax, x_from, y_from, xs_to, y_to, lw=LW_ARROW):
    """单点分叉到多框：箭头 → 水平分叉线 → 每个框顶部短竖线。"""
    if len(xs_to) == 1:
        _arrow(ax, x_from, y_from, xs_to[0], y_to, lw=lw)
        return
    y_mid = (y_from + y_to) / 2
    _arrow(ax, x_from, y_from, x_from, y_mid, lw=lw)
    ax.plot([min(xs_to), max(xs_to)], [y_mid, y_mid], 'k-', linewidth=lw, zorder=1)
    for x in xs_to:
        ax.plot([x, x], [y_mid, y_to], 'k-', linewidth=lw, zorder=1)


def draw_fig1():
    fig, ax = plt.subplots(figsize=(12, 16))
    _setup(ax, xlim=(0, 100), ylim=(-6, 108))

    layer_h = 10.5     # 框高适中
    gap = 7.5          # 层间距大，箭头长
    left_label_x = 2
    box_center_x = 56
    box_w = 74

    layers = [
        ("输入层",       ["光学遥感", "SAR遥感", "多光谱遥感", "无人机", "激光雷达", "地面传感器"]),
        ("网格划分层",   ["离散格网系统\n(多级网格单元)"]),
        ("编码层",       ["专用输入头", "时间编码器", "多路径时空\n特征提取单元"]),
        ("嵌入层",       ["VMF瓶颈层\n(训练保留原始幅度\n推理L2归一化+噪声)"]),
        ("解码层",       ["动态目标解码器", "静态目标解码器"]),
        ("损失层",       ["重建损失", "反坍缩损失", "一致性损失", "时序对比损失"]),
    ]

    centers = []
    box_centers = []   # 每层每个框的中心x坐标列表
    for idx, (name, items) in enumerate(layers):
        cy = 100 - idx * (layer_h + gap)
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

    # ── 层间连接：每个框都参与 ──
    for i in range(len(centers) - 1):
        y1 = centers[i] - layer_h/2
        y2 = centers[i+1] + layer_h/2
        _converge_arrows(ax, box_centers[i], y1, box_center_x, y2, lw=1.0)

    # ── 编码层内部横向连接 ──
    enc_idx = 2
    enc_cy = centers[enc_idx]
    enc_xs = box_centers[enc_idx]
    arrow_y = enc_cy - layer_h/2 - 2.0
    for j in range(len(enc_xs) - 1):
        ax.annotate('', xy=(enc_xs[j+1] - 1.5, arrow_y),
                    xytext=(enc_xs[j] + 1.5, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    # ── 解码层 → 损失层：每个解码框分叉到所有损失框 ──
    dec_idx = 4
    loss_idx = 5
    dec_xs = box_centers[dec_idx]
    loss_xs = box_centers[loss_idx]
    y1 = centers[dec_idx] - layer_h/2
    y2 = centers[loss_idx] + layer_h/2
    # 左解码框 → 损失层左半部分
    _diverge_arrows(ax, dec_xs[0], y1, loss_xs[:2], y2, lw=0.9)
    # 右解码框 → 损失层右半部分
    _diverge_arrows(ax, dec_xs[1], y1, loss_xs[2:], y2, lw=0.9)

    # ── 最底层输出 ──
    out_y = centers[-1] - layer_h/2 - 4.0
    _rect(ax, box_center_x - 18, out_y - 2.5, 36, 4, lw=LW_THIN)
    _text(ax, box_center_x, out_y - 0.5, "通用嵌入向量", fontsize=FS_BOX)
    _converge_arrows(ax, box_centers[-1], centers[-1] - layer_h/2,
                     box_center_x, out_y + 2.5, lw=1.0)

    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1")


if __name__ == "__main__":
    print("开始绘制专利附图 v3 修复4...")
    draw_fig1()
    print(f"\n完成！输出目录: {OUTPUT_DIR}")
