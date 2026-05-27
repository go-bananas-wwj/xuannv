#!/usr/bin/env python3
"""
专利附图 v3 修复2 — 图1框宽/箭头/底部遮挡，图4可扩展位置与spec文字
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
LW_ARROW = 0.8
FS_BOX = 9
FS_LABEL = 8
FS_LAYER = 10


def _rect(ax, x, y, w, h, lw=LW_MAIN):
    rect = mpatches.Rectangle((x, y), w, h, fill=False,
                               edgecolor='black', linewidth=lw, zorder=2)
    ax.add_patch(rect)
    return rect


def _text(ax, x, y, s, fontsize=FS_BOX, ha='center', va='center', bold=False):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize,
            color='black', fontweight=weight, zorder=3, linespacing=1.15)


def _arrow(ax, x1, y1, x2, y2, style='->', lw=LW_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color='black', lw=lw,
                                connectionstyle='arc3,rad=0'))


def _setup(ax, xlim=(0, 100), ylim=(0, 100)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_facecolor('white')


# ═══════════════════════════════════════════════════════════════
# 图1: 系统整体架构图 — 修复：框加宽、箭头不挡字、底部不遮
# ═══════════════════════════════════════════════════════════════
def draw_fig1():
    fig, ax = plt.subplots(figsize=(11, 15))
    _setup(ax, xlim=(0, 100), ylim=(-5, 110))

    layer_h = 13
    gap = 3
    left_label_x = 2
    box_center_x = 56
    box_w = 72          # 大幅加宽，6个框每个约10宽+间距

    layers = [
        ("输入层",       ["光学遥感", "SAR遥感", "多光谱遥感", "无人机", "激光雷达", "地面传感器"]),
        ("网格划分层",   ["离散格网系统\n(多级网格单元)"]),
        ("编码层",       ["专用输入头", "时间编码器", "多路径时空\n特征提取单元"]),
        ("嵌入层",       ["VMF瓶颈层\n(训练保留原始幅度\n推理L2归一化+噪声)"]),
        ("解码层",       ["动态目标解码器", "静态目标解码器"]),
        ("损失层",       ["重建损失", "反坍缩损失", "一致性损失", "时序对比损失"]),
    ]

    centers = []
    box_coords = []  # 记录每层每个框的坐标，用于画内部箭头
    for idx, (name, items) in enumerate(layers):
        cy = 100 - idx * (layer_h + gap)
        centers.append(cy)
        _text(ax, left_label_x, cy, name, fontsize=FS_LAYER, ha='left', bold=True)

        n = len(items)
        if n == 1:
            _rect(ax, box_center_x - box_w/2, cy - layer_h/2, box_w, layer_h)
            _text(ax, box_center_x, cy, items[0], fontsize=FS_BOX)
            box_coords.append([(box_center_x - box_w/2, box_w)])
        else:
            item_w = (box_w - (n - 1) * 1.5) / n
            start_x = box_center_x - box_w/2
            coords = []
            for i, item in enumerate(items):
                x = start_x + i * (item_w + 1.5)
                _rect(ax, x, cy - layer_h/2, item_w, layer_h)
                _text(ax, x + item_w/2, cy, item, fontsize=FS_BOX - 1)
                coords.append((x, item_w))
            box_coords.append(coords)

    # 层间垂直箭头
    for i in range(len(centers) - 1):
        _arrow(ax, box_center_x, centers[i] - layer_h/2 - 0.4,
               box_center_x, centers[i+1] + layer_h/2 + 0.4)

    # 最底层输出框（确保不超出画布底部）
    out_y = centers[-1] - layer_h/2 - 2.5
    _rect(ax, box_center_x - 20, out_y - 2.5, 40, 4, lw=LW_THIN)
    _text(ax, box_center_x, out_y - 0.5, "通用嵌入向量 (64维)", fontsize=FS_BOX)
    _arrow(ax, box_center_x, centers[-1] - layer_h/2 - 0.4,
           box_center_x, out_y + 1.5)

    # 编码层内部横向箭头 — 画在框的底部边缘下方，不穿过文字
    enc_idx = 2
    enc_cy = centers[enc_idx]
    enc_coords = box_coords[enc_idx]
    arrow_y = enc_cy - layer_h/2 - 1.2  # 框底下方
    for i in range(len(enc_coords) - 1):
        x1 = enc_coords[i][0] + enc_coords[i][1]
        x2 = enc_coords[i+1][0]
        _arrow(ax, x1 + 0.3, arrow_y, x2 - 0.3, arrow_y)

    # 解码层两个框各自指向损失层对应框
    dec_idx = 4
    loss_idx = 5
    dec_coords = box_coords[dec_idx]
    loss_coords = box_coords[loss_idx]
    for dc, lc in zip(dec_coords, loss_coords):
        x1 = dc[0] + dc[1]/2
        x2 = lc[0] + lc[1]/2
        _arrow(ax, x1, centers[dec_idx] - layer_h/2 - 0.4,
               x2, centers[loss_idx] + layer_h/2 + 0.4)

    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1")


# ═══════════════════════════════════════════════════════════════
# 图4: 专用输入头 — 修复：左侧加"输入数据"大框+可扩展，spec写箭头上方
# ═══════════════════════════════════════════════════════════════
def draw_fig4():
    fig, ax = plt.subplots(figsize=(12, 9))
    _setup(ax)

    sources = [
        ("卫星光学影像", "6ch 10m"),
        ("SAR雷达", "2ch 10m"),
        ("多光谱数据", "6ch 30m"),
        ("无人机航拍", "3ch RGB 0.1m"),
        ("激光雷达", "1ch高程 1m"),
    ]
    src_x, src_w, src_h = 10, 18, 7
    src_y0, src_gap = 86, 12.5
    ys = []
    for i, (name, spec) in enumerate(sources):
        y = src_y0 - i * src_gap
        ys.append(y)
        _rect(ax, src_x, y - src_h/2, src_w, src_h)
        _text(ax, src_x + src_w/2, y, name, fontsize=FS_BOX)
        # spec 写在框上方，不挡箭头
        _text(ax, src_x + src_w/2, y + src_h/2 + 1.8, spec, fontsize=FS_LABEL - 1)

    # 左侧大虚线框：把所有输入数据源框起来
    big_top = ys[0] + src_h/2 + 3.5
    big_bottom = ys[-1] - src_h/2 - 1
    big_h = big_top - big_bottom
    big_rect = mpatches.Rectangle((src_x - 2, big_bottom), src_w + 4, big_h,
                                   fill=False, edgecolor='black', linewidth=LW_THIN,
                                   linestyle='--', zorder=2)
    ax.add_patch(big_rect)
    _text(ax, src_x + src_w/2, big_top + 1.5, "输入数据", fontsize=FS_LAYER, bold=True)
    # 可扩展写在大框下方
    _text(ax, src_x + src_w/2, big_bottom - 2, "可扩展", fontsize=FS_LABEL + 1, bold=True)

    # 中间专用输入头
    head_x, head_w, head_h = 48, 14, 6.5
    for y in ys:
        _rect(ax, head_x, y - head_h/2, head_w, head_h)
        _text(ax, head_x + head_w/2, y, "专用\n输入头", fontsize=FS_BOX - 1)

    # 右侧统一潜在空间
    right_x, right_w = 74, 22
    right_y = ys[2]
    right_h = ys[0] + src_h/2 - (ys[-1] - src_h/2) + 4
    _rect(ax, right_x, right_y - right_h/2, right_w, right_h, lw=LW_MAIN + 0.3)
    _text(ax, right_x + right_w/2, right_y,
          "统一潜在空间\n(统一通道维度\n统一空间分辨率)", fontsize=FS_BOX)

    # 箭头：数据源 → 输入头（水平，spec已移到上方不挡）
    for y in ys:
        _arrow(ax, src_x + src_w, y, head_x, y)

    # 箭头：输入头 → 统一空间
    for y in ys:
        _arrow(ax, head_x + head_w, y, right_x, right_y)

    _text(ax, 50, 6, "注：新增数据源时，只需增加对应的专用输入头", fontsize=FS_LABEL)

    fig.savefig(OUTPUT_DIR / "fig4_input_heads.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig4_input_heads.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig4")


if __name__ == "__main__":
    print("开始绘制专利附图 v3 修复2...")
    draw_fig1()
    draw_fig4()
    print(f"\n完成！输出目录: {OUTPUT_DIR}")
