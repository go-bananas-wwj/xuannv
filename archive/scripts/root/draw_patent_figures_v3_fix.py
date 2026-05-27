#!/usr/bin/env python3
"""
专利附图 v3 修复版
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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
FS_HEADER = 10


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
# 图1: 系统整体架构图 — 修复：增大框高、加宽画布
# ═══════════════════════════════════════════════════════════════
def draw_fig1():
    fig, ax = plt.subplots(figsize=(9, 14))
    _setup(ax)

    layer_h = 14       # 增大框高
    gap = 3.5
    left_label_x = 3
    box_center_x = 54
    box_w = 40         # 加宽一点

    layers = [
        ("输入层",       ["光学遥感", "SAR遥感", "多光谱遥感", "无人机", "激光雷达", "地面传感器"]),
        ("网格划分层",   ["离散格网系统\n(多级网格单元)"]),
        ("编码层",       ["专用输入头", "时间编码器", "多路径时空\n特征提取单元"]),
        ("嵌入层",       ["VMF瓶颈层\n(训练保留原始幅度\n推理L2归一化+噪声)"]),
        ("解码层",       ["动态目标解码器", "静态目标解码器"]),
        ("损失层",       ["重建损失", "反坍缩损失", "一致性损失", "时序对比损失"]),
    ]

    centers = []
    for idx, (name, items) in enumerate(layers):
        cy = 93 - idx * (layer_h + gap)
        centers.append(cy)
        _text(ax, left_label_x, cy, name, fontsize=FS_LAYER, ha='left', bold=True)

        n = len(items)
        if n == 1:
            _rect(ax, box_center_x - box_w/2, cy - layer_h/2, box_w, layer_h)
            _text(ax, box_center_x, cy, items[0], fontsize=FS_BOX)
        else:
            item_w = (box_w - (n - 1) * 2) / n
            start_x = box_center_x - box_w/2
            for i, item in enumerate(items):
                x = start_x + i * (item_w + 2)
                _rect(ax, x, cy - layer_h/2, item_w, layer_h)
                _text(ax, x + item_w/2, cy, item, fontsize=FS_BOX - 1)

    # 层间箭头
    for i in range(len(centers) - 1):
        _arrow(ax, box_center_x, centers[i] - layer_h/2 - 0.5,
               box_center_x, centers[i+1] + layer_h/2 + 0.5)

    # 最底层输出
    bottom_y = centers[-1] - layer_h/2 - 3
    _rect(ax, box_center_x - 20, bottom_y - 3, 40, 4, lw=LW_THIN)
    _text(ax, box_center_x, bottom_y - 1, "通用嵌入向量 (64维)", fontsize=FS_BOX)
    _arrow(ax, box_center_x, centers[-1] - layer_h/2 - 0.5,
           box_center_x, bottom_y + 1)

    # 编码层内部横向箭头
    enc_cy = centers[2]
    item_w = (box_w - 2 * 2) / 3
    start_x = box_center_x - box_w/2
    x1 = start_x + item_w + 1
    x2 = start_x + 2 * item_w + 2 + 1
    _arrow(ax, x1, enc_cy, x2, enc_cy)
    x0 = start_x + item_w - 1
    _arrow(ax, x0, enc_cy, start_x + item_w + 1 - 1, enc_cy)

    # 解码层→损失层
    dec_cy = centers[4]
    loss_cy = centers[5]
    item_w_dec = (box_w - 1 * 2) / 2
    start_x_dec = box_center_x - box_w/2
    x_left = start_x_dec + item_w_dec/2
    x_right = start_x_dec + item_w_dec + 2 + item_w_dec/2
    _arrow(ax, x_left, dec_cy - layer_h/2 - 0.5, x_left, loss_cy + layer_h/2 + 0.5)
    _arrow(ax, x_right, dec_cy - layer_h/2 - 0.5, x_right, loss_cy + layer_h/2 + 0.5)

    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1")


# ═══════════════════════════════════════════════════════════════
# 图2: 数据流程图 — 修复：添加三列顶部标题
# ═══════════════════════════════════════════════════════════════
def draw_fig2():
    fig, ax = plt.subplots(figsize=(12, 9))
    _setup(ax)

    # 三列顶部标题
    _text(ax, 16, 95, "多源输入数据", fontsize=FS_HEADER + 1, bold=True)
    _text(ax, 55, 95, "模型训练流程", fontsize=FS_HEADER + 1, bold=True)
    _text(ax, 86, 95, "下游任务复用", fontsize=FS_HEADER + 1, bold=True)

    # 分隔线
    ax.plot([34, 34], [8, 92], color='black', linewidth=LW_THIN, linestyle='--')
    ax.plot([74, 74], [8, 92], color='black', linewidth=LW_THIN, linestyle='--')

    # 左侧数据源
    sources = [
        "卫星光学影像", "SAR雷达", "多光谱数据",
        "无人机航拍", "激光雷达点云", "地面传感器数据"
    ]
    src_x, src_w, src_h = 8, 16, 7
    src_y0, src_gap = 82, 12
    for i, name in enumerate(sources):
        y = src_y0 - i * src_gap
        _rect(ax, src_x, y - src_h/2, src_w, src_h)
        _text(ax, src_x + src_w/2, y, name, fontsize=FS_BOX)

    # 中间处理流程
    mids = [
        ("离散格网\n对齐", 78),
        ("专用输入头\n映射到统一潜空间", 62),
        ("编码层\n时间编码+时空特征提取", 46),
        ("嵌入层\nVMF瓶颈层\n64维嵌入向量", 28),
    ]
    mid_x, mid_w = 44, 22
    for name, y in mids:
        h = 11 if '\n' in name else 8
        _rect(ax, mid_x, y - h/2, mid_w, h)
        _text(ax, mid_x + mid_w/2, y, name, fontsize=FS_BOX)

    # 右侧任务
    tasks = [
        "变化检测", "土地覆盖分类", "农田变化监测",
        "水体动态追踪", "灾害受损评估", "城市扩张监测"
    ]
    task_x, task_w, task_h = 78, 16, 7
    for i, name in enumerate(tasks):
        y = src_y0 - i * src_gap
        _rect(ax, task_x, y - task_h/2, task_w, task_h)
        _text(ax, task_x + task_w/2, y, name, fontsize=FS_BOX)

    # 左侧→中间 汇聚箭头
    mid_y_center = mids[1][1]
    for i, name in enumerate(sources):
        y = src_y0 - i * src_gap
        _arrow(ax, src_x + src_w, y, mid_x, mid_y_center, lw=LW_THIN)

    # 中间流程箭头
    for i in range(len(mids) - 1):
        y1 = mids[i][1]
        y2 = mids[i+1][1]
        h1 = 11 if '\n' in mids[i][0] else 8
        h2 = 11 if '\n' in mids[i+1][0] else 8
        _arrow(ax, mid_x + mid_w/2, y1 - h1/2 - 0.5, mid_x + mid_w/2, y2 + h2/2 + 0.5)

    # 中间→右侧 发散箭头
    mid_y_out = mids[-1][1]
    for i, name in enumerate(tasks):
        y = src_y0 - i * src_gap
        _arrow(ax, mid_x + mid_w, mid_y_out, task_x, y, lw=LW_THIN)

    # 底部说明
    _text(ax, 52, 5, "同一嵌入向量支持多个下游任务", fontsize=FS_LABEL + 1, bold=True)

    fig.savefig(OUTPUT_DIR / "fig2_data_flow.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig2_data_flow.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig2")


# ═══════════════════════════════════════════════════════════════
# 图4: 专用输入头 — 修复：调整"可扩展"位置避免重叠
# ═══════════════════════════════════════════════════════════════
def draw_fig4():
    fig, ax = plt.subplots(figsize=(11, 8))
    _setup(ax)

    sources = [
        ("卫星光学影像", "6通道 10m"),
        ("SAR雷达", "2通道 10m"),
        ("多光谱数据", "6通道 30m"),
        ("无人机航拍", "3通道RGB 0.1m"),
        ("激光雷达", "1通道高程 1m"),
    ]
    src_x, src_w, src_h = 10, 18, 8
    src_y0, src_gap = 88, 13.5
    for i, (name, spec) in enumerate(sources):
        y = src_y0 - i * src_gap
        _rect(ax, src_x, y - src_h/2, src_w, src_h)
        _text(ax, src_x + src_w/2, y, name, fontsize=FS_BOX)
        _text(ax, src_x + src_w + 1.5, y, spec, fontsize=FS_LABEL - 1, ha='left')

    # 中间专用输入头
    head_x, head_w, head_h = 48, 14, 7
    for i in range(len(sources)):
        y = src_y0 - i * src_gap
        _rect(ax, head_x, y - head_h/2, head_w, head_h)
        _text(ax, head_x + head_w/2, y, "专用\n输入头", fontsize=FS_BOX - 1)

    # 可扩展标注 — 移到最右侧独立位置，不重叠
    _text(ax, 92, 88, "可扩展", fontsize=FS_LABEL + 1, bold=True, ha='right')
    _arrow(ax, head_x + head_w + 1, 88, 90, 88, lw=LW_THIN)

    # 右侧统一潜在空间
    right_x, right_w = 72, 22
    right_y = src_y0 - 2 * src_gap
    right_h = src_y0 - (len(sources)-1)*src_gap - right_y + src_h/2
    _rect(ax, right_x, right_y - right_h/2, right_w, right_h, lw=LW_MAIN + 0.3)
    _text(ax, right_x + right_w/2, right_y,
          "统一潜在空间\n(统一通道维度\n统一空间分辨率)", fontsize=FS_BOX)

    # 箭头
    for i in range(len(sources)):
        y = src_y0 - i * src_gap
        _arrow(ax, src_x + src_w, y, head_x, y)
        _arrow(ax, head_x + head_w, y, right_x, right_y)

    _text(ax, 50, 8, "注：新增数据源时，只需增加对应的专用输入头", fontsize=FS_LABEL)

    fig.savefig(OUTPUT_DIR / "fig4_input_heads.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig4_input_heads.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig4")


# ═══════════════════════════════════════════════════════════════
# 图5: 双窗口时序对比 — 修复：补全流程逻辑
# ═══════════════════════════════════════════════════════════════
def draw_fig5():
    fig, ax = plt.subplots(figsize=(12, 8))
    _setup(ax)

    # 顶部：同一地点
    _rect(ax, 44, 90, 14, 5, lw=LW_THIN)
    _text(ax, 51, 92.5, "同一地点", fontsize=FS_BOX, bold=True)

    # 时间轴
    ax.plot([6, 96], [52, 52], 'k-', linewidth=LW_MAIN)
    ax.plot([6, 6], [50, 54], 'k-', linewidth=LW_MAIN)
    ax.plot([96, 96], [50, 54], 'k-', linewidth=LW_MAIN)
    _text(ax, 6, 47, "2023.04", fontsize=FS_LABEL)
    _text(ax, 96, 47, "2025.10", fontsize=FS_LABEL)
    _text(ax, 51, 47, "时间轴", fontsize=FS_BOX, bold=True)

    # 窗口1 (实线)
    _rect(ax, 16, 65, 28, 9)
    _text(ax, 30, 71.5, "时间窗口1", fontsize=FS_BOX, bold=True)
    _text(ax, 30, 68, "不重叠", fontsize=FS_LABEL - 1)

    # 窗口2 (虚线)
    w2 = mpatches.Rectangle((58, 65), 28, 9, fill=False,
                             edgecolor='black', linewidth=LW_MAIN, linestyle='--')
    ax.add_patch(w2)
    _text(ax, 72, 71.5, "时间窗口2", fontsize=FS_BOX, bold=True)

    # 编码器
    _rect(ax, 22, 38, 18, 8)
    _text(ax, 31, 42, "编码器", fontsize=FS_BOX, bold=True)
    _rect(ax, 62, 38, 18, 8)
    _text(ax, 71, 42, "编码器", fontsize=FS_BOX, bold=True)

    # 嵌入向量
    _rect(ax, 22, 18, 18, 8)
    _text(ax, 31, 22, "嵌入向量1", fontsize=FS_BOX)
    _rect(ax, 62, 18, 18, 8)
    _text(ax, 71, 22, "嵌入向量2", fontsize=FS_BOX)

    # 损失框
    _rect(ax, 44, 5, 16, 7)
    _text(ax, 52, 9.5, "时序对比损失", fontsize=FS_BOX, bold=True)
    _text(ax, 52, 6.5, "(约束差异)", fontsize=FS_LABEL)

    # 箭头：同一地点 → 两个窗口
    _arrow(ax, 48, 90, 30, 74)
    _arrow(ax, 54, 90, 72, 74)

    # 箭头：窗口 → 编码器
    _arrow(ax, 30, 65, 31, 46)
    _arrow(ax, 72, 65, 71, 46)

    # 箭头：编码器 → 嵌入向量
    _arrow(ax, 31, 38, 31, 26)
    _arrow(ax, 71, 38, 71, 26)

    # 箭头：嵌入向量 → 损失
    _arrow(ax, 31, 18, 44, 9)
    _arrow(ax, 71, 18, 60, 9)

    fig.savefig(OUTPUT_DIR / "fig5_dual_window.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig5_dual_window.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig5")


if __name__ == "__main__":
    print("开始绘制专利附图 v3 修复...")
    draw_fig1()
    draw_fig2()
    draw_fig4()
    draw_fig5()
    print(f"\n完成！输出目录: {OUTPUT_DIR}")
