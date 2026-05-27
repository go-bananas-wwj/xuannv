#!/usr/bin/env python3
"""
专利附图绘制 v3 — 专业黑白线条图
- 纯黑白，无颜色、无阴影、无渐变
- 直角矩形，线条清晰锐利
- 统一线宽与字号
- 图内不标注图题（图题在文档中）
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path

# ── 全局样式 ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("/workspace/xuannv/docs/patent_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 统一参数
LW_MAIN = 1.0      # 主框线宽
LW_THIN = 0.6      # 细线
LW_ARROW = 0.8     # 箭头线宽
FS_BOX = 9         # 框内文字
FS_LABEL = 8       # 标注文字
FS_LAYER = 10      # 层标签


def _rect(ax, x, y, w, h, lw=LW_MAIN):
    """直角矩形（专利图标准）。"""
    rect = mpatches.Rectangle((x, y), w, h, fill=False,
                               edgecolor='black', linewidth=lw, zorder=2)
    ax.add_patch(rect)
    return rect


def _text(ax, x, y, s, fontsize=FS_BOX, ha='center', va='center', bold=False):
    """居中文本。"""
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize,
            color='black', fontweight=weight, zorder=3,
            linespacing=1.2)


def _arrow(ax, x1, y1, x2, y2, style='->', lw=LW_ARROW):
    """简洁箭头。"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color='black', lw=lw,
                                connectionstyle='arc3,rad=0'))


def _setup(ax, xlim=(0, 100), ylim=(0, 100)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_facecolor('white')


# ═══════════════════════════════════════════════════════════════
# 图1: 系统整体架构图
# ═══════════════════════════════════════════════════════════════
def draw_fig1():
    fig, ax = plt.subplots(figsize=(8, 12))
    _setup(ax)

    # 六层纵向排列，每层高度约13，间距3
    layer_h = 11
    gap = 3.5
    left_label_x = 3
    box_center_x = 52
    box_w = 36

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
        cy = 92 - idx * (layer_h + gap)
        centers.append(cy)
        # 左侧层名
        _text(ax, left_label_x, cy, name, fontsize=FS_LAYER, ha='left', bold=True)

        n = len(items)
        if n == 1:
            _rect(ax, box_center_x - box_w/2, cy - layer_h/2, box_w, layer_h)
            _text(ax, box_center_x, cy, items[0], fontsize=FS_BOX)
        else:
            item_w = (box_w - (n - 1) * 2) / n
            start_x = box_center_x - box_w/2
            xs = []
            for i, item in enumerate(items):
                x = start_x + i * (item_w + 2)
                _rect(ax, x, cy - layer_h/2, item_w, layer_h)
                _text(ax, x + item_w/2, cy, item, fontsize=FS_BOX - 1)
                xs.append(x + item_w/2)

    # 层间垂直箭头
    for i in range(len(centers) - 1):
        _arrow(ax, box_center_x, centers[i] - layer_h/2 - 0.5,
               box_center_x, centers[i+1] + layer_h/2 + 0.5)

    # 最底层输出
    bottom_y = centers[-1] - layer_h/2 - 2.5
    _rect(ax, box_center_x - 18, bottom_y - 3, 36, 4, lw=LW_THIN)
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

    # 解码层两个框各自指向损失层对应框
    dec_cy = centers[4]
    loss_cy = centers[5]
    item_w_dec = (box_w - 1 * 2) / 2
    start_x_dec = box_center_x - box_w/2
    x_left = start_x_dec + item_w_dec/2
    x_right = start_x_dec + item_w_dec + 2 + item_w_dec/2
    _arrow(ax, x_left, dec_cy - layer_h/2 - 0.5,
           x_left, loss_cy + layer_h/2 + 0.5)
    _arrow(ax, x_right, dec_cy - layer_h/2 - 0.5,
           x_right, loss_cy + layer_h/2 + 0.5)

    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1")


# ═══════════════════════════════════════════════════════════════
# 图2: 多源数据到多任务复用的数据流程图
# ═══════════════════════════════════════════════════════════════
def draw_fig2():
    fig, ax = plt.subplots(figsize=(12, 8))
    _setup(ax)

    # 左侧数据源
    sources = [
        "卫星光学影像", "SAR雷达", "多光谱数据",
        "无人机航拍", "激光雷达点云", "地面传感器数据"
    ]
    src_x, src_w, src_h = 8, 16, 7
    src_y0, src_gap = 86, 12.5
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
    mid_y_center = mids[1][1]  # 专用输入头 y
    for i, name in enumerate(sources):
        y = src_y0 - i * src_gap
        _arrow(ax, src_x + src_w, y, mid_x, mid_y_center, lw=LW_THIN)

    # 中间流程箭头
    for i in range(len(mids) - 1):
        y1 = mids[i][1]
        y2 = mids[i+1][1]
        h1 = 11 if '\n' in mids[i][0] else 8
        h2 = 11 if '\n' in mids[i+1][0] else 8
        _arrow(ax, mid_x + mid_w/2, y1 - h1/2 - 0.5,
               mid_x + mid_w/2, y2 + h2/2 + 0.5)

    # 中间→右侧 发散箭头
    mid_y_out = mids[-1][1]  # 嵌入层 y
    for i, name in enumerate(tasks):
        y = src_y0 - i * src_gap
        _arrow(ax, mid_x + mid_w, mid_y_out, task_x, y, lw=LW_THIN)

    # 底部说明
    _text(ax, 52, 6, "同一嵌入向量支持多个下游任务", fontsize=FS_LABEL + 1, bold=True)

    fig.savefig(OUTPUT_DIR / "fig2_data_flow.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig2_data_flow.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig2")


# ═══════════════════════════════════════════════════════════════
# 图3: 离散格网划分示意图
# ═══════════════════════════════════════════════════════════════
def draw_fig3():
    fig, ax = plt.subplots(figsize=(10, 10))
    _setup(ax)

    # 一级网格 (左上)
    _text(ax, 28, 95, "一级网格 (coarse)", fontsize=FS_LAYER, bold=True)
    g1_x, g1_y = 8, 68
    g1_w, g1_h = 40, 24
    for i in range(5):
        for j in range(4):
            rect = mpatches.Rectangle((g1_x + j*10, g1_y - i*4.8), 10, 4.8,
                                       fill=False, edgecolor='black', linewidth=LW_MAIN)
            ax.add_patch(rect)

    # 二级网格 (右上)
    _text(ax, 75, 62, "二级网格", fontsize=FS_LAYER, bold=True)
    g2_x, g2_y = 60, 38
    for i in range(4):
        for j in range(4):
            rect = mpatches.Rectangle((g2_x + j*7.5, g2_y - i*6), 7.5, 6,
                                       fill=False, edgecolor='black', linewidth=LW_MAIN)
            ax.add_patch(rect)

    # 三级网格 (下方)
    _text(ax, 50, 30, "三级网格 (fine)", fontsize=FS_LAYER, bold=True)
    g3_x, g3_y = 35, 10
    for i in range(4):
        for j in range(4):
            rect = mpatches.Rectangle((g3_x + j*7.5, g3_y - i*5), 7.5, 5,
                                       fill=False, edgecolor='black', linewidth=LW_THIN)
            ax.add_patch(rect)
            if (i, j) in [(0,0), (0,3), (1,1), (2,2), (3,0), (3,3)]:
                ax.plot(g3_x + j*7.5 + 3.75, g3_y - i*5 + 2.5, 'ko', markersize=3)

    # 箭头与标注
    _arrow(ax, 48, 52, 58, 52, lw=LW_MAIN)
    _text(ax, 53, 54, "细分", fontsize=FS_LABEL)
    _arrow(ax, 68, 38, 58, 30, lw=LW_MAIN)

    _text(ax, 50, 3, "网格单元：数据组织的基本单位", fontsize=FS_BOX, bold=True)

    fig.savefig(OUTPUT_DIR / "fig3_grid_division.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig3_grid_division.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig3")


# ═══════════════════════════════════════════════════════════════
# 图4: 专用输入头示意图
# ═══════════════════════════════════════════════════════════════
def draw_fig4():
    fig, ax = plt.subplots(figsize=(11, 8))
    _setup(ax)

    # 左侧数据源
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
        _text(ax, src_x + src_w + 2, y, spec, fontsize=FS_LABEL - 1, ha='left')

    # 中间专用输入头
    head_x, head_w, head_h = 48, 14, 7
    for i in range(len(sources)):
        y = src_y0 - i * src_gap
        _rect(ax, head_x, y - head_h/2, head_w, head_h)
        _text(ax, head_x + head_w/2, y, "专用\n输入头", fontsize=FS_BOX - 1)

    # 可扩展标注
    _text(ax, head_x + head_w/2, src_y0 - 2.5 * src_gap + 7,
          "可扩展", fontsize=FS_LABEL, bold=True,
          ha='center', va='bottom')
    ax.plot([head_x, head_x + head_w], [src_y0 - 2.5 * src_gap + 6.5] * 2,
            'k-', linewidth=LW_THIN)

    # 右侧统一潜在空间
    right_x, right_w = 74, 20
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

    # 底部注释
    _text(ax, 50, 8,
          "注：新增数据源时，只需增加对应的专用输入头",
          fontsize=FS_LABEL)

    fig.savefig(OUTPUT_DIR / "fig4_input_heads.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig4_input_heads.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig4")


# ═══════════════════════════════════════════════════════════════
# 图5: 双窗口时序对比示意图
# ═══════════════════════════════════════════════════════════════
def draw_fig5():
    fig, ax = plt.subplots(figsize=(10, 7))
    _setup(ax)

    # 时间轴
    ax.plot([8, 92], [48, 48], 'k-', linewidth=LW_MAIN)
    ax.plot([8, 8], [46, 50], 'k-', linewidth=LW_MAIN)
    ax.plot([92, 92], [46, 50], 'k-', linewidth=LW_MAIN)
    _text(ax, 8, 43, "2023.04", fontsize=FS_LABEL)
    _text(ax, 92, 43, "2025.10", fontsize=FS_LABEL)
    _text(ax, 50, 43, "时间轴", fontsize=FS_BOX, bold=True)

    # 窗口1
    w1 = mpatches.Rectangle((18, 62), 30, 10, fill=False,
                             edgecolor='black', linewidth=LW_MAIN)
    ax.add_patch(w1)
    _text(ax, 33, 68, "时间窗口1", fontsize=FS_BOX, bold=True)
    _text(ax, 33, 64, "不重叠", fontsize=FS_LABEL - 1)

    # 窗口2
    w2 = mpatches.Rectangle((58, 62), 30, 10, fill=False,
                             edgecolor='black', linewidth=LW_MAIN, linestyle='--')
    ax.add_patch(w2)
    _text(ax, 73, 68, "时间窗口2", fontsize=FS_BOX, bold=True)

    # 嵌入向量1,2
    _rect(ax, 20, 22, 24, 10)
    _text(ax, 32, 27, "嵌入向量1\nemb_w1", fontsize=FS_BOX)
    _rect(ax, 58, 22, 24, 10)
    _text(ax, 70, 27, "嵌入向量2\nemb_w2", fontsize=FS_BOX)

    # 箭头：时间轴→窗口
    _arrow(ax, 33, 48, 33, 62)
    _arrow(ax, 73, 48, 73, 62)

    # 箭头：窗口→嵌入向量
    _arrow(ax, 33, 62, 32, 32)
    _arrow(ax, 73, 62, 70, 32)

    # 双向箭头：嵌入向量之间
    ax.annotate('', xy=(58, 27), xytext=(44, 27),
                arrowprops=dict(arrowstyle='<->', color='black', lw=LW_MAIN))
    _text(ax, 50, 20, "时序对比损失", fontsize=FS_LABEL, bold=True)

    fig.savefig(OUTPUT_DIR / "fig5_dual_window.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig5_dual_window.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig5")


# ═══════════════════════════════════════════════════════════════
# 图6: 预归一化两阶段策略对比图
# ═══════════════════════════════════════════════════════════════
def draw_fig6():
    fig, ax = plt.subplots(figsize=(10, 7))
    _setup(ax)

    # 两列标题
    _text(ax, 28, 92, "训练阶段", fontsize=FS_LAYER + 1, bold=True)
    _text(ax, 72, 92, "推理阶段", fontsize=FS_LAYER + 1, bold=True)

    # 左侧：训练阶段
    _rect(ax, 12, 65, 32, 14)
    _text(ax, 28, 72, "嵌入向量\n(原始幅度空间)", fontsize=FS_BOX)
    _arrow(ax, 28, 65, 28, 58)
    _rect(ax, 12, 38, 32, 18)
    _text(ax, 28, 47, "反坍缩损失计算\n(原始幅度空间)", fontsize=FS_BOX)
    _text(ax, 28, 28, "梯度始终非零", fontsize=FS_LABEL + 1, bold=True)

    # 右侧：推理阶段
    _rect(ax, 58, 65, 32, 14)
    _text(ax, 74, 72, "嵌入向量\n(原始幅度空间)", fontsize=FS_BOX)
    _arrow(ax, 74, 65, 74, 58)
    _rect(ax, 58, 38, 32, 18)
    _text(ax, 74, 47, "L2归一化 + 分布噪声\n→ 单位球面嵌入", fontsize=FS_BOX)
    _text(ax, 74, 28, "标准化输出", fontsize=FS_LABEL + 1, bold=True)

    # 中间 VS
    _text(ax, 50, 55, "VS", fontsize=16, bold=True)

    # 底部说明
    _text(ax, 50, 10,
          "现有技术：训练阶段即施加归一化，导致反坍缩梯度消失",
          fontsize=FS_LABEL, bold=False)

    fig.savefig(OUTPUT_DIR / "fig6_two_stage.png", dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig6_two_stage.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig6")


if __name__ == "__main__":
    print("开始绘制专利附图 v3 (专业黑白线条图)...")
    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    draw_fig6()
    print(f"\n全部完成！输出目录: {OUTPUT_DIR}")
