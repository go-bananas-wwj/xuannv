#!/usr/bin/env python3
"""
专利附图绘制 v2 — 使用 nature-figure skill 规范
- data 坐标系精确定位，避免 figsize 缩放漂移
- FancyArrowPatch + annotate 精确箭头
- gridspec 分层布局，无遮挡
- 输出 SVG + PNG (300dpi)
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ── Nature-figure skill 规范 ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.frameon'] = False

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1":   "#F6CFCB",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
}

OUTPUT_DIR = Path("/workspace/xuannv/docs/patent_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_box(ax, x, y, w, h, text, fontsize=10, boxcolor='white', textcolor='black',
              edgecolor='black', linewidth=1.5, radius=0.02, ha='center', va='center'):
    """在 data 坐标系中绘制圆角矩形框 + 文本。"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=boxcolor, edgecolor=edgecolor,
                         linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=textcolor, fontweight='normal', zorder=3,
            linespacing=1.3)
    return box


def _draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.2, style='-|>'):
    """在 data 坐标系中绘制箭头。"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))


def _setup_ax(ax, xlim=(0, 100), ylim=(0, 100)):
    """初始化 axes：白底、无边框、无刻度。"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_facecolor('white')


# ═══════════════════════════════════════════════════════════════
# 图1: 系统整体架构图
# ═══════════════════════════════════════════════════════════════
def draw_fig1():
    fig, ax = plt.subplots(figsize=(10, 14))
    _setup_ax(ax)

    # 六层垂直排列，每层一个区域
    layers = [
        ("输入层", 88, ["光学遥感", "SAR遥感", "多光谱遥感", "无人机", "激光雷达", "地面传感器"], 6),
        ("网格划分层", 72, ["离散格网系统\n(多级网格单元)"], 1),
        ("编码层", 56, ["专用输入头", "时间编码器", "多路径时空\n特征提取单元"], 3),
        ("嵌入层", 42, ["VMF瓶颈层\n(训练:保留原始幅度\n推理:L2归一化+噪声)"], 1),
        ("解码层", 28, ["动态目标解码器", "静态目标解码器"], 2),
        ("损失层", 14, ["重建损失", "反坍缩损失", "一致性损失", "时序对比损失"], 4),
    ]

    layer_boxes = []
    for name, cy, items, n in layers:
        # 画层标签（左侧）
        ax.text(2, cy, name, ha='left', va='center', fontsize=12,
                fontweight='bold', color=PALETTE['neutral_black'])

        # 画该层的框
        if n == 1:
            _draw_box(ax, 55, cy, 38, 10, items[0], fontsize=9)
            layer_boxes.append((55, cy))
        else:
            # 调整间距避免右侧截断
            box_w = 11.5
            gap = 1.5
            total_w = n * box_w + (n-1) * gap
            start_x = 55 - total_w / 2 + box_w / 2
            xs = []
            for i, item in enumerate(items):
                x = start_x + i * (box_w + gap)
                _draw_box(ax, x, cy, box_w, 8, item, fontsize=8)
                xs.append(x)
            layer_boxes.append((xs, cy))

    # 画层间箭头（从上到下）
    arrow_x = 55
    _draw_arrow(ax, arrow_x, 83, arrow_x, 77)   # 输入→网格
    _draw_arrow(ax, arrow_x, 67, arrow_x, 61)   # 网格→编码
    _draw_arrow(ax, arrow_x, 51, arrow_x, 47)   # 编码→嵌入
    _draw_arrow(ax, arrow_x, 37, arrow_x, 33)   # 嵌入→解码
    _draw_arrow(ax, arrow_x, 23, arrow_x, 19)   # 解码→损失

    # 最底层指向最终输出
    _draw_arrow(ax, arrow_x, 9, arrow_x, 5)
    _draw_box(ax, arrow_x, 2.5, 24, 4, "通用嵌入向量 (64维)", fontsize=9,
              boxcolor='#f0f0f0', edgecolor=PALETTE['neutral_dark'])

    # 编码层内部连接
    enc_xs = [43, 55, 67]
    for i in range(len(enc_xs)-1):
        _draw_arrow(ax, enc_xs[i]+6, 56, enc_xs[i+1]-6, 56)

    # 解码层两个框各自指向损失层
    _draw_arrow(ax, 49, 24, 49, 19)
    _draw_arrow(ax, 61, 24, 61, 19)

    ax.set_title("图1  系统整体架构图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig1 完成")


# ═══════════════════════════════════════════════════════════════
# 图2: 多源数据到多任务复用的数据流程图
# ═══════════════════════════════════════════════════════════════
def draw_fig2():
    fig, ax = plt.subplots(figsize=(14, 9))
    _setup_ax(ax, xlim=(0, 100), ylim=(0, 100))

    # 左侧：多源输入
    sources = [
        ("卫星光学影像", 90),
        ("SAR雷达", 78),
        ("多光谱数据", 66),
        ("无人机航拍", 54),
        ("激光雷达点云", 42),
        ("地面传感器数据", 30),
    ]
    for name, y in sources:
        _draw_box(ax, 12, y, 18, 8, name, fontsize=9)

    # 中间处理流程
    mid_steps = [
        ("离散格网\n对齐", 78),
        ("专用输入头\n映射到统一潜空间", 62),
        ("编码层\n时间编码+时空特征提取", 46),
        ("嵌入层\nVMF瓶颈层\n64维嵌入向量", 30),
    ]
    for name, y in mid_steps:
        _draw_box(ax, 50, y, 22, 10, name, fontsize=9,
                  boxcolor='#f8f8f8', edgecolor=PALETTE['blue_main'])

    # 右侧：多任务
    tasks = [
        ("变化检测", 90),
        ("土地覆盖分类", 78),
        ("农田变化监测", 66),
        ("水体动态追踪", 54),
        ("灾害受损评估", 42),
        ("城市扩张监测", 30),
    ]
    for name, y in tasks:
        _draw_box(ax, 88, y, 16, 8, name, fontsize=9,
                  boxcolor='#f0f8ff', edgecolor=PALETTE['teal'])

    # 左侧 → 中间（汇聚箭头）
    for _, y in sources:
        _draw_arrow(ax, 21, y, 39, 62, color=PALETTE['neutral_mid'], lw=0.8)

    # 中间流程箭头
    for i in range(len(mid_steps)-1):
        _draw_arrow(ax, 50, mid_steps[i][1]-5, 50, mid_steps[i+1][1]+5)

    # 中间 → 右侧（发散箭头）
    for _, y in tasks:
        _draw_arrow(ax, 61, 30, 80, y, color=PALETTE['neutral_mid'], lw=0.8)

    # 同一嵌入向量标注
    ax.text(72, 18, "同一嵌入向量\n支持多个任务", ha='center', va='center',
            fontsize=9, color=PALETTE['neutral_dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=PALETTE['neutral_mid'], linewidth=1))

    ax.set_title("图2  多源数据到多任务复用的数据流程图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(OUTPUT_DIR / "fig2_data_flow.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig2_data_flow.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig2 完成")


# ═══════════════════════════════════════════════════════════════
# 图3: 离散格网划分示意图
# ═══════════════════════════════════════════════════════════════
def draw_fig3():
    fig, ax = plt.subplots(figsize=(10, 10))
    _setup_ax(ax)

    # 一级网格 coarse (左上)
    ax.text(30, 95, "一级网格 (coarse)", ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i in range(5):
        for j in range(4):
            rect = mpatches.Rectangle((10 + j*10, 68 - i*8), 10, 8,
                                       fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

    # 标注：不同分辨率数据归入对应网格
    ax.text(5, 52, "不同分辨率\n数据归入\n对应网格", ha='center', va='center',
            fontsize=9, color=PALETTE['neutral_dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=PALETTE['neutral_mid']))

    # 二级网格 (右上)
    ax.text(75, 62, "二级网格", ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i in range(4):
        for j in range(4):
            rect = mpatches.Rectangle((60 + j*7.5, 38 - i*7), 7.5, 7,
                                       fill=False, edgecolor='black', linewidth=1.2)
            ax.add_patch(rect)

    # 箭头：一级 → 二级
    _draw_arrow(ax, 50, 52, 58, 52, lw=1.5)
    ax.text(54, 54, "空间细分", ha='center', va='bottom', fontsize=9,
            color=PALETTE['neutral_dark'])

    # 三级网格 fine (下方居中，独立区域)
    ax.text(50, 30, "三级网格 (fine)", ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i in range(4):
        for j in range(4):
            rect = mpatches.Rectangle((35 + j*7.5, 10 - i*5), 7.5, 5,
                                       fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            # 采样点
            if (i, j) in [(0,0), (0,3), (1,1), (2,2), (3,0), (3,3)]:
                ax.plot(35 + j*7.5 + 3.75, 10 - i*5 + 2.5, 'ko', markersize=4)

    # 箭头：二级 → 三级
    _draw_arrow(ax, 68, 38, 58, 30, lw=1.5)

    # 底部说明
    ax.text(50, 3, "网格单元：数据组织的基本单位", ha='center', va='center',
            fontsize=10, color=PALETTE['neutral_dark'])

    ax.set_title("图3  离散格网划分示意图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(OUTPUT_DIR / "fig3_grid_division.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig3_grid_division.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig3 完成")


# ═══════════════════════════════════════════════════════════════
# 图4: 专用输入头示意图
# ═══════════════════════════════════════════════════════════════
def draw_fig4():
    fig, ax = plt.subplots(figsize=(12, 9))
    _setup_ax(ax)

    # 左侧：数据源
    sources = [
        ("卫星光学影像", "6通道\n10m分辨率", 88),
        ("SAR雷达", "2通道\n10m分辨率", 74),
        ("多光谱数据", "6通道\n30m分辨率", 60),
        ("无人机航拍", "3通道RGB\n0.1m分辨率", 46),
        ("激光雷达", "1通道高程\n1m分辨率", 32),
    ]
    for name, spec, y in sources:
        _draw_box(ax, 18, y, 20, 10, name, fontsize=9)
        ax.text(35, y, spec, ha='left', va='center', fontsize=8,
                color=PALETTE['neutral_mid'])

    # 中间：专用输入头
    heads_y = [88, 74, 60, 46, 32]
    for y in heads_y:
        _draw_box(ax, 58, y, 16, 8, "专用\n输入头", fontsize=9,
                  boxcolor='#f0f0f8', edgecolor=PALETTE['blue_main'])

    # 多光谱标注可扩展（放在框上方，不重叠）
    ax.text(58, 64.5, "可扩展", ha='center', va='bottom', fontsize=8,
            color=PALETTE['red_strong'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=PALETTE['red_strong'], linewidth=1.2))

    # 右侧：统一潜在空间
    _draw_box(ax, 88, 60, 20, 30, "统一潜在空间\n(统一通道维度\n统一空间分辨率)",
              fontsize=10, boxcolor='#e8f4e8', edgecolor=PALETTE['green_3'],
              linewidth=2)

    # 箭头：数据源 → 输入头
    for _, _, y in sources:
        _draw_arrow(ax, 28, y, 50, y, lw=1)

    # 箭头：输入头 → 统一空间
    for y in heads_y:
        _draw_arrow(ax, 66, y, 78, 60, lw=1)

    # 注释
    ax.text(58, 18, "注：新增数据源时，只需增加对应的专用输入头",
            ha='center', va='center', fontsize=9, color=PALETTE['neutral_dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=PALETTE['neutral_mid']))

    ax.set_title("图4  专用输入头示意图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(OUTPUT_DIR / "fig4_input_heads.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig4_input_heads.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig4 完成")


# ═══════════════════════════════════════════════════════════════
# 图5: 双窗口时序对比示意图
# ═══════════════════════════════════════════════════════════════
def draw_fig5():
    fig, ax = plt.subplots(figsize=(12, 8))
    _setup_ax(ax)

    # 同一地点标签
    _draw_box(ax, 50, 92, 16, 6, "同一地点", fontsize=10,
              boxcolor='#f0f0f0', edgecolor=PALETTE['neutral_dark'])

    # 时间轴
    ax.plot([5, 95], [55, 55], 'k-', linewidth=2)
    ax.plot([5, 5], [53, 57], 'k-', linewidth=2)  # 左端点
    ax.plot([95, 95], [53, 57], 'k-', linewidth=2)  # 右端点
    ax.text(5, 48, "2023.04", ha='center', va='top', fontsize=9)
    ax.text(95, 48, "2025.10", ha='center', va='top', fontsize=9)
    ax.text(50, 48, "时间轴", ha='center', va='top', fontsize=10, fontweight='bold')

    # 时间窗口1 (实线框)
    w1 = FancyBboxPatch((15, 62), 30, 10, boxstyle="round,pad=0,rounding_size=0.5",
                         facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(w1)
    ax.text(30, 67, "时间窗口1", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(30, 63, "不重叠", ha='center', va='center', fontsize=8, color=PALETTE['red_strong'])

    # 时间窗口2 (虚线框)
    w2 = FancyBboxPatch((55, 62), 30, 10, boxstyle="round,pad=0,rounding_size=0.5",
                         facecolor='white', edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(w2)
    ax.text(70, 67, "时间窗口2", ha='center', va='center', fontsize=10, fontweight='bold')

    # 从"同一地点"指向两个窗口
    _draw_arrow(ax, 50, 89, 30, 72)
    _draw_arrow(ax, 50, 89, 70, 72)

    # 从时间轴指向上方的窗口（箭头从时间轴向上指）
    _draw_arrow(ax, 30, 55, 30, 62)
    _draw_arrow(ax, 70, 55, 70, 62)

    # 嵌入向量
    _draw_box(ax, 25, 28, 22, 10, "嵌入向量1\nemb_w1", fontsize=10,
              boxcolor='#e8f0f8', edgecolor=PALETTE['blue_main'])
    _draw_box(ax, 75, 28, 22, 10, "嵌入向量2\nemb_w2", fontsize=10,
              boxcolor='#f8e8e8', edgecolor=PALETTE['red_strong'])

    # 窗口 → 嵌入向量
    _draw_arrow(ax, 30, 62, 25, 33)
    _draw_arrow(ax, 70, 62, 75, 33)

    # 嵌入向量之间：时序对比损失
    ax.annotate('', xy=(75, 28), xytext=(25, 28),
                arrowprops=dict(arrowstyle='<->', color=PALETTE['neutral_dark'],
                                lw=1.5, connectionstyle='arc3,rad=0'))
    ax.text(50, 22, "时序对比损失\n(约束差异)", ha='center', va='top',
            fontsize=9, color=PALETTE['neutral_dark'])

    ax.set_title("图5  双窗口时序对比示意图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(OUTPUT_DIR / "fig5_dual_window.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig5_dual_window.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig5 完成")


# ═══════════════════════════════════════════════════════════════
# 图6: 预归一化两阶段策略对比图
# ═══════════════════════════════════════════════════════════════
def draw_fig6():
    fig, ax = plt.subplots(figsize=(12, 8))
    _setup_ax(ax)

    # 标题
    ax.text(25, 92, "训练阶段", ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(75, 92, "推理阶段", ha='center', va='center', fontsize=13, fontweight='bold')

    # 左侧：训练阶段
    _draw_box(ax, 25, 72, 35, 14, "嵌入向量\n(原始幅度空间)", fontsize=10)
    _draw_arrow(ax, 25, 65, 25, 58)
    _draw_box(ax, 25, 50, 35, 14,
              "反坍缩损失计算\n(原始幅度空间中)", fontsize=10,
              boxcolor='#f8f0f0', edgecolor=PALETTE['red_strong'])
    ax.text(25, 35, "梯度始终非零\n⇒ 有效反坍缩", ha='center', va='center',
            fontsize=9, color=PALETTE['green_3'], fontweight='bold')

    # 中间：VS
    ax.text(50, 62, "VS", ha='center', va='center', fontsize=20,
            fontweight='bold', color=PALETTE['neutral_mid'])

    # 右侧：推理阶段
    _draw_box(ax, 75, 72, 35, 14, "嵌入向量\n(原始幅度空间)", fontsize=10)
    _draw_arrow(ax, 75, 65, 75, 58)
    _draw_box(ax, 75, 50, 35, 14,
              "L2归一化 + 分布噪声\n→ 单位球面嵌入", fontsize=10,
              boxcolor='#f0f0f8', edgecolor=PALETTE['blue_main'])
    ax.text(75, 35, "标准化输出\n⇒ 便于下游使用", ha='center', va='center',
            fontsize=9, color=PALETTE['blue_secondary'], fontweight='bold')

    # 底部说明
    ax.text(50, 15, "现有技术：训练阶段即施加归一化，导致反坍缩梯度消失",
            ha='center', va='center', fontsize=10, color=PALETTE['neutral_dark'],
            style='italic')

    ax.set_title("图6  预归一化两阶段策略对比图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(OUTPUT_DIR / "fig6_two_stage.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "fig6_two_stage.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("✓ fig6 完成")


if __name__ == "__main__":
    print("开始绘制专利附图 v2...")
    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    draw_fig6()
    print(f"\n全部完成！输出目录: {OUTPUT_DIR}")
