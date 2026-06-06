#!/usr/bin/env python3
"""
海淀区训练可视化分析脚本
生成训练过程趋势图
"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
font_path = '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from pathlib import Path


def main():
    output_dir = Path('/workspace/xuannv/outputs/exp_v2_haidian_baseline_8card_100ep')
    save_dir = output_dir / 'visualization'
    save_dir.mkdir(exist_ok=True)

    # 加载数据
    with open(output_dir / 'epoch_metrics.json') as f:
        epochs = json.load(f)

    ep = [e['epoch'] for e in epochs]
    l2unif = [e['l2unif'] for e in epochs]
    recon = [e['recon'] for e in epochs]
    std_mean = [e['std_mean'] for e in epochs]
    total = [e['total'] for e in epochs]
    var = [e['var'] for e in epochs]
    cov = [e['cov'] for e in epochs]
    consist = [e['consist'] for e in epochs]
    lr = [e['lr'] for e in epochs]

    # 评分
    score = [s - r * 0.5 for s, r in zip(std_mean, recon)]

    # ========== 图1: 核心指标综合趋势 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('海淀区训练 - 核心指标趋势 (Epoch 1-100)', fontsize=14, fontweight='bold')

    # l2unif
    ax = axes[0, 0]
    ax.plot(ep, l2unif, 'b-', linewidth=1.2, label='l2unif')
    ax.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5, label='collapse警戒线(-0.5)')
    ax.axvline(x=20, color='g', linestyle='--', alpha=0.5, label='干预点(E20)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('l2unif')
    ax.set_title('L2 Uniformity (越低越好)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    # recon
    ax = axes[0, 1]
    ax.plot(ep, recon, 'g-', linewidth=1.2, label='recon')
    ax.axvline(x=20, color='g', linestyle='--', alpha=0.5, label='干预点(E20)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('recon')
    ax.set_title('Reconstruction Loss (越低越好)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    # std_mean
    ax = axes[1, 0]
    ax.plot(ep, std_mean, 'm-', linewidth=1.2, label='std_mean')
    ax.axvline(x=20, color='g', linestyle='--', alpha=0.5, label='干预点(E20)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('std_mean')
    ax.set_title('Embedding Std Mean (越高越好)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    # score
    ax = axes[1, 1]
    ax.plot(ep, score, 'r-', linewidth=1.2, label='score=std_mean-recon*0.5')
    best_idx = int(np.argmax(score))
    ax.scatter([ep[best_idx]], [score[best_idx]], color='red', s=80, zorder=5, label=f'Best E{best_idx+1}')
    ax.axvline(x=20, color='g', linestyle='--', alpha=0.5, label='干预点(E20)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('综合评分 (越高越好)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    plt.tight_layout()
    plt.savefig(save_dir / '01_core_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {save_dir / "01_core_metrics.png"}')

    # ========== 图2: 损失分解 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ep, total, 'k-', linewidth=1.5, label='total', alpha=0.8)
    ax.plot(ep, recon, 'g-', linewidth=1.0, label='recon')
    ax.plot(ep, [v * 0.3 for v in var], 'orange', linewidth=1.0, label='var*0.3')
    ax.plot(ep, [c * 0.1 for c in cov], 'purple', linewidth=1.0, label='cov*0.1')
    ax.plot(ep, consist, 'c-', linewidth=1.0, label='consist')
    ax.axvline(x=20, color='g', linestyle='--', alpha=0.5, label='干预点(E20)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('损失分解趋势')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)
    plt.tight_layout()
    plt.savefig(save_dir / '02_loss_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {save_dir / "02_loss_decomposition.png"}')

    # ========== 图3: l2unif 与 std_mean 对比 ==========
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('l2unif', color=color1)
    ax1.plot(ep, l2unif, color=color1, linewidth=1.5, label='l2unif')
    ax1.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=20, color='g', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:purple'
    ax2.set_ylabel('std_mean', color=color2)
    ax2.plot(ep, std_mean, color=color2, linewidth=1.5, linestyle='--', label='std_mean')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('l2unif (蓝) vs std_mean (紫) - Uniformity-Variance Trade-off')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig(save_dir / '03_unif_vs_stdmean.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {save_dir / "03_unif_vs_stdmean.png"}')

    # ========== 图4: 学习率与 lr ==========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(ep, lr, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.set_title('学习率衰减')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)
    plt.tight_layout()
    plt.savefig(save_dir / '04_lr_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {save_dir / "04_lr_schedule.png"}')

    # ========== 图5: 分阶段对比 ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('训练三阶段对比', fontsize=12, fontweight='bold')

    stages = [
        ('Epoch 1-20\n(初期)', 0, 20),
        ('Epoch 21-56\n(平台期)', 20, 56),
        ('Epoch 57-100\n(改善期)', 56, 100),
    ]

    for idx, (title, start, end) in enumerate(stages):
        ax = axes[idx]
        sub_ep = ep[start:end]
        sub_l2 = l2unif[start:end]
        sub_re = recon[start:end]
        sub_sm = std_mean[start:end]

        ax.plot(sub_ep, sub_l2, 'b-', linewidth=1.5, label='l2unif')
        ax_twin = ax.twinx()
        ax_twin.plot(sub_ep, sub_re, 'g-', linewidth=1.0, linestyle='--', label='recon')
        ax_twin.plot(sub_ep, sub_sm, 'm-', linewidth=1.0, linestyle='--', label='std_mean')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('l2unif', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y')
        ax.grid(True, alpha=0.3)

    fig.legend(['l2unif', 'recon', 'std_mean'], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    plt.savefig(save_dir / '05_stage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {save_dir / "05_stage_comparison.png"}')

    # ========== 统计分析 ==========
    print('\n' + '='*60)
    print('训练统计分析')
    print('='*60)

    # 阶段统计
    def stage_stats(name, start, end):
        s_l2 = l2unif[start:end]
        s_re = recon[start:end]
        s_sm = std_mean[start:end]
        print(f'\n{name} (E{start+1}-{end}):')
        print(f'  l2unif: {s_l2[0]:.4f} -> {s_l2[-1]:.4f} (变化 {s_l2[-1]-s_l2[0]:+.4f})')
        print(f'  recon:  {s_re[0]:.4f} -> {s_re[-1]:.4f} (变化 {s_re[-1]-s_re[0]:+.4f})')
        print(f'  std_mean: {s_sm[0]:.4f} -> {s_sm[-1]:.4f} (变化 {s_sm[-1]-s_sm[0]:+.4f})')

    stage_stats('初期', 0, 20)
    stage_stats('平台期', 20, 56)
    stage_stats('改善期', 56, 100)

    print(f'\n全程 (E1-100):')
    print(f'  l2unif: {l2unif[0]:.4f} -> {l2unif[-1]:.4f} (变化 {l2unif[-1]-l2unif[0]:+.4f})')
    print(f'  recon:  {recon[0]:.4f} -> {recon[-1]:.4f} (变化 {recon[-1]-recon[0]:+.4f})')
    print(f'  std_mean: {std_mean[0]:.4f} -> {std_mean[-1]:.4f} (变化 {std_mean[-1]-std_mean[0]:+.4f})')

    best_score_idx = int(np.argmax(score))
    print(f'\n最佳评分 Epoch: {best_score_idx + 1}')
    print(f'  score={score[best_score_idx]:.4f} (std_mean={std_mean[best_score_idx]:.4f}, recon={recon[best_score_idx]:.4f})')

    best_l2_idx = int(np.argmin(l2unif))
    print(f'\n最佳 l2unif Epoch: {best_l2_idx + 1}')
    print(f'  l2unif={l2unif[best_l2_idx]:.4f}')

    print(f'\n最终 Epoch 100:')
    print(f'  l2unif={l2unif[-1]:.4f}, recon={recon[-1]:.4f}, std_mean={std_mean[-1]:.4f}')

    print(f'\n所有图表已保存至: {save_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
