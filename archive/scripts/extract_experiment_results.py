#!/usr/bin/env python3
"""提取所有V13实验的epoch结果并生成对比表格."""
import sys
from pathlib import Path
import re

EXPERIMENTS = {
    'v13_exp1_spatial_unif': 'Exp1: Spatial Uniformity',
    'v13_exp2_high_weight': 'Exp2: High Weight',
    'v13_exp3_vicreg_fix': 'Exp3: VICReg Fix',
    'v13_exp4_combined': 'Exp4: Combined',
    'v13_exp5_prenorm_raw': 'Exp5: Pre-norm Raw',
}

def extract_epochs(log_path):
    """从日志文件中提取每个epoch的指标."""
    if not log_path.exists():
        return []
    epochs = []
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配: [HH:MM:SS] Epoch NNN/10 | total=... recon=... consist=... var=... cov=... l2unif=... bank=... active=... std_mean=... lr=...
            m = re.search(r'\[\d{2}:\d{2}:\d{2}\] Epoch (\d+)/\d+ \| total=([\d.]+) recon=([\d.]+) consist=([\d.]+) var=([\d.]+) cov=([\d.]+) l2unif=([\d.]+) bank=\d+/\d+ active=(\d+)/\d+ std_mean=([\d.]+)', line)
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'total': float(m.group(2)),
                    'recon': float(m.group(3)),
                    'consist': float(m.group(4)),
                    'var': float(m.group(5)),
                    'cov': float(m.group(6)),
                    'l2unif': float(m.group(7)),
                    'active_dims': int(m.group(8)),
                    'std_mean': float(m.group(9)),
                })
    return epochs

def main():
    print("# V13 Uniformity 实验结果汇总\n")
    print("| 实验 | Epoch | total | recon | consist | var | cov | l2unif | active_dims | std_mean |")
    print("|------|-------|-------|-------|---------|-----|-----|--------|-------------|----------|")
    
    for exp_name, exp_label in EXPERIMENTS.items():
        log_path = Path(f'/workspace/outputs/{exp_name}/train.log')
        epochs = extract_epochs(log_path)
        for e in epochs:
            print(f"| {exp_label} | {e['epoch']} | {e['total']:.4f} | {e['recon']:.4f} | {e['consist']:.4f} | {e['var']:.4f} | {e['cov']:.4f} | {e['l2unif']:.4f} | {e['active_dims']} | {e['std_mean']:.4f} |")
        if not epochs:
            print(f"| {exp_label} | N/A | - | - | - | - | - | - | - | - |")

if __name__ == '__main__':
    main()
