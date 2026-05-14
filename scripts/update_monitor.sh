#!/bin/bash
cd /workspace/xuannv
/root/miniconda3/envs/xuannv/bin/python -c "
import os, re
from datetime import datetime

EXPERIMENTS = {
    'expA': '/workspace/outputs/v12_expA_baseline/train.log',
    'expB': '/workspace/outputs/v12_expB_inter01/train.log',
    'expC': '/workspace/outputs/v12_expC_inter02/train.log',
    'expD': '/workspace/outputs/v12_expD_recon01/train.log',
}

def parse_epoch_summary(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in reversed(lines[-200:]):
        m = re.search(r'Epoch\s+(\d+).*recon=([\d.]+).*inter_var=([\d.]+).*active=(\d+).*std_mean=([\d.]+)', line)
        if m:
            return {
                'epoch': int(m.group(1)), 'recon': float(m.group(2)),
                'inter_var': float(m.group(3)), 'active': int(m.group(4)),
                'std_mean': float(m.group(5)),
            }
    return None

def parse_step_logs(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in reversed(lines[-100:]):
        m = re.search(r'inter=\[(\d+)/(\d+):([\d.]+)\]', line)
        if m:
            return {'inter_active': int(m.group(1)), 'inter_std': float(m.group(3))}
    return None

report = [f'# 4实验并行监控报告\n生成时间: {datetime.now().isoformat()}\n\n']
report.append('| 实验 | Epoch | Recon | Spatial | SpatialStd | InterActive | InterStd | InterVar | 状态 |\n')
report.append('|------|-------|-------|---------|------------|-------------|----------|----------|------|\n')

for name, path in EXPERIMENTS.items():
    epoch_data = parse_epoch_summary(path)
    step_data = parse_step_logs(path)
    if epoch_data:
        e = epoch_data['epoch']; r = epoch_data['recon']
        a = epoch_data['active']; sm = epoch_data['std_mean']; iv = epoch_data['inter_var']
        ia = step_data['inter_active'] if step_data else '?'
        is_ = step_data['inter_std'] if step_data else '?'
        status = '✅' if e >= 50 else '🔄'
        report.append(f'| {name} | {e} | {r:.4f} | {a}/128 | {sm:.4f} | {ia}/128 | {is_} | {iv:.4f} | {status} |\n')
    else:
        report.append(f'| {name} | - | - | - | - | - | - | - | ⏳ |\n')

out = ''.join(report)
with open('/workspace/outputs/4exp_monitor_report.md', 'w') as f:
    f.write(out)

print(out)
"
