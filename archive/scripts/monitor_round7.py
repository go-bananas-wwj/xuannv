#!/usr/bin/env python3
"""Round7 8卡并行实验长期监控脚本.

每5分钟读取8个实验的train.log，提取关键指标并输出到监控日志。
"""
import os
import sys
import time
import re
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

EXPERIMENTS = [
    "round7_exp1_pre_norm_strong",
    "round7_exp2_pre_norm_low_recon",
    "round7_exp3_no_teacher",
    "round7_exp4_vicreg_heavy",
    "round7_exp5_l2_space_ctrl",
    "round7_exp6_high_recon_ctrl",
    "round7_exp7_strong_orth",
    "round7_exp8_full_combo",
]

SHORT_NAMES = ["E1_preN", "E2_lowR", "E3_noTS", "E4_vicH", "E5_l2C", "E6_hiR", "E7_orth", "E8_full"]

MONITOR_LOG = "/workspace/outputs/round7_monitor.log"


def parse_log_line(line: str) -> dict:
    """从train.log的一行提取关键指标."""
    result = {}
    
    # Step
    m = re.search(r'\[Step (\d+)\]', line)
    if m:
        result['step'] = int(m.group(1))
    
    # Epoch
    m = re.search(r'Epoch (\d+)', line)
    if m:
        result['epoch'] = int(m.group(1))
    
    # recon
    m = re.search(r'recon=([\d.]+)', line)
    if m:
        result['recon'] = float(m.group(1))
    
    # l2unif / raw_unif
    m = re.search(r'l2unif=(-?[\d.]+)', line)
    if m:
        result['l2unif'] = float(m.group(1))
    
    m = re.search(r'raw_unif=(-?[\d.]+)', line)
    if m:
        result['raw_unif'] = float(m.group(1))
    
    # spatial active_dims
    m = re.search(r'spatial=\[(\d+)/(\d+):([\d.]+)\]', line)
    if m:
        result['spatial_active'] = int(m.group(1))
        result['spatial_total'] = int(m.group(2))
        result['spatial_std'] = float(m.group(3))
    
    # inter active_dims
    m = re.search(r'inter=\[(\d+)/(\d+):([\d.]+)\]', line)
    if m:
        result['inter_active'] = int(m.group(1))
        result['inter_std'] = float(m.group(3))
    
    # orth
    m = re.search(r'orth=([\d.]+)', line)
    if m:
        result['orth'] = float(m.group(1))
    
    # var
    m = re.search(r'var=([\d.]+)', line)
    if m:
        result['var'] = float(m.group(1))
    
    # cov
    m = re.search(r'cov=([\d.]+)', line)
    if m:
        result['cov'] = float(m.group(1))
    
    # decorr
    m = re.search(r'decorr=([\d.]+)', line)
    if m:
        result['decorr'] = float(m.group(1))
    
    # consist
    m = re.search(r'consist=([\d.]+)', line)
    if m:
        result['consist'] = float(m.group(1))
    
    # lr
    m = re.search(r'lr=([\d.e+-]+)', line)
    if m:
        result['lr'] = float(m.group(1))
    
    return result


def check_anomaly(data: dict, short_name: str) -> list:
    """检查异常指标."""
    alerts = []
    
    spatial_active = data.get('spatial_active', 128)
    spatial_std = data.get('spatial_std', 1.0)
    l2unif = data.get('l2unif', 0.0)
    
    if spatial_active < 100:
        alerts.append(f"🔴 {short_name}: active_dims={spatial_active} (坍缩预警)")
    elif spatial_active < 120:
        alerts.append(f"🟡 {short_name}: active_dims={spatial_active} (轻微坍缩)")
    
    if spatial_std < 0.1:
        alerts.append(f"🔴 {short_name}: std={spatial_std:.4f} (严重坍缩)")
    
    if l2unif > 0.5:
        alerts.append(f"🔴 {short_name}: l2unif={l2unif:.4f} (uniformity恶化)")
    
    return alerts


def monitor_once():
    """执行一次监控轮询."""
    lines = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n{'='*100}")
    lines.append(f"[{timestamp}] Round7 监控报告")
    lines.append(f"{'='*100}")
    
    # 表头
    header = f"{'Exp':<8} {'Epoch':>5} {'Step':>5} {'Recon':>7} {'Unif':>7} {'Active':>8} {'Std':>6} {'Orth':>6} {'Var':>6} {'Cov':>6} {'Dec':>6} {'状态':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    
    all_alerts = []
    
    for exp_name, short_name in zip(EXPERIMENTS, SHORT_NAMES):
        log_path = Path(f"/workspace/outputs/{exp_name}/train.log")
        if not log_path.exists():
            lines.append(f"{short_name:<8} {'N/A':>5} {'N/A':>5} {'N/A':>7} {'N/A':>7} {'N/A':>8} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'⏳':>6}")
            continue
        
        # 读取最后非空行
        with open(log_path, 'r') as f:
            log_lines = f.readlines()
        
        # 找到最后一个包含Step的行
        data = {}
        for line in reversed(log_lines):
            line = line.strip()
            if not line or '[Step' not in line:
                continue
            data = parse_log_line(line)
            if data:
                break
        
        if not data:
            lines.append(f"{short_name:<8} {'?':>5} {'?':>5} {'?':>7} {'?':>7} {'?':>8} {'?':>6} {'?':>6} {'?':>6} {'?':>6} {'?':>6} {'?':>6}")
            continue
        
        epoch = data.get('epoch', '?')
        step = data.get('step', '?')
        recon = data.get('recon', 0.0)
        unif = data.get('l2unif', data.get('raw_unif', 0.0))
        active = f"{data.get('spatial_active', 0)}/{data.get('spatial_total', 128)}"
        std = data.get('spatial_std', 0.0)
        orth = data.get('orth', 0.0)
        var = data.get('var', 0.0)
        cov = data.get('cov', 0.0)
        decorr = data.get('decorr', 0.0)
        
        # 状态判断
        status = "✅"
        alerts = check_anomaly(data, short_name)
        if alerts:
            status = "⚠️"
            all_alerts.extend(alerts)
        
        line = f"{short_name:<8} {epoch:>5} {step:>5} {recon:>7.4f} {unif:>7.4f} {active:>8} {std:>6.4f} {orth:>6.4f} {var:>6.4f} {cov:>6.4f} {decorr:>6.2f} {status:>6}"
        lines.append(line)
    
    lines.append("-" * len(header))
    
    # 输出异常告警
    if all_alerts:
        lines.append("")
        lines.append("⚠️ 异常告警:")
        for alert in all_alerts:
            lines.append(f"  {alert}")
    else:
        lines.append("")
        lines.append("✅ 所有实验指标正常")
    
    output = "\n".join(lines)
    
    # 写入日志文件
    with open(MONITOR_LOG, "a") as f:
        f.write(output + "\n")
    
    # 同时打印到stdout
    print(output)


def main():
    print(f"Round7 监控启动，日志: {MONITOR_LOG}")
    print("监控间隔: 300秒（5分钟）")
    print("按 Ctrl+C 停止\n")
    
    # 立即执行一次
    monitor_once()
    
    while True:
        try:
            time.sleep(300)
            monitor_once()
        except KeyboardInterrupt:
            print("\n监控停止")
            break
        except Exception as e:
            print(f"监控异常: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
