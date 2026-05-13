#!/usr/bin/env python3
"""Kimi Background Task 监控器 — 训练完成后自动退出"""
import os, re, time, json, subprocess
from datetime import datetime

EXPERIMENTS = {
    "recon005": "/workspace/outputs/v12_recon005_4card_0513/train.log",
    "recon010": "/workspace/outputs/v12_recon010_4card_0513/train.log",
}
TARGET_EPOCHS = 20

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    active = recon = epoch = None
    for line in reversed(lines[-400:]):
        m = re.search(r'Epoch\s+(\d+)', line)
        if m and epoch is None: epoch = int(m.group(1))
        m = re.search(r'active=([\d]+)', line)
        if m and active is None: active = int(m.group(1))
        m = re.search(r'recon=([\d.]+)', line)
        if m and recon is None: recon = float(m.group(1))
        if active is not None and recon is not None:
            break
    return {'epoch': epoch, 'active': active, 'recon': recon}

def is_training_alive():
    result = subprocess.run(['pgrep', '-f', 'train_ddp_v12'], capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() != ''

print(f"[{datetime.now().strftime('%H:%M:%S')}] V12 Background Monitor 启动")
print(f"监控实验: {', '.join(EXPERIMENTS.keys())}")
print(f"目标 epoch: {TARGET_EPOCHS}")
print("")

while True:
    now = datetime.now().strftime('%H:%M:%S')
    states = {}
    done_count = 0
    
    for name, path in EXPERIMENTS.items():
        s = parse_log(path)
        states[name] = s
        epoch_str = s.get('epoch', '?') if s else '?'
        active_str = s.get('active', '?') if s else '?'
        recon_str = f"{s['recon']:.4f}" if s and s.get('recon') is not None else '?'
        
        if s and s.get('epoch') and s['epoch'] >= TARGET_EPOCHS:
            done_count += 1
            print(f"[{now}] {name}: E{epoch_str} active={active_str} recon={recon_str} ✅")
        else:
            print(f"[{now}] {name}: E{epoch_str} active={active_str} recon={recon_str}")
    
    # 保存报告
    report_lines = [f"# V12 Background Monitor\n生成时间: {datetime.now().isoformat()}\n"]
    for name, s in states.items():
        if s:
            e = s.get('epoch', '?'); a = s.get('active', '?'); r = s.get('recon')
            rstr = f"{r:.4f}" if r is not None else '?'
            report_lines.append(f"- **{name}**: Epoch={e}, active={a}, recon={rstr}")
        else:
            report_lines.append(f"- **{name}**: 等待中")
    with open('/workspace/outputs/v12_bg_monitor_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    if done_count == len(EXPERIMENTS):
        print(f"\n[{now}] 所有实验已达到 {TARGET_EPOCHS} epochs！监控结束。")
        break
    
    if not is_training_alive() and any(s and s.get('epoch', 0) > 1 for s in states.values() if s):
        print(f"\n[{now}] 训练进程已消失，监控结束。")
        break
    
    print(f"[{now}] 等待 5 分钟后下一次检查...\n")
    time.sleep(300)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 监控器退出")
