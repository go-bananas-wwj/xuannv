#!/usr/bin/env python3
"""Kimi Background Task 监控器 — 训练完成后自动退出"""
import os, re, time, json, subprocess
from datetime import datetime

EXPERIMENTS = {
    "recon005": "/workspace/outputs/v12_recon005_4card_0513/train.log",
    "recon010": "/workspace/outputs/v12_recon010_4card_0513/train.log",
}
TARGET_EPOCHS = 20  # 目标完成 epoch

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    active = recon = epoch = None
    for line in reversed(lines[-300:]):
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
    """检查是否有 train_ddp_v12 进程在运行"""
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
        if s and s.get('epoch') and s['epoch'] >= TARGET_EPOCHS:
            done_count += 1
            print(f"[{now}] {name}: ✅ Epoch {s['epoch']} 完成 (active={s['active']}, recon={s['recon']:.4f})")
        elif s:
            print(f"[{now}] {name}: E{s.get('epoch','?')} active={s.get('active','?')} recon={s.get('recon','?'):.4f}")
        else:
            print(f"[{now}] {name}: 等待训练中...")
    
    # 保存报告
    report_lines = [f"# V12 Background Monitor\n生成时间: {datetime.now().isoformat()}\n"]
    for name, s in states.items():
        if s:
            report_lines.append(f"- **{name}**: Epoch={s.get('epoch','?')}, active={s.get('active','?')}, recon={s.get('recon','?'):.4f}")
        else:
            report_lines.append(f"- **{name}**: 等待中")
    with open('/workspace/outputs/v12_bg_monitor_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    # 检查是否全部完成或训练已停止
    if done_count == len(EXPERIMENTS):
        print(f"\n[{now}] 🎉 所有实验已达到 {TARGET_EPOCHS} epochs！监控结束。")
        break
    
    if not is_training_alive() and any(s and s.get('epoch', 0) > 1 for s in states.values()):
        print(f"\n[{now}] ⚠️ 训练进程已消失，监控结束。")
        break
    
    print(f"[{now}] 等待 5 分钟后下一次检查...\n")
    time.sleep(300)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 监控器退出")
