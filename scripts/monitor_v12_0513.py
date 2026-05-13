#!/usr/bin/env python3
"""V12 0513 实验后台监控器"""
import os, re, time, json
from datetime import datetime

experiments = {
    "recon005": "/workspace/outputs/v12_recon005_4card_0513/train.log",
    "recon010": "/workspace/outputs/v12_recon010_4card_0513/train.log",
}

def parse_log(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except:
        return None
    
    active = None; recon = None; epoch = None
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

while True:
    now = datetime.now().strftime('%H:%M:%S')
    states = {}
    for name, path in experiments.items():
        s = parse_log(path)
        states[name] = s
        if s:
            print(f"[{now}] {name}: E{s.get('epoch','?')} active={s.get('active','?')} recon={s.get('recon','?'):.4f}" if s.get('recon') else f"[{now}] {name}: initializing...")
        else:
            print(f"[{now}] {name}: no log yet")
    
    # 保存报告
    report = f"# V12 0513 监控报告\n生成时间: {datetime.now().isoformat()}\n\n"
    for name, s in states.items():
        if s:
            report += f"- **{name}**: Epoch={s.get('epoch','?')}, active={s.get('active','?')}, recon={s.get('recon','?'):.4f}\n" if s.get('recon') else f"- **{name}**: initializing...\n"
        else:
            report += f"- **{name}**: no data\n"
    
    with open('/workspace/outputs/v12_0513_monitor_report.md', 'w') as f:
        f.write(report)
    
    # 保存历史
    hist_path = '/workspace/outputs/v12_0513_monitor_history.json'
    try:
        hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
        hist.append({'time': datetime.now().isoformat(), 'states': states})
        json.dump(hist, open(hist_path, 'w'), indent=2)
    except: pass
    
    time.sleep(300)
