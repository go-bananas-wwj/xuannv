#!/usr/bin/env python3
"""4实验并行监控器"""
import os, re, time, json, subprocess
from datetime import datetime

EXPERIMENTS = {
    "expA": "/workspace/outputs/v12_expA_baseline/train.log",
    "expB": "/workspace/outputs/v12_expB_inter01/train.log",
    "expC": "/workspace/outputs/v12_expC_inter02/train.log",
    "expD": "/workspace/outputs/v12_expD_recon01/train.log",
}
TARGET_EPOCHS = 15

def parse_epoch_summary(path):
    """解析 epoch 摘要行"""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # 找最后一个 Epoch 行
    for line in reversed(lines[-200:]):
        m = re.search(r'Epoch\s+(\d+).*recon=([\d.]+).*active=(\d+).*std_mean=([\d.]+).*inter_var=([\d.]+)', line)
        if m:
            return {
                'epoch': int(m.group(1)),
                'recon': float(m.group(2)),
                'active': int(m.group(3)),
                'std_mean': float(m.group(4)),
                'inter_var': float(m.group(5)),
            }
    return None

def parse_step_logs(path):
    """解析 step 日志中的 inter_active"""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # 找最后一个 [Step] 行
    for line in reversed(lines[-100:]):
        m = re.search(r'inter=\[(\d+)/(\d+):([\d.]+)\]', line)
        if m:
            return {
                'inter_active': int(m.group(1)),
                'inter_std': float(m.group(3)),
            }
    return None

def is_training_alive():
    result = subprocess.run(['pgrep', '-f', 'train_ddp_v12'], capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() != ''

print(f"[{datetime.now().strftime('%H:%M:%S')}] 4实验监控器启动")
print(f"目标 epoch: {TARGET_EPOCHS}")
print("")

while True:
    now = datetime.now().strftime('%H:%M:%S')
    states = {}
    done_count = 0
    
    for name, path in EXPERIMENTS.items():
        epoch_data = parse_epoch_summary(path)
        step_data = parse_step_logs(path)
        
        if epoch_data:
            states[name] = {**epoch_data, **(step_data or {})}
            e = epoch_data['epoch']
            r = epoch_data['recon']
            a = epoch_data['active']
            s = epoch_data['std_mean']
            ia = step_data['inter_active'] if step_data else '?'
            is_ = step_data['inter_std'] if step_data else '?'
            
            if e >= TARGET_EPOCHS:
                done_count += 1
                print(f"[{now}] {name}: E{e:02d} recon={r:.4f} spatial={a}/128:{s:.4f} inter={ia}/128:{is_} ✅")
            else:
                print(f"[{now}] {name}: E{e:02d} recon={r:.4f} spatial={a}/128:{s:.4f} inter={ia}/128:{is_}")
        else:
            print(f"[{now}] {name}: 等待训练中...")
    
    # 保存报告
    report = [f"# 4实验并行监控报告\n生成时间: {datetime.now().isoformat()}\n\n"]
    report.append("| 实验 | Epoch | Recon | Spatial | SpatialStd | InterActive | InterStd | 状态 |\n")
    report.append("|------|-------|-------|---------|------------|-------------|----------|------|\n")
    
    for name, path in EXPERIMENTS.items():
        s = states.get(name)
        if s:
            e = s.get('epoch', '?')
            r = s.get('recon', '?')
            a = s.get('active', '?')
            sm = s.get('std_mean', '?')
            ia = s.get('inter_active', '?')
            is_ = s.get('inter_std', '?')
            status = "✅" if e != '?' and e >= TARGET_EPOCHS else "🔄"
            report.append(f"| {name} | {e} | {r:.4f} | {a}/128 | {sm:.4f} | {ia}/128 | {is_} | {status} |\n")
        else:
            report.append(f"| {name} | - | - | - | - | - | - | ⏳ |\n")
    
    with open('/workspace/outputs/4exp_monitor_report.md', 'w') as f:
        f.write(''.join(report))
    
    # 保存历史
    hist_path = '/workspace/outputs/4exp_monitor_history.json'
    try:
        hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
        hist.append({'time': datetime.now().isoformat(), 'states': states})
        json.dump(hist, open(hist_path, 'w'), indent=2)
    except: pass
    
    if done_count == len(EXPERIMENTS):
        print(f"\n[{now}] 所有实验已达到 {TARGET_EPOCHS} epochs！监控结束。")
        break
    
    if not is_training_alive() and any(s and s.get('epoch', 0) > 1 for s in states.values() if s):
        print(f"\n[{now}] 训练进程已消失，监控结束。")
        break
    
    print(f"[{now}] 等待 5 分钟后下一次检查...\n")
    time.sleep(300)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 监控器退出")
