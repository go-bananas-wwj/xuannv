#!/usr/bin/env python3
"""自动监控训练并在检测到NaN时修复重启."""
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

LOG_DIR = "/workspace/outputs/aef_qwen_v4_official"
CHECKPOINT = "/workspace/outputs/aef_qwen_v4_official/epoch_best_epoch8.pt"
CONFIG = "/workspace/xuannv/configs/qwen_v4_official.yaml"
TMUX_SESSION = "v4_official"

def get_latest_log():
    logs = sorted(Path(LOG_DIR).glob("train_*.log"))
    return logs[-1] if logs else None

def check_nan_in_log(log_path):
    """检查最近50行是否有NaN/Inf epoch输出."""
    if not log_path.exists():
        return False, None
    lines = log_path.read_text().splitlines()
    recent = lines[-50:]
    for line in recent:
        # Match epoch summary line with nan
        if "total=nan" in line.lower() or "total=inf" in line.lower():
            return True, line
    return False, None

def is_training_running():
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True
    )
    if result.returncode != 0:
        return False
    # Check if python process is actually running inside
    result2 = subprocess.run(
        f"tmux list-panes -t {TMUX_SESSION} -F '#{{pane_pid}}'",
        shell=True, capture_output=True, text=True
    )
    if result2.returncode != 0:
        return False
    pane_pid = result2.stdout.strip()
    if not pane_pid:
        return False
    # Check if there are child python processes
    result3 = subprocess.run(
        f"ps --ppid {pane_pid} -o comm= | grep -q python",
        shell=True, capture_output=True
    )
    # Also check if torchrun/python is in the process tree
    result4 = subprocess.run(
        f"pgrep -f 'train_ddp_v4_official'",
        shell=True, capture_output=True
    )
    return result4.returncode == 0

def kill_training():
    subprocess.run(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; pkill -f 'train_ddp_v4_official' 2>/dev/null; sleep 2; pkill -9 -f 'train_ddp_v4_official' 2>/dev/null", shell=True)
    time.sleep(3)

def start_training():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"{LOG_DIR}/train_{timestamp}.log"
    cmd = (
        f"cd /workspace/xuannv && "
        f"CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 "
        f"scripts/train/train_ddp_v4_official.py --config {CONFIG} "
        f"--resume {CHECKPOINT} 2>&1 | tee -a {logfile}"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, "bash", "-c", cmd],
        capture_output=True
    )
    print(f"[{datetime.now()}] Started new training, log: {logfile}")
    return logfile

def fix_and_restart():
    """检测到NaN时的修复逻辑."""
    print(f"[{datetime.now()}] NaN detected! Killing training...")
    kill_training()
    
    # 读取当前config，进一步降低risky weights
    config_path = Path(CONFIG)
    config_text = config_path.read_text()
    
    # 降低decorrelation weight
    if "decorrelation_weight: 0.05" in config_text:
        config_text = config_text.replace("decorrelation_weight: 0.05", "decorrelation_weight: 0.02")
        print("[{datetime.now()}] Reduced decorrelation_weight 0.05 -> 0.02")
    elif "decorrelation_weight: 0.1" in config_text:
        config_text = config_text.replace("decorrelation_weight: 0.1", "decorrelation_weight: 0.02")
        print("[{datetime.now()}] Reduced decorrelation_weight 0.1 -> 0.02")
    
    # 降低classification weight
    if "classification_weight: 0.03" in config_text:
        config_text = config_text.replace("classification_weight: 0.03", "classification_weight: 0.01")
        print("[{datetime.now()}] Reduced classification_weight 0.03 -> 0.01")
    elif "classification_weight: 0.05" in config_text:
        config_text = config_text.replace("classification_weight: 0.05", "classification_weight: 0.01")
        print("[{datetime.now()}] Reduced classification_weight 0.05 -> 0.01")
    
    config_path.write_text(config_text)
    
    # 重新启动
    new_log = start_training()
    return new_log

def main():
    print(f"[{datetime.now()}] Monitor started. Watching {LOG_DIR}")
    current_log = get_latest_log()
    if current_log:
        print(f"[{datetime.now()}] Current log: {current_log}")
    
    nan_count = 0
    restart_count = 0
    max_restarts = 3
    
    while True:
        time.sleep(600)
        
        # Check if training is still running
        if not is_training_running():
            current_log = get_latest_log()
            if current_log:
                # Check if log shows completion
                log_text = current_log.read_text()
                if "Training complete" in log_text:
                    print(f"[{datetime.now()}] Training completed successfully!")
                    break
                elif "Epoch 300" in log_text or "Epoch 300/300" in log_text:
                    print(f"[{datetime.now()}] Training reached final epoch!")
                    break
            
            print(f"[{datetime.now()}] Training not running, restarting...")
            current_log = start_training()
            restart_count += 1
            if restart_count > max_restarts:
                print(f"[{datetime.now()}] Too many restarts ({restart_count}), giving up.")
                break
            continue
        
        # Training is running, check for NaN
        current_log = get_latest_log()
        if current_log:
            has_nan, nan_line = check_nan_in_log(current_log)
            if has_nan:
                nan_count += 1
                print(f"[{datetime.now()}] NaN found: {nan_line}")
                if nan_count >= 1:  # 检测到1次就修复
                    if restart_count >= max_restarts:
                        print(f"[{datetime.now()}] Max restarts reached. Stopping monitor.")
                        break
                    current_log = fix_and_restart()
                    restart_count += 1
                    nan_count = 0
            else:
                nan_count = max(0, nan_count - 1)  # 衰减
        
        # Print status every 2 minutes
        if int(time.time()) % 120 < 35:
            current_log = get_latest_log()
            if current_log:
                lines = current_log.read_text().splitlines()
                # Find last epoch line
                for line in reversed(lines[-30:]):
                    if "Epoch" in line and "total=" in line:
                        print(f"[{datetime.now()}] Status: {line.strip()}")
                        break

if __name__ == "__main__":
    main()
