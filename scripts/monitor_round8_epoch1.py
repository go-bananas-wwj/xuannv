#!/usr/bin/env python3
"""Monitor Round 8 experiments — alert when any reaches epoch 1."""
import time
import glob
import os
from datetime import datetime

OUTPUT_BASE = "/workspace/outputs"
EXPS = [f"round8_single_exp{i}" for i in range(1, 9)]

def find_epoch1(name):
    wandb_dirs = sorted(glob.glob(f"{OUTPUT_BASE}/{name}/wandb/run-*"), key=os.path.getmtime, reverse=True)
    if not wandb_dirs:
        return None
    log_path = os.path.join(wandb_dirs[0], "files", "output.log")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            if "Epoch 1/" in line:
                return line.strip()
    return None

print(f"[{datetime.now()}] Monitoring started...")
while True:
    for exp in EXPS:
        result = find_epoch1(exp)
        if result:
            msg = f"\n{'='*60}\n[{datetime.now()}] 🎉 {exp} FIRST EPOCH COMPLETE!\n{result}\n{'='*60}\n"
            print(msg)
            with open(f"{OUTPUT_BASE}/round8_epoch1_alert.log", "a") as f:
                f.write(msg + "\n")
            # Don't exit — keep monitoring others
    time.sleep(120)
