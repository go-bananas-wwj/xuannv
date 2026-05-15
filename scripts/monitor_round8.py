#!/usr/bin/env python3
"""Monitor Round 8 experiments and report when first epoch completes."""
import time
import glob
import os
from datetime import datetime

OUTPUT_BASE = "/workspace/outputs"
EXPS = [f"round8_single_exp{i}" for i in range(1, 9)]

def check_exp(name):
    wandb_dirs = sorted(glob.glob(f"{OUTPUT_BASE}/{name}/wandb/run-*"), key=os.path.getmtime, reverse=True)
    if not wandb_dirs:
        return "no wandb dir"
    latest = wandb_dirs[0]
    log_path = os.path.join(latest, "files", "output.log")
    if not os.path.exists(log_path):
        return "no output.log"
    with open(log_path) as f:
        lines = f.readlines()
    epoch_lines = [l for l in lines if "Epoch " in l and "/" in l]
    oom_lines = [l for l in lines if "out of memory" in l or "OOM" in l]
    if epoch_lines:
        return f"EPOCH: {epoch_lines[-1].strip()}"
    if oom_lines:
        return f"OOM: {len(oom_lines)} times"
    return f"running ({len(lines)} lines)"

with open("/workspace/outputs/round8_monitor.log", "a") as f:
    f.write(f"\n=== Monitor started at {datetime.now()} ===\n")
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{now}]\n")
        for exp in EXPS:
            status = check_exp(exp)
            f.write(f"  {exp}: {status}\n")
        f.flush()
        time.sleep(300)
