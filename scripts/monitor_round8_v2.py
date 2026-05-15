#!/usr/bin/env python3
"""Monitor Round 8 experiments via tmux output — alert when any reaches epoch 1."""
import subprocess
import time
from datetime import datetime

EXPS = [f"r8s{i}" for i in range(1, 9)]

def check_tmux(session):
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout
        for line in output.splitlines():
            if "Epoch 1/20" in line:
                return line.strip()
        return None
    except Exception:
        return None

print(f"[{datetime.now()}] Monitoring started...")
while True:
    for exp in EXPS:
        result = check_tmux(exp)
        if result:
            msg = f"\n{'='*60}\n[{datetime.now()}] 🎉 {exp} FIRST EPOCH COMPLETE!\n{result}\n{'='*60}\n"
            print(msg)
            with open("/workspace/outputs/round8_epoch1_alert_v2.log", "a") as f:
                f.write(msg + "\n")
    time.sleep(60)
