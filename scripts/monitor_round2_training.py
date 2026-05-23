#!/usr/bin/env python3
"""Monitor Round 2 training and auto-run evaluation when complete."""
import subprocess
import time
import os
import sys

LOG_FILE = "/workspace/outputs/round2_0517/round2_cross_temporal/train.log"
CHECKPOINT_DIR = "/workspace/outputs/round2_0517/round2_cross_temporal"
CONFIG = "configs/round2_cross_temporal.yaml"

def get_last_epoch():
    try:
        result = subprocess.run(
            ["grep", "-oE", r"Epoch [0-9]+/[0-9]+", LOG_FILE],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if lines and lines[-1]:
            # Parse "Epoch 005/050" -> epoch 5
            parts = lines[-1].replace("Epoch ", "").split("/")
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 50

def check_training_complete():
    try:
        with open(LOG_FILE, "r") as f:
            content = f.read()
        return "Training complete." in content or "[train] Training complete." in content
    except Exception:
        return False

def find_best_checkpoint():
    """Find the best checkpoint file."""
    import glob
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_best*.pt"))
    if checkpoints:
        # Sort by modification time, newest first
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]
    # Fallback to any epoch checkpoint
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt"))
    if checkpoints:
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]
    return None

def run_evaluation(checkpoint_path):
    """Run the standardized evaluation pipeline."""
    print(f"[Monitor] Training complete! Best checkpoint: {checkpoint_path}")
    print("[Monitor] Running standardized evaluation pipeline...")
    
    cmd = [
        sys.executable,
        "scripts/eval/run_full_pipeline.py",
        "--config", CONFIG,
        "--checkpoint", checkpoint_path,
        "--device", "npu:0",
    ]
    
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    
    subprocess.run(cmd, cwd="/workspace/xuannv", env=env)

print("[Monitor] Starting Round 2 training monitor...")
print(f"[Monitor] Log file: {LOG_FILE}")

last_epoch, total = get_last_epoch()
print(f"[Monitor] Initial status: Epoch {last_epoch}/{total}")

while True:
    time.sleep(60)  # Check every minute
    
    if check_training_complete():
        ckpt = find_best_checkpoint()
        if ckpt:
            run_evaluation(ckpt)
        else:
            print("[Monitor] Training complete but no checkpoint found!")
        break
    
    epoch, total = get_last_epoch()
    if epoch > last_epoch:
        print(f"[Monitor] Progress: Epoch {epoch}/{total}")
        last_epoch = epoch

print("[Monitor] Done.")
