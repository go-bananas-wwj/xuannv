#!/usr/bin/env python3
"""
ExpD 训练存活监控 — 只检测进程是否存活/卡住，指标异常不干预
"""
import os, time, re, subprocess
from pathlib import Path
from datetime import datetime

TRAIN_NAME = "expD_train"
LOG_DIR = Path("/workspace/outputs/v2_skipL2_7target_lowrecon_7card_0520")
CONFIG = "/workspace/xuannv/configs/xuannv_v2_expD_7target_lowrecon.yaml"
SCRIPT = "/workspace/xuannv/scripts/train/train_ddp_xuannv_v2.py"
NPUS, NPROC = "0,1,2,3,4,5,6", 7
CHECK_INTERVAL = 300  # 5分钟检查一次
MAX_NO_OUTPUT_SEC = 900  # 15分钟无输出视为卡住


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "keepalive.log", "a") as f:
        f.write(line + "\n")


def is_alive():
    try:
        r = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True, timeout=5)
        return TRAIN_NAME in r.stdout
    except:
        return False


def get_last_step():
    try:
        r = subprocess.run(["tmux", "capture-pane", "-t", TRAIN_NAME, "-p", "-J"],
                          capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().split("\n")
        for line in reversed(lines):
            m = re.search(r"\[Step\s+(\d+)\]", line)
            if m:
                return int(m.group(1))
    except:
        pass
    return -1


def find_latest_ckpt():
    ckpts = list(LOG_DIR.glob("epoch_*.pt"))
    if not ckpts:
        return None
    regular = [p for p in ckpts if "best" not in p.name]
    if regular:
        def num(p):
            m = re.search(r'epoch_(\d+)', p.name)
            return int(m.group(1)) if m else 0
        return max(regular, key=num)
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def clean_cache():
    for f in LOG_DIR.glob("dataset_cache_*.pt"):
        f.unlink()
        log(f"Removed cache: {f.name}")


def restart(resume_ckpt=None):
    env = f"export ASCEND_RT_VISIBLE_DEVICES={NPUS} && export HCCL_CONNECT_TIMEOUT=600 && export HCCL_EXEC_TIMEOUT=600"
    conda = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv"
    if resume_ckpt:
        cmd = f"torchrun --nproc_per_node={NPROC} {SCRIPT} --config {CONFIG} --resume {resume_ckpt} --save-every 20"
        log(f"RESTART with resume: {resume_ckpt.name}")
    else:
        cmd = f"torchrun --nproc_per_node={NPROC} {SCRIPT} --config {CONFIG} --save-every 20"
        log("RESTART from scratch")
    full = f"{env} && {conda} && cd /workspace/xuannv && {cmd}"
    subprocess.run(["tmux", "new-session", "-d", "-s", TRAIN_NAME, "-c", "/workspace/xuannv"], timeout=10)
    subprocess.run(["tmux", "send-keys", "-t", TRAIN_NAME, full, "Enter"], timeout=10)
    log("Training restarted")


def stop():
    try:
        subprocess.run(["tmux", "send-keys", "-t", TRAIN_NAME, "C-c"], timeout=10)
        time.sleep(5)
        r = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True, timeout=5)
        if TRAIN_NAME in r.stdout:
            subprocess.run(["tmux", "kill-session", "-t", TRAIN_NAME], timeout=10)
            time.sleep(2)
    except Exception as e:
        log(f"Stop error: {e}")


# 主循环
log("=" * 50)
log("Keepalive monitor started")
log("=" * 50)

prev_step = -1
prev_step_time = time.time()

while True:
    try:
        alive = is_alive()
        if not alive:
            log("CRITICAL: Training session GONE!")
            ckpt = find_latest_ckpt()
            clean_cache()
            restart(ckpt)
            prev_step = -1
            prev_step_time = time.time()
            time.sleep(60)
            continue

        step = get_last_step()
        if step >= 0:
            if step != prev_step:
                prev_step = step
                prev_step_time = time.time()
            else:
                stuck = time.time() - prev_step_time
                if stuck > MAX_NO_OUTPUT_SEC:
                    log(f"CRITICAL: Stuck at step {step} for {int(stuck)}s!")
                    stop()
                    clean_cache()
                    ckpt = find_latest_ckpt()
                    restart(ckpt)
                    prev_step = -1
                    prev_step_time = time.time()
                    time.sleep(60)
                    continue

    except Exception as e:
        log(f"Monitor error: {e}")

    time.sleep(CHECK_INTERVAL)
