#!/usr/bin/env python3
"""V12 训练监控脚本 — 定时检查训练状态、汇报指标、自动恢复.

用法:
    python scripts/monitor_v12_training.py --interval 300

功能:
1. 定期检查训练进程是否存活
2. 自动汇报最新指标到 stdout
3. 检测到报错时，自动找到最新 checkpoint 并恢复训练
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

# 配置
OUTPUT_DIR = Path("/workspace/outputs/xuannv_v12_clean")
LOG_FILE = OUTPUT_DIR / "train.log"
TMUX_SESSION = "v12_train"
CONFIG = "configs/xuannv_v12_clean.yaml"
TRAIN_SCRIPT = "scripts/train/train_ddp_v12.py"


def run_cmd(cmd: str) -> str:
    """运行 shell 命令并返回输出."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def check_training_alive() -> bool:
    """检查训练进程是否存活."""
    output = run_cmd(f"tmux has-session -t {TMUX_SESSION} 2>/dev/null && echo 'ALIVE' || echo 'DEAD'")
    if output != "ALIVE":
        return False
    # 检查 torchrun 进程
    output = run_cmd("pgrep -f 'train_ddp_v12.py' | wc -l")
    try:
        n_procs = int(output)
        return n_procs >= 8  # 应该有 8 个子进程 + 1 个 torchrun
    except ValueError:
        return False


def get_latest_metrics() -> dict:
    """从 tmux 输出和日志中提取最新指标."""
    metrics = {}
    
    # 从 tmux 获取最新输出
    tmux_output = run_cmd(f"tmux capture-pane -t {TMUX_SESSION} -p 2>/dev/null | grep -E 'Epoch|Step' | tail -5")
    if tmux_output:
        # 解析 Epoch 行
        epoch_match = re.search(r'Epoch\s+(\d+).*total=([\d.]+)\s+recon=([\d.]+)\s+consist=([\d.]+)\s+uniform=([\d.]+)\s+lr=([\d.e+-]+)', tmux_output)
        if epoch_match:
            metrics["epoch"] = int(epoch_match.group(1))
            metrics["total"] = float(epoch_match.group(2))
            metrics["recon"] = float(epoch_match.group(3))
            metrics["consist"] = float(epoch_match.group(4))
            metrics["uniform"] = float(epoch_match.group(5))
            metrics["lr"] = float(epoch_match.group(6))
        
        # 解析 Step 行
        step_match = re.search(r'Step\s+(\d+).*recon=([\d.]+)\s+consist=([\d.]+)\s+uniform=([\d.]+)', tmux_output)
        if step_match:
            metrics["step"] = int(step_match.group(1))
            if "recon" not in metrics:
                metrics["recon"] = float(step_match.group(2))
                metrics["consist"] = float(step_match.group(3))
                metrics["uniform"] = float(step_match.group(4))
    
    # 从日志文件获取
    if LOG_FILE.exists():
        log_content = run_cmd(f"tail -5 {LOG_FILE}")
        epoch_match = re.search(r'Epoch\s+(\d+).*total=([\d.]+)\s+recon=([\d.]+)\s+consist=([\d.]+)\s+uniform=([\d.]+)', log_content)
        if epoch_match:
            metrics["epoch"] = int(epoch_match.group(1))
            metrics["total"] = float(epoch_match.group(2))
            metrics["recon"] = float(epoch_match.group(3))
            metrics["consist"] = float(epoch_match.group(4))
            metrics["uniform"] = float(epoch_match.group(5))
    
    return metrics


def check_for_errors() -> tuple[bool, str]:
    """检查训练是否报错."""
    # 从 tmux 输出检查
    tmux_output = run_cmd(f"tmux capture-pane -t {TMUX_SESSION} -p 2>/dev/null | tail -100")
    
    error_patterns = [
        r"RuntimeError",
        r"IndexError",
        r"KeyError",
        r"AttributeError",
        r"ValueError",
        r"torch.distributed.elastic.multiprocessing.errors.ChildFailedError",
        r"exitcode\s*:\s*1",
        r"FAILED",
        r"NPU out of memory",
        r"OOM",
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, tmux_output, re.IGNORECASE):
            # 获取错误上下文
            error_lines = []
            for line in tmux_output.split("\n"):
                if re.search(pattern, line, re.IGNORECASE):
                    error_lines.append(line.strip())
            return True, " | ".join(error_lines[-3:])
    
    return False, ""


def find_latest_checkpoint() -> Path | None:
    """找到最新的 checkpoint."""
    ckpts = list(OUTPUT_DIR.glob("epoch_best_*.pt"))
    if not ckpts:
        ckpts = list(OUTPUT_DIR.glob("epoch_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def resume_training(checkpoint: Path | None = None) -> None:
    """恢复训练."""
    # 先终止旧的 session
    run_cmd(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null")
    time.sleep(2)
    
    cmd = f"conda activate xuannv && torchrun --nproc_per_node=8 {TRAIN_SCRIPT} --config {CONFIG} --save-every 20"
    if checkpoint:
        cmd += f" --resume {checkpoint}"
    
    run_cmd(f"tmux new-session -d -s {TMUX_SESSION} -c /workspace/xuannv")
    run_cmd(f"tmux send-keys -t {TMUX_SESSION} '{cmd}' Enter")
    print(f"[monitor] Training resumed{' from ' + str(checkpoint) if checkpoint else ''}")


def print_status(metrics: dict, alive: bool, error: bool, error_msg: str) -> None:
    """打印当前状态."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if error:
        print(f"\n{'='*60}")
        print(f"[{now}] 🚨 TRAINING ERROR DETECTED")
        print(f"{'='*60}")
        print(f"Error: {error_msg}")
    elif not alive:
        print(f"\n{'='*60}")
        print(f"[{now}] ⚠️ TRAINING NOT ALIVE")
        print(f"{'='*60}")
    else:
        epoch = metrics.get("epoch", "?")
        step = metrics.get("step", "?")
        recon = metrics.get("recon", "?")
        consist = metrics.get("consist", "?")
        uniform = metrics.get("uniform", "?")
        lr = metrics.get("lr", "?")
        
        # Uniform 状态判断
        if isinstance(uniform, float):
            if uniform > 0.95:
                uniform_status = "🔴 严重坍缩"
            elif uniform > 0.8:
                uniform_status = "🟠 中度坍缩"
            elif uniform > 0.5:
                uniform_status = "🟡 轻度坍缩"
            else:
                uniform_status = "🟢 良好"
        else:
            uniform_status = "?"
        
        print(f"[{now}] Epoch={epoch} Step={step} | recon={recon:.4f} consist={consist:.4f} uniform={uniform:.4f}({uniform_status}) lr={lr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300, help="检查间隔（秒）")
    parser.add_argument("--auto-resume", action="store_true", default=True, help="检测到错误时自动恢复")
    args = parser.parse_args()
    
    print(f"[monitor] V12 训练监控启动，检查间隔: {args.interval}秒")
    print(f"[monitor] 输出目录: {OUTPUT_DIR}")
    print(f"[monitor] 自动恢复: {args.auto_resume}")
    
    last_epoch = 0
    
    while True:
        time.sleep(args.interval)
        
        alive = check_training_alive()
        metrics = get_latest_metrics()
        error, error_msg = check_for_errors()
        
        # 打印状态
        print_status(metrics, alive, error, error_msg)
        
        # 检测是否需要恢复
        if (not alive or error) and args.auto_resume:
            print(f"[monitor] 尝试恢复训练...")
            ckpt = find_latest_checkpoint()
            if ckpt:
                print(f"[monitor] 找到最新 checkpoint: {ckpt}")
            else:
                print(f"[monitor] 未找到 checkpoint，从头开始")
            resume_training(ckpt)
            time.sleep(30)  # 等待训练启动


if __name__ == "__main__":
    main()
