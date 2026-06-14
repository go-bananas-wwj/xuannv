#!/usr/bin/env python3
"""训练健康后台监控（core 任务）.

持续检查：
- tmux session / 训练进程是否存活
- 最新 train_*.log 中的 step、loss、erank 等指标
- NaN / Inf
- erank 连续过低（坍缩预警）
- 长时间没有新 step（卡住/ hung）

用法:
    python scripts/monitor_training_core.py <tmux_session> <output_dir> [--interval 300]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime


def log(msg: str, log_path: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest_train_log(output_dir: str) -> str | None:
    files = glob.glob(os.path.join(output_dir, "train_*.log"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_last_step(log_path: str) -> dict | None:
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # 从后往前找第一个 [Step ...] 行
    for line in reversed(lines):
        line = line.replace("\r", "\n").strip()
        if not line.startswith("[Step"):
            continue
        # 支持 v2b 格式:
        # [Step 123/200] total=-4.123 recon=... cls=... var=... cov=... l2unif=... erank=... aef=[...] olmo=[...] lr=...
        m = re.search(
            r"\[Step\s+(\d+)/(\d+)\]\s+"
            r"total=([\-\d\.]+)\s+"
            r"recon=([\-\d\.]+)\s+"
            r"cls=([\-\d\.]+)\s+"
            r"var=([\-\d\.]+)\s+"
            r"cov=([\-\d\.]+)\s+"
            r"l2unif=([\-\d\.]+)\s+"
            r"erank=([\-\d\.]+)",
            line,
        )
        if not m:
            continue
        return {
            "step": int(m.group(1)),
            "total_steps": int(m.group(2)),
            "total": float(m.group(3)),
            "recon": float(m.group(4)),
            "cls": float(m.group(5)),
            "var": float(m.group(6)),
            "cov": float(m.group(7)),
            "l2unif": float(m.group(8)),
            "erank": float(m.group(9)),
            "raw": line,
        }
    return None


def has_nan_or_inf(log_path: str, n_lines: int = 100) -> bool:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    text = "".join(lines[-n_lines:]).lower()
    return "nan" in text or "inf" in text


def tmux_alive(session: str) -> bool:
    try:
        subprocess.check_output(
            ["tmux", "has-session", "-t", session], stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="后台训练健康监控")
    parser.add_argument("session", help="tmux session 名称")
    parser.add_argument("output_dir", help="训练输出目录（含 train_*.log）")
    parser.add_argument("--interval", type=int, default=300, help="检查间隔（秒）")
    parser.add_argument("--erank-threshold", type=float, default=5.0, help="erank 告警阈值")
    parser.add_argument(
        "--erank-consecutive", type=int, default=3, help="erank 连续低于阈值次数才告警"
    )
    parser.add_argument(
        "--stuck-threshold", type=int, default=600, help="无新 step 超时（秒）"
    )
    args = parser.parse_args()

    monitor_log = os.path.join(
        "/workspace/xuannv/logs", f"monitor_{args.session}_core.log"
    )
    os.makedirs(os.path.dirname(monitor_log), exist_ok=True)

    log(
        f"开始监控 session={args.session} output_dir={args.output_dir} interval={args.interval}s",
        monitor_log,
    )

    last_step = -1
    last_step_time = time.time()
    erank_low_count = 0

    while True:
        try:
            if not tmux_alive(args.session):
                log(
                    f"[ALERT] tmux session '{args.session}' 不存在，训练可能已退出",
                    monitor_log,
                )
                time.sleep(args.interval)
                continue

            log_path = latest_train_log(args.output_dir)
            if log_path is None:
                log(f"[WARN] {args.output_dir} 下未找到 train_*.log", monitor_log)
                time.sleep(args.interval)
                continue

            info = parse_last_step(log_path)
            if info is None:
                log(f"[WARN] 无法从 {log_path} 解析 step 指标", monitor_log)
                time.sleep(args.interval)
                continue

            cur_step = info["step"]
            epoch = cur_step // info["total_steps"]
            if cur_step != last_step:
                last_step = cur_step
                last_step_time = time.time()
                erank_low_count = 0 if info["erank"] >= args.erank_threshold else 1
            else:
                if info["erank"] < args.erank_threshold:
                    erank_low_count += 1

            idle_sec = time.time() - last_step_time
            status = (
                f"Epoch {epoch} Step {cur_step}/{info['total_steps']} | "
                f"total={info['total']:.3f} recon={info['recon']:.3f} "
                f"var={info['var']:.3f} cov={info['cov']:.3f} "
                f"l2unif={info['l2unif']:.3f} erank={info['erank']:.1f} "
                f"idle={idle_sec/60:.1f}min"
            )
            log(status, monitor_log)

            if has_nan_or_inf(log_path):
                log("[ALERT] 最近日志中出现 NaN / Inf！", monitor_log)

            if erank_low_count >= args.erank_consecutive:
                log(
                    f"[ALERT] erank 连续 {erank_low_count} 次 < {args.erank_threshold} "
                    f"（当前 {info['erank']:.1f}），embedding 可能坍缩",
                    monitor_log,
                )

            if idle_sec > args.stuck_threshold:
                log(
                    f"[ALERT] 超过 {args.stuck_threshold}s 没有新 step，训练可能卡住 "
                    f"（已 idle {idle_sec:.0f}s）",
                    monitor_log,
                )

        except Exception as e:
            log(f"[ERROR] 监控异常: {e}", monitor_log)

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
