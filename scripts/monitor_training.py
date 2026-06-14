from __future__ import annotations

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


def get_latest_log(log_dir: str) -> str | None:
    files = glob.glob(os.path.join(log_dir, "train_*.log"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_last_step(log_path: str) -> dict | None:
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # find last [Step ...] line
    for line in reversed(lines):
        line = line.replace("\r", "\n").strip()
        if not line.startswith("[Step"):
            continue
        m = re.search(
            r"\[Step\s+(\d+)/(\d+)\]\s+total=([\-\d\.]+)\s+recon=([\-\d\.]+)\s+cls=([\-\d\.]+)\s+var=([\-\d\.]+)\s+cov=([\-\d\.]+)\s+l2unif=([\-\d\.]+)\s+erank=([\-\d\.]+)",
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


def has_nan_or_inf(log_path: str, n_lines: int = 50) -> bool:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for line in lines[-n_lines:]:
        if "nan" in line.lower() or "inf" in line.lower():
            return True
    return False


def tmux_alive(session: str) -> bool:
    try:
        subprocess.check_output(
            ["tmux", "has-session", "-t", session], stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/monitor_training.py <tmux_session> <log_dir>")
        return 1
    session = sys.argv[1]
    log_dir = sys.argv[2]
    monitor_log = os.path.join("/workspace/xuannv/logs", f"monitor_{session}.log")
    os.makedirs(os.path.dirname(monitor_log), exist_ok=True)

    log(f"开始监控 session={session} log_dir={log_dir}", monitor_log)
    last_step = -1
    last_step_time = time.time()
    erank_low_count = 0

    while True:
        try:
            if not tmux_alive(session):
                log(f"[ALERT] tmux session '{session}' 不存在，训练可能已退出", monitor_log)
                time.sleep(60)
                continue

            log_path = get_latest_log(log_dir)
            if log_path is None:
                log("[WARN] 未找到 train_*.log", monitor_log)
                time.sleep(60)
                continue

            step_info = parse_last_step(log_path)
            if step_info is None:
                log(f"[WARN] 无法从 {log_path} 解析 step", monitor_log)
                time.sleep(60)
                continue

            cur_step = step_info["step"]
            if cur_step != last_step:
                last_step = cur_step
                last_step_time = time.time()

            elapsed_no_step = time.time() - last_step_time
            epoch = cur_step // step_info["total_steps"]
            status = (
                f"Epoch {epoch} Step {cur_step}/{step_info['total_steps']} | "
                f"total={step_info['total']:.3f} recon={step_info['recon']:.3f} "
                f"var={step_info['var']:.3f} cov={step_info['cov']:.3f} "
                f"l2unif={step_info['l2unif']:.3f} erank={step_info['erank']:.1f} "
                f"idle={elapsed_no_step/60:.1f}min"
            )
            log(status, monitor_log)

            if has_nan_or_inf(log_path):
                log("[ALERT] 检测到 NaN/Inf，请检查训练！", monitor_log)

            if step_info["erank"] < 5.0:
                erank_low_count += 1
                if erank_low_count >= 3:
                    log(
                        f"[ALERT] erank 连续 {erank_low_count} 次低于 5.0（当前 {step_info['erank']:.1f}），embedding 可能仍在坍缩",
                        monitor_log,
                    )
            else:
                erank_low_count = 0

            if elapsed_no_step > 600:
                log(
                    f"[ALERT] 超过 {elapsed_no_step/60:.1f} 分钟没有新 step，训练可能卡住",
                    monitor_log,
                )

        except Exception as e:
            log(f"[ERROR] 监控异常: {e}", monitor_log)

        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
