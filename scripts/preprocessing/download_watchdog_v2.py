#!/usr/bin/env python3
"""
v2 下载看门狗 - 基于 PID 文件监控，避免 pgrep 命令行截断问题
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
WORKSPACE = "/workspace/raw/national_china"
SCRIPT = "/workspace/xuannv/scripts/preprocessing/download_from_planetary_computer_v2.py"
PYTHON = "/root/miniconda3/envs/xuannv/bin/python"

CHECK_INTERVAL = 60  # 秒
STALL_THRESHOLD = 15 * 60  # 15 分钟
MAX_RESTARTS = 10
COOLDOWN = 3 * 60  # 3 分钟冷却

SOURCES = {
    "s2": {
        "log": f"{WORKSPACE}/download_s2_v2.log",
        "args": [
            "--source", "s2", "--patches",
            f"{WORKSPACE}/patches_meta.json",
            "--output", WORKSPACE,
            "--workers", "16",
            "--date-start", "2025-01-01",
            "--date-end", "2026-06-01",
        ],
    },
    "s1": {
        "log": f"{WORKSPACE}/download_s1_v2.log",
        "args": [
            "--source", "s1", "--patches",
            f"{WORKSPACE}/patches_meta.json",
            "--output", WORKSPACE,
            "--workers", "16",
            "--date-start", "2025-01-01",
            "--date-end", "2026-06-01",
        ],
    },
    "landsat": {
        "log": f"{WORKSPACE}/download_landsat_v2.log",
        "args": [
            "--source", "landsat", "--patches",
            f"{WORKSPACE}/patches_meta.json",
            "--output", WORKSPACE,
            "--workers", "16",
            "--date-start", "2025-01-01",
            "--date-end", "2026-06-01",
        ],
    },
    "dem": {
        "log": f"{WORKSPACE}/download_dem_v2.log",
        "args": [
            "--source", "dem", "--patches",
            f"{WORKSPACE}/patches_meta.json",
            "--output", WORKSPACE,
            "--workers", "16",
        ],
    },
    "worldcover": {
        "log": f"{WORKSPACE}/download_worldcover_v2.log",
        "args": [
            "--source", "worldcover", "--patches",
            f"{WORKSPACE}/patches_meta.json",
            "--output", WORKSPACE,
            "--workers", "16",
        ],
    },
}

WATCHDOG_PIDFILE = f"{WORKSPACE}/watchdog_v2.pid"


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{WORKSPACE}/watchdog_v2.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pidfile_path(source: str) -> str:
    return f"{WORKSPACE}/download_{source}_v2.pid"


def read_pid(source: str) -> int | None:
    try:
        with open(pidfile_path(source), "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_pid(source: str, pid: int) -> None:
    with open(pidfile_path(source), "w", encoding="utf-8") as f:
        f.write(str(pid))


def is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def log_mtime(path: str) -> datetime | None:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


def latest_progress(path: str) -> tuple[int | None, int | None]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-200:]
        for line in reversed(lines):
            m = re.search(r"\[(\d+)/(\d+)\]", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None, None


def kill_residual(source: str) -> None:
    """通过 pgrep 查找并杀死所有残留进程。"""
    pattern = f"download_from_planetary_computer_v2.py --source {source}"
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return
        pids = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(2)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1)
    except Exception as e:
        _log(f"[WARN] 终止 {source} 残留进程失败: {e}")


def restart(source: str, cfg: dict) -> bool:
    _log(f"[RESTART] 正在重启 {source} ...")
    kill_residual(source)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_path = cfg["log"]
    log_file = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [PYTHON, SCRIPT] + cfg["args"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd="/workspace/xuannv",
            env=env,
        )
        write_pid(source, proc.pid)
        _log(f"[RESTART] {source} 已启动，新 PID={proc.pid}")
        return True
    except Exception as e:
        _log(f"[ERROR] 重启 {source} 失败: {e}")
        return False


def write_watchdog_pid(pid: int) -> None:
    with open(WATCHDOG_PIDFILE, "w", encoding="utf-8") as f:
        f.write(str(pid))


def remove_watchdog_pid() -> None:
    try:
        os.remove(WATCHDOG_PIDFILE)
    except Exception:
        pass


def main() -> None:
    write_watchdog_pid(os.getpid())
    _log("=" * 60)
    _log("v2 下载看门狗启动 (PID 文件版)")
    _log(f"检查间隔: {CHECK_INTERVAL}s | 停滞阈值: {STALL_THRESHOLD // 60}分钟 | 最大重启: {MAX_RESTARTS}")
    _log("=" * 60)

    restart_counts = {s: 0 for s in SOURCES}
    cooldown_until = {s: datetime.min for s in SOURCES}
    last_progress = {s: (None, None) for s in SOURCES}
    cycle = 0

    try:
        while True:
            cycle += 1
            time.sleep(CHECK_INTERVAL)
            now = datetime.now()
            report_lines: list[str] = []
            any_restarted = False

            for source, cfg in SOURCES.items():
                pid = read_pid(source)
                alive = is_alive(pid)
                mtime = log_mtime(cfg["log"])
                prog_cur, prog_total = latest_progress(cfg["log"])

                if not alive:
                    status = "🔴 丢失"
                    reason = f"PID={pid} 不存在"
                elif mtime is None:
                    status = "🔴 无日志"
                    reason = "日志文件不存在"
                elif (now - mtime).total_seconds() > STALL_THRESHOLD:
                    status = "🔴 停滞"
                    reason = f"日志 {(now - mtime).seconds // 60} 分钟未更新"
                else:
                    status = "✅ 正常"
                    reason = ""

                last_progress[source] = (prog_cur, prog_total)
                report_lines.append(
                    f"  {source:8s} {status:8s} | {prog_cur or '?'}/{prog_total or '?'} | {reason}"
                )

                if "🔴" in status:
                    if now < cooldown_until[source]:
                        report_lines[-1] += " (冷却中)"
                        continue
                    if restart_counts[source] >= MAX_RESTARTS:
                        report_lines[-1] += " (已达最大重启次数)"
                        continue

                    if restart(source, cfg):
                        restart_counts[source] += 1
                        cooldown_until[source] = now + timedelta(seconds=COOLDOWN)
                        any_restarted = True

            if cycle % 5 == 0 or any_restarted:
                _log("-" * 50)
                _log(f"周期 #{cycle} 状态报告")
                for line in report_lines:
                    _log(line)
                _log(f"重启历史: {restart_counts}")
                _log("-" * 50)
    finally:
        remove_watchdog_pid()
        _log("[EXIT] 看门狗退出")


if __name__ == "__main__":
    main()
