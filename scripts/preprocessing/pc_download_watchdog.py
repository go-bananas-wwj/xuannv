#!/usr/bin/env python3
"""
Planetary Computer 下载看门狗
- 检测 tmux 下载 session 是否健康
- 发现 SAS token 过期（大量 RasterioIOError）时自动重启
- 监控全0数据检测是否正常工作
- 每 30 分钟运行一次（建议用 cron 或 while sleep 循环）

用法:
    python scripts/preprocessing/pc_download_watchdog.py
    # 或后台循环运行:
    nohup bash -c 'while true; do python scripts/preprocessing/pc_download_watchdog.py; sleep 1800; done' > /workspace/outputs/pc_watchdog.log 2>&1 &
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 配置
TMUX_SESSIONS = ["pc_qiqihar_all", "pc_daqing", "pc_haidian"]
ERROR_WINDOW_LINES = 50      # 检查最近 N 行输出
ERROR_THRESHOLD = 5          # 最近 N 行中出现 >= 此数量的 token 过期错误则判定过期
RESTART_WAIT_S = 10          # 停止后等待秒数
LOG_FILE = Path("/workspace/outputs/pc_download_watchdog.log")

TOKEN_EXPIRY_PATTERNS = [
    "not recognized as being in a supported file format",
    "RasterioIOError",
    "Read failed",
    "HTTP response code: 403",
]

ZERO_DATA_PATTERNS = [
    "删除全0数据文件",
    "删除损坏/空文件",
    "All-zero data",
]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_tmux_output(session: str, n_lines: int = ERROR_WINDOW_LINES) -> str:
    """获取 tmux session 的最后 N 行输出"""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", session],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        return "\n".join(lines[-n_lines:])
    except Exception as e:
        return ""


def count_errors(output: str) -> tuple[int, int]:
    """返回 (token_expiry_errors, zero_data_fixes)"""
    token_errors = sum(1 for p in TOKEN_EXPIRY_PATTERNS if p in output)
    # 更精确：统计 RasterioIOError 出现次数
    exact_token = output.count("RasterioIOError")
    exact_403 = output.count("HTTP response code: 403")
    exact_unsupported = output.count("not recognized as being in a supported file format")

    zero_fixes = sum(1 for p in ZERO_DATA_PATTERNS if p in output)
    exact_zero = output.count("删除全0数据文件") + output.count("删除损坏/空文件")

    return exact_token + exact_403 + exact_unsupported, exact_zero


def is_session_alive(session: str) -> bool:
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=True, capture_output=True, timeout=5
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_session_command(session: str) -> str:
    """从 tmux session 中提取运行的命令"""
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
            capture_output=True, text=True, timeout=5
        )
        pid = result.stdout.strip().split("\n")[0]
        # 获取该 pid 的命令行
        result2 = subprocess.run(
            ["ps", "-o", "args=", "-p", pid],
            capture_output=True, text=True, timeout=5
        )
        cmd = result2.stdout.strip()
        # 提取 python 命令部分
        if "python" in cmd:
            # 去掉 bash -c '...' 包装
            if "bash -c" in cmd:
                # 尝试提取内部命令
                idx = cmd.find("python -u")
                if idx >= 0:
                    return cmd[idx:].rstrip("'")
            return cmd
    except Exception:
        pass
    return ""


def kill_session(session: str):
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, "C-c"],
            capture_output=True, timeout=5
        )
        time.sleep(2)
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True, timeout=5
        )
        log(f"  → 已停止 session: {session}")
    except Exception as e:
        log(f"  → 停止 session {session} 出错: {e}")


def restart_session(session: str, cmd: str):
    try:
        # 确保 session 已不存在
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True, timeout=5
        )
    except Exception:
        pass

    # 构建新的 tmux session，使用与之前相同的命令
    # 命令格式应该是: python -u scripts/preprocessing/download_from_planetary_computer.py ...
    tmux_cmd = f"bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv && cd /workspace/xuannv && {cmd}' 2>&1 | tee -a /workspace/outputs/{session}.log"

    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, tmux_cmd],
            capture_output=True, timeout=10
        )
        log(f"  → 已重启 session: {session}")
    except Exception as e:
        log(f"  → 重启 session {session} 出错: {e}")


def main():
    log("=" * 60)
    log("PC 下载看门狗检查开始")

    restarted = []
    for session in TMUX_SESSIONS:
        if not is_session_alive(session):
            log(f"[{session}] ⚠️ session 不存在")
            continue

        output = get_tmux_output(session)
        token_err_count, zero_fix_count = count_errors(output)

        status_emoji = "✅" if token_err_count < ERROR_THRESHOLD else "🔥"
        log(f"[{session}] {status_emoji} 最近 {ERROR_WINDOW_LINES} 行中: "
            f"token过期错误={token_err_count}, 全0修复={zero_fix_count}")

        if token_err_count >= ERROR_THRESHOLD:
            log(f"  → 判定为 token 过期，准备重启...")
            cmd = get_session_command(session)
            if not cmd:
                log(f"  → 无法提取命令，跳过重启")
                continue
            kill_session(session)
            time.sleep(RESTART_WAIT_S)
            restart_session(session, cmd)
            restarted.append(session)

    if restarted:
        log(f"看门狗完成: 已重启 {len(restarted)} 个 session: {', '.join(restarted)}")
    else:
        log("看门狗完成: 所有 session 健康")


if __name__ == "__main__":
    main()
