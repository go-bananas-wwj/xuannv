#!/usr/bin/env python3
"""
Planetary Computer 下载看门狗 V2
- 检测 tmux 下载 session 是否健康
- session 不存在时自动创建并启动
- 文件数量停滞检测（30分钟0增长则重启）
- 发现 SAS token 过期（大量 RasterioIOError）时自动重启
- 每 15 分钟运行一次

用法:
    python scripts/preprocessing/pc_download_watchdog.py
    # 或后台循环运行:
    nohup bash -c 'while true; do python scripts/preprocessing/pc_download_watchdog.py; sleep 900; done' > /workspace/xuannv/outputs/pc_watchdog.log 2>&1 &
"""

import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

LOG_FILE = Path("/workspace/xuannv/outputs/pc_download_watchdog.log")
STATE_FILE = Path("/workspace/xuannv/outputs/pc_download_watchdog_state.json")
DATA_ROOT = Path("/workspace/xuannv/data_raw/phase2_heilongjiang")

# Session 注册表：session 名 -> 启动命令
# 注意：命令中不需要包含 tmux 相关部分，看门狗会自动包装
SESSION_REGISTRY = {
    # 海淀
    "pc_haidian_s2": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/haidian/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources s2 --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_haidian_s1": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/haidian/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources s1 --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_haidian_landsat": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/haidian/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources landsat --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_haidian_dem": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/haidian/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources dem --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_haidian_worldcover": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/haidian/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources worldcover --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    # 大庆
    "pc_daqing_s2": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/daqing/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources s2 --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_daqing_s1": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/daqing/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources s1 --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_daqing_landsat": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/daqing/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources landsat --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_daqing_dem": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/daqing/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources dem --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_daqing_worldcover": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/daqing/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources worldcover --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    # 齐齐哈尔（DEM/WorldCover 已完成，只需 S2 + Landsat）
    "pc_qiqihar_s2": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/qiqihar/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources s2 --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
    "pc_qiqihar_landsat": "python -u scripts/preprocessing/download_from_planetary_computer.py --patches /workspace/xuannv/data_raw/phase2_heilongjiang/qiqihar/patches_meta.json --output /workspace/xuannv/data_raw/phase2_heilongjiang --sources landsat --date-start 2023-01-01 --date-end 2025-10-31 --workers 2",
}

# 文件数量停滞检测配置
STALL_THRESHOLD_MINUTES = 30  # 30 分钟无增长则判定为停滞
STALL_CHECK_SOURCES = {
    "pc_haidian_s2": ("haidian", "s2"),
    "pc_haidian_s1": ("haidian", "s1"),
    "pc_haidian_landsat": ("haidian", "landsat"),
    "pc_haidian_dem": ("haidian", "dem"),
    "pc_haidian_worldcover": ("haidian", "worldcover"),
    "pc_daqing_s2": ("daqing", "s2"),
    "pc_daqing_s1": ("daqing", "s1"),
    "pc_daqing_landsat": ("daqing", "landsat"),
    "pc_daqing_dem": ("daqing", "dem"),
    "pc_daqing_worldcover": ("daqing", "worldcover"),
    "pc_qiqihar_s2": ("qiqihar", "s2"),
    "pc_qiqihar_landsat": ("qiqihar", "landsat"),
}

ERROR_WINDOW_LINES = 50
ERROR_THRESHOLD = 8  # 最近 50 行中出现 >= 8 个 token 过期错误则重启
RESTART_WAIT_S = 5

TOKEN_EXPIRY_PATTERNS = [
    "not recognized as being in a supported file format",
    "RasterioIOError",
    "Read failed",
    "HTTP response code: 403",
]

# ---------------------------------------------------------------------------
# 日志与状态
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# tmux / 进程操作
# ---------------------------------------------------------------------------

def is_session_alive(session: str) -> bool:
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=True, capture_output=True, timeout=5
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_tmux_output(session: str, n_lines: int = ERROR_WINDOW_LINES) -> str:
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", session],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def count_token_errors(output: str) -> int:
    exact_token = output.count("RasterioIOError")
    exact_403 = output.count("HTTP response code: 403")
    exact_unsupported = output.count("not recognized as being in a supported file format")
    return exact_token + exact_403 + exact_unsupported


def count_file_growth(city: str, source: str) -> int:
    """统计某个 city+source 的 .tif 文件数量"""
    dir_path = DATA_ROOT / city / source
    if not dir_path.exists():
        return 0
    try:
        return sum(1 for _ in dir_path.rglob("*.tif"))
    except Exception:
        return 0


def kill_session(session: str):
    try:
        subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], capture_output=True, timeout=5)
        time.sleep(1)
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=5)
        log(f"  → 已停止 session: {session}")
    except Exception as e:
        log(f"  → 停止 session {session} 出错: {e}")


def create_session(session: str, cmd: str):
    """创建新的 tmux session 并启动命令"""
    try:
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=5)
    except Exception:
        pass

    # 包装命令：激活 conda 环境，cd 到项目目录，执行命令，tee 到日志
    tmux_cmd = (
        f"bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate xuannv && cd /workspace/xuannv && {cmd}' "
        f"2>&1 | tee -a /workspace/xuannv/outputs/{session}.log"
    )

    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, tmux_cmd],
            capture_output=True, timeout=10
        )
        log(f"  → 已创建并启动 session: {session}")
        return True
    except Exception as e:
        log(f"  → 创建 session {session} 出错: {e}")
        return False


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main():
    state = load_state()
    now_ts = time.time()

    log("=" * 60)
    log("PC 下载看门狗 V2 检查开始")

    restarted = []
    created = []

    for session, cmd in SESSION_REGISTRY.items():
        alive = is_session_alive(session)

        if not alive:
            log(f"[{session}] ⚠️ session 不存在，自动创建...")
            if create_session(session, cmd):
                created.append(session)
                state[session] = {"created_at": now_ts, "last_file_count": 0, "last_check_ts": now_ts}
            continue

        # 获取 tmux 输出并统计 token 错误
        output = get_tmux_output(session)
        token_err_count = count_token_errors(output)

        # 文件数量停滞检测
        stalled = False
        if session in STALL_CHECK_SOURCES:
            city, source = STALL_CHECK_SOURCES[session]
            current_count = count_file_growth(city, source)
            prev_state = state.get(session, {})
            prev_count = prev_state.get("last_file_count", 0)
            prev_ts = prev_state.get("last_check_ts", 0)

            # 更新状态
            state[session] = {
                "last_file_count": current_count,
                "last_check_ts": now_ts,
            }

            # 如果之前已有记录，检查增长
            if prev_ts > 0:
                elapsed_min = (now_ts - prev_ts) / 60
                if elapsed_min >= STALL_THRESHOLD_MINUTES:
                    growth = current_count - prev_count
                    if growth == 0:
                        stalled = True
                        log(f"[{session}] 🔥 文件数量停滞: {prev_count} -> {current_count} ({elapsed_min:.0f}分钟无增长)")

            log(f"[{session}] ✅ token错误={token_err_count}, 文件数={current_count}(+{current_count - prev_count})")
        else:
            log(f"[{session}] ✅ token错误={token_err_count}")

        # 判定是否需要重启
        need_restart = token_err_count >= ERROR_THRESHOLD or stalled

        if need_restart:
            reason = "token过期" if token_err_count >= ERROR_THRESHOLD else "文件停滞"
            log(f"  → 判定为 {reason}，准备重启...")
            kill_session(session)
            time.sleep(RESTART_WAIT_S)
            if create_session(session, cmd):
                restarted.append(session)
                state[session] = {"restarted_at": now_ts, "last_file_count": 0, "last_check_ts": now_ts}

    save_state(state)

    if created:
        log(f"看门狗完成: 已创建 {len(created)} 个 session: {', '.join(created)}")
    if restarted:
        log(f"看门狗完成: 已重启 {len(restarted)} 个 session: {', '.join(restarted)}")
    if not created and not restarted:
        log("看门狗完成: 所有 session 健康")


if __name__ == "__main__":
    main()
