#!/usr/bin/env python3
"""
下载看门狗 — 自动监控和恢复 Planetary Computer 下载
========================================================

功能:
1. 监控所有 national_* tmux session 的下载进度
2. 检测下载是否卡死/暂停（通过检查日志更新时间和文件生成速度）
3. 自动重启卡死的下载进程（利用断点续传）
4. 每 10 分钟输出一次进度报告
5. 支持优雅退出 (Ctrl+C / SIGTERM)

用法:
    # 前台运行（推荐用于观察）
    python scripts/preprocessing/download_watchdog.py

    # 后台运行
nohup python scripts/preprocessing/download_watchdog.py > /workspace/xuannv/data_raw/national_china/watchdog.log 2>&1 &

监控指标:
    - 日志更新时间: 超过 15 分钟无更新 → 触发重启
    - 文件生成速度: 10 分钟内无新 .tif 生成 → 触发重启
    - tmux session 存在性: session 丢失 → 自动重建并重启
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
WATCH_INTERVAL = 60           # 检查间隔（秒）
STALL_THRESHOLD_MIN = 15      # 超过 N 分钟无进度视为卡死
RESTART_COOLDOWN_MIN = 3      # 重启后冷却时间（分钟），避免频繁重启
REPORT_INTERVAL_MIN = 10      # 进度报告间隔（分钟）

NATIONAL_ROOT = Path("/workspace/xuannv/data_raw/national_china")
SESSIONS = {
    "national_download": {
        "label": "S2",
        "log": NATIONAL_ROOT / "download_s2.log",
        "data_dir": NATIONAL_ROOT / "national_china" / "s2",
        "script": NATIONAL_ROOT / "download_s2.sh",
    },
    "national_s1": {
        "label": "S1",
        "log": NATIONAL_ROOT / "download_s1.log",
        "data_dir": NATIONAL_ROOT / "national_china" / "s1",
        "script": NATIONAL_ROOT / "download_s1.sh",
    },
    "national_landsat": {
        "label": "Landsat",
        "log": NATIONAL_ROOT / "download_landsat.log",
        "data_dir": NATIONAL_ROOT / "national_china" / "landsat",
        "script": NATIONAL_ROOT / "download_landsat.sh",
    },
    "national_static": {
        "label": "DEM+WorldCover",
        "log": NATIONAL_ROOT / "download_static.log",
        "data_dir": NATIONAL_ROOT / "national_china" / "dem",
        "script": NATIONAL_ROOT / "download_static.sh",
    },
}

STATE_FILE = NATIONAL_ROOT / ".watchdog_state.json"

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_running = True

def _handle_sigterm(signum, frame):
    global _running
    print(f"\n[{_now()}] 收到信号 {signum}，优雅退出...")
    _running = False

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _get_tmux_pane_text(session: str) -> str:
    """读取 tmux session 的 pane 内容"""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception:
        return ""


def _tmux_session_exists(session: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_log_mtime(log_path: Path) -> datetime | None:
    """获取日志文件最后修改时间"""
    if not log_path.exists():
        return None
    try:
        mtime = log_path.stat().st_mtime
        return datetime.fromtimestamp(mtime)
    except Exception:
        return None


def _count_recent_tifs(data_dir: Path, minutes: int = 10) -> int:
    """统计最近 N 分钟内生成/修改的 .tif 文件数"""
    if not data_dir.exists():
        return 0
    cutoff = time.time() - minutes * 60
    count = 0
    for tif in data_dir.rglob("*.tif"):
        try:
            if tif.stat().st_mtime > cutoff:
                count += 1
        except Exception:
            continue
    return count


def _count_total_tifs(data_dir: Path) -> int:
    """统计目录下所有 .tif 文件数"""
    if not data_dir.exists():
        return 0
    return len(list(data_dir.rglob("*.tif")))


def _count_patches(data_dir: Path) -> int:
    """统计有数据的 patch 数量"""
    if not data_dir.exists():
        return 0
    return len([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("patch_")])


def _get_dir_size(data_dir: Path) -> str:
    """获取目录大小"""
    if not data_dir.exists():
        return "0B"
    try:
        result = subprocess.run(
            ["du", "-sh", str(data_dir)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.split()[0]
    except Exception:
        return "?"


def _kill_tmux_session(session: str):
    """强制杀死 tmux session"""
    try:
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=10)
        time.sleep(2)
    except Exception:
        pass


def _restart_session(session: str, script: Path):
    """重建 tmux session 并启动下载脚本"""
    # 先确保 session 不存在
    _kill_tmux_session(session)
    time.sleep(1)

    # 创建新 session
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "-c", str(NATIONAL_ROOT)],
            capture_output=True, timeout=10, check=True
        )
        time.sleep(1)
        # 启动脚本
        subprocess.run(
            ["tmux", "send-keys", "-t", session, f"{script}", "Enter"],
            capture_output=True, timeout=5
        )
        return True
    except Exception as e:
        print(f"[{_now()}] [ERROR] 重启 {session} 失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 核心看门狗逻辑
# ---------------------------------------------------------------------------
class DownloadWatchdog:
    def __init__(self):
        self.state = _read_state()
        self.last_report_time = datetime.min
        self.restart_cooldown = {}  # session -> cooldown expiry
        self.cycle_count = 0

    def check_session(self, session: str, cfg: dict) -> dict:
        """检查单个 session 的健康状态"""
        result = {
            "session": session,
            "label": cfg["label"],
            "exists": False,
            "log_exists": False,
            "log_mins_since_update": None,
            "recent_tifs": 0,
            "total_tifs": 0,
            "patches": 0,
            "size": "0",
            "status": "unknown",
            "action": None,
            "reason": None,
        }

        # 1. session 是否存在
        result["exists"] = _tmux_session_exists(session)

        # 2. 日志状态
        log_path = cfg["log"]
        result["log_exists"] = log_path.exists()
        log_mtime = _get_log_mtime(log_path)
        if log_mtime:
            result["log_mins_since_update"] = (datetime.now() - log_mtime).total_seconds() / 60

        # 3. 文件生成状态
        data_dir = cfg["data_dir"]
        result["recent_tifs"] = _count_recent_tifs(data_dir, minutes=STALL_THRESHOLD_MIN)
        result["total_tifs"] = _count_total_tifs(data_dir)
        result["patches"] = _count_patches(data_dir)
        result["size"] = _get_dir_size(data_dir)

        # 4. 判断状态
        now = datetime.now()

        # 检查是否在冷却期
        if session in self.restart_cooldown and now < self.restart_cooldown[session]:
            result["status"] = "cooldown"
            result["reason"] = f"冷却期中 ({(self.restart_cooldown[session] - now).seconds // 60} min left)"
            return result

        # 判断卡死条件
        is_stalled = False
        stall_reasons = []

        if not result["exists"]:
            is_stalled = True
            stall_reasons.append("tmux session 丢失")

        elif result["log_mins_since_update"] is not None and result["log_mins_since_update"] > STALL_THRESHOLD_MIN:
            # 日志长时间未更新，但还要确认没有新文件生成
            if result["recent_tifs"] == 0:
                is_stalled = True
                stall_reasons.append(f"日志 {result['log_mins_since_update']:.0f} 分钟未更新且无新文件")

        elif result["recent_tifs"] == 0 and result["log_exists"] and result["log_mins_since_update"] is not None:
            # 有日志但长时间无新文件
            if result["log_mins_since_update"] > STALL_THRESHOLD_MIN:
                is_stalled = True
                stall_reasons.append(f"{STALL_THRESHOLD_MIN} 分钟无新 .tif 生成")

        if is_stalled:
            result["status"] = "stalled"
            result["reason"] = "; ".join(stall_reasons)
            result["action"] = "restart"
        else:
            result["status"] = "healthy"
            if result["log_mins_since_update"] is not None:
                result["reason"] = f"日志 {result['log_mins_since_update']:.0f} min 前更新, 最近 {result['recent_tifs']} 个新文件"

        # 确保 reason 不为 None
        if result.get("reason") is None:
            result["reason"] = ""

        return result

    def restart_if_needed(self, session: str, cfg: dict) -> bool:
        """如果需要，重启 session"""
        script = cfg["script"]
        if not script.exists():
            print(f"[{_now()}] [ERROR] {session}: 脚本不存在 {script}")
            return False

        print(f"[{_now()}] [RESTART] 正在重启 {session} ({cfg['label']})...")
        ok = _restart_session(session, script)
        if ok:
            # 设置冷却期
            self.restart_cooldown[session] = datetime.now() + timedelta(minutes=RESTART_COOLDOWN_MIN)
            # 记录重启历史
            restarts = self.state.setdefault("restarts", {})
            restarts[session] = restarts.get(session, 0) + 1
            _write_state(self.state)
            print(f"[{_now()}] [RESTART] {session} 重启成功，冷却 {RESTART_COOLDOWN_MIN} 分钟")
        else:
            print(f"[{_now()}] [RESTART] {session} 重启失败!")
        return ok

    def print_report(self, results: list[dict]):
        """打印进度报告"""
        total_patches = sum(r["patches"] for r in results)
        total_tifs = sum(r["total_tifs"] for r in results)
        total_size = sum([
            int(r["size"].replace("G", "000").replace("M", "").replace("K", "").replace("B", "0").strip())
            for r in results if r["size"][0].isdigit()
        ])

        print(f"\n{'='*70}")
        print(f"[{_now()}] 下载进度报告 (周期 #{self.cycle_count})")
        print(f"{'='*70}")
        print(f"{'Session':<20} {'Status':<10} {'Patches':>8} {'Frames':>8} {'Size':>8} {'Reason':<30}")
        print("-" * 70)
        for r in results:
            status_emoji = {"healthy": "🟢", "stalled": "🔴", "cooldown": "🟡", "unknown": "⚪"}.get(r["status"], "?")
            print(f"{r['session']:<20} {status_emoji} {r['status']:<8} {r['patches']:>8} {r['total_tifs']:>8} {r['size']:>8} {r.get('reason', '')[:28]:<30}")
        print("-" * 70)
        print(f"{'TOTAL':<20} {'':<10} {total_patches:>8} {total_tifs:>8}")

        # 重启历史
        restarts = self.state.get("restarts", {})
        if restarts:
            print(f"\n重启历史: {restarts}")
        print(f"{'='*70}\n")

    def run(self):
        print(f"[{_now()}] 下载看门狗启动")
        print(f"  监控 session: {list(SESSIONS.keys())}")
        print(f"  检查间隔: {WATCH_INTERVAL}s")
        print(f"  卡死阈值: {STALL_THRESHOLD_MIN}min 无更新")
        print(f"  重启冷却: {RESTART_COOLDOWN_MIN}min")
        print(f"  进度报告: 每 {REPORT_INTERVAL_MIN}min")
        print(f"  按 Ctrl+C 退出\n")

        while _running:
            self.cycle_count += 1
            results = []
            need_restart = []

            for session, cfg in SESSIONS.items():
                try:
                    r = self.check_session(session, cfg)
                    results.append(r)
                    if r.get("action") == "restart":
                        need_restart.append((session, cfg))
                except Exception as e:
                    print(f"[{_now()}] [ERROR] 检查 {session} 时异常: {e}")
                    traceback.print_exc()

            # 重启卡死的 session
            for session, cfg in need_restart:
                self.restart_if_needed(session, cfg)

            # 定期报告
            now = datetime.now()
            if (now - self.last_report_time).total_seconds() >= REPORT_INTERVAL_MIN * 60:
                self.print_report(results)
                self.last_report_time = now

            # 等待下一轮
            for _ in range(WATCH_INTERVAL):
                if not _running:
                    break
                time.sleep(1)

        print(f"\n[{_now()}] 看门狗已停止。总计运行 {self.cycle_count} 个检查周期。")


if __name__ == "__main__":
    watchdog = DownloadWatchdog()
    watchdog.run()
