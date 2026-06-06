#!/usr/bin/env python3
"""
BaiduPCS-Go 并行下载看门狗 (v2 - 4进程并行版)
监控4个子目录下载进程，任意一个卡死/退出则自动重启
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import logging
from pathlib import Path

# ─────────────────────── 配置区 ───────────────────────
BASE_PATH   = "/海淀区sar/玄女底座"
LOCAL_DEST  = "/workspace/xuannv/data_raw/haidian_sar"
LOG_FILE    = f"{LOCAL_DEST}/watchdog.log"
THREADS     = 10
STALL_TIMEOUT = 600   # 10分钟无进度判定卡死
CHECK_INTERVAL = 60   # 每60秒检查一次

SUBDIRS = [
    "北京朝阳角反",
    "北京市门头沟区大台-干涉",
    "中国北京市点位1_干涉",
    "中国北京市点位2_干涉",
]
# ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog")


def get_dir_stats(path: str) -> tuple[int, int]:
    total_files = total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            if not f.endswith(".zip"):
                continue
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
                total_files += 1
            except OSError:
                pass
    return total_files, total_bytes


def kill_proc(pid: int):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
            time.sleep(1)
        except ProcessLookupError:
            break


def start_one(subdir: str) -> subprocess.Popen:
    cmd = [
        "BaiduPCS-Go", "download",
        f"{BASE_PATH}/{subdir}/",
        "--saveto", LOCAL_DEST,
        "-p", str(THREADS),
    ]
    log.info(f"启动: {subdir}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def main():
    os.makedirs(LOCAL_DEST, exist_ok=True)
    log.info("=" * 60)
    log.info("Watchdog v2 启动（4进程并行模式）")
    log.info(f"  目标目录: {LOCAL_DEST}")
    log.info(f"  卡死超时: {STALL_TIMEOUT}s  检查间隔: {CHECK_INTERVAL}s")
    log.info("=" * 60)

    procs = {d: start_one(d) for d in SUBDIRS}
    prev_stats = get_dir_stats(LOCAL_DEST)
    last_progress_time = time.time()

    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            cur_stats = get_dir_stats(LOCAL_DEST)

            # 进度汇报
            if cur_stats != prev_stats:
                delta_f = cur_stats[0] - prev_stats[0]
                delta_mb = (cur_stats[1] - prev_stats[1]) / 1024**2
                log.info(
                    f"📥 进度: {cur_stats[0]} ZIP / "
                    f"{cur_stats[1]/1024**3:.2f} GB  "
                    f"(+{delta_f} 文件, +{delta_mb:.0f} MB)"
                )
                prev_stats = cur_stats
                last_progress_time = time.time()

            # 卡死检测
            stall = time.time() - last_progress_time
            if stall > STALL_TIMEOUT and cur_stats[0] > 0:
                log.warning(f"⚠️  {stall:.0f}s 无进度，重启所有进程...")
                for d, p in procs.items():
                    kill_proc(p.pid)
                time.sleep(3)
                procs = {d: start_one(d) for d in SUBDIRS}
                last_progress_time = time.time()
                continue

            # 检查各进程存活，自动重启退出的
            for d in list(procs.keys()):
                ret = procs[d].poll()
                if ret is not None:
                    if ret == 0:
                        log.info(f"✅ {d} 下载完成")
                        del procs[d]
                    else:
                        log.warning(f"⚠️  {d} 意外退出(ret={ret})，重启中...")
                        procs[d] = start_one(d)
                        last_progress_time = time.time()

            # 全部完成
            if not procs:
                log.info("🎉 所有子目录下载完成！")
                final = get_dir_stats(LOCAL_DEST)
                log.info(f"   共 {final[0]} 个ZIP，{final[1]/1024**3:.2f} GB")
                break

            # 心跳
            alive = [d for d, p in procs.items() if p.poll() is None]
            log.info(f"💓 存活进程: {len(alive)}/4 | ZIP: {cur_stats[0]} | "
                     f"{cur_stats[1]/1024**3:.2f}GB | 无进度: {stall:.0f}s")

    except KeyboardInterrupt:
        log.info("收到停止信号，清理进程...")
    finally:
        for d, p in procs.items():
            kill_proc(p.pid)
        log.info("Watchdog 已退出")


if __name__ == "__main__":
    main()
