#!/usr/bin/env python3
"""
PC 下载进度监控脚本
每 5 分钟统计一次各城市各源的文件数量，计算速率和 ETA

用法:
    python monitor_pc_download.py
    # 或后台运行:
    nohup python monitor_pc_download.py > /dev/null 2>&1 &
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CITIES = {
    "qiqihar": {
        "name": "齐齐哈尔",
        "sources": ["s2", "s1", "landsat", "dem", "worldcover"],
        "total_patches": 400,
    },
    "daqing": {
        "name": "大庆",
        "sources": ["s2", "s1", "landsat", "dem", "worldcover"],
        "total_patches": 400,
    },
    "haidian": {
        "name": "海淀",
        "sources": ["s2", "s1", "landsat", "dem", "worldcover"],
        "total_patches": 400,
    },
}

DATA_ROOT = Path("/workspace/raw/heilongjiang_new")
LOG_FILE = Path("/workspace/outputs/pc_download_monitor.log")
CHECK_INTERVAL = 300  # 5 分钟

# ---------------------------------------------------------------------------
# 统计函数
# ---------------------------------------------------------------------------

def count_source(city: str, source: str) -> tuple[int, int]:
    """统计某个城市某个源的 patch 数和总帧数"""
    source_dir = DATA_ROOT / city / source
    if not source_dir.exists():
        return 0, 0
    
    patches = [d for d in source_dir.glob("patch_*") if d.is_dir()]
    total_frames = sum(len(list(p.glob("*.tif"))) for p in patches)
    return len(patches), total_frames


def get_snapshot() -> dict:
    """获取当前快照"""
    snapshot = {}
    for city_key, cfg in CITIES.items():
        snapshot[city_key] = {}
        for source in cfg["sources"]:
            n_patches, n_frames = count_source(city_key, source)
            snapshot[city_key][source] = {
                "patches": n_patches,
                "frames": n_frames,
            }
    return snapshot


def format_report(current: dict, previous: dict | None, elapsed_min: float) -> str:
    """格式化报告"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"PC 下载进度报告 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    # 检查 tmux session
    import subprocess
    try:
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=5)
        sessions = [line.split(":")[0] for line in result.stdout.strip().split("\n") if line]
        pc_sessions = [s for s in sessions if s.startswith("pc_")]
        lines.append(f"活跃 tmux sessions: {', '.join(pc_sessions) if pc_sessions else '⚠️ 无'}")
    except Exception:
        lines.append("活跃 tmux sessions: 无法检测")
    lines.append("")
    
    for city_key, cfg in CITIES.items():
        city_name = cfg["name"]
        total_patches = cfg["total_patches"]
        lines.append(f"【{city_name}】")
        
        for source in cfg["sources"]:
            curr = current[city_key][source]
            n_patches = curr["patches"]
            n_frames = curr["frames"]
            
            # 计算速率和 ETA
            if previous is not None and elapsed_min > 0:
                prev = previous[city_key][source]
                patch_delta = n_patches - prev["patches"]
                frame_delta = n_frames - prev["frames"]
                patch_rate = patch_delta / (elapsed_min / 60)  # patches/hour
                frame_rate = frame_delta / (elapsed_min / 60)  # frames/hour
                
                remaining = total_patches - n_patches
                if patch_rate > 0 and remaining > 0:
                    eta_hours = remaining / patch_rate
                    eta_str = f"ETA={eta_hours:.1f}h"
                else:
                    eta_str = "ETA=∞"
                
                rate_str = f"+{patch_delta}p/{frame_delta}f ({patch_rate:.1f}p/h, {frame_rate:.0f}f/h)"
            else:
                rate_str = "baseline"
                eta_str = "ETA=--"
            
            status = "✅" if n_patches >= total_patches else "🔄"
            lines.append(f"  {status} {source:12s}: {n_patches:3d}/{total_patches} patches, {n_frames:5d} frames | {rate_str} | {eta_str}")
        
        lines.append("")
    
    lines.append(f"下次检查: {(datetime.now().timestamp() + CHECK_INTERVAL):.0f} ({CHECK_INTERVAL//60}分钟后)")
    lines.append("")
    return "\n".join(lines)


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    previous = None
    previous_time = None
    
    print(f"PC 下载监控启动 | 日志: {LOG_FILE}")
    print(f"检查间隔: {CHECK_INTERVAL//60} 分钟")
    print("=" * 60)
    
    while True:
        current_time = time.time()
        current = get_snapshot()
        
        if previous is not None:
            elapsed_min = (current_time - previous_time) / 60
        else:
            elapsed_min = 0
        
        report = format_report(current, previous, elapsed_min)
        
        # 写入日志
        with open(LOG_FILE, "a") as f:
            f.write(report + "\n")
        
        # 同时输出到 stdout
        print(report)
        
        previous = current
        previous_time = current_time
        
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
