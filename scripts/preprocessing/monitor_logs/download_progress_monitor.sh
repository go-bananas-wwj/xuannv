#!/bin/bash
# 全国数据下载进度监控脚本
BASE="/workspace/xuannv/data_raw/national_china/national_china"
LOG="/workspace/xuannv/data_raw/national_china/download_progress_monitor.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进度检查" >> "$LOG"

for src in s1 landsat; do
  dir="$BASE/$src"
  if [ -d "$dir" ]; then
    patches_with_tif=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -exec sh -c '[ -n "$(find "$1" -name "*.tif" -print -quit)" ]' _ {} \; -print 2>/dev/null | wc -l)
    tifs_5m=$(find "$dir" -name "*.tif" -mmin -5 2>/dev/null | wc -l)
    tifs_30m=$(find "$dir" -name "*.tif" -mmin -30 2>/dev/null | wc -l)
    
    # 检查进程
    pidfile="/workspace/xuannv/data_raw/national_china/download_${src}_v2.pid"
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile")
      if kill -0 "$pid" 2>/dev/null; then
        status="alive"
      else
        status="DEAD"
      fi
    else
      status="no_pidfile"
    fi
    
    echo "  $src: $patches_with_tif/5000 patches | 5min+$tifs_5m tifs | 30min+$tifs_30m tifs | proc=$status" >> "$LOG"
  fi
done

# 异常检测：如果 30 分钟内无新 tif 且进程 alive，记录告警
for src in s1 landsat; do
  dir="$BASE/$src"
  tifs_30m=$(find "$dir" -name "*.tif" -mmin -30 2>/dev/null | wc -l)
  pidfile="/workspace/xuannv/data_raw/national_china/download_${src}_v2.pid"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null && [ "$tifs_30m" -eq 0 ]; then
      echo "  [WARN] $src: 进程 alive 但 30min 无新 tif！可能 STAC 超时卡住。" >> "$LOG"
    fi
  fi
done

echo "---" >> "$LOG"
