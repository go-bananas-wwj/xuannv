#!/bin/bash
# Round7 E50 监控循环 - 每 5 分钟刷新一次，共跑 30 次(150分钟)
# 用法: bash scripts/monitor_e50_loop.sh

LOG="/workspace/outputs/e50_monitor.log"
CONDA="/root/miniconda3/bin/conda"

echo "=== E50 监控启动 $(date '+%Y-%m-%d %H:%M:%S') ===" > "$LOG"

for i in $(seq 1 30); do
    echo "" >> "$LOG"
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') 第 $i 次刷新 ---" >> "$LOG"
    
    eval "$($CONDA shell.bash hook)"
    conda activate xuannv
    cd /workspace/xuannv
    python scripts/monitor_e50_progress.py --once >> "$LOG" 2>&1
    
    # 检查是否全部完成
    if grep -q "所有实验已完成 E50" "$LOG" 2>/dev/null; then
        echo "✅ 全部完成，监控退出" >> "$LOG"
        exit 0
    fi
    
    sleep 300
done

echo "监控循环结束 $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
