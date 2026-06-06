#!/bin/bash
# 等待下载完成并自动执行替换和统计量计算

DL_DIR="/workspace/xuannv/data_raw/haidian/s1_pc_download"
SCENES_DIR="/workspace/xuannv/data_raw/haidian/scenes"
LOG="/workspace/xuannv/logs/s1_replace_$(date +%Y%m%d_%H%M%S).log"

echo "Waiting for download to complete..." >> "$LOG"

# 等待下载进程结束
while pgrep -f "download_s1_pc_haidian.py" > /dev/null; do
    sleep 30
done

echo "Download finished. Starting replacement..." >> "$LOG"

cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

# 1. 替换数据
python scripts/preprocessing/replace_s1_with_pc.py >> "$LOG" 2>&1

# 2. 重新计算统计量
python scripts/preprocessing/compute_s1_stats_pc.py >> "$LOG" 2>&1

echo "All done at $(date)" >> "$LOG"
