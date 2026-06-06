#!/bin/bash
# S1 PC 大规模下载脚本
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

LOG_FILE="/workspace/xuannv/logs/dl_s1_pc_$(date +%Y%m%d_%H%M%S).log"

echo "Starting S1 PC download at $(date)" > "$LOG_FILE"
python scripts/preprocessing/download_s1_pc_haidian.py \
    --patches /workspace/xuannv/data_raw/haidian/olmoearth/patches_meta.json \
    --output /workspace/xuannv/data_raw/haidian/s1_pc_download \
    --workers 8 \
    --date-start 2025-01-01 \
    --date-end 2026-04-30 \
    >> "$LOG_FILE" 2>&1

echo "Download finished at $(date)" >> "$LOG_FILE"
