#!/bin/bash
# Planetary Computer 下载 wrapper（用于后台运行）
source /opt/conda/etc/profile.d/conda.sh
conda activate xuannv
cd /workspace/xuannv
SOURCE="$1"
LOG="/workspace/outputs/${SOURCE}_download_2026.log"
python scripts/preprocessing/download_from_planetary_computer.py \
    --patches /workspace/raw/harbin_scenes/patches_meta.json \
    --output /workspace/raw/harbin_scenes \
    --sources "$SOURCE" --workers 4 \
    --date-start 2026-01-01 --date-end 2026-05-31 \
    2>&1 | tee -a "$LOG"
