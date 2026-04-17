#!/bin/bash
# ============================================================
# DDP v4 训练看门狗 — 自动崩溃恢复
# ============================================================
# 用法:
#   cd /workspace/xuannv
#   bash watchdog.sh
#
# 功能:
#   - 自动查找最新 checkpoint 并 resume
#   - 崩溃/OOM 后等待 30s 自动重启
#   - 最大重试次数: 10 (防止无限循环)
#   - 训练成功完成后退出
# ============================================================

set -e

CONFIG="configs/qwen_v4_cd_upgrade.yaml"
OUTPUT_DIR="/workspace/outputs/aef_qwen_v4_cd_upgrade"
LOG_FILE="$OUTPUT_DIR/watchdog.log"
MAX_RETRIES=10
RETRY_COUNT=0

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_latest_checkpoint() {
    local latest=""
    if [ -d "$OUTPUT_DIR" ]; then
        latest=$(ls -t "$OUTPUT_DIR"/epoch_*.pt 2>/dev/null | head -1)
    fi
    echo "$latest"
}

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    LATEST_CKPT=$(get_latest_checkpoint)
    RESUME_ARG=""
    
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume $LATEST_CKPT"
        log "[Retry $RETRY_COUNT/$MAX_RETRIES] Resuming from: $LATEST_CKPT"
    else
        log "[Retry $RETRY_COUNT/$MAX_RETRIES] No checkpoint found, starting from scratch."
    fi
    
    log "[Retry $RETRY_COUNT/$MAX_RETRIES] Launching training..."
    
    set +e
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
        scripts/train/train_ddp_v4.py \
        --config "$CONFIG" \
        $RESUME_ARG
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "Training completed successfully."
        exit 0
    else
        log "Training failed with exit code $EXIT_CODE."
        
        # 检查是否是 OOM
        if grep -q "CUDA out of memory" "$OUTPUT_DIR/train.log" 2>/dev/null; then
            log "Detected OOM error. Consider reducing batch size or enabling gradient checkpointing."
        fi
        
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            log "Waiting 30s before restart..."
            sleep 30
        else
            log "Max retries ($MAX_RETRIES) reached. Giving up."
            exit 1
        fi
    fi
done
