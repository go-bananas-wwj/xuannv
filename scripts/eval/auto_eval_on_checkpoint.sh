#!/usr/bin/env bash
# auto_eval_on_checkpoint.sh — 当新 checkpoint 保存时自动评估
# 用法: bash scripts/eval/auto_eval_on_checkpoint.sh --config configs/config_v14_anti_collapse.yaml --output-dir /workspace/outputs/exp_v14_anti_collapse_0605

set -euo pipefail

CONFIG="configs/config_v14_anti_collapse.yaml"
OUTPUT_DIR=""
DEVICE="npu:7"  # 使用空闲的 NPU 7 进行评估，不干扰训练
CHECKPOINT_DIR=""
LAST_EPOCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG="$2";       shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
    --device)       DEVICE="$2";       shift 2 ;;
    *) echo "[错误] 未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "[错误] 必须指定 --output-dir"
  exit 1
fi

CHECKPOINT_DIR="$OUTPUT_DIR"

echo "=== 自动评估监控启动 ==="
echo "Config: $CONFIG"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Eval device: $DEVICE"
echo "按 Ctrl+C 停止"
echo ""

while true; do
  # 查找最新的 epoch_*.pt（非 best/last）
  LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/epoch_*.pt 2>/dev/null | grep -v "epoch_best" | grep -v "epoch_last" | head -1 || true)
  
  if [[ -n "$LATEST_CKPT" ]]; then
    EPOCH_NUM=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K[0-9]+' || echo "0")
    
    if [[ "$EPOCH_NUM" -gt "$LAST_EPOCH" ]]; then
      echo "[$(date)] 发现新 checkpoint: $LATEST_CKPT (epoch $EPOCH_NUM)"
      LAST_EPOCH=$EPOCH_NUM
      
      # 运行 AUC 评估（最快，单卡即可）
      echo "[$(date)] 开始 AUC 评估..."
      python scripts/eval/auc_eval.py \
        --config "$CONFIG" \
        --checkpoint "$LATEST_CKPT" \
        --device "$DEVICE" \
        --output-dir "$CHECKPOINT_DIR/eval_epoch_${EPOCH_NUM}" \
        2>&1 | tee "$CHECKPOINT_DIR/eval_epoch_${EPOCH_NUM}/auc.log" || echo "[警告] AUC 评估失败"
      
      echo "[$(date)] Epoch $EPOCH_NUM 评估完成"
      echo ""
    fi
  fi
  
  # 每 5 分钟检查一次
  sleep 300
done
