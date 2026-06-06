#!/bin/bash
# v14 评估流水线 — 训练完成后一键执行

set -e

CONFIG="configs/config_v14_anti_collapse.yaml"
CKPT_DIR="/workspace/xuannv/outputs/exp_v14_anti_collapse_0605"
DEVICE="npu:0"

# 找到最新的 best checkpoint
BEST_CKPT=$(ls -t ${CKPT_DIR}/epoch_best_*.pt 2>/dev/null | head -1)
LATEST_CKPT=$(ls -t ${CKPT_DIR}/epoch_*.pt 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "未找到 best checkpoint，使用最新: $LATEST_CKPT"
    CKPT="$LATEST_CKPT"
else
    echo "使用 best checkpoint: $BEST_CKPT"
    CKPT="$BEST_CKPT"
fi

echo "=== 1. 提取 Embedding ==="
python scripts/eval/extract_embeddings.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-dir "$CKPT_DIR/eval" \
    --device "$DEVICE"

echo "=== 2. MLP 评估 ==="
python scripts/eval/evaluate_mlp_v2.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-dir "$CKPT_DIR/eval/mlp" \
    --device "$DEVICE"

echo "=== 3. 变化检测 AUC 评估 ==="
python scripts/eval/evaluate_cd_v2.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-dir "$CKPT_DIR/eval/cd" \
    --device "$DEVICE"

echo "=== 评估完成 ==="
