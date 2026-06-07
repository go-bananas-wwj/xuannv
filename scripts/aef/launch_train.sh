#!/bin/bash
# AEF 训练启动脚本
set -e

cd /workspace/xuannv

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

# NPU 设置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 可选：从 checkpoint resume
RESUME=""
# RESUME="--resume /workspace/xuannv/outputs/aef_haidian/step_010000.pt"

echo "Starting AEF training on 8 NPUs..."
torchrun \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    src/aef/train.py \
    --config configs/config_aef_haidian.yaml \
    --batch-size 4 \
    --num-workers 0 \
    --max-steps 100000 \
    --lr 0.0001 \
    --warmup-steps 2000 \
    --log-every 50 \
    --save-every 5000 \
    --eval-every 5000 \
    --output-dir /workspace/xuannv/outputs/aef_haidian \
    --seed 42 \
    $RESUME

echo "Training completed or stopped."
