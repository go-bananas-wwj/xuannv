#!/bin/bash
# V6 Enhanced Temporal 训练启动脚本 — GPU 6/7 双卡
# 从 V5 best checkpoint 软重启

cd /workspace/xuannv

export CUDA_VISIBLE_DEVICES=6,7

torchrun \
  --nproc_per_node=2 \
  --master_port=29505 \
  scripts/train/train_ddp_v6.py \
  --config configs/qwen_v6_enhanced_temporal.yaml \
  --soft-restart /workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt \
  --save-every 20
