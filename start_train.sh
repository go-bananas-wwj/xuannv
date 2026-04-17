#!/bin/bash
cd /workspace/xuannv
export CUDA_VISIBLE_DEVICES=5,6,7
exec torchrun --nproc_per_node=3 scripts/train/train_ddp.py \
  --config configs/qwen_v1_scenes.yaml \
  --save-every 50 --warmup-epochs 10
