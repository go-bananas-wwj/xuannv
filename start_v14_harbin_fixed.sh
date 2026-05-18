#!/bin/bash
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
    scripts/train/train_ddp_v14.py \
    --config configs/v14/v14_harbin_fixed_s2.yaml \
    --save-every 10 \
    --epochs 50 \
    2>&1 | tee /workspace/outputs/v14_harbin_fixed_s2/train.log
