#!/bin/bash
# 4组并行消融实验 v2: 测试行shuffle + y-gradient惩罚

cd /workspace/xuannv/aef_reference
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

COMMON="--batch-size 2 --max-steps 200 --save-every 50 --eval-every 50 --grad-accum-steps 2 --log-every 10 --warmup-steps 200 --seed 42"

# 实验A: 行shuffle + 空间蒸馏 + 重建 (卡0,1)
SPATIAL_DISTILL=true DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 Y_GRAD_WEIGHT=0.0 \
  ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  train.py $COMMON --output-dir outputs/ablation_v2_exp1_rowshuffle_spatial &

# 实验B: 行shuffle + 全局蒸馏 + 重建 (卡2,3)
SPATIAL_DISTILL=false DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 Y_GRAD_WEIGHT=0.0 \
  ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
  train.py $COMMON --output-dir outputs/ablation_v2_exp2_rowshuffle_global &

# 实验C: 行shuffle + y-gradient惩罚 + 空间蒸馏 (卡4,5)
SPATIAL_DISTILL=true DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 Y_GRAD_WEIGHT=1.0 \
  ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29502 \
  train.py $COMMON --output-dir outputs/ablation_v2_exp3_ygrad_spatial &

# 实验D: 行shuffle + y-gradient惩罚 + 全局蒸馏 (卡6,7)
SPATIAL_DISTILL=false DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 Y_GRAD_WEIGHT=1.0 \
  ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29503 \
  train.py $COMMON --output-dir outputs/ablation_v2_exp4_ygrad_global &

wait
echo "All 4 v2 experiments completed!"
