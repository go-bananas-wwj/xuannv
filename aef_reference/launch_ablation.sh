#!/bin/bash
# 4组并行消融实验，每组2卡，200 step

cd /workspace/xuannv/aef_reference
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

COMMON="--batch-size 2 --max-steps 200 --save-every 50 --eval-every 50 --grad-accum-steps 2 --log-every 10 --warmup-steps 200 --seed 42"

# 实验1: 空间蒸馏 + 高重建 (卡0,1)
SPATIAL_DISTILL=true DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 \
  ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  train.py $COMMON --output-dir outputs/ablation_exp1_spatial_recon &

# 实验2: 空间蒸馏 + 低重建 (卡2,3)
SPATIAL_DISTILL=true DISTILL_WEIGHT=10.0 RECON_WEIGHT=0.01 UNIFORMITY_WEIGHT=0.5 \
  ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
  train.py $COMMON --output-dir outputs/ablation_exp2_spatial_only &

# 实验3: 全局蒸馏 + 高重建 (卡4,5)
SPATIAL_DISTILL=false DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 \
  ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29502 \
  train.py $COMMON --output-dir outputs/ablation_exp3_global_recon &

# 实验4: 空间蒸馏 + 高重建 + 无uniformity (卡6,7)
SPATIAL_DISTILL=true DISTILL_WEIGHT=5.0 RECON_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.0 \
  ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29503 \
  train.py $COMMON --output-dir outputs/ablation_exp4_spatial_recon_nounif &

wait
echo "All 4 experiments completed!"
