#!/bin/bash
# Ablation v4: Low distill weight + high recon/uniformity to learn from data
# 2 groups, 200 steps each, 2 cards per group

cd /workspace/xuannv
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xuannv
export ASCEND_LAUNCH_BLOCKING=1

COMMON_ARGS="--batch-size 2 --max-steps 200 --save-every 50 --eval-every 200 --log-every 10 --seed 42 --distill-warmup-steps 0 --grad-accum-steps 2"

# Group 7: spatial_distill=true, distill=0.1, recon=5.0, uniformity=2.0 (cards 0,1)
ASCEND_RT_VISIBLE_DEVICES=0,1 \
SPATIAL_DISTILL=true RECON_WEIGHT=5.0 DISTILL_WEIGHT=0.1 UNIFORMITY_WEIGHT=2.0 CONSISTENCY_WEIGHT=0.02 Y_GRAD_WEIGHT=0.0 \
torchrun --nproc_per_node=2 --master_port=29700 \
  aef_reference/train.py $COMMON_ARGS \
  --output-dir aef_reference/outputs/ablation_v4_spatial_distill0.1 &
PID7=$!

# Group 8: spatial_distill=false, distill=0.1, recon=5.0, uniformity=2.0 (cards 2,3)
ASCEND_RT_VISIBLE_DEVICES=2,3 \
SPATIAL_DISTILL=false RECON_WEIGHT=5.0 DISTILL_WEIGHT=0.1 UNIFORMITY_WEIGHT=2.0 CONSISTENCY_WEIGHT=0.02 Y_GRAD_WEIGHT=0.0 \
torchrun --nproc_per_node=2 --master_port=29701 \
  aef_reference/train.py $COMMON_ARGS \
  --output-dir aef_reference/outputs/ablation_v4_global_distill0.1 &
PID8=$!

echo "Launched 2 ablation v4 experiments:"
echo "  exp7 (spatial_distill+distill0.1+recon5.0) on GPUs 0,1 -> PID $PID7"
echo "  exp8 (global_distill+distill0.1+recon5.0) on GPUs 2,3 -> PID $PID8"
echo ""
echo "Wait for completion..."
wait $PID7
wait $PID8

echo "All v4 experiments complete!"
