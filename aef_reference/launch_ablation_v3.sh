#!/bin/bash
# Ablation v3: Test decoder spatial conv fix
# 2 groups, 200 steps each, 2 cards per group

cd /workspace/xuannv
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xuannv
export ASCEND_LAUNCH_BLOCKING=1

COMMON_ARGS="--batch-size 2 --max-steps 200 --save-every 50 --eval-every 200 --log-every 10 --seed 42 --distill-warmup-steps 0 --grad-accum-steps 2"

# Group 5: spatial_distill=true, recon=1.0, decoder spatial conv (cards 0,1)
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29600 \
  aef_reference/train.py $COMMON_ARGS \
  --output-dir aef_reference/outputs/ablation_v3_spatial_recon1.0 &
PID5=$!

# Group 6: spatial_distill=true, recon=5.0, decoder spatial conv (cards 2,3)
ASCEND_RT_VISIBLE_DEVICES=2,3 \
SPATIAL_DISTILL=true RECON_WEIGHT=5.0 DISTILL_WEIGHT=1.0 UNIFORMITY_WEIGHT=0.5 CONSISTENCY_WEIGHT=0.02 Y_GRAD_WEIGHT=0.0 \
torchrun --nproc_per_node=2 --master_port=29601 \
  aef_reference/train.py $COMMON_ARGS \
  --output-dir aef_reference/outputs/ablation_v3_spatial_recon5.0 &
PID6=$!

echo "Launched 2 ablation v3 experiments:"
echo "  exp5 (spatial+recon1.0+decoder_conv) on GPUs 0,1 -> PID $PID5"
echo "  exp6 (spatial+recon5.0+decoder_conv) on GPUs 2,3 -> PID $PID6"
echo ""
echo "Wait for completion..."
wait $PID5
wait $PID6

echo "All v3 experiments complete!"
