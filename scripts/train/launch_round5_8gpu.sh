#!/bin/bash
# Round 5: 6个实验并行，各用1张NPU (torchrun --nproc_per_node=1)
# 用法: cd /workspace/xuannv && bash scripts/train/launch_round5_8gpu.sh

set -e

EXPERIMENTS=(
    round5_kappa_temporal_mild
    round5_baseline_40ep
    round5_consist_mild
    round5_no_consist
    round5_kappa_baseline
    round5_temporal_plus_recon
)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

cd /workspace/xuannv

for i in {0..5}; do
    EXP="${EXPERIMENTS[$i]}"
    GPU=$i
    SESSION="r5_${EXP}"
    
    echo "[$i] Launching $EXP on NPU $GPU..."
    
    # Kill existing session
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 0.5
    
    # Create new session with training command (torchrun single-process)
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv "bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv && export ASCEND_RT_VISIBLE_DEVICES=$GPU && torchrun --nproc_per_node=1 --master_port=$((29600 + i)) scripts/train/train_ddp_v13.py --config configs/round5_8gpu/${EXP}.yaml --save-every 10'"
done

echo ""
echo "All 6 Round 5 experiments launched!"
echo "Monitor with: tmux list-sessions | grep r5_"
echo "View log: tmux capture-pane -t r5_round5_kappa_temporal_mild -p | tail -20"
