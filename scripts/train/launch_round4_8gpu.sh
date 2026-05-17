#!/bin/bash
# Round 4: 8个实验并行，各用1张NPU (torchrun --nproc_per_node=1)
# 用法: cd /workspace/xuannv && bash scripts/train/launch_round4_8gpu.sh

set -e

EXPERIMENTS=(
    round4_full_vicreg_baseline
    round4_full_high_var
    round4_full_high_temporal
    round4_full_low_recon
    round4_full_high_consist
    round4_full_high_kappa
    round4_full_emb128
    round4_full_low_decoder
)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

cd /workspace/xuannv

for i in {0..7}; do
    EXP="${EXPERIMENTS[$i]}"
    GPU=$i
    SESSION="r4_${EXP}"
    
    echo "[$i] Launching $EXP on NPU $GPU..."
    
    # Kill existing session
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 0.5
    
    # Create new session with training command (torchrun single-process)
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv "bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv && export ASCEND_RT_VISIBLE_DEVICES=$GPU && torchrun --nproc_per_node=1 --master_port=$((29500 + i)) scripts/train/train_ddp_v13.py --config configs/round4_8gpu/${EXP}.yaml --save-every 10'"
done

echo ""
echo "All 8 experiments launched!"
echo "Monitor with: tmux list-sessions | grep r4_"
echo "View log: tmux capture-pane -t r4_round4_full_vicreg_baseline -p | tail -20"
