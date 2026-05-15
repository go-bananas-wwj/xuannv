#!/bin/bash
cd /workspace/xuannv

for i in {1..8}; do
    npu_idx=$((i-1))
    port=$((29500 + npu_idx))
    session="r8s${i}"
    
    # Kill existing session if any
    tmux kill-session -t "$session" 2>/dev/null
    
    # Create new session and launch
    tmux new-session -d -s "$session" -c /workspace/xuannv
    tmux send-keys -t "$session" "export ASCEND_RT_VISIBLE_DEVICES=${npu_idx}" Enter
    tmux send-keys -t "$session" "torchrun --nproc_per_node=1 --master_port=${port} scripts/train/train_ddp_v7.py --config configs/round8_single_exp${i}.yaml --save-every 5" Enter
    
    echo "Launched exp${i} on NPU ${npu_idx} (port ${port}) in tmux ${session}"
done

echo "All 8 experiments launched. Use 'tmux ls | grep r8s' to list sessions."
