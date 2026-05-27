#!/bin/bash
# Round 9 训练启动器 — 4 实验 × 2 NPU

cd /workspace/xuannv

for i in {1..4}; do
    npu_start=$(( (i-1) * 2 ))
    npu_end=$(( npu_start + 1 ))
    port=$(( 29500 + i * 10 ))
    session="r9_exp${i}"
    
    # Kill existing session if any
    tmux kill-session -t "$session" 2>/dev/null
    
    # Create new session
    tmux new-session -d -s "$session" -c /workspace/xuannv
    tmux send-keys -t "$session" "export ASCEND_RT_VISIBLE_DEVICES=${npu_start},${npu_end}" Enter
    tmux send-keys -t "$session" "conda activate xuannv" Enter
    tmux send-keys -t "$session" "torchrun --nproc_per_node=2 --master_port=${port} scripts/train/train_ddp_v7.py --config configs/round9_exp${i}_*.yaml --save-every 20" Enter
    
    echo "Launched Round 9 exp${i} on NPU ${npu_start}-${npu_end} (port ${port}) in tmux ${session}"
done

echo ""
echo "All 4 experiments launched. Use 'tmux ls | grep r9' to list sessions."
echo "Monitor: tmux capture-pane -t r9_exp1 -p | tail -20"
