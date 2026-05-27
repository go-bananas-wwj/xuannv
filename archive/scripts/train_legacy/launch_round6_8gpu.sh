#!/bin/bash
# Round 6 训练并行启动脚本
# 6个实验在NPU 2-7上并行启动，每个100 epochs

set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

EXPERIMENTS=(
    r6_consist_kappa5k
    r6_consist_mild_100ep
    r6_temporal_consist_k5k
    r6_high_consist_k5k
    r6_no_consist_k5k
    r6_temporal_recon_consist
)

# NPU 2-7 分配给6个实验
ASCEND_DEVICES=(2 3 4 5 6 7)

echo "========================================"
echo "Launching Round 6 (6 experiments × 100ep)"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    GPU="${ASCEND_DEVICES[$i]}"
    CONFIG="configs/round6_8gpu/${EXP}.yaml"
    SESSION="r6_${EXP}"
    
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] $EXP -> NPU $GPU (100 epochs)"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${GPU}" Enter
    tmux send-keys -t "$SESSION" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv" Enter
    tmux send-keys -t "$SESSION" "torchrun --nproc_per_node=1 --master_port=$((29700 + i)) scripts/train/train_ddp_v13.py --config ${CONFIG} --save-every 20" Enter
    
    sleep 0.5
done

echo ""
echo "========================================"
echo "All Round 6 experiments launched!"
echo "========================================"
echo ""
echo "Monitor with:"
echo "  tmux list-sessions | grep r6_"
echo "  npu-smi info"
echo ""
echo "Experiments:"
for EXP in "${EXPERIMENTS[@]}"; do
    echo "  - $EXP"
done
