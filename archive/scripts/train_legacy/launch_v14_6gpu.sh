#!/bin/bash
# V14 单卡×6快速筛选启动脚本
# 6个实验在 NPU 0-5 上并行启动，每个 20 epochs

set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

EXPERIMENTS=(
    v14_multi_shared_ts
    v14_multi_baseline
    v14_harbin_only
    v14_high_consist
    v14_high_batch
    v14_high_temporal
)

# NPU 0-5 分配给6个实验
ASCEND_DEVICES=(0 1 2 3 4 5)

echo "========================================"
echo "Launching V14 (6 experiments × 20ep)"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    GPU="${ASCEND_DEVICES[$i]}"
    CONFIG="configs/v14/${EXP}.yaml"
    SESSION="v14_${EXP}"
    
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] $EXP -> NPU $GPU (20 epochs)"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${GPU}" Enter
    tmux send-keys -t "$SESSION" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv" Enter
    tmux send-keys -t "$SESSION" "torchrun --nproc_per_node=1 --master_port=$((29700 + i)) scripts/train/train_ddp_v14.py --config ${CONFIG} --save-every 10 --epochs 20" Enter
    
    sleep 0.5
done

echo ""
echo "========================================"
echo "All V14 experiments launched!"
echo "========================================"
echo ""
echo "Monitor with:"
echo "  tmux list-sessions | grep v14_"
echo "  npu-smi info"
echo ""
echo "Experiments:"
for EXP in "${EXPERIMENTS[@]}"; do
    echo "  - $EXP"
done
