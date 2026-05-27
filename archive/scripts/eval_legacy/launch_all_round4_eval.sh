#!/bin/bash
# 7卡并行启动 Round 4 全部实验的 embedding 提取（带断点续传）
# 用法: bash scripts/eval/launch_all_round4_eval.sh

set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

EXPERIMENT_ROOT="/workspace/outputs/xuannv_round2"
CONFIG_ROOT="configs/round4_8gpu"

# 7个实验的配置（排除 emb128）
declare -a EXP_NAMES=(
    "round4_full_vicreg_baseline"
    "round4_full_high_var"
    "round4_full_high_temporal"
    "round4_full_low_recon"
    "round4_full_high_consist"
    "round4_full_high_kappa"
    "round4_full_low_decoder"
)

# NPU 分配 (0-6)
declare -a DEVICES=(
    "npu:0"
    "npu:1"
    "npu:2"
    "npu:3"
    "npu:4"
    "npu:5"
    "npu:6"
)

# ASCEND 设备索引 (0-6)
declare -a ASCEND_DEVICES=(0 1 2 3 4 5 6)

echo "========================================"
echo "Launching 7 experiments in parallel"
echo "========================================"

for i in "${!EXP_NAMES[@]}"; do
    EXP="${EXP_NAMES[$i]}"
    DEVICE="${DEVICES[$i]}"
    ASCEND_DEV="${ASCEND_DEVICES[$i]}"
    CONFIG="${CONFIG_ROOT}/${EXP}.yaml"
    CKPT="${EXPERIMENT_ROOT}/${EXP}/epoch_best_epoch20.pt"
    
    if [[ ! -f "$CKPT" ]]; then
        echo "[SKIP] $EXP: checkpoint not found at $CKPT"
        continue
    fi
    
    SESSION="eval_${EXP}"
    
    # 如果 session 已存在，先杀掉
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] $EXP -> $DEVICE (tmux: $SESSION)"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV}" Enter
    tmux send-keys -t "$SESSION" "conda activate xuannv" Enter
    tmux send-keys -t "$SESSION" "python scripts/eval/run_extraction_resilient.py --exp-name ${EXP} --config ${CONFIG} --checkpoint ${CKPT} --device npu:0 --batch-size 16 --save-every 500" Enter
    
    sleep 0.5
done

echo ""
echo "========================================"
echo "All launched! Monitor with:"
echo "  tmux list-sessions | grep eval_"
echo "  tmux attach -t eval_round4_full_vicreg_baseline"
echo "========================================"
