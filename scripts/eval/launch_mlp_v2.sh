#!/bin/bash
# 只重新跑 MLP（修复 JRC Water 标签过滤后）
set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

EXPERIMENT_ROOT="/workspace/outputs/xuannv_round2"

declare -a EXP_NAMES=(
    "round4_full_vicreg_baseline"
    "round4_full_high_var"
    "round4_full_high_temporal"
    "round4_full_low_recon"
    "round4_full_high_consist"
    "round4_full_high_kappa"
    "round4_full_low_decoder"
)

declare -a ASCEND_DEVICES=(0 1 2 3 4 5 6)

for i in "${!EXP_NAMES[@]}"; do
    EXP="${EXP_NAMES[$i]}"
    ASCEND_DEV="${ASCEND_DEVICES[$i]}"
    EMB_FILE="${EXPERIMENT_ROOT}/${EXP}/evaluation/embeddings/patch_embeddings.npz"
    OUT_DIR="${EXPERIMENT_ROOT}/${EXP}/evaluation/downstream"
    
    SESSION="mlp_${EXP}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] MLP $EXP -> npu:0 (ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV})"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV}" Enter
    tmux send-keys -t "$SESSION" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv" Enter
    tmux send-keys -t "$SESSION" "python scripts/eval/evaluate_mlp_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR} --device npu:0 --epochs 50 --month 6 > ${EXPERIMENT_ROOT}/${EXP}/evaluation/mlp_v2.log 2>&1 && echo 'MLP DONE' || echo 'MLP FAILED'" Enter
    sleep 0.3
done

echo "All MLP v2 launched!"
