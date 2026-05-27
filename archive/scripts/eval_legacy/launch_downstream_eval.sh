#!/bin/bash
# 7卡并行启动 Round 4 全部实验的下游评估
# 用法: bash scripts/eval/launch_downstream_eval.sh

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

echo "========================================"
echo "Launching downstream evaluation"
echo "========================================"

for i in "${!EXP_NAMES[@]}"; do
    EXP="${EXP_NAMES[$i]}"
    ASCEND_DEV="${ASCEND_DEVICES[$i]}"
    EMB_FILE="${EXPERIMENT_ROOT}/${EXP}/evaluation/embeddings/patch_embeddings.npz"
    OUT_DIR="${EXPERIMENT_ROOT}/${EXP}/evaluation"
    
    if [[ ! -f "$EMB_FILE" ]]; then
        echo "[SKIP] $EXP: embedding not found"
        continue
    fi
    
    SESSION="downstream_${EXP}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] $EXP -> npu:0 (ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV})"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV}" Enter
    tmux send-keys -t "$SESSION" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv" Enter
    
    # KNN
    tmux send-keys -t "$SESSION" "echo '=== KNN ===' && python scripts/eval/evaluate_knn_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR}/downstream --device npu:0 --k 5 --month 6 > ${OUT_DIR}/knn.log 2>&1 && echo 'KNN DONE' || echo 'KNN FAILED'" Enter
    
    # MLP
    tmux send-keys -t "$SESSION" "echo '=== MLP ===' && python scripts/eval/evaluate_mlp_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR}/downstream --device npu:0 --epochs 50 --month 6 > ${OUT_DIR}/mlp.log 2>&1 && echo 'MLP DONE' || echo 'MLP FAILED'" Enter
    
    # CD
    tmux send-keys -t "$SESSION" "echo '=== CD ===' && python scripts/eval/evaluate_cd_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR}/change_detection > ${OUT_DIR}/cd.log 2>&1 && echo 'CD DONE' || echo 'CD FAILED'" Enter
    
    tmux send-keys -t "$SESSION" "echo 'ALL DOWNSTREAM DONE for ${EXP}'" Enter
    
    sleep 0.5
done

echo ""
echo "========================================"
echo "All downstream launched!"
echo "========================================"
