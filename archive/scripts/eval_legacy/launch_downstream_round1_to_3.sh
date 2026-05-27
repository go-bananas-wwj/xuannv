#!/bin/bash
# 并行启动 Round 1-3 下游评估（7卡并行）
# 用法: bash scripts/eval/launch_downstream_round1_to_3.sh

set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

declare -a EXP_DIRS=(
    "/workspace/outputs/xuannv_round1/aef_baseline"
    "/workspace/outputs/xuannv_round1/aef_high_kappa"
    "/workspace/outputs/xuannv_round1/aef_skip_l2"
    "/workspace/outputs/xuannv_round1/aef_no_uniform"
    "/workspace/outputs/xuannv_round2/round2_cross_temporal"
    "/workspace/outputs/xuannv_round2/round3_vicreg_temporal"
)

declare -a ASCEND_DEVICES=(0 1 2 3 4 5 6)

echo "========================================"
echo "Launching Round 1-3 downstream eval v2"
echo "========================================"

for i in "${!EXP_DIRS[@]}"; do
    EXP_DIR="${EXP_DIRS[$i]}"
    ASCEND_DEV="${ASCEND_DEVICES[$i]}"
    EMB_FILE="${EXP_DIR}/evaluation_v2/embeddings/patch_embeddings.npz"
    OUT_DIR="${EXP_DIR}/evaluation_v2"
    NAME=$(basename "$EXP_DIR")
    
    if [[ ! -f "$EMB_FILE" ]]; then
        echo "[SKIP] $NAME: embedding not found at $EMB_FILE"
        continue
    fi
    
    SESSION="downstream_r123_${NAME}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] $NAME -> npu:0 (ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV})"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEV}" Enter
    tmux send-keys -t "$SESSION" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv" Enter
    
    # KNN
    tmux send-keys -t "$SESSION" "echo '=== KNN ===' && python scripts/eval/evaluate_knn_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR}/downstream --device npu:0 --k 5 --month 6 > ${OUT_DIR}/knn_v2.log 2>&1 && echo 'KNN DONE' || echo 'KNN FAILED'" Enter
    
    # MLP
    tmux send-keys -t "$SESSION" "echo '=== MLP ===' && python scripts/eval/evaluate_mlp_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR}/downstream --device npu:0 --epochs 50 --month 6 > ${OUT_DIR}/mlp_v2.log 2>&1 && echo 'MLP DONE' || echo 'MLP FAILED'" Enter
    
    # CD
    tmux send-keys -t "$SESSION" "echo '=== CD ===' && python scripts/eval/evaluate_cd_v2.py --embedding-file ${EMB_FILE} --output-dir ${OUT_DIR}/change_detection > ${OUT_DIR}/cd_v2.log 2>&1 && echo 'CD DONE' || echo 'CD FAILED'" Enter
    
    tmux send-keys -t "$SESSION" "echo 'ALL DOWNSTREAM DONE for ${NAME}'" Enter
    
    sleep 0.5
done

echo ""
echo "========================================"
echo "All downstream v2 launched!"
echo "========================================"
