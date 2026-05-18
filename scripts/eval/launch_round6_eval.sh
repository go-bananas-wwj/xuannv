#!/bin/bash
# Round 6 下游评估并行启动脚本
# 6个100-epoch实验在NPU 0-5上并行评估

set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

declare -a EXP_NAMES=(
    "r6_consist_kappa5k"
    "r6_consist_mild_100ep"
    "r6_temporal_consist_k5k"
    "r6_high_consist_k5k"
    "r6_no_consist_k5k"
    "r6_temporal_recon_consist"
)

declare -a ASCEND_DEVICES=(0 1 2 3 4 5)

echo "========================================"
echo "Launching Round 6 downstream evaluation"
echo "========================================"

for i in "${!EXP_NAMES[@]}"; do
    EXP="${EXP_NAMES[$i]}"
    GPU="${ASCEND_DEVICES[$i]}"
    CONFIG="configs/round6_8gpu/${EXP}.yaml"
    CKPT="/workspace/outputs/round6/${EXP}/epoch_best_epoch92.pt"
    OUT_DIR="/workspace/outputs/round6/${EXP}/evaluation"
    EMB_DIR="${OUT_DIR}/embeddings"
    
    if [[ ! -f "$CKPT" ]]; then
        echo "[SKIP] $EXP: checkpoint not found: $CKPT"
        continue
    fi
    
    mkdir -p "$OUT_DIR"
    
    SESSION="eval_${EXP}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    
    echo "[START] $EXP -> npu:0 (ASCEND_RT_VISIBLE_DEVICES=${GPU})"
    
    tmux new-session -d -s "$SESSION" -c /workspace/xuannv
    tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=${GPU}" Enter
    tmux send-keys -t "$SESSION" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv" Enter
    
    # 1. 提取embedding
    tmux send-keys -t "$SESSION" "echo '=== [1/5] Extract embeddings ===' && python scripts/eval/extract_embeddings_v2.py --config ${CONFIG} --checkpoint ${CKPT} --output-dir ${EMB_DIR} --device npu:0 --batch-size 8 > ${OUT_DIR}/extract.log 2>&1 && echo 'EXTRACT DONE' || echo 'EXTRACT FAILED'" Enter
    
    # 2. KNN
    tmux send-keys -t "$SESSION" "echo '=== [2/5] KNN ===' && python scripts/eval/evaluate_knn_v2.py --embedding-file ${EMB_DIR}/patch_embeddings.npz --output-dir ${OUT_DIR}/downstream --device npu:0 --k 5 --month 6 > ${OUT_DIR}/knn.log 2>&1 && echo 'KNN DONE' || echo 'KNN FAILED'" Enter
    
    # 3. MLP
    tmux send-keys -t "$SESSION" "echo '=== [3/5] MLP ===' && python scripts/eval/evaluate_mlp_v2.py --embedding-file ${EMB_DIR}/patch_embeddings.npz --output-dir ${OUT_DIR}/downstream --device npu:0 --epochs 50 --month 6 > ${OUT_DIR}/mlp.log 2>&1 && echo 'MLP DONE' || echo 'MLP FAILED'" Enter
    
    # 4. CD (LR cosine AUC)
    tmux send-keys -t "$SESSION" "echo '=== [4/5] CD ===' && python scripts/eval/evaluate_cd_v2.py --embedding-file ${EMB_DIR}/patch_embeddings.npz --output-dir ${OUT_DIR}/change_detection > ${OUT_DIR}/cd.log 2>&1 && echo 'CD DONE' || echo 'CD FAILED'" Enter
    
    # 5. Semantic segmentation (LR K-Fold)
    tmux send-keys -t "$SESSION" "echo '=== [5/5] Semantic Segmentation ===' && python scripts/eval/comprehensive_downstream_eval_v2.py --config ${CONFIG} --checkpoint ${CKPT} --device npu:0 --output ${OUT_DIR}/semantic_results.json --folds 3 > ${OUT_DIR}/semantic.log 2>&1 && echo 'SEMANTIC DONE' || echo 'SEMANTIC FAILED'" Enter
    
    tmux send-keys -t "$SESSION" "echo 'ALL EVALUATION DONE for ${EXP}'" Enter
    
    sleep 0.5
done

echo ""
echo "========================================"
echo "All Round 6 evaluations launched!"
echo "========================================"
echo "Monitor with: tmux list-sessions | grep eval_r6"
