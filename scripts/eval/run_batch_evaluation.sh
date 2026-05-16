#!/bin/bash
# Round 8 批量评估启动器 — 8 实验并行评估，每实验 1 NPU

set -e

OUTPUT_BASE="/workspace/outputs"
CONFIG_BASE="/workspace/xuannv/configs"

# 8 实验配置
declare -a EXP_NAMES=(
    "round8_single_exp1"
    "round8_single_exp2"
    "round8_single_exp3"
    "round8_single_exp4"
    "round8_single_exp5"
    "round8_single_exp6"
    "round8_single_exp7"
    "round8_single_exp8"
)

N_FOLDS=3
CD_EPOCHS=50

echo "========================================"
echo "  Round 8 批量评估启动"
echo "  实验数: ${#EXP_NAMES[@]}"
echo "  每实验: 1 NPU"
echo "========================================"

for i in "${!EXP_NAMES[@]}"; do
    exp="${EXP_NAMES[$i]}"
    device_idx=$i
    cfg="${CONFIG_BASE}/round8_single_exp$((i+1)).yaml"
    ckpt="${OUTPUT_BASE}/${exp}/epoch_19.pt"
    out="${OUTPUT_BASE}/${exp}/eval_results.json"
    precomp="${OUTPUT_BASE}/${exp}/precomputed_embeddings.pt"

    if [ ! -f "$ckpt" ]; then
        echo "  跳过 $exp: checkpoint 不存在 ($ckpt)"
        continue
    fi

    session="eval_${exp}"
    echo "  [$i] 启动 $session on NPU $device_idx"

    tmux new-session -d -s "$session" -c /workspace/xuannv || true
    tmux send-keys -t "$session" "export ASCEND_RT_VISIBLE_DEVICES=${device_idx}" Enter
    tmux send-keys -t "$session" "conda activate xuannv" Enter
    tmux send-keys -t "$session" "cd /workspace/xuannv" Enter

    if [ -f "$precomp" ]; then
        tmux send-keys -t "$session" "python scripts/eval/full_evaluation_pipeline.py --config $cfg --checkpoint $ckpt --output $out --device npu:0 --precomputed $precomp --sem-folds $N_FOLDS --cd-epochs $CD_EPOCHS" Enter
    else
        tmux send-keys -t "$session" "python scripts/eval/full_evaluation_pipeline.py --config $cfg --checkpoint $ckpt --output $out --device npu:0 --sem-folds $N_FOLDS --cd-epochs $CD_EPOCHS" Enter
    fi
done

echo ""
echo "  全部已启动!"
echo "  查看: tmux list-sessions"
echo "  监控: tmux capture-pane -t eval_round8_single_exp1 -p | tail -20"
echo ""
