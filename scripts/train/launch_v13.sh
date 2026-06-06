#!/bin/bash
# V13 标准训练启动器
# 用法:
#   ./scripts/train/launch_v13.sh                     # 使用默认 4 卡，configs/config.yaml
#   ./scripts/train//launch_v13.sh --npus 0,1,2,3     # 指定 NPU 编号
#   ./scripts/train/launch_v13.sh --config configs/v14/v14_multi_baseline.yaml
#   ./scripts/train/launch_v13.sh --resume /workspace/xuannv/outputs/exp_xxx/epoch_best_xxx.pt

set -e
cd "$(dirname "$0")/../.."

# 默认参数
NPUS="0,1,2,3"
CONFIG="configs/config.yaml"
SAVE_EVERY=20
WARMUP_EPOCHS=10
RESUME=""
SESSION="v13_train"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --npus)     NPUS="$2";         shift 2 ;;
        --config)   CONFIG="$2";       shift 2 ;;
        --save-every) SAVE_EVERY="$2"; shift 2 ;;
        --warmup)   WARMUP_EPOCHS="$2"; shift 2 ;;
        --resume)   RESUME="$2";       shift 2 ;;
        --session)  SESSION="$2";      shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

NPROC=$(echo "$NPUS" | tr ',' '\n' | wc -l)
RESUME_FLAG=""
if [[ -n "$RESUME" ]]; then
    RESUME_FLAG="--resume $RESUME"
fi

echo "==============================="
echo "  V13 训练启动"
echo "  NPUs:    $NPUS ($NPROC 卡)"
echo "  Config:  $CONFIG"
echo "  Session: $SESSION"
echo "  Resume:  ${RESUME:-无}"
echo "==============================="

# 启动 tmux session
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$(pwd)"
tmux send-keys -t "$SESSION" "export ASCEND_RT_VISIBLE_DEVICES=$NPUS" Enter
tmux send-keys -t "$SESSION" "conda activate xuannv" Enter
tmux send-keys -t "$SESSION" "torchrun --nproc_per_node=$NPROC scripts/train/train_ddp_v13.py --config $CONFIG --save-every $SAVE_EVERY --warmup-epochs $WARMUP_EPOCHS $RESUME_FLAG" Enter

echo ""
echo "已在 tmux session '$SESSION' 中启动。"
echo "查看日志: tmux attach -t $SESSION"
echo "后台监控: tmux capture-pane -t $SESSION -p | tail -20"
