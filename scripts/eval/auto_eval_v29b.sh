#!/bin/bash
# 自动评估脚本：等待 v29b 训练结束，然后自动提取 embedding + MLP + AUC 评估
# 训练占用所有 8 张 NPU，只有训练结束后才能进行 NPU 推理

set -e
OUTDIR=/workspace/outputs/exp_v29b_haidian_tc_simple_0601
CONFIG=configs/config_haidian_v29b.yaml
LOG=$OUTDIR/v29b_train.log
EVAL_LOG=$OUTDIR/auto_eval.log

cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

echo "[$(date '+%H:%M:%S')] 自动评估监控启动（等待训练完成）" | tee -a $EVAL_LOG

# ── 等待训练达到指定 epoch 并保存 checkpoint ─────────────────────────────────
wait_checkpoint() {
    local epoch=$1
    local ckpt="$OUTDIR/epoch_${epoch}.pt"
    echo "[$(date '+%H:%M:%S')] 等待 epoch_${epoch}.pt..." | tee -a $EVAL_LOG
    while [ ! -f "$ckpt" ]; do
        sleep 300
    done
    echo "[$(date '+%H:%M:%S')] epoch_${epoch}.pt 已就绪" | tee -a $EVAL_LOG
}

# ── 等待训练进程结束（NPU 空出） ─────────────────────────────────────────────
wait_training_done() {
    echo "[$(date '+%H:%M:%S')] 等待训练进程结束..." | tee -a $EVAL_LOG
    while pgrep -f "train.py.*config_haidian_v29b" > /dev/null 2>&1; do
        sleep 60
    done
    sleep 30  # 多等 30 秒确保进程完全退出
    echo "[$(date '+%H:%M:%S')] 训练进程已结束，NPU 可用" | tee -a $EVAL_LOG
}

# ── 运行单个 checkpoint 的完整评估 ────────────────────────────────────────────
eval_checkpoint() {
    local epoch=$1
    local ckpt="$OUTDIR/epoch_${epoch}.pt"
    local EVAL_OUT="$OUTDIR/eval_epoch${epoch}"
    
    if [ ! -f "$ckpt" ]; then
        echo "[$(date '+%H:%M:%S')] 跳过 epoch_${epoch}（checkpoint 不存在）" | tee -a $EVAL_LOG
        return
    fi
    
    echo "[$(date '+%H:%M:%S')] === 开始 epoch_${epoch} 评估 ===" | tee -a $EVAL_LOG
    mkdir -p $EVAL_OUT

    # 1. 提取 embedding（用 NPU 0-3，4 卡并行）
    echo "[$(date '+%H:%M:%S')] [1/3] 提取 embedding..." | tee -a $EVAL_LOG
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python scripts/eval/extract_embeddings.py \
        --config $CONFIG \
        --checkpoint $ckpt \
        --output-dir $EVAL_OUT \
        --format npz \
        2>&1 | tee -a $EVAL_LOG

    # 找到生成的 npz 文件
    NPZ_FILE=$(ls $EVAL_OUT/patch_embeddings*.npz 2>/dev/null | head -1)
    if [ -z "$NPZ_FILE" ]; then
        echo "[$(date '+%H:%M:%S')] 警告：embedding 文件未找到，尝试内联评估" | tee -a $EVAL_LOG
        NPZ_FILE=""
    fi

    # 2. MLP 下游评估
    echo "[$(date '+%H:%M:%S')] [2/3] MLP 下游评估..." | tee -a $EVAL_LOG
    if [ -n "$NPZ_FILE" ]; then
        # 从 npz 文件评估（CPU，快速）
        python scripts/eval/mlp_eval_haidian.py \
            --embedding-file $NPZ_FILE \
            --output-dir $EVAL_OUT/mlp/ \
            2>&1 | tee -a $EVAL_LOG
    else
        # 内联评估（NPU）
        ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval/mlp_eval_haidian.py \
            --config $CONFIG \
            --checkpoint $ckpt \
            --device npu:0 \
            --output-dir $EVAL_OUT/mlp/ \
            2>&1 | tee -a $EVAL_LOG
    fi

    # 3. AUC 变化检测评估
    echo "[$(date '+%H:%M:%S')] [3/3] AUC 变化检测评估..." | tee -a $EVAL_LOG
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval/auc_eval.py \
        --config $CONFIG \
        --checkpoint $ckpt \
        --device npu:0 \
        2>&1 | tee -a $EVAL_LOG

    echo "[$(date '+%H:%M:%S')] === epoch_${epoch} 评估完成 ===" | tee -a $EVAL_LOG
}

# ── 主流程 ─────────────────────────────────────────────────────────────────────

# 1. 等待 epoch 80（训练最终完成）
wait_checkpoint 80

# 2. 等待训练进程结束（NPU 完全释放）
wait_training_done

# 3. 评估 epoch 40（中间检查点）
eval_checkpoint 40

# 4. 评估 epoch 80（最终结果）
eval_checkpoint 80

# 5. 如果存在 best checkpoint，也评估
BEST_CKPT=$(ls $OUTDIR/epoch_best_epoch*.pt 2>/dev/null | sort -t'h' -k3 -n | tail -1)
if [ -n "$BEST_CKPT" ]; then
    echo "[$(date '+%H:%M:%S')] 评估最佳 checkpoint: $BEST_CKPT" | tee -a $EVAL_LOG
    EPOCH_NUM=$(echo $BEST_CKPT | grep -oP 'epoch\K[0-9]+')
    EVAL_OUT="$OUTDIR/eval_best"
    mkdir -p $EVAL_OUT

    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval/auc_eval.py \
        --config $CONFIG \
        --checkpoint $BEST_CKPT \
        --device npu:0 \
        2>&1 | tee -a $EVAL_LOG

    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval/mlp_eval_haidian.py \
        --config $CONFIG \
        --checkpoint $BEST_CKPT \
        --device npu:0 \
        --output-dir $EVAL_OUT/mlp/ \
        2>&1 | tee -a $EVAL_LOG
fi

echo "[$(date '+%H:%M:%S')] 所有评估完成！结果在 $OUTDIR/" | tee -a $EVAL_LOG
