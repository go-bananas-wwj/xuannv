#!/bin/bash
# V7 Embedding 维度消融实验自动化脚本
# 运行顺序: 128(基线) → 64 → 32 → 16 → 8
# 建议分批次运行，避免占用全部 8 卡超过一周

set -e

DIMS=(128 64 32 16 8)
CONFIG_DIR="configs/ablation"
OUTPUT_BASE="/workspace/outputs"
WANDB_PROJECT="xuannv-backbone-ablation"

# 检查 NPU 可用
npu-smi info | head -20
echo "=========================================="
echo "Embedding 维度消融实验"
echo "对比组: ${DIMS[@]}"
echo "开始时间: $(date)"
echo "=========================================="

for dim in "${DIMS[@]}"; do
    EXP_NAME="v7_ablation_dim${dim}"
    CONFIG="${CONFIG_DIR}/v7_dim${dim}.yaml"
    OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    
    echo ""
    echo "=========================================="
    echo "[$(date)] 开始训练: ${dim}-dim"
    echo "Config: ${CONFIG}"
    echo "Output: ${OUTPUT_DIR}"
    echo "=========================================="
    
    # 检查是否已有完成的 checkpoint
    BEST_CKPT=$(ls ${OUTPUT_DIR}/epoch_best_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$BEST_CKPT" ]; then
        echo "[SKIP] ${dim}-dim 已有 checkpoint: ${BEST_CKPT}"
    else
        # 8 卡 DDP 训练
        ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            torchrun --nproc_per_node=8 \
            scripts/train/train_ddp_v7.py \
            --config ${CONFIG} \
            --save-every 20 \
            --wandb-project ${WANDB_PROJECT} \
            --wandb-run-name ${EXP_NAME}
    fi
    
    # 找到最佳 checkpoint
    BEST_CKPT=$(ls ${OUTPUT_DIR}/epoch_best_*.pt 2>/dev/null | head -1 || true)
    if [ -z "$BEST_CKPT" ]; then
        echo "[WARN] ${dim}-dim 未找到最佳 checkpoint，跳过评估"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "[$(date)] 开始评估: ${dim}-dim"
    echo "Checkpoint: ${BEST_CKPT}"
    echo "=========================================="
    
    # 1. 变化检测 AUC
    echo "[Eval] Change Detection AUC..."
    python scripts/eval/validate_v7_level1_bare.py \
        --checkpoint ${BEST_CKPT} \
        --config ${CONFIG} \
        2>&1 | tee ${OUTPUT_DIR}/eval_cd_auc.log
    
    # 2. Embedding 空间质量分析
    if [ -f "scripts/eval/analyze_v7_embedding_quality.py" ]; then
        echo "[Eval] Embedding Quality Analysis..."
        python scripts/eval/analyze_v7_embedding_quality.py \
            --checkpoint ${BEST_CKPT} \
            2>&1 | tee ${OUTPUT_DIR}/eval_embedding_quality.log
    fi
    
    echo ""
    echo "[$(date)] ${dim}-dim 完成"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "全部消融实验完成!"
echo "结束时间: $(date)"
echo "=========================================="

# 汇总结果
echo ""
echo "===== 结果汇总 ====="
for dim in "${DIMS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/v7_ablation_dim${dim}"
    LOG="${OUTPUT_DIR}/eval_cd_auc.log"
    if [ -f "$LOG" ]; then
        AUC=$(grep -oP 'AUC[:=]\s*\K[0-9.]+' "$LOG" | head -1 || echo "N/A")
        echo "${dim}-dim: AUC=${AUC}"
    else
        echo "${dim}-dim: 评估未完成"
    fi
done
