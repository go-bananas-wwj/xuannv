#!/bin/bash
# Round7 E50 完成后一键运行全面分析
# 用法: bash scripts/eval/run_round7_full_analysis.sh

set -e

OUTPUT_BASE="/workspace/outputs"
ANALYSIS_DIR="${OUTPUT_BASE}/round7_analysis_$(date +%m%d_%H%M)"
mkdir -p "$ANALYSIS_DIR"

echo "========================================"
echo " Round7 E50 全面分析"
echo " 输出目录: $ANALYSIS_DIR"
echo "========================================"

eval "$(conda shell.bash hook)"
conda activate xuannv
cd /workspace/xuannv

# ── Step 1: AUC 验证 ──
echo ""
echo "【Step 1/2】AUC 批量验证 (pre-norm + l2-norm)..."
echo "  预计时间: 8实验 × 5分钟 ≈ 40 分钟"
python scripts/eval/batch_auc_validate.py \
    --experiments 1,2,3,4,5,6,7,8 \
    --n-samples 200 \
    --device npu:0 \
    --output "${ANALYSIS_DIR}/round7_auc_results.json"

echo "  ✅ AUC 验证完成"

# ── Step 2: Embedding 全面分析 ──
echo ""
echo "【Step 2/2】Embedding 全面分析..."
echo "  预计时间: 8实验 × 10分钟 ≈ 80 分钟 (CPU)"
python scripts/eval/analyze_embeddings_comprehensive.py \
    --experiments 1,2,3,4,5,6,7,8 \
    --n-batches 100 \
    --device npu:0 \
    --output "${ANALYSIS_DIR}/round7_embedding_analysis.json"

echo "  ✅ Embedding 分析完成"

# ── 汇总 ──
echo ""
echo "========================================"
echo " 分析完成！"
echo "========================================"
echo ""
echo "输出文件:"
echo "  AUC 报告:      ${ANALYSIS_DIR}/round7_auc_results.json"
echo "  Embedding 报告: ${ANALYSIS_DIR}/round7_embedding_analysis.json"
echo ""
echo "查看命令:"
echo "  cat ${ANALYSIS_DIR}/round7_auc_results.json | python -m json.tool | less"
echo "  cat ${ANALYSIS_DIR}/round7_embedding_analysis.json | python -m json.tool | less"
