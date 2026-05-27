#!/bin/bash
# Round7 方案 B: 4实验 Few-Shot 批量评估
# 实验: exp2(low_recon), exp3(no_teacher), exp5(l2_ctrl), exp7(strong_orth)

set -e

EXPS="2 3 5 7"
OUTPUT_BASE="/workspace/outputs/round7_downstream_eval"
mkdir -p "$OUTPUT_BASE"

# Bare AUC 已在后台运行，这里跳过
# 只跑 CD Few-Shot 和 LandCover Few-Shot

for EXP in $EXPS; do
    echo ""
    echo "========================================"
    echo " 实验 exp$EXP"
    echo "========================================"

    # 1. 变化检测 Few-Shot
    echo "[1/2] 变化检测 Few-Shot..."
    python scripts/eval/fewshot_change_detection.py \
        --experiment $EXP \
        --k-shots 1,5,10,20 \
        --n-splits 5 \
        --device npu:0 \
        --output "$OUTPUT_BASE/exp${EXP}_cd_fewshot.json"

    # 2. 土地利用 Few-Shot
    echo "[2/2] 土地利用 Few-Shot..."
    python scripts/eval/fewshot_landcover.py \
        --experiment $EXP \
        --k-shots 5,20,50,100 \
        --pixels-per-patch 500 \
        --n-splits 3 \
        --device npu:0 \
        --output "$OUTPUT_BASE/exp${EXP}_landcover_fewshot.json"

done

echo ""
echo "========================================"
echo " 方案 B 评估完成！"
echo "========================================"
echo ""
echo "输出文件:"
for EXP in $EXPS; do
    echo "  exp${EXP}_cd_fewshot.json"
    echo "  exp${EXP}_landcover_fewshot.json"
done
