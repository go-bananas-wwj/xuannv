#!/bin/bash
# MLP 下游分类消融实验 — 对比不同 Head/损失组合

set -e

cd /workspace/xuannv

EMB="/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation/embeddings/patch_embeddings.npz"
OUT="/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation/mlp_ablation"
mkdir -p "$OUT"

PYTHON="/root/miniconda3/envs/xuannv/bin/python"
SCRIPT="scripts/eval/evaluate_mlp_v2.py"

echo "======================================"
echo "MLP 下游分类消融实验"
echo "======================================"

# 1. Baseline (原始配置)
echo ""
echo "[1/5] Baseline (MLP, no class weight)"
$PYTHON $SCRIPT \
    --embedding-file "$EMB" \
    --output-dir "$OUT/baseline" \
    --device cpu \
    --epochs 50 \
    --head-type mlp \
    > "$OUT/01_baseline.log" 2>&1
echo "  Done: $(cat $OUT/baseline/mlp_summary.json 2>/dev/null | grep accuracy | head -3)"

# 2. + Class Weight
echo ""
echo "[2/5] + Class Weight"
$PYTHON $SCRIPT \
    --embedding-file "$EMB" \
    --output-dir "$OUT/class_weight" \
    --device cpu \
    --epochs 50 \
    --head-type mlp \
    --use-class-weight \
    > "$OUT/02_class_weight.log" 2>&1
echo "  Done: $(cat $OUT/class_weight/mlp_summary.json 2>/dev/null | grep accuracy | head -3)"

# 3. + Class Weight + Focal
echo ""
echo "[3/5] + Class Weight + Focal Loss"
$PYTHON $SCRIPT \
    --embedding-file "$EMB" \
    --output-dir "$OUT/class_weight_focal" \
    --device cpu \
    --epochs 50 \
    --head-type mlp \
    --use-class-weight \
    --use-focal \
    > "$OUT/03_class_weight_focal.log" 2>&1
echo "  Done: $(cat $OUT/class_weight_focal/mlp_summary.json 2>/dev/null | grep accuracy | head -3)"

# 4. MLPv2 (deeper)
echo ""
echo "[4/5] MLPv2 (Deeper MLP) + Class Weight"
$PYTHON $SCRIPT \
    --embedding-file "$EMB" \
    --output-dir "$OUT/mlpv2" \
    --device cpu \
    --epochs 50 \
    --head-type mlpv2 \
    --hidden-dim 512 \
    --use-class-weight \
    > "$OUT/04_mlpv2.log" 2>&1
echo "  Done: $(cat $OUT/mlpv2/mlp_summary.json 2>/dev/null | grep accuracy | head -3)"

# 5. JRC Water 单独验证（修复后）
echo ""
echo "[5/5] JRC Water 修复验证"
$PYTHON $SCRIPT \
    --embedding-file "$EMB" \
    --output-dir "$OUT/jrc_fixed" \
    --device cpu \
    --epochs 50 \
    --head-type mlp \
    --use-class-weight \
    > "$OUT/05_jrc_fixed.log" 2>&1
echo "  Done: $(cat $OUT/jrc_fixed/mlp_summary.json 2>/dev/null | grep accuracy | head -3)"

echo ""
echo "======================================"
echo "汇总结果"
echo "======================================"

for d in baseline class_weight class_weight_focal mlpv2 jrc_fixed; do
    echo ""
    echo "--- $d ---"
    cat "$OUT/$d/mlp_summary.json" 2>/dev/null || echo "  无结果"
done

echo ""
echo "全部完成! 输出目录: $OUT"
