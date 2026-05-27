#!/bin/bash
# MLP 下游分类消融实验 — 串行批量运行
set -e

cd /workspace/xuannv

EMB="/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation/embeddings/patch_embeddings.npz"
OUT="/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation/mlp_ablation"
mkdir -p "$OUT"

export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

PYTHON="/root/miniconda3/envs/xuannv/bin/python"
SCRIPT="scripts/eval/evaluate_mlp_v2.py"

# 实验配置: (名称, 额外参数...)
run_exp() {
    local name="$1"
    shift
    local exp_out="$OUT/$name"
    local log="$OUT/${name}.log"
    
    echo ""
    echo "============================================================"
    echo "[Experiment] $name"
    echo "============================================================"
    echo "Output: $exp_out"
    echo "Log: $log"
    
    $PYTHON $SCRIPT \
        --embedding-file "$EMB" \
        --output-dir "$exp_out" \
        --device npu:0 \
        --epochs 30 \
        "$@" \
        2>&1 | tee "$log"
    
    echo "✅ $name 完成"
}

# 1. Baseline (原始配置，用于对比)
run_exp "baseline" \
    --head-type mlp

# 2. + Class Weight (逆频率加权)
run_exp "class_weight" \
    --head-type mlp \
    --use-class-weight

# 3. + Class Weight + Focal Loss
run_exp "class_weight_focal" \
    --head-type mlp \
    --use-class-weight \
    --use-focal

# 4. MLPv2 (更深的 MLP) + Class Weight
run_exp "mlpv2" \
    --head-type mlpv2 \
    --hidden-dim 512 \
    --use-class-weight

echo ""
echo "============================================================"
echo "全部完成! 输出目录: $OUT"
echo "============================================================"

# 汇总
for d in baseline class_weight class_weight_focal mlpv2; do
    echo ""
    echo "--- $d ---"
    cat "$OUT/$d/mlp_summary.json" 2>/dev/null || echo "  无结果"
done
