#!/bin/bash
# 继续提取 Round 1-3 剩余实验的 embedding
set -e
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv

export ASCEND_RT_VISIBLE_DEVICES=0

# 剩余实验: 配置|checkpoint|输出目录
declare -a EXPERIMENTS=(
    "configs/aef_skip_l2.yaml|/workspace/outputs/xuannv_round1/aef_skip_l2/epoch_best_epoch17.pt|/workspace/outputs/xuannv_round1/aef_skip_l2/evaluation_v2/embeddings"
    "configs/aef_no_uniform.yaml|/workspace/outputs/xuannv_round1/aef_no_uniform/epoch_best_epoch18.pt|/workspace/outputs/xuannv_round1/aef_no_uniform/evaluation_v2/embeddings"
    "configs/round2_cross_temporal.yaml|/workspace/outputs/xuannv_round2/round2_cross_temporal/epoch_best_epoch42.pt|/workspace/outputs/xuannv_round2/round2_cross_temporal/evaluation_v2/embeddings"
    "configs/round3_vicreg_temporal.yaml|/workspace/outputs/xuannv_round2/round3_vicreg_temporal/epoch_best_epoch48.pt|/workspace/outputs/xuannv_round2/round3_vicreg_temporal/evaluation_v2/embeddings"
)

total=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r config checkpoint output_dir <<< "${EXPERIMENTS[$i]}"
    name=$(echo "$output_dir" | grep -oP '(?<=/outputs/)[^/]+/[^/]+' | head -1)
    echo ""
    echo "========================================"
    echo "[$((i+1))/$total] Extracting: $name"
    echo "Config: $config"
    echo "Checkpoint: $checkpoint"
    echo "Output: $output_dir"
    echo "========================================"
    
    mkdir -p "$output_dir"
    
    python scripts/eval/extract_embeddings_v2.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --output-dir "$output_dir" \
        --device npu:0 \
        --batch-size 16 \
        --save-every 500
    
    echo "[$((i+1))/$total] DONE: $name"
done

echo ""
echo "========================================"
echo "All extractions completed!"
echo "========================================"
