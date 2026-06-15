#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate xuannv
fi

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python scripts/visualize_head_comparison.py \
    --pred-dir "$PROD_DIR/outputs/merged_construction_ablation" \
    --output-dir "$PROD_DIR/visualizations/merged_construction" \
    --scene-dir "/workspace/xuannv/data_raw/beijing/planetscene" \
    --cache "$PROD_DIR/outputs/head_ablation/.cache/embeddings.npz" \
    --heads linear,mlp_torch,unet \
    --task shigongjiandu
