#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate xuannv
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python scripts/compare_heads.py \
    --model-dir "$PROD_DIR/model" \
    --label-dir "/workspace/xuannv/haidian_label/labeljson" \
    --output-dir "$PROD_DIR/outputs/head_ablation" \
    --device npu:0 \
    --mode bitemporal \
    --heads linear,mlp_torch,unet
