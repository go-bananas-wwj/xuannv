#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python -m xuannv_v1.worldcover_knn \
    --model-dir "$PROD_DIR/model" \
    --label-dir "/workspace/xuannv/data_raw/haidian/scenes" \
    --output-dir "$PROD_DIR/outputs/worldcover" \
    --device npu:0 \
    --k 5
