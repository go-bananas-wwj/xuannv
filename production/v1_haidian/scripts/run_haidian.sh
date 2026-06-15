#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python -m xuannv_v1.haidian_tasks \
    --model-dir "$PROD_DIR/model" \
    --label-dir "/workspace/xuannv/haidian_label/labeljson" \
    --output-dir "$PROD_DIR/outputs/haidian" \
    --device npu:0 \
    --mode bitemporal \
    --classifier linear
