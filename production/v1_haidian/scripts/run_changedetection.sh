#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python -m xuannv_v1.changedetection \
    --model-dir "$PROD_DIR/model" \
    --output-dir "$PROD_DIR/outputs/changedetection" \
    --device npu:0 \
    --periods june,aug,September,October \
    --annot-dir "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件" \
    --grid "/workspace/index/harbin/grid/harbin_grid.geojson"
