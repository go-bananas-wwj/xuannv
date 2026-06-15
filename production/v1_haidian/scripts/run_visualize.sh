#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

# 非交互式 shell 中 conda 命令不可用，需要显式初始化。
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate xuannv
fi

# 非交互式 shell 需要显式 source CANN 环境，否则 NPU 算子编译阶段会报
# "No module named 'tbe'" / ACL_PRECISION_MODE 初始化失败。
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python scripts/visualize_haidian.py \
    --model-dir "$PROD_DIR/model" \
    --output-dir "$PROD_DIR/visualizations" \
    --pred-dir "$PROD_DIR/outputs/haidian" \
    --device npu:0
