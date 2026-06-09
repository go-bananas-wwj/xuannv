#!/bin/bash
# 监控训练完成并生成可视化
OUTPUT_DIR="/workspace/xuannv/aef_reference/outputs/aef_distill_control_exp"
CHECKPOINT="${OUTPUT_DIR}/step_000200_seed42.pt"
VIZ_SCRIPT="/workspace/xuannv/aef_reference/viz_control.py"

while [ ! -f "$CHECKPOINT" ]; do
    sleep 30
    echo "Waiting for ${CHECKPOINT}..."
done

echo "Checkpoint found! Generating visualization..."
cd /workspace/xuannv/aef_reference
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xuannv
ASCEND_RT_VISIBLE_DEVICES=0 python "$VIZ_SCRIPT"
echo "Done!"
