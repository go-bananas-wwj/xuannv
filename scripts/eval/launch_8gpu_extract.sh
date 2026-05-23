#!/bin/bash
# 8卡并行提取 ExpE embedding

cd /workspace/xuannv
OUTPUT_DIR="/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/eval/embeddings_all_months"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*

CONFIG="configs/xuannv_v2_expE_pure_recon.yaml"
CHECKPOINT="/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/epoch_best_epoch52.pt"
PYTHON="/root/miniconda3/envs/xuannv/bin/python"
WORKER="scripts/eval/extract_expE_worker.py"

echo "=== 启动 8 GPU 并行提取 ==="
echo "输出目录: $OUTPUT_DIR"

for gpu in $(seq 0 7); do
    PATCHES=$(cat /tmp/gpu${gpu}_patches.txt)
    LOG="/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/eval/extract_gpu${gpu}.log"
    
    ASCEND_RT_VISIBLE_DEVICES=$gpu \
        PYTHONUNBUFFERED=1 \
        $PYTHON $WORKER \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --device npu:0 \
        --batch-size 4 \
        --output-dir $OUTPUT_DIR \
        --patches "$PATCHES" \
        > $LOG 2>&1 &
    
    echo "GPU $gpu: PID=$! patches=${gpu}00..${gpu}52 log=$LOG"
done

echo ""
echo "=== 所有 GPU 已启动 ==="
echo "监控命令: tail -f /workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/eval/extract_gpu*.log"
