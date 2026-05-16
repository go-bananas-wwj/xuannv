#!/bin/bash
LOG="/workspace/outputs/xuannv_round2/round2_cross_temporal/train.log"
echo "Waiting for training to complete..."
while true; do
    if grep -q "Training complete." "$LOG" 2>/dev/null; then
        echo "Training complete!"
        exit 0
    fi
    sleep 30
done
