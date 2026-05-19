#!/bin/bash
LOG_A="/workspace/outputs/xuannv_v2_expA_skipL2/train.log"
LOG_B="/workspace/outputs/xuannv_v2_expB_noSkipL2/train.log"
REPORT="/workspace/outputs/xuannv_v2_epoch2_report.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始监控前2个 epoch..." > "$REPORT"

while true; do
    epoch_a=$(grep -c "Epoch 001" "$LOG_A" 2>/dev/null || echo 0)
    epoch_b=$(grep -c "Epoch 001" "$LOG_B" 2>/dev/null || echo 0)
    
    if [ "$epoch_a" -gt 0 ] && [ "$epoch_b" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Epoch 1 完成!" >> "$REPORT"
        grep "Epoch 001" "$LOG_A" >> "$REPORT"
        grep "Epoch 001" "$LOG_B" >> "$REPORT"
    fi
    
    epoch2_a=$(grep -c "Epoch 002" "$LOG_A" 2>/dev/null || echo 0)
    epoch2_b=$(grep -c "Epoch 002" "$LOG_B" 2>/dev/null || echo 0)
    
    if [ "$epoch2_a" -gt 0 ] && [ "$epoch2_b" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Epoch 2 完成!" >> "$REPORT"
        grep "Epoch 002" "$LOG_A" >> "$REPORT"
        grep "Epoch 002" "$LOG_B" >> "$REPORT"
        break
    fi
    
    steps_a=$(grep -c "\[Step" "$LOG_A" 2>/dev/null || echo 0)
    steps_b=$(grep -c "\[Step" "$LOG_B" 2>/dev/null || echo 0)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待中... ExpA steps=$steps_a ExpB steps=$steps_b" >> "$REPORT"
    
    sleep 60
done
