#!/bin/bash
# 监控实验epoch进度，每20分钟记录一次
LOG=/workspace/xuannv/epoch_monitor.log
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1200
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> $LOG
    for exp in v13_exp1_spatial_unif v13_exp2_high_weight v13_exp3_vicreg_fix v13_exp4_combined; do
        echo "--- $exp ---" >> $LOG
        tail -1 /workspace/outputs/$exp/train.log 2>/dev/null >> $LOG
    done
    echo "" >> $LOG
done
