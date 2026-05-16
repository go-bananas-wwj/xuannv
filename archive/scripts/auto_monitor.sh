#!/bin/bash
LOG=/workspace/xuannv/auto_monitor.log
while true; do
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> $LOG
    for exp in v13_exp1 v13_exp2 v13_exp3 v13_exp4; do
        echo "--- $exp ---" >> $LOG
        tmux capture-pane -t $exp -p -S -100 2>/dev/null | grep -E "Epoch [0-9]+|Step [0-9]+" | tail -3 >> $LOG
    done
    echo "" >> $LOG
    sleep 300
done
