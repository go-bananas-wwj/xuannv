#!/bin/bash
LOG=/workspace/xuannv/exp5_monitor.log
while true; do
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> $LOG
    echo "--- v13_exp5 ---" >> $LOG
    tmux capture-pane -t v13_exp5 -p -S -100 2>/dev/null | grep -E "Epoch [0-9]+|Step [0-9]+" | tail -3 >> $LOG
    echo "" >> $LOG
    sleep 300
done
