#!/bin/bash
# 监控所有V13实验的进度
cd /workspace/xuannv

LOGFILE=/workspace/xuannv/experiment_progress.log
echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> $LOGFILE

for exp in v13_exp1 v13_exp2 v13_exp3 v13_exp4; do
    echo "--- $exp ---" >> $LOGFILE
    tmux capture-pane -t $exp -p 2>/dev/null | tail -5 >> $LOGFILE
    echo "" >> $LOGFILE
done

echo "" >> $LOGFILE
