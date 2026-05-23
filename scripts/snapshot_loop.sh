#!/bin/bash
# 每10分钟抓取训练快照
OUTDIR="/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/snapshots"
mkdir -p "$OUTDIR"
while true; do
    TS=$(date +%Y%m%d_%H%M%S)
    tmux capture-pane -t expD_train -p -J 2>/dev/null | grep "\[Step" | tail -5 > "$OUTDIR/snap_${TS}.txt"
    sleep 600
done
