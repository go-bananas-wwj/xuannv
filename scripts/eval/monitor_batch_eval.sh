#!/bin/bash
# 监控批量评估进度

while true; do
    echo "=== $(date) ==="
    for i in {1..8}; do
        session="eval_round8_single_exp${i}"
        if tmux has-session -t "$session" 2>/dev/null; then
            # 检查是否还在运行
            pane_output=$(tmux capture-pane -t "$session" -p 2>/dev/null | tail -5)
            if echo "$pane_output" | grep -q "评估完成"; then
                echo "  [$i] DONE"
            else
                progress=$(echo "$pane_output" | grep -o '[0-9]*/[0-9]*' | tail -1)
                echo "  [$i] RUNNING  progress: $progress"
            fi
        else
            echo "  [$i] SESSION NOT FOUND"
        fi
    done
    echo ""
    sleep 60
done
