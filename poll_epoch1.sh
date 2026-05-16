#!/bin/bash
while true; do
    date
    for i in {1..8}; do
        result=$(tmux capture-pane -t "r8s${i}" -p 2>/dev/null | grep "Epoch 1/20" | tail -1)
        if [ -n "$result" ]; then
            echo "🎉 r8s$i: $result"
        else
            echo "r8s$i: still running..."
        fi
    done
    echo "---"
    sleep 300
done
