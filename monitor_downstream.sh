#!/bin/bash
while true; do
    date
    for i in {1..8}; do
        log="/workspace/outputs/round8_single_exp${i}/eval_downstream.log"
        if [ -f "$log" ]; then
            last_line=$(tail -1 "$log" 2>/dev/null)
            echo "exp$i: $last_line"
        else
            echo "exp$i: no log"
        fi
    done
    echo "---"
    sleep 60
done
