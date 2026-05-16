#!/bin/bash
LOG="/workspace/outputs/mini_batch_monitor.log"
echo "=== Mini Batch Monitor Started: $(date) ===" > "$LOG"

while true; do
  all_done=true
  report="\n=== $(date '+%H:%M:%S') ===\n"
  
  for name in mb_exp1_baseline mb_exp2_spatial mb_exp3_no_skip_l2 mb_exp4_high_uniform mb_exp5_high_recon mb_exp6_high_variance mb_exp7_no_teacher mb_exp8_combined; do
    output=$(tmux capture-pane -t "$name" -p 2>/dev/null)
    epoch_info=$(echo "$output" | grep -E "Epoch [0-9]+/" | tail -1)
    dots=$(echo "$output" | grep -o '\.' | wc -l)
    
    if [ -n "$epoch_info" ]; then
      report+="$name: $epoch_info | dots=$dots\n"
      # Check if epoch >= 5
      current_epoch=$(echo "$epoch_info" | grep -oP 'Epoch \K[0-9]+')
      if [ -n "$current_epoch" ] && [ "$current_epoch" -lt 5 ]; then
        all_done=false
      fi
    else
      report+="$name: epoch1 in progress (dots=$dots)\n"
      all_done=false
    fi
  done
  
  echo -e "$report" >> "$LOG"
  
  if [ "$all_done" = true ]; then
    echo -e "\n*** ALL EXPERIMENTS REACHED EPOCH 5 ***" >> "$LOG"
    echo -e "$report" | tee /workspace/outputs/mini_batch_epoch5_report.txt
    break
  fi
  
  sleep 120
done
