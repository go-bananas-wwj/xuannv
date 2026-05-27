#!/bin/bash
# 并行启动 8 个 KNN 下游评估，每实验一张 NPU
cd /workspace/xuannv
EXPS=(aef_baseline aef_high_consist aef_no_static aef_skip_l2 aef_diff_recon aef_high_kappa aef_cyclic_unif aef_no_uniform)
for i in {0..7}; do
  exp="${EXPS[$i]}"
  log="/workspace/outputs/xuannv_round1/${exp}_knn_v2.log"
  echo "[$(date)] Launch $exp on NPU $i -> $log"
  (
    export ASCEND_RT_VISIBLE_DEVICES=$i
    /root/miniconda3/envs/xuannv/bin/python scripts/eval/eval_downstream_knn_v2.py --experiment "$exp" --device npu:0 > "$log" 2>&1
    echo "[$(date)] $exp DONE"
  ) &
done
wait
echo "[$(date)] All 8 experiments completed"
