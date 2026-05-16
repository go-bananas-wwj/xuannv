#!/bin/bash
# 并行运行 Round 8 八组实验的 AUC 评估
# 注意：ASCEND_RT_VISIBLE_DEVICES 将物理 NPU 映射为逻辑 npu:0

cd /workspace/xuannv
PYTHON="/root/miniconda3/envs/xuannv/bin/python3"

for i in {1..8}; do
    npu_idx=$((i-1))
    config="configs/round8_single_exp${i}.yaml"
    checkpoint="/workspace/outputs/round8_single_exp${i}/epoch_19.pt"
    output="/workspace/outputs/round8_single_exp${i}/auc_result.json"
    log="/workspace/outputs/round8_single_exp${i}/eval_auc.log"
    
    echo "启动 exp${i} 评估 (物理 NPU ${npu_idx}, 逻辑 npu:0)..."
    ASCEND_RT_VISIBLE_DEVICES=${npu_idx} "${PYTHON}" scripts/eval/validate_v12_auc.py \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --device "npu:0" \
        --output "${output}" \
        > "${log}" 2>&1 &
    
    pids[$i]=$!
done

echo "所有 8 个评估任务已启动，PID: ${pids[@]}"
echo "等待完成（预计 15-30 分钟）..."
wait

echo "全部评估完成！"

# 汇总结果
echo ""
echo "========================================"
echo "AUC 结果汇总"
echo "========================================"
for i in {1..8}; do
    output="/workspace/outputs/round8_single_exp${i}/auc_result.json"
    if [ -f "$output" ]; then
        auc_mean=$("${PYTHON}" -c "import json; d=json.load(open('$output')); print(f'{d.get(\"auc_mean\", \"N/A\")}')")
        auc_median=$("${PYTHON}" -c "import json; d=json.load(open('$output')); print(f'{d.get(\"auc_median\", \"N/A\")}')")
        n_patches=$("${PYTHON}" -c "import json; d=json.load(open('$output')); print(d.get(\"n_patches\", 0))")
        printf "exp%2d: mean=%s | median=%s | patches=%s\n" "$i" "$auc_mean" "$auc_median" "$n_patches"
    else
        echo "exp${i}: 结果文件未生成"
    fi
done
