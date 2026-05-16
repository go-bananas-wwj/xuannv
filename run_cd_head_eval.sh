#!/bin/bash
# 并行运行 Round 8 八组实验的 CD Head 评估

cd /workspace/xuannv
PYTHON="/root/miniconda3/envs/xuannv/bin/python3"

for i in {1..8}; do
    npu_idx=$((i-1))
    config="configs/round8_single_exp${i}.yaml"
    checkpoint="/workspace/outputs/round8_single_exp${i}/epoch_19.pt"
    log="/workspace/outputs/round8_single_exp${i}/eval_cdhead.log"
    
    echo "启动 exp${i} CD Head 评估 (NPU ${npu_idx})..."
    ASCEND_RT_VISIBLE_DEVICES=${npu_idx} "${PYTHON}" scripts/eval/train_cd_head_v8.py \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --device "npu:0" \
        --epochs 100 \
        --lr 1e-3 \
        --folds 5 \
        > "${log}" 2>&1 &
    
    pids[$i]=$!
done

echo "所有 8 个 CD Head 评估已启动，PID: ${pids[@]}"
echo "等待完成（预计 20-40 分钟）..."
wait

echo "全部 CD Head 评估完成！"

# 提取结果
echo ""
echo "========================================"
echo "CD Head AUC 结果汇总"
echo "========================================"
for i in {1..8}; do
    log="/workspace/outputs/round8_single_exp${i}/eval_cdhead.log"
    if [ -f "$log" ]; then
        auc=$(grep "Mean AUC" "$log" 2>/dev/null | tail -1)
        if [ -n "$auc" ]; then
            echo "exp${i}: $auc"
        else
            echo "exp${i}: (结果解析中...)"
            grep -E "AUC|mean|median" "$log" 2>/dev/null | tail -5
        fi
    else
        echo "exp${i}: 日志文件不存在"
    fi
    echo "---"
done
