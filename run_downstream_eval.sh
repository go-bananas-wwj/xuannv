#!/bin/bash
# 并行运行 Round 8 八组实验的综合下游任务评估

cd /workspace/xuannv
PYTHON="/root/miniconda3/envs/xuannv/bin/python3"

for i in {1..8}; do
    npu_idx=$((i-1))
    config="configs/round8_single_exp${i}.yaml"
    checkpoint="/workspace/outputs/round8_single_exp${i}/epoch_19.pt"
    output="/workspace/outputs/round8_single_exp${i}/downstream_results.json"
    log="/workspace/outputs/round8_single_exp${i}/eval_downstream.log"
    
    echo "启动 exp${i} 下游任务评估 (NPU ${npu_idx})..."
    ASCEND_RT_VISIBLE_DEVICES=${npu_idx} "${PYTHON}" scripts/eval/comprehensive_downstream_eval.py \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --device "npu:0" \
        --output "${output}" \
        --folds 5 \
        > "${log}" 2>&1 &
    
    pids[$i]=$!
done

echo "所有 8 个下游任务评估已启动，PID: ${pids[@]}"
echo "等待完成（预计 20-40 分钟）..."
wait

echo "全部下游任务评估完成！"

# 汇总结果
echo ""
echo "========================================"
echo "下游任务评估结果汇总"
echo "========================================"
PYTHON="/root/miniconda3/envs/xuannv/bin/python3"
"${PYTHON}" << 'PYEOF'
import json

print(f"{'Exp':<5} {'WorldCover':<20} {'DynamicWorld':<20} {'JRC_Water':<30} {'OSM_Buildings':<30}")
print("-" * 105)

for i in range(1, 9):
    path = f"/workspace/outputs/round8_single_exp{i}/downstream_results.json"
    try:
        with open(path) as f:
            d = json.load(f)
        
        wc = d.get('worldcover', {})
        dw = d.get('dynamic_world', {})
        jrc = d.get('jrc_water', {})
        osm = d.get('osm_buildings', {})
        
        wc_str = f"mIoU={wc.get('miou', 'N/A'):.3f}" if 'miou' in wc else str(wc.get('error', 'N/A'))
        dw_str = f"mIoU={dw.get('miou', 'N/A'):.3f}" if 'miou' in dw else str(dw.get('error', 'N/A'))
        jrc_str = f"IoU={jrc.get('iou', 'N/A'):.3f},F1={jrc.get('f1', 'N/A'):.3f}" if 'iou' in jrc else str(jrc.get('error', 'N/A'))
        osm_str = f"IoU={osm.get('iou', 'N/A'):.3f},F1={osm.get('f1', 'N/A'):.3f}" if 'iou' in osm else str(osm.get('error', 'N/A'))
        
        print(f"exp{i:<4} {wc_str:<20} {dw_str:<20} {jrc_str:<30} {osm_str:<30}")
    except Exception as e:
        print(f"exp{i:<4} (错误: {e})")
PYEOF
