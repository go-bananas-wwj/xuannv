#!/bin/bash
cd /workspace/xuannv
PYTHON="/root/miniconda3/envs/xuannv/bin/python3"

for i in {1..8}; do
    npu_idx=$((i-1))
    config="configs/round8_single_exp${i}.yaml"
    checkpoint="/workspace/outputs/round8_single_exp${i}/epoch_19.pt"
    output="/workspace/outputs/round8_single_exp${i}/downstream_results.json"
    log="/workspace/outputs/round8_single_exp${i}/eval_downstream_v2.log"
    
    echo "启动 exp${i} 下游评估 V2 (NPU ${npu_idx})..."
    ASCEND_RT_VISIBLE_DEVICES=${npu_idx} "${PYTHON}" scripts/eval/comprehensive_downstream_eval_v2.py \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --device "npu:0" \
        --output "${output}" \
        --folds 3 \
        > "${log}" 2>&1 &
    
    pids[$i]=$!
done

echo "所有 8 个评估已启动，PID: ${pids[@]}"
echo "等待完成..."
wait

echo "全部完成！"

echo ""
echo "========================================"
echo "下游任务结果汇总"
echo "========================================"
"${PYTHON}" << 'PYEOF'
import json
print(f"{'Exp':<5} {'WorldCover':<22} {'DynamicWorld':<22} {'JRC_Water':<28} {'OSM_Buildings':<28}")
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
        wc_s = f"mIoU={wc.get('miou', 'N/A'):.3f}" if 'miou' in wc else str(wc.get('error', 'N/A'))
        dw_s = f"mIoU={dw.get('miou', 'N/A'):.3f}" if 'miou' in dw else str(dw.get('error', 'N/A'))
        jrc_s = f"IoU={jrc.get('iou', 'N/A'):.3f},F1={jrc.get('f1', 'N/A'):.3f}" if 'iou' in jrc else str(jrc.get('error', 'N/A'))
        osm_s = f"IoU={osm.get('iou', 'N/A'):.3f},F1={osm.get('f1', 'N/A'):.3f}" if 'iou' in osm else str(osm.get('error', 'N/A'))
        print(f"exp{i:<4} {wc_s:<22} {dw_s:<22} {jrc_s:<28} {osm_s:<28}")
    except Exception as e:
        print(f"exp{i:<4} (错误: {e})")
PYEOF
