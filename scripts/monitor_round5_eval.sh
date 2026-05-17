#!/bin/bash
# Round 5 下游评估监控脚本

REPORT_FILE="/workspace/outputs/round5/eval_monitor_report.md"
mkdir -p /workspace/outputs/round5

declare -a EXP_NAMES=(
    "round5_consist_mild"
    "round5_no_consist"
    "round5_kappa_baseline"
    "round5_temporal_plus_recon"
)

while true; do
    ALL_DONE=true
    STATUS=""
    
    for EXP in "${EXP_NAMES[@]}"; do
        OUT_DIR="/workspace/outputs/round5/${EXP}/evaluation"
        
        # 检查各步骤是否完成
        EXTRACT_DONE="⏳"
        KNN_DONE="⏳"
        MLP_DONE="⏳"
        CD_DONE="⏳"
        SEM_DONE="⏳"
        
        if [[ -f "${OUT_DIR}/extract.log" ]]; then
            if grep -q "EXTRACT DONE\|已保存\|patch_embeddings.npz" "${OUT_DIR}/extract.log" 2>/dev/null; then
                EXTRACT_DONE="✅"
            elif grep -q "EXTRACT FAILED\|Error\|Traceback" "${OUT_DIR}/extract.log" 2>/dev/null; then
                EXTRACT_DONE="❌"
                ALL_DONE=false
            else
                EXTRACT_DONE="🔄"
                ALL_DONE=false
            fi
        else
            ALL_DONE=false
        fi
        
        if [[ -f "${OUT_DIR}/knn.log" ]]; then
            if grep -q "KNN DONE" "${OUT_DIR}/knn.log" 2>/dev/null; then
                KNN_DONE="✅"
            elif grep -q "KNN FAILED\|Error\|Traceback" "${OUT_DIR}/knn.log" 2>/dev/null; then
                KNN_DONE="❌"
            else
                KNN_DONE="🔄"
                ALL_DONE=false
            fi
        else
            ALL_DONE=false
        fi
        
        if [[ -f "${OUT_DIR}/mlp.log" ]]; then
            if grep -q "MLP DONE" "${OUT_DIR}/mlp.log" 2>/dev/null; then
                MLP_DONE="✅"
            elif grep -q "MLP FAILED\|Error\|Traceback" "${OUT_DIR}/mlp.log" 2>/dev/null; then
                MLP_DONE="❌"
            else
                MLP_DONE="🔄"
                ALL_DONE=false
            fi
        else
            ALL_DONE=false
        fi
        
        if [[ -f "${OUT_DIR}/cd.log" ]]; then
            if grep -q "CD DONE" "${OUT_DIR}/cd.log" 2>/dev/null; then
                CD_DONE="✅"
            elif grep -q "CD FAILED\|Error\|Traceback" "${OUT_DIR}/cd.log" 2>/dev/null; then
                CD_DONE="❌"
            else
                CD_DONE="🔄"
                ALL_DONE=false
            fi
        else
            ALL_DONE=false
        fi
        
        if [[ -f "${OUT_DIR}/semantic.log" ]]; then
            if grep -q "SEMANTIC DONE\|结果已保存" "${OUT_DIR}/semantic.log" 2>/dev/null; then
                SEM_DONE="✅"
            elif grep -q "SEMANTIC FAILED\|Error\|Traceback" "${OUT_DIR}/semantic.log" 2>/dev/null; then
                SEM_DONE="❌"
            else
                SEM_DONE="🔄"
                ALL_DONE=false
            fi
        else
            ALL_DONE=false
        fi
        
        STATUS="${STATUS}| ${EXP} | ${EXTRACT_DONE} | ${KNN_DONE} | ${MLP_DONE} | ${CD_DONE} | ${SEM_DONE} |\n"
    done
    
    # 生成报告
    cat > "$REPORT_FILE" << EOF
# Round 5 下游评估监控

更新时间: $(date '+%Y-%m-%d %H:%M:%S')

| 实验 | Extract | KNN | MLP | CD | Semantic |
|------|---------|-----|-----|----|----------|
$(echo -e "$STATUS")
EOF
    
    if $ALL_DONE; then
        echo "$(date '+%H:%M:%S') ALL EVALUATION COMPLETE!" >> "$REPORT_FILE"
        break
    fi
    
    sleep 30
done

echo "$(date '+%H:%M:%S') 监控结束，所有评估完成。" >> "$REPORT_FILE"
