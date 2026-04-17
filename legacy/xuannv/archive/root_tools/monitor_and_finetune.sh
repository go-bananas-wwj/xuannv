#!/bin/bash
# ============================================================
# AEF_qwen 训练监控 + 自动微调脚本
# ============================================================
# 功能:
#   1. 持续监控训练日志，检查训练是否正常运行
#   2. 训练结束后自动评估效果
#   3. 如果效果不佳，自动启动 Phase 2 微调
#   4. 所有结果写入报告文件
# ============================================================

LOG_DIR="/workspace/logs"
TRAIN_LOG="$LOG_DIR/qwen_v1_train_v3.log"
REPORT="$LOG_DIR/qwen_v1_training_report.txt"
AUF_QWEN_DIR="/workspace/xuannv"
OUTPUT_DIR="/workspace/outputs/aef_qwen_v1"

# ──────────────────────────────────────────────
# 配置: 判断训练是否"合格"的阈值
# ──────────────────────────────────────────────
MIN_RAW_UNIF=-3.0      # RawUnif 必须低于此值 (越负越好)
MIN_PRE_UNIF=-3.0      # PreUnif 必须低于此值
MAX_RECON=2.5          # Recon 必须低于此值
MAX_DECOR=25.0         # Decor 可接受上限
MIN_EPOCH=300          # 至少要训练到多少 epoch 才算完成

# Phase 2 微调配置 (如果训练效果不达标)
PHASE2_DECOR_WEIGHT=0.05
PHASE2_LR=0.00001
PHASE2_EPOCHS=100

# ──────────────────────────────────────────────
# 函数: 写报告
# ──────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$REPORT"
}

# ──────────────────────────────────────────────
# 函数: 获取最新 epoch 的指标
# ──────────────────────────────────────────────
get_latest_metrics() {
    local line
    line=$(grep "Epoch [0-9]" "$TRAIN_LOG" 2>/dev/null | grep -v "Traceback" | grep -v "SignalException" | tail -1)

    if [ -z "$line" ]; then
        echo ""
        return
    fi

    local epoch recon runif punif decor var orth
    epoch=$(echo "$line" | grep -oP 'Epoch \K[0-9]+')
    recon=$(echo "$line" | grep -oP 'Recon:\K[0-9.]+')
    runif=$(echo "$line" | grep -oP 'RawUnif:\K[-0-9.]+')
    punif=$(echo "$line" | grep -oP 'PreUnif:\K[-0-9.]+')
    decor=$(echo "$line" | grep -oP 'Decor:\K[0-9.]+')
    var=$(echo "$line" | grep -oP 'Var:\K[0-9.]+')
    orth=$(echo "$line" | grep -oP 'Orth:\K[0-9.]+')

    echo "$epoch $recon $runif $punif $decor $var $orth"
}

# ──────────────────────────────────────────────
# 函数: 检查训练进程是否在运行
# ──────────────────────────────────────────────
is_training_running() {
    ps aux | grep "train_ddp" | grep -v grep | grep -q "qwen_v1"
    return $?
}

# ──────────────────────────────────────────────
# 函数: 训练结束后评估效果
# ──────────────────────────────────────────────
evaluate_training() {
    log "========================================="
    log "训练已结束，开始评估最终效果..."
    log "========================================="

    local metrics
    metrics=$(get_latest_metrics)

    if [ -z "$metrics" ]; then
        log "ERROR: 无法读取最终指标！"
        return 1
    fi

    local epoch recon runif punif decor var orth
    read -r epoch recon runif punif decor var orth <<< "$metrics"

    log ""
    log "--- 最终训练指标 (Epoch $epoch) ---"
    log "  Recon:   $recon  (目标 < $MAX_RECON)"
    log "  RawUnif: $runif  (目标 < $MIN_RAW_UNIF)"
    log "  PreUnif: $punif  (目标 < $MIN_PRE_UNIF)"
    log "  Decor:   $decor  (目标 < $MAX_DECOR)"
    log "  Var:     $var"
    log "  Orth:    $orth"
    log ""

    # ── 判断是否需要微调 ──
    local need_finetune=false
    local reasons=""

    # 检查 epoch 数
    if [ "$epoch" -lt "$MIN_EPOCH" ]; then
        need_finetune=true
        reasons="${reasons}训练未完成 (仅 $epoch/$MIN_EPOCH epoch)\n"
    fi

    # 检查 RawUnif (使用 bash 浮点比较)
    if python3 -c "import sys; sys.exit(0 if float("$runif") > float("$MIN_RAW_UNIF") else 1)"; then
        need_finetune=true
        reasons="${reasons}RawUnif=$runif 不够分散 (目标 < $MIN_RAW_UNIF)\n"
    fi

    # 检查 PreUnif
    if python3 -c "import sys; sys.exit(0 if float("$punif") > float("$MIN_PRE_UNIF") else 1)"; then
        need_finetune=true
        reasons="${reasons}PreUnif=$punif 不够分散 (目标 < $MIN_PRE_UNIF)\n"
    fi

    # 检查 Recon
    if python3 -c "import sys; sys.exit(0 if float("$recon") > float("$MAX_RECON") else 1)"; then
        need_finetune=true
        reasons="${reasons}Recon=$recon 重建质量不足 (目标 < $MAX_RECON)\n"
    fi

    if [ "$need_finetune" = true ]; then
        log "❌ 训练效果不达标，原因:"
        log "$(echo -e "$reasons")"
        return 1
    else
        log "✅ 训练效果达标！无需微调。"
        return 0
    fi
}

# ──────────────────────────────────────────────
# 函数: 启动 Phase 2 微调
# ──────────────────────────────────────────────
start_phase2_finetune() {
    log "========================================="
    log "启动 Phase 2 自动微调..."
    log "========================================="

    # 找到最好的 checkpoint
    local best_ckpt="$OUTPUT_DIR/best.pt"
    if [ ! -f "$best_ckpt" ]; then
        # 如果 best.pt 不存在，用最新的 checkpoint
        best_ckpt=$(ls -t "$OUTPUT_DIR"/epoch_*.pt 2>/dev/null | head -1)
        if [ -z "$best_ckpt" ]; then
            log "ERROR: 找不到任何 checkpoint，无法微调！"
            return 1
        fi
    fi

    log "使用 checkpoint: $best_ckpt"

    # 创建 Phase 2 配置文件
    local phase2_config="$AUF_QWEN_DIR/configs/qwen_v1_phase2_auto.yaml"
    cat > "$phase2_config" << EOF
# Phase 2 自动微调 (由监控脚本生成)
_base_: qwen_v1_scenes.yaml

training:
  decorrelation_weight: $PHASE2_DECOR_WEIGHT   # 提高去相关权重
  lr: $PHASE2_LR                              # 低学习率微调
  epochs: $PHASE2_EPOCHS                       # 微调 epoch 数
  warmup_epochs: 5
  recon_warmup_epochs: 5
  save_every: 20
  checkpoint_interval: 20
  early_stop_patience: 50
  best_balanced_uniform_min: -4.5
  best_balanced_uniform_max: -2.0
EOF

    log "Phase 2 配置: $phase2_config"
    log "  decor_weight: $PHASE2_DECOR_WEIGHT, lr: $PHASE2_LR, epochs: $PHASE2_EPOCHS"

    # 启动微调
    local phase2_log="$LOG_DIR/qwen_v1_phase2_auto.log"
    cd "$AUF_QWEN_DIR"
    setsid bash -c "CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 scripts/train_ddp.py \
        --config configs/qwen_v1_phase2_auto.yaml \
        --resume $best_ckpt \
        --save-every 20 --warmup-epochs 5" \
        > "$phase2_log" 2>&1 &
    local pid=$!
    disown $pid

    log "Phase 2 微调已启动 (PID: $pid, 日志: $phase2_log)"
    log "预计耗时: ~$(($PHASE2_EPOCHS * 70 / 60)) 小时"
    log "========================================="
}

# ──────────────────────────────────────────────
# 主循环: 监控训练
# ──────────────────────────────────────────────
log "========================================="
log "AEF_qwen 训练监控脚本启动"
log "训练日志: $TRAIN_LOG"
log "报告文件: $REPORT"
log "========================================="
log ""

# 初始检查: 训练是否还在运行
if ! is_training_running; then
    log "训练进程未运行，直接评估..."
    evaluate_training
    if [ $? -ne 0 ]; then
        start_phase2_finetune
    fi
    exit 0
fi

last_epoch=0
consecutive_failures=0

while true; do
    # 1. 检查训练是否仍在运行
    if ! is_training_running; then
        log ""
        log "========================================="
        log "训练进程已结束"
        log "========================================="
        break
    fi

    # 2. 获取最新指标
    metrics=$(get_latest_metrics)

    if [ -z "$metrics" ]; then
        consecutive_failures=$((consecutive_failures + 1))
        if [ $consecutive_failures -gt 5 ]; then
            log "WARNING: 连续 5 次无法读取指标，训练可能已卡死"
            break
        fi
        sleep 60
        continue
    fi

    consecutive_failures=0

    # 解析指标
    read -r epoch recon runif punif decor var orth <<< "$metrics"

    # 3. 每 10 个 epoch 打印一次进度
    if [ "$epoch" != "$last_epoch" ]; then
        if [ $((epoch % 10)) -eq 0 ] || [ "$epoch" -lt "$last_epoch" ]; then
            log "Epoch $epoch/400 | Recon=$recon | RawUnif=$runif | PreUnif=$punif | Decor=$decor | Var=$var | Orth=$orth"
        fi
        last_epoch=$epoch
    fi

    # 4. 异常检测: Recon 连续上升 (可能过拟合)
    # 5. 异常检测: RawUnif 变正 (坍缩风险)
    if python3 -c "import sys; sys.exit(0 if float("$runif") > -1.0 else 1)"; then
        log "⚠️ WARNING: RawUnif=$runif 接近零，可能存在坍缩风险！"
    fi

    sleep 60
done

# ──────────────────────────────────────────────
# 训练结束后的最终评估
# ──────────────────────────────────────────────
evaluate_training
eval_result=$?

if [ $eval_result -ne 0 ]; then
    log ""
    log "========================================="
    log "训练效果不达标，启动 Phase 2 自动微调..."
    log "========================================="
    start_phase2_finetune
else
    log ""
    log "========================================="
    log "训练效果良好，无需微调。"
    log "最佳模型: $OUTPUT_DIR/best.pt"
    log "========================================="
fi

log ""
log "监控脚本完成。"
