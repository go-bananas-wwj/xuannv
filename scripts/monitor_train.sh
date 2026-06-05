#!/bin/bash
# 训练监控脚本 - 检查 tmux 训练会话状态
SESSION="train_v14"
echo "=== $(date) ==="
echo "--- tmux 状态 ---"
tmux has-session -t $SESSION 2>/dev/null && echo "tmux session $SESSION: 运行中" || echo "tmux session $SESSION: 不存在"
echo ""
echo "--- 最近10行日志 ---"
tail -n 10 /workspace/outputs/exp_v14_anti_collapse_0605/train_*.log 2>/dev/null | tail -20
echo ""
echo "--- NPU 状态 ---"
npu-smi info | grep -E "NPU [0-7]|AICore|Process id"
echo ""
echo "--- Checkpoint 文件 ---"
ls -lt /workspace/outputs/exp_v14_anti_collapse_0605/epoch_*.pt 2>/dev/null | head -5
echo ""
