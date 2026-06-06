#!/bin/bash
# ================================================
# 海淀区SAR数据下载启动脚本
# 在 tmux 会话中运行，通过 Watchdog 保证传输稳定
# ================================================

SESSION="sar_download"
CONDA_ENV="xuannv"
WORK_DIR="/workspace/xuannv"

# 如果 tmux 会话已存在则关闭旧的
tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "检测到旧的 $SESSION 会话，正在关闭..."
    tmux kill-session -t "$SESSION"
    sleep 1
}

echo "创建 tmux 会话: $SESSION"
tmux new-session -d -s "$SESSION" -c "$WORK_DIR"

# 激活 conda 并启动 watchdog
tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
sleep 1
tmux send-keys -t "$SESSION" "python3 scripts/download_watchdog.py 2>&1 | tee /workspace/xuannv/data_raw/haidian_sar/watchdog_stdout.log" Enter

echo ""
echo "✅ 下载任务已在 tmux 会话 '$SESSION' 中启动"
echo ""
echo "  查看实时进度:   tmux attach -t $SESSION"
echo "  查看日志:       tail -f /workspace/xuannv/data_raw/haidian_sar/watchdog.log"
echo "  查看文件统计:   watch -n10 'find /workspace/xuannv/data_raw/haidian_sar -name \"*.zip\" | wc -l'"
echo "  查看磁盘占用:   watch -n30 'du -sh /workspace/xuannv/data_raw/haidian_sar/'"
echo "  停止下载:       tmux kill-session -t $SESSION"
