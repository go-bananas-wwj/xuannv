#!/bin/bash
# V12 训练自动监控 — 每10分钟调用 Kimi CLI 检查状态
# 用法: nohup bash scripts/auto_monitor_v12.sh &

OUTPUT_DIR="/workspace/outputs/xuannv_v12_clean"
LOG_FILE="$OUTPUT_DIR/auto_monitor.log"
CONFIG="/workspace/xuannv/configs/xuannv_v12_clean.yaml"
MONITOR_STATUS="/workspace/xuannv/monitor_status.txt"

echo "[$(date)] 自动监控启动，间隔: 600秒" >> "$LOG_FILE"

while true; do
    echo "" >> "$LOG_FILE"
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG_FILE"
    
    # 准备提示词
    PROMPT="你是V12训练项目的AI监控助手。请读取以下文件并汇报训练状态：

1. /workspace/xuannv/monitor_status.txt — 最新的训练进程和指标
2. /workspace/outputs/xuannv_v12_clean/train.log — 训练日志

**Uniformity 指标说明（关键）**：
- uniform 值域 [0, 1]，0=embedding完美分散（好），1=完全坍缩（坏）
- uniform < 0.3: 🟢 优秀
- uniform 0.3~0.6: 🟡 及格
- uniform 0.6~0.8: 🟠 轻度坍缩
- uniform 0.8~0.95: 🔴 中度坍缩
- uniform > 0.95: 🚨 严重坍缩

请用简洁的中文汇报：
1. 最新 Epoch/Step 的指标（recon/consist/uniform/lr）
2. Uniform 状态判断（用上述等级）
3. 与前一次相比的趋势变化
4. 如果有异常（uniform > 0.95 或进程退出），请指出"

    # 调用 Kimi CLI（非交互式）
    # 用 grep 过滤出 TextPart 的文本内容
    kimi -p "$PROMPT" -w /workspace/xuannv --print 2>/dev/null | \
        python3 -c "
import sys, re
raw = sys.stdin.read()
# 提取 TextPart 中的 text 字段
m = re.search(r\"text='([^']*(?:\\'[^']*)*)'\", raw, re.DOTALL)
if m:
    text = m.group(1)
    # 转义单引号
    text = text.replace(\"\\'\", \"'\")
    print(text)
else:
    print('[监控] Kimi 调用未返回有效文本')
" >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "等待 600 秒..." >> "$LOG_FILE"
    
    sleep 600
done
