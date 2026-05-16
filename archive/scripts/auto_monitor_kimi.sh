#!/bin/bash
# V12 训练自动监控 — 每10分钟调用 Kimi CLI 检查并改进
# 用法: nohup bash scripts/auto_monitor_kimi.sh &

OUTPUT_DIR="/workspace/outputs/xuannv_v12_clean"
LOG_FILE="$OUTPUT_DIR/auto_monitor_kimi.log"
INSTRUCTIONS="/workspace/xuannv/MONITOR_INSTRUCTIONS.md"

echo "[$(date)] 自动监控启动，间隔: 600秒" >> "$LOG_FILE"
echo "[$(date)] 指令文档: $INSTRUCTIONS" >> "$LOG_FILE"

while true; do
    echo "" >> "$LOG_FILE"
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG_FILE"
    
    # 构建提示词：先读取指令文档，然后附加当前状态
    PROMPT="请严格遵循以下指令文档执行监控任务：

文档路径: $INSTRUCTIONS

请先读取该文档，然后按文档中的步骤执行：
1. 检查训练进程是否存活
2. 读取最新训练指标
3. 检查 NPU 状态
4. 判断训练是否正常
5. 如果正常，汇报状态并继续监控
6. 如果异常，搜索改进方案，修改代码，resume 训练，验证效果

当前项目路径: /workspace/xuannv
输出目录: $OUTPUT_DIR

请用简洁的中文汇报执行结果。"

    # 调用 Kimi CLI，过滤出文本内容
    kimi -p "$PROMPT" -w /workspace/xuannv --print 2>/dev/null | \
        python3 -c "
import sys, re
raw = sys.stdin.read()
m = re.search(r\"text='([^']*(?:\\'[^']*)*)'\", raw, re.DOTALL)
if m:
    text = m.group(1)
    text = text.replace(\"\\'\", \"'\")
    print(text)
else:
    print('[监控] Kimi 未返回有效文本，原始输出:')
    print(raw[:500])
" >> "$LOG_FILE" 2>&1

    echo "" >> "$LOG_FILE"
    echo "等待 600 秒..." >> "$LOG_FILE"
    
    sleep 600
done
