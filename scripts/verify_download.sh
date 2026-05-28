#!/bin/bash
# 下载完成后校验脚本：对比网盘 vs 本地文件数
# 用法: bash scripts/verify_download.sh

DEST="/workspace/raw/haidian_sar"
REPORT="$DEST/verify_report.txt"

echo "========================================"  | tee "$REPORT"
echo "下载完整性校验 - $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$REPORT"
echo "========================================" | tee -a "$REPORT"

# 本地统计
echo "" | tee -a "$REPORT"
echo "[本地文件统计]" | tee -a "$REPORT"
for DIR in "北京朝阳角反" "北京市门头沟区大台-干涉" "中国北京市点位1_干涉" "中国北京市点位2_干涉"; do
    LOCAL_COUNT=$(find "$DEST/$DIR" -name "*.zip" 2>/dev/null | wc -l)
    LOCAL_SIZE=$(du -sh "$DEST/$DIR" 2>/dev/null | cut -f1)
    echo "  $DIR: $LOCAL_COUNT 个ZIP, $LOCAL_SIZE" | tee -a "$REPORT"
done

TOTAL_LOCAL=$(find "$DEST" -name "*.zip" | wc -l)
TOTAL_SIZE=$(du -sh "$DEST" | cut -f1)
echo "" | tee -a "$REPORT"
echo "  合计: $TOTAL_LOCAL 个ZIP, $TOTAL_SIZE" | tee -a "$REPORT"

# 预期数量 (网盘基线)
echo "" | tee -a "$REPORT"
echo "[预期文件数（基于网盘结构）]" | tee -a "$REPORT"
echo "  北京朝阳角反:      80 子目录 × 2 = 160 个ZIP" | tee -a "$REPORT"
echo "  门头沟大台-干涉:    5 子目录 × 4 = 20  个ZIP" | tee -a "$REPORT"
echo "  点位1_干涉:        44 子目录 × 4 = 176 个ZIP" | tee -a "$REPORT"
echo "  点位2_干涉:        40 子目录 × 4 = 160 个ZIP" | tee -a "$REPORT"
echo "  预期合计:                           516 个ZIP" | tee -a "$REPORT"

echo "" | tee -a "$REPORT"
echo "[结论]" | tee -a "$REPORT"
if [ "$TOTAL_LOCAL" -ge 510 ]; then
    echo "  ✅ 下载完整 ($TOTAL_LOCAL/~516)" | tee -a "$REPORT"
elif [ "$TOTAL_LOCAL" -gt 0 ]; then
    echo "  ⚠️  下载不完整 ($TOTAL_LOCAL/~516)，仍在下载中或有文件丢失" | tee -a "$REPORT"
else
    echo "  ❌ 本地无文件" | tee -a "$REPORT"
fi
echo "========================================"  | tee -a "$REPORT"
