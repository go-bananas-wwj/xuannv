#!/bin/bash
# S2 11-12月下载监控 + 完成后自动处理

LOG="/workspace/xuannv/logs/s2_novdec_pipeline.log"
mkdir -p "$(dirname "$LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== 监控启动 ==="
log "目标: 424 patch 的 S2 2025年11-12月数据"

# 循环检查进度
while true; do
    # 统计进度
    python3 << 'PYEOF'
import os
import sys

base = '/workspace/raw/harbin_scenes/s2'
patches = sorted([d for d in os.listdir(base) if d.startswith('patch_')])
nov_frames = 0
dec_frames = 0
both_patches = 0

for p in patches:
    files = os.listdir(os.path.join(base, p))
    has_nov = any(f.startswith('202511') for f in files)
    has_dec = any(f.startswith('202512') for f in files)
    nov_frames += sum(1 for f in files if f.startswith('202511'))
    dec_frames += sum(1 for f in files if f.startswith('202512'))
    if has_nov and has_dec: both_patches += 1

print(f"{both_patches}|{nov_frames}|{dec_frames}")
PYEOF
    > /tmp/s2_progress.txt

    progress=$(cat /tmp/s2_progress.txt)
    both=$(echo "$progress" | cut -d'|' -f1)
    nov=$(echo "$progress" | cut -d'|' -f2)
    dec=$(echo "$progress" | cut -d'|' -f3)
    
    log "进度: ${both}/424 patch 完成 | 11月: ${nov}帧 | 12月: ${dec}帧"
    
    # 检查是否完成
    if [ "$both" -ge 424 ]; then
        log "=== 下载完成！开始后续处理 ==="
        break
    fi
    
    # 检查下载进程是否还在运行
    if ! pgrep -f "download_from_planetary_computer.py.*2025-12-31" > /dev/null; then
        log "警告: 下载进程已退出，检查是否完成..."
        if [ "$both" -lt 400 ]; then
            log "错误: 下载异常中断，仅完成 ${both}/424"
            exit 1
        fi
        break
    fi
    
    # 每3分钟检查一次
    sleep 180
done

# ===== 后续处理 =====
log "=== Step 1: 验证下载完整性 ==="
python3 << 'PYEOF'
import os
from collections import Counter

base = "/workspace/raw/harbin_scenes/s2"
patches = sorted([d for d in os.listdir(base) if d.startswith("patch_")])
nov_counts = []
dec_counts = []

for p in patches:
    files = os.listdir(os.path.join(base, p))
    nov = sum(1 for f in files if f.startswith("202511"))
    dec = sum(1 for f in files if f.startswith("202512"))
    nov_counts.append(nov)
    dec_counts.append(dec)

print(f"Patch总数: {len(patches)}")
print(f"11月: {sum(nov_counts)}帧, 分布: {dict(Counter(nov_counts))}")
print(f"12月: {sum(dec_counts)}帧, 分布: {dict(Counter(dec_counts))}")
PYEOF

log "=== Step 2: 增量云筛选（仅处理11-12月新数据）==="
cd /workspace/xuannv
conda activate xuannv

# 为11-12月数据单独运行云筛选
python3 << 'PYEOF'
import os
import sys
import shutil
from pathlib import Path

# 创建临时目录
TMP_RAW = "/tmp/s2_novdec_raw"
TMP_FILTERED = "/tmp/s2_novdec_filtered"
DST = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2"

os.makedirs(TMP_RAW, exist_ok=True)
os.makedirs(TMP_FILTERED, exist_ok=True)

# 复制11-12月原始数据到临时目录
base = "/workspace/raw/harbin_scenes/s2"
patches = sorted([d for d in os.listdir(base) if d.startswith("patch_")])
copied = 0

for p in patches:
    src_dir = os.path.join(base, p)
    tmp_dir = os.path.join(TMP_RAW, p)
    os.makedirs(tmp_dir, exist_ok=True)
    
    for f in os.listdir(src_dir):
        if f.startswith("202511") or f.startswith("202512"):
            src = os.path.join(src_dir, f)
            dst = os.path.join(tmp_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

print(f"复制了 {copied} 个文件到临时目录")
PYEOF

# 运行云筛选
python scripts/preprocessing/filter_cloudy_frames.py \
    --input-dir /tmp/s2_novdec_raw \
    --output-dir /tmp/s2_novdec_filtered \
    --max-per-month 2 \
    --cloud-threshold 0.3 \
    --workers 16

log "=== Step 3: 合并云筛选结果 ==="
python3 << 'PYEOF'
import os
import shutil

SRC = "/tmp/s2_novdec_filtered"
DST = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2"

copied = 0
for p in os.listdir(SRC):
    if not p.startswith("patch_"): continue
    src_dir = os.path.join(SRC, p)
    dst_dir = os.path.join(DST, p)
    os.makedirs(dst_dir, exist_ok=True)
    
    for f in os.listdir(src_dir):
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

print(f"合并完成: 复制了 {copied} 个文件到 cloud_filtered")
PYEOF

log "=== Step 4: 重新计算 S2 统计数据 ==="
python scripts/preprocessing/compute_statistics.py \
    --data-dir /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered \
    --output-dir /workspace/statistics/harbin_scenes \
    --sources s2

log "=== Step 5: 清理训练缓存 ==="
find /workspace/outputs -name "dataset_cache_*.pt" -delete 2>/dev/null
log "已删除所有 dataset_cache"

log "=== Step 6: 清理临时文件 ==="
rm -rf /tmp/s2_novdec_raw /tmp/s2_novdec_filtered

log "=== 全部完成！==="
log "新数据已就绪，训练下次启动时将自动加载"

# 最终统计
python3 << 'PYEOF'
import os
base = "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2"
patches = sorted([d for d in os.listdir(base) if d.startswith("patch_")])
total = sum(len(os.listdir(os.path.join(base, p))) for p in patches)
nov = sum(1 for p in patches for f in os.listdir(os.path.join(base, p)) if f.startswith("202511"))
dec = sum(1 for p in patches for f in os.listdir(os.path.join(base, p)) if f.startswith("202512"))
print(f"\n最终统计:")
print(f"  总Patch: {len(patches)}")
print(f"  云筛选后总帧数: {total}")
print(f"  11月帧数: {nov}")
print(f"  12月帧数: {dec}")
PYEOF

log "=== 流水线结束 ==="
