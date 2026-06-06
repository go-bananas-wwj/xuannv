# P0 执行计划：S2 2025年11-12月数据补下载

> 状态: 待执行（等你确认）
> 前提: 所有依赖已安装，网络可达，不影响训练

---

## 一、前置调研结论（已完成）

### 1.1 缺失确认

| 检查项 | 结果 |
|--------|------|
| S2 2025年11月原始数据 | **0/424 patch** ❌ |
| S2 2025年12月原始数据 | **0/424 patch** ❌ |
| 根因 | `DATE_END="2025-10-31"` 人为日期限制 |
| 是否云筛选导致 | **否**（原始数据就不存在） |

### 1.2 环境确认

| 检查项 | 结果 |
|--------|------|
| 目标环境 | `conda activate xuannv` |
| pystac_client | 0.9.0 ✅ |
| planetary_computer | 已安装 ✅ |
| stackstac | 已安装 ✅ |
| odc-stac | 已安装 ✅ |
| Planetary Computer API | HTTP 200 ✅ |
| 代理 | `127.0.0.1:7890` ✅ |
| 磁盘空间 | 1.4TB 剩余 ✅ |

### 1.3 训练状态确认

| 训练任务 | NPU | 状态 |
|----------|-----|------|
| ExpA (skipL2) | 0-3 | 运行中 |
| ExpB (noSkipL2) | 4-6 | 运行中 |
| **下载占用** | **无** | **纯 CPU + 网络** |

**结论: 下载不会与训练竞争任何资源。**

### 1.4 已生成的元数据

文件: `/workspace/raw/harbin_scenes/patches_meta.json`
```json
{
  "city": "",
  "city_name": "哈尔滨",
  "n_patches": 424,
  "patch_size_m": 1280,
  "crs": "EPSG:32652",
  "patches": [
    {"id": 0, "utm_bounds": [...], "center_lonlat": [126.524, 45.750]},
    ...
  ]
}
```

**注意:** `city=""` 已手动设置，确保下载路径为 `harbin_scenes/s2/patch_000000/`（与现有数据结构一致）。

---

## 二、下载脚本技术说明

### 2.1 脚本位置

`scripts/preprocessing/download_from_planetary_computer.py`（523行）

### 2.2 核心流程

```
patches_meta.json
    ↓
[ThreadPoolExecutor] × N workers
    ↓
对每个 patch:
  1. STAC API 搜索（按月拆分窗口，避免超时）
  2. 过滤 Landsat-7（仅 Landsat）
  3. 对每个 item:
     a. 检查本地是否已存在（断点续传）
     b. 重新签名 SAS token
     c. stackstac/odc-stac 下载并裁剪
     d. 保存为 GeoTIFF
  4. 返回统计信息
```

### 2.3 断点续传机制

```python
out_path = patch_dir / f"{date_str}.tif"
if out_path.exists():
    # 正常文件 → 跳过
    # 损坏/全0/旧格式 → 删除重新下载
    skipped += 1
    continue
```

**这意味着:**
- 中断后可随时重启，已下载的文件不会重复下载
- 本次只下载 11-12 月数据，现有 1-10 月数据完全不受影响

### 2.4 输出格式

| 属性 | 值 |
|------|----|
| 分辨率 | 10m |
| 尺寸 | 128×128 像素 |
| 波段 | B02, B03, B04, B05, B06, B07 |
| 数据类型 | float32（已除以 10000） |
| 投影 | EPSG:32652 |
| 压缩 | LZW |

**与现有 GEE 下载的数据格式完全一致。**

---

## 三、执行步骤（精确命令）

### Step 1: 创建 tmux 会话（0分钟）

```bash
tmux new-session -d -s s2_novdec_download -c /workspace/xuannv
```

### Step 2: 启动下载（0分钟）

```bash
tmux send-keys -t s2_novdec_download 'conda activate xuannv' Enter
tmux send-keys -t s2_novdec_download 'python scripts/preprocessing/download_from_planetary_computer.py \
    --patches /workspace/raw/harbin_scenes/patches_meta.json \
    --output /workspace/raw/harbin_scenes \
    --sources s2 \
    --date-start 2025-11-01 \
    --date-end 2025-12-31 \
    --workers 4' Enter
```

**参数解析:**

| 参数 | 值 | 说明 |
|------|----|----|
| `--patches` | `/workspace/raw/harbin_scenes/patches_meta.json` | 424 patch 的地理元数据 |
| `--output` | `/workspace/raw/harbin_scenes` | 输出根目录 |
| `--sources` | `s2` | 仅 Sentinel-2 |
| `--date-start` | `2025-11-01` | 包含11月全部 |
| `--date-end` | `2025-12-31` | 包含12月全部 |
| `--workers` | `4` | 4线程并行（受 PC 限流，再高无益） |

### Step 3: 监控（持续进行）

```bash
# 方式1: 实时 attach
tmux attach -t s2_novdec_download
# 按 Ctrl+B 然后 D 退出（保持后台运行）

# 方式2: 查看最新输出（不 attach）
tmux capture-pane -t s2_novdec_download -p | tail -50

# 方式3: 统计已下载文件数
watch -n 60 'echo "=== 2025年11月 ==="; find /workspace/raw/harbin_scenes/s2 -name "202511*.tif" | wc -l; echo "=== 2025年12月 ==="; find /workspace/raw/harbin_scenes/s2 -name "202512*.tif" | wc -l'
```

**预期输出示例:**
```
城市:  (哈尔滨)
总 patch 数: 424, 本次处理: 424
数据源: ['s2']
时间范围: 2025-11-01 ~ 2025-12-31
S2 divide(10000): True
并行 workers: 4
============================================================
[10/424] patch_000009 | s2:3/3 | elapsed=2.1m eta=85.3m
[20/424] patch_000019 | s2:2/2 | elapsed=4.3m eta=82.1m
...
```

### Step 4: 验证下载完整性（下载完成后）

```bash
# 统计每个 patch 的 11-12 月帧数
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
print(f"\n2025年11月:")
print(f"  有数据的patch: {sum(1 for c in nov_counts if c > 0)}/{len(patches)}")
print(f"  总帧数: {sum(nov_counts)}")
print(f"  每patch帧数分布: {Counter(nov_counts)}")

print(f"\n2025年12月:")
print(f"  有数据的patch: {sum(1 for c in dec_counts if c > 0)}/{len(patches)}")
print(f"  总帧数: {sum(dec_counts)}")
print(f"  每patch帧数分布: {Counter(dec_counts)}")
PYEOF
```

**预期结果:**
```
Patch总数: 424
2025年11月:
  有数据的patch: 424/424
  总帧数: ~1200
  每patch帧数分布: Counter({2: 300, 3: 100, 4: 24})

2025年12月:
  有数据的patch: 424/424
  总帧数: ~1200
  每patch帧数分布: Counter({2: 350, 3: 60, 4: 14})
```

---

## 四、下载后处理步骤

### Phase A: 云筛选（~30分钟）

新下载的 S2 原始数据需要经过与现有数据一致的云筛选。

**方案选择（需要你决定）:**

**方案A1 — 全量重新云筛选（推荐）**
```bash
# 删除旧云筛选结果
rm -rf /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2

# 重新运行全量云筛选
python scripts/preprocessing/filter_cloudy_frames.py \
    --input-dir /workspace/raw/harbin_scenes/s2 \
    --output-dir /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2 \
    --max-per-month 2 \
    --cloud-threshold 0.3 \
    --workers 16
```
- **优点**: 统一处理，结果一致
- **缺点**: 需重新处理全部 29,707→9,321 帧，耗时较长

**方案A2 — 仅筛选新数据（快速）**
```bash
# 对新下载的 11-12 月数据单独筛选
mkdir -p /tmp/s2_novdec_raw
for p in /workspace/raw/harbin_scenes/s2/patch_*; do
    pid=$(basename $p)
    mkdir -p /tmp/s2_novdec_raw/$pid
    cp $p/202511*.tif /tmp/s2_novdec_raw/$pid/ 2>/dev/null
    cp $p/202512*.tif /tmp/s2_novdec_raw/$pid/ 2>/dev/null
done

python scripts/preprocessing/filter_cloudy_frames.py \
    --input-dir /tmp/s2_novdec_raw \
    --output-dir /tmp/s2_novdec_filtered \
    --max-per-month 2 \
    --cloud-threshold 0.3 \
    --workers 16

# 将筛选结果复制到目标目录
for p in /tmp/s2_novdec_filtered/patch_*; do
    pid=$(basename $p)
    mkdir -p /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2/$pid
    cp $p/*.tif /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2/$pid/
done

# 清理临时目录
rm -rf /tmp/s2_novdec_raw /tmp/s2_novdec_filtered
```
- **优点**: 只处理新数据，速度快
- **缺点**: 需要确认筛选逻辑和现有数据一致

### Phase B: 重新计算统计（~10分钟）

```bash
python scripts/preprocessing/compute_statistics.py \
    --data-dir /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered \
    --output-dir /workspace/statistics/harbin \
    --sources s2
```

### Phase C: 清理训练缓存（~1分钟）

```bash
# 删除所有 dataset_cache，强制训练重新加载
find /workspace/outputs -name "dataset_cache_*.pt" -delete 2>/dev/null
```

### Phase D: 更新 scene_index.json（可选）

```bash
# 如有需要，更新 scene_index.json 包含新的 11-12 月日期
python3 << 'PYEOF'
import json
import os

base = "/workspace/raw/harbin_scenes"
with open(f"{base}/scene_index.json") as f:
    index = json.load(f)

# 收集 S2 所有日期
all_dates = set()
for p in os.listdir(f"{base}/s2"):
    if not p.startswith("patch_"): continue
    for f in os.listdir(f"{base}/s2/{p}"):
        if f.endswith(".tif"):
            all_dates.add(f.replace(".tif", ""))

index["s2"] = sorted(list(all_dates))
with open(f"{base}/scene_index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"Updated scene_index.json: S2 now has {len(index['s2'])} dates")
PYEOF
```

---

## 五、完整时间线

```
T+0h    用户确认，立即执行
        ├── 创建 tmux 会话
        ├── 启动下载（后台运行）
        └── 开始监控

T+0~8h  下载进行中（后台，不影响训练）
        ├── 每 10 个 patch 打印进度
        ├── 可随时查看 tmux 日志
        └── 断网/中断后可自动续传

T+8h    下载完成，执行后处理
        ├── [方案A1/A2] 云筛选（~30分钟）
        ├── 重新计算统计（~10分钟）
        ├── 清理缓存（~1分钟）
        └── 更新索引（~1分钟）

T+8.75h 全部完成
        └── 训练下次启动时自动加载新数据
```

---

## 六、风险与回滚

### 6.1 风险矩阵

| 风险 | 可能性 | 影响 | 自动缓解 |
|------|--------|------|---------|
| PC 限流/403 | 中 | 速度变慢 | ✅ 自动重试 + token刷新 |
| 网络中断 | 中 | 暂时停止 | ✅ tmux 保持，断点续传 |
| 冬季云覆盖高 | 高 | 有效帧少 | ⚠️ 正常，数据质量由云筛选处理 |
| 磁盘满 | 极低 | 下载失败 | ✅ 1.4TB >> 3GB |

### 6.2 回滚方案

```bash
# 随时停止下载
tmux kill-session -t s2_novdec_download

# 删除本次下载的 11-12 月数据（不影响现有数据）
find /workspace/raw/harbin_scenes/s2 -name "202511*.tif" -delete
find /workspace/raw/harbin_scenes/s2 -name "202512*.tif" -delete

# 如果已做云筛选，删除对应的云筛选结果
find /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2 -name "202511*.tif" -delete
find /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2 -name "202512*.tif" -delete
```

---

## 七、待你确认的事项

### 必须确认（执行前）

1. **是否执行 P0 下载？** → 确认后我立即启动

### 可选确认（下载完成后）

2. **云筛选方案**:
   - **方案A1**: 全量重新筛选（更统一，耗时更长）
   - **方案A2**: 仅筛选新数据（更快，需验证一致性）
   - 你倾向哪个？

3. **P1 验证**: 是否同时验证 S1 2025年7月在 PC 上的可用性？（5分钟，无风险）

4. **训练重启**: 数据补全后是否立即重启训练？还是等当前 epoch 自然结束？

---

## 八、执行命令汇总（一键复制）

```bash
# === 创建并启动下载会话 ===
tmux new-session -d -s s2_novdec_download -c /workspace/xuannv
tmux send-keys -t s2_novdec_download 'conda activate xuannv' Enter
tmux send-keys -t s2_novdec_download 'python scripts/preprocessing/download_from_planetary_computer.py \
    --patches /workspace/raw/harbin_scenes/patches_meta.json \
    --output /workspace/raw/harbin_scenes \
    --sources s2 \
    --date-start 2025-11-01 \
    --date-end 2025-12-31 \
    --workers 4' Enter

# === 监控 ===
tmux capture-pane -t s2_novdec_download -p | tail -30

# === 查看进度统计 ===
watch -n 60 'find /workspace/raw/harbin_scenes/s2 -name "202511*.tif" | wc -l; find /workspace/raw/harbin_scenes/s2 -name "202512*.tif" | wc -l'
```
