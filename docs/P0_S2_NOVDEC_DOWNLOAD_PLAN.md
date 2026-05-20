# P0 补全计划：S2 2025年11-12月数据下载

> 生成时间: 2026-05-16
> 目标: 补全哈尔滨 424 patch 的 Sentinel-2 2025年11-12月数据

---

## 一、当前状态确认

### 1.1 训练状态（下载不影响）

| 项目 | 状态 |
|------|------|
| ExpA (skipL2) | NPU 0-3 运行中 ✅ |
| ExpB (noSkipL2) | NPU 4-6 运行中 ✅ |
| NPU 7 | 空闲 ✅ |
| **下载使用资源** | **纯 CPU + 网络，不使用 NPU** |
| 磁盘剩余 | **1.4TB** ✅ 充足 |

**结论: 下载操作不会影响正在进行的训练。**

### 1.2 缺失数据确认

```
S2 2025年11月: 0/424 patch 有数据 ❌
S2 2025年12月: 0/424 patch 有数据 ❌
根因: 下载脚本 DATE_END="2025-10-31" 人为限制
```

---

## 二、下载代码说明

### 2.1 使用脚本

`scripts/preprocessing/download_from_planetary_computer.py`（523行）

**核心机制:**
- 数据源: Microsoft Planetary Computer STAC API
- 下载方式: COG (Cloud Optimized GeoTIFF) 按需裁剪
- 断点续传: ✅ 已存在的 `.tif` 自动跳过
- 自动重试: ✅ 失败时自动重试 3 次
- Token 刷新: ✅ SAS token 过期自动重新签名
- 多线程: ✅ 支持 `--workers` 并行下载

**关键代码片段:**

```python
# 日期范围（默认值）
DATE_START = "2023-01-01"
DATE_END = "2025-10-31"   # ← 这就是缺失11-12月的原因

# S2 集合和波段
COLLECTION_MAP = {"s2": "sentinel-2-l2a"}
ASSET_MAP = {"s2": ["B02", "B03", "B04", "B05", "B06", "B07"]}
RESOLUTION_MAP = {"s2": 10}

# 搜索时自动按月拆分窗口，避免服务器超时
# 下载时自动除以 10000 保持与 GEE 格式一致
```

**断点续传逻辑:**
```python
out_path = patch_dir / f"{date_str}.tif"
if out_path.exists():
    # 检查格式、文件大小、是否全0
    # 正常的文件 → skipped += 1（跳过）
    # 损坏的文件 → 删除重新下载
```

### 2.2 已生成的 Patch 元数据

从现有 424 个 patch 的 TIFF 文件中提取了地理信息：

```
文件: /workspace/raw/harbin_scenes/patches_meta.json
- CRS: EPSG:32652 (UTM Zone 52N)
- Patch size: 1280m (128像素 @ 10m分辨率)
- 范围: 126.51°E ~ 126.72°E, 45.75°N ~ 46.01°N
```

---

## 三、执行步骤

### Step 0: 安装依赖（~5分钟）

```bash
conda activate xuannv
pip install pystac-client planetary-computer stackstac odc-stac -q
```

**注意:** 这些依赖在 `scripts/preprocessing/download_from_planetary_computer.py` 头部有说明。

### Step 1: 执行补下载（~6-8小时）

```bash
cd /workspace/xuannv
conda activate xuannv

# 使用 tmux 后台运行（避免 nohup 问题）
tmux new-session -d -s s2_novdec_download
tmux send-keys -t s2_novdec_download 'cd /workspace/xuannv && conda activate xuannv' Enter
tmux send-keys -t s2_novdec_download 'python scripts/preprocessing/download_from_planetary_computer.py \
    --patches /workspace/raw/harbin_scenes/patches_meta.json \
    --output /workspace/raw/harbin_scenes \
    --sources s2 \
    --date-start 2025-11-01 \
    --date-end 2025-12-31 \
    --workers 4' Enter
```

**参数说明:**
| 参数 | 值 | 说明 |
|------|----|----|
| `--patches` | `/workspace/raw/harbin_scenes/patches_meta.json` | Patch 地理元数据 |
| `--output` | `/workspace/raw/harbin_scenes` | 输出根目录（与现有数据同目录） |
| `--sources` | `s2` | 仅下载 Sentinel-2 |
| `--date-start` | `2025-11-01` | 从11月1日开始 |
| `--date-end` | `2025-12-31` | 到12月31日结束 |
| `--workers` | `4` | 4线程并行 |

**预期输出路径:**
```
/workspace/raw/harbin_scenes/harbin/s2/patch_000000/20251108.tif
/workspace/raw/harbin_scenes/harbin/s2/patch_000000/20251118.tif
...
```

### Step 2: 监控下载进度

```bash
# 查看实时日志
tmux attach -t s2_novdec_download

# 或查看最近输出
tmux capture-pane -t s2_novdec_download -p | tail -30

# 查看已下载文件数
find /workspace/raw/harbin_scenes/harbin/s2 -name "202511*.tif" | wc -l
find /workspace/raw/harbin_scenes/harbin/s2 -name "202512*.tif" | wc -l
```

### Step 3: 云筛选（~30分钟）

新下载的 S2 数据需要经过云筛选，与现有数据保持一致：

```bash
# 注意：这里需要确认 filter_cloudy_frames.py 是否支持增量处理
# 或者重新对所有 S2 数据运行云筛选
python scripts/preprocessing/filter_cloudy_frames.py \
    --input-dir /workspace/raw/harbin_scenes/harbin/s2 \
    --output-dir /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2 \
    --max-per-month 2 \
    --cloud-threshold 0.3 \
    --workers 16
```

**需要确认的问题:**
- `filter_cloudy_frames.py` 是否会覆盖现有云筛选数据？
- 还是需要只对 11-12 月的新数据运行筛选？

### Step 4: 重新计算统计数据（~10分钟）

```bash
python scripts/preprocessing/compute_statistics.py \
    --data-dir /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered \
    --output-dir /workspace/statistics/harbin_scenes \
    --sources s2
```

### Step 5: 删除旧缓存（~1分钟）

```bash
# 删除训练缓存，强制重新加载
cd /workspace/outputs
find . -name "dataset_cache_*.pt" -delete 2>/dev/null
```

---

## 四、预计数据量

| 项目 | 估算 |
|------|------|
| 每 patch 每月帧数 | 2-4 帧（Sentinel-2 5天重访） |
| 总帧数 | 424 × 2月 × ~3帧 = **~2,500帧** |
| 每帧大小 | ~1.2MB (128×128×6 band float32) |
| 原始数据总量 | **~3GB** |
| 云筛选后 | **~1GB**（保留每月最clear的2帧） |
| 磁盘剩余 | 1.4TB → **充足** |

---

## 五、风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| Planetary Computer 下载配额/限流 | 中 | 速度变慢 | 已设置自动重试 + token刷新 |
| 网络中断 | 中 | 下载中断 | tmux 保持会话，断点续传 |
| 11-12月哈尔滨积雪/云覆盖高 | 高 | 有效帧少 | 这是正常的冬季数据特征 |
| 依赖安装失败 | 低 | 无法下载 | 使用 conda 环境，已有代理 |
| 磁盘空间不足 | 极低 | 下载失败 | 剩余 1.4TB，远超 3GB 需求 |

---

## 六、时间线

```
T+0h    : 用户确认，开始安装依赖
T+0.5h  : 启动 tmux 下载会话
T+0.5~8h: 后台下载（不影响训练）
T+8h    : 下载完成，开始云筛选
T+8.5h  : 重新计算统计，删除缓存
T+9h    : 完成，训练下次启动时自动加载新数据
```

---

## 七、回滚方案

如果下载出现问题，可以随时停止：

```bash
# 停止下载
tmux kill-session -t s2_novdec_download

# 删除已下载的部分数据（可选）
find /workspace/raw/harbin_scenes/harbin/s2 -name "202511*.tif" -delete
find /workspace/raw/harbin_scenes/harbin/s2 -name "202512*.tif" -delete
```

**注意:** 断点续传机制意味着即使中断，已下载的文件不会丢失，下次运行会自动跳过。

---

## 八、待确认事项

在执行前需要你确认：

1. **云筛选策略**: 新下载的 11-12 月数据是否需要运行 `filter_cloudy_frames.py`？还是直接复制到 `harbin_scenes_cloud_filtered/s2`？
2. **训练重启**: 数据补全后是否需要立即重启训练？还是等当前 epoch 结束？
3. **P1/P2 任务**: 是否同步验证 S1 7月 和 Landsat 3月/7月 的补全可行性？
