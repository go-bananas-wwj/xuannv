# Phase 2 数据重新下载 — 改进计划 V13

## 背景

用户误删了 `/workspace/raw/phase2_heilongjiang/` 下齐齐哈尔、大庆、海淀三个城市的全部下载数据（S2/S1/Landsat/DEM/WorldCover）。需要从零重新开始下载。

结合前两轮下载遇到的问题，制定本计划，目标是一次性下载到位，不再反复重启。

---

## 前两轮问题总结

| 问题 | 影响 | 根因 |
|------|------|------|
| **SAS token 48分钟过期** | 后半程 patches 大量 403/"not recognized" 失败 | `pc.sign()` TTL 仅 0.8h，遍历 400p 需 2-3h |
| **STAC 搜索超时** | 部分 patches 的 items 完全找不到 | PC API `maximum allowed time` 响应慢 |
| **全0数据文件** | 无效文件被保存，后续被跳过 | stackstac 偶发返回全0数组，未检测 |
| **串行处理5个源** | S2 卡住导致 Landsat/DEM 无法开始 | 单进程串行处理所有 sources |
| **workers=4 并发过高** | 可能触发 API 限流 | ThreadPoolExecutor 4 workers |
| **进程结束不自动恢复** | 需人工反复重启 | 看门狗只检查，不自动创建 session |
| **批量删除旧数据** | 用户误操作清空目录 | 无备份/保护机制 |

---

## 改进方案（分三层）

### 第一层：脚本核心改进（`download_from_planetary_computer.py`）

#### 1.1 自动重签名 + 403 自动重试

**问题**：`pc.sign()` TTL 仅 48 分钟。

**方案**：
- 在 `download_source_stackstac()` / `download_source_odcstac()` 中，**捕获 `RasterioIOError` / HTTP 403**
- 如果遇到 403 或 "not recognized"，**自动重新调用 `pc.sign(item)` 获取新 token，然后重试**
- 最多重试 3 次，每次重试前都重新签名

```python
# 伪代码
for retry in range(3):
    try:
        item = pc.sign(item)  # 每次重试都重新签名
        data = download_item(item)
        break
    except (RasterioIOError, HTTPError) as e:
        if retry < 2:
            time.sleep(2)
            continue
        raise
```

#### 1.2 STAC 搜索超时自动重试

**问题**：`The request exceeded the maximum allowed time`。

**方案**：
- `search_items()` 添加 `@retry` 装饰器
- 超时后等待 5-10 秒重试，最多 3 次
- 如果最终仍超时，记录该 patch+source 为 "skipped"，不崩溃

#### 1.3 全0数据检测（已修复，保留）

- 下载后检查 `np.all(data_np == 0)`
- 全0则抛出异常，不保存文件

#### 1.4 降低并发 workers

**问题**：workers=4 可能导致 API 限流 + token 竞争。

**方案**：
- 降低到 **workers=2**，减少并发压力
- 单 worker 处理速度足够（网络 I/O 是瓶颈）

#### 1.5 每个 source 独立进程（并行化）

**问题**：单进程串行处理 S2→S1→Landsat→DEM→WC，S2 卡住导致其他源无法开始。

**方案**：
- 不再一个进程处理所有 5 个 source
- 每个 source 启动一个独立 tmux session
- 例如：
  - `pc_daqing_s2`
  - `pc_daqing_landsat`
  - `pc_daqing_s1`
  - `pc_daqing_dem`
  - `pc_daqing_worldcover`

这样：
- S2 和 Landsat 可以并行下载
- token 过期只影响一个 source，不影响其他
- 每个 source 的 400 patches 独立遍历，速度更快

**注意**：DEM 和 WorldCover 每个 patch 只有 1 帧，可以快速完成。

---

### 第二层：看门狗增强（`pc_download_watchdog.py`）

#### 2.1 自动创建缺失 session

**问题**：session 不存在时只报警，不恢复。

**方案**：
- 如果检测到 session 不存在，**自动创建并启动**
- 维护一个 session→命令的映射表，自动重启

#### 2.2 更频繁检查

- 从 30 分钟改为 **15 分钟** 检查一次
- token 48 分钟过期，15 分钟检查可以及时发现

#### 2.3 文件数量停滞检测

- 如果 30 分钟内某 session 的文件数量增长为 0，判定为卡住
- 自动 kill + 重启

---

### 第三层：部署策略

#### 3.1 分阶段启动

不是一次性启动所有 session，而是：

**Phase 1（先启动，最快完成）**：
- DEM + WorldCover（每个 patch 1 帧，几小时完成）

**Phase 2（并行启动）**：
- S2 + Landsat（主要耗时源）

**Phase 3（最后）**：
- S1（数据量小，且大庆/海淀 S1 coverage 有 gap）

#### 3.2 先小城市验证

先启动 **海淀**（数据量较小）验证改进效果，确认正常后再启动大庆。

#### 3.3 数据保护

下载完成后建议：
- 设置目录只读权限（防止误删）
- 创建完成标记文件（`.download_complete`）

---

## 实施步骤

### Step 1：修改脚本（~15 分钟）

1. 修改 `download_from_planetary_computer.py`：
   - 添加 403 自动重试 + 重新签名
   - 添加 STAC 搜索重试
   - 支持 `--sources` 只传一个 source
   - workers 默认改为 2

2. 修改 `pc_download_watchdog.py`：
   - 自动创建缺失 session
   - 检查间隔改为 15 分钟
   - 文件数量停滞检测

3. git commit

### Step 2：创建数据目录（~5 分钟）

```bash
mkdir -p /workspace/raw/phase2_heilongjiang/{qiqihar,daqing,haidian}/{s2,s1,landsat,dem,worldcover}
```

### Step 3：先启动海淀 DEM + WorldCover 验证（~1 小时）

```bash
# 海淀 DEM
tmux new-session -d -s pc_haidian_dem "python ... --sources dem --workers 2"

# 海淀 WorldCover
tmux new-session -d -s pc_haidian_wc "python ... --sources worldcover --workers 2"
```

验证：
- token 是否自动刷新
- 403 是否自动重试
- 文件是否正常写入

### Step 4：启动海淀 S2 + Landsat（~10-20 小时）

```bash
tmux new-session -d -s pc_haidian_s2 "python ... --sources s2 --workers 2"
tmux new-session -d -s pc_haidian_landsat "python ... --sources landsat --workers 2"
```

### Step 5：启动大庆（海淀验证通过后）

```bash
# DEM + WC 先启动
tmux new-session -d -s pc_daqing_dem "python ... --sources dem --workers 2"
tmux new-session -d -s pc_daqing_wc "python ... --sources worldcover --workers 2"

# 然后 S2 + Landsat
tmux new-session -d -s pc_daqing_s2 "python ... --sources s2 --workers 2"
tmux new-session -d -s pc_daqing_landsat "python ... --sources landsat --workers 2"
```

### Step 6：启动齐齐哈尔

```bash
# 齐齐哈尔 S2 + Landsat（DEM/WC 已完成，不需要重新下载）
tmux new-session -d -s pc_qiqihar_s2 "python ... --sources s2 --workers 2"
tmux new-session -d -s pc_qiqihar_landsat "python ... --sources landsat --workers 2"
```

### Step 7：看门狗自动守护

看门狗会：
- 每 15 分钟检查所有 session
- 发现缺失自动创建
- 发现停滞自动重启

---

## 预期效果

| 指标 | 之前 | 改进后 |
|------|------|--------|
| Token 过期处理 | 手动重启 | **自动重试+重签名** |
| STAC 超时 | 跳过该 patch | **自动重试 3 次** |
| 源并行度 | 串行 5 源 | **每源独立进程** |
| Workers | 4 | **2（降低限流）** |
| 进程恢复 | 手动 | **看门狗自动创建** |
| 检查间隔 | 30 分钟 | **15 分钟** |
| 完成标记 | 无 | **`.download_complete`** |

---

## 风险与兜底

1. **PC API 完全不可用**：重试 3 次后仍失败，记录到失败列表，后续手动处理
2. **磁盘空间不足**：当前 2.0T/3.5T，三个城市预计需要 ~500GB，足够
3. **误删再次发生**：下载完成后设置只读权限

---

## 下一步

请确认以上计划后，我将开始 **Step 1**（修改脚本）。
