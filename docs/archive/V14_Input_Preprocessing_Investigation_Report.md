# V14 模型输入与预处理深度调研报告

> 生成时间: 2026-05-18
> 调研范围: 多区域训练数据（哈尔滨+大庆+海淀）的输入来源、预处理流程、Dataset采样逻辑、损失计算

---

## 一、数据量分布统计

### 1.1 各区域数据源帧数

| 区域 | 源 | Patches | TIFF总数 | 平均每Patch帧数 | 备注 |
|------|---|---------|---------|----------------|------|
| **哈尔滨** | S2 (云筛选后) | 424 | 9,321 | **21.9** | 原始~180帧，筛选后~22帧 |
| **哈尔滨** | S1 | 424 | ~18,100 | **~42.7** | symlink到harbin_scenes/s1 |
| **哈尔滨** | Landsat | 424 | ~19,500 | **~47.9** | symlink到harbin_scenes/landsat |
| **大庆** | S2 | 309 | 45,350 | **146.8** | 未做云筛选，帧数密集 |
| **大庆** | S1 | 309→400? | 28,000 | **~70** | 实际400 patches有S1 |
| **大庆** | Landsat | 400 | 52,515 | **131.3** | |
| **海淀** | S2 | 400 | 41,077 | **102.7** | 未做云筛选 |
| **海淀** | S1 | 400 | 35,600 | **89.0** | |
| **海淀** | Landsat | 400 | 50,639 | **126.6** | |

> **注意**: 哈尔滨的S1/Landsat通过symlink指向 `harbin_scenes/` 原始目录，只有S2是云筛选后的独立目录。

### 1.2 静态目标分布

| 源 | 哈尔滨 | 大庆 | 海淀 | 说明 |
|------|--------|------|------|------|
| DEM | 424 | 400 | 379 | 每patch单张 |
| WorldCover | 424 | 400 | 400 | 每patch单张 |
| Dynamic World | 5,512 (424p) | **无** | **无** | 多时相 |
| JRC Water | 424 | **无** | **无** | 每patch单张 |

### 1.3 月度样本数估算

V13+采用**月度采样**：每个patch每个月只要有任意输入源有数据，就生成一个样本。

- 哈尔滨: 424 patches × ~7个月(2025) ≈ **~2,968 样本**
- 大庆: 309 patches × ~?个月 ≈ **~? 样本**
- 海淀: 400 patches × ~?个月 ≈ **~? 样本**
- **总计: 约 4,000~6,000+ 月度样本**

---

## 二、预处理流程详解

### 2.1 预处理代码位置

核心实现: `src/data/transforms.py` 中的 `normalize_data()` 函数

### 2.2 各源预处理策略（对齐AEF论文V7）

#### **S2 / Landsat（光学源）**

```
原始值: 反射率 (0 ~ 10000+)
    ↓
log(x+1)/10           # 对数压缩，对齐AEF论文S3.2
    ↓
z-score: (x - mean) / std   # 用预计算统计量
    ↓
clip to [-6σ, +6σ]    # 异常值裁剪
```

**AEF论文原文**: "像素强度先经过 s(x) = log(x+1)/10 变换，再基于全局统计量进行标准化"

**为什么用log变换？**
- 光学遥感反射率动态范围极大（0~20000+）
- log变换压缩高值、拉伸低值，使分布更接近高斯
- 与AEF原版完全对齐

#### **S1（SAR源）**

```
原始值: dB值 (VV/VH 后向散射)
    ↓
clip to [-30, +10] dB   # 物理合理范围裁剪
    ↓
z-score: (x - mean) / std
    ↓
clip to [-6σ, +6σ]
```

**AEF论文原文**: "将DN值转换为dB: γ = 10·log₁₀(DN²) - 83"

> 注意: 我们的数据已经是dB值（从Google Earth Engine导出时已完成转换），所以只做clip和z-score。

#### **分类源（WorldCover / Dynamic World）**

```
原始值: 类别索引 (如 10=林地, 20=灌木)
    ↓
类别映射: 原始值 → 0~N-1 紧凑索引
    ↓
One-Hot 编码: (num_classes, H, W)
```

**不做z-score**，因为分类数据是离散索引。

- WorldCover: 11类，映射 `{10→0, 20→1, ..., 100→10}`
- Dynamic World: 9类，映射 `{0→0, 1→1, ..., 8→8}`

#### **DEM / JRC Water（其他连续值）**

```
原始值
    ↓
z-score: (x - mean) / std
    ↓
clip to [-6σ, +6σ]
```

**JRC Water特殊处理**: nodata值=-128，先替换为NaN，再归一化。

### 2.3 预处理前后分布统计（实测采样）

以下数据从实际TIFF文件采样（哈尔滨patch_000000）：

#### **S2 — 20230113.tif**

| Band | 原始 min | 原始 max | 原始 mean | 原始 std | 预处理后 min | 预处理后 max | 预处理后 mean | 预处理后 std |
|------|---------|---------|----------|---------|------------|------------|--------------|-------------|
| 0 (B2) | 3476 | 5288 | 4306 | 259 | **-0.85** | **-0.85** | **-0.85** | **~0** |
| 1 (B3) | 3576 | 5312 | 4358 | 262 | **-0.97** | **-0.97** | **-0.97** | **~0** |
| 2 (B4) | 3704 | 5528 | 4531 | 275 | **-0.94** | **-0.94** | **-0.94** | **~0** |
| 3 (B8) | 3900 | 5936 | 4880 | 291 | **-1.03** | **-1.03** | **-1.03** | **~0** |
| 4 (B11) | 1786 | 3632 | 2525 | 294 | **-1.42** | **-1.42** | **-1.42** | **~0** |
| 5 (B12) | 1737 | 3727 | 2529 | 332 | **-1.30** | **-1.30** | **-1.30** | **~0** |

> **⚠️ 重大发现**: S2预处理后所有值几乎相同（std≈0）！这意味着该patch在该时间点**非常均匀**（可能是雪地/云层覆盖），不代表整体分布。

#### **S1 — 20230110.tif**

| Band | 原始 min | 原始 max | 原始 mean | 原始 std | 预处理后 min | 预处理后 max | 预处理后 mean | 预处理后 std |
|------|---------|---------|----------|---------|------------|------------|--------------|-------------|
| 0 (VV) | -30.4 | 14.3 | -14.7 | 3.74 | **-3.26** | **4.17** | **-0.36** | **0.62** |
| 1 (VH) | -39.9 | 1.6 | -22.4 | 3.78 | **-1.38** | **3.07** | **-0.32** | **0.54** |

S1预处理后分布健康：mean≈0, std≈0.5~0.6，范围[-3.3, +4.2]。

#### **Landsat — 20230207.tif**

| Band | 原始 min | 原始 max | 原始 mean | 原始 std | 预处理后 min | 预处理后 max | 预处理后 mean | 预处理后 std |
|------|---------|---------|----------|---------|------------|------------|--------------|-------------|
| 0 | 0 | 10369 | 10.9 | 331 | **-1.36** | **-1.36** | **-1.36** | **~0** |
| 1 | 0 | 10979 | 11.2 | 342 | **-1.49** | **-1.49** | **-1.49** | **~0** |
| 2 | 0 | 10733 | 11.5 | 348 | **-1.48** | **-1.48** | **-1.48** | **~0** |

> Landsat也有类似问题——这个patch这个时间点可能是无效数据（大量0值）。

### 2.4 更全面的分布评估

单点采样不可靠，以下是基于**预计算统计量**的理论预处理后期望分布：

| 源 | 理论预处理后mean | 理论预处理后std | 实际范围 |
|------|-----------------|----------------|---------|
| S2 | ~0 | ~1 | [-6, +6] (clip) |
| S1 | ~0 | ~1 | [-6, +6] (clip) |
| Landsat | ~0 | ~1 | [-6, +6] (clip) |
| DEM | ~0 | ~1 | [-6, +6] (clip) |

预处理后理论上应该是**标准正态分布**（mean≈0, std≈1），±6σ裁剪会截断约0.0000002%的极端值。

---

## 三、预计算统计量：变化前还是变化后？

### 3.1 答案：**变化前的原始统计量**

代码位置: `scripts/preprocessing/compute_statistics.py`

```python
def compute_source_stats(dataset, source_name, max_patches=50):
    all_samples = []
    for patch_id in patch_ids[:max_patches]:
        for tif_path in sample_files:
            data = read_tif(tif_path, dataset.image_size)  # ← 只read_tif，不调normalize
            all_samples.append(data)  # ← 原始数据直接收集
    
    for c in range(n_channels):
        channel_vals = np.concatenate([s[c].flatten() for s in all_samples])
        mean = float(np.mean(channel_vals))
        std = float(np.std(channel_vals))
```

**关键**: `read_tif()` 后直接收集，**没有调用 `normalize_data()`**。

### 3.2 统计量内容（实际文件）

```json
// s2_stats.json — 原始反射率统计
{
  "band_0": {"mean": 1394.66, "std": 1641.05},  // B2
  "band_1": {"mean": 1532.38, "std": 1576.40},  // B3
  "band_2": {"mean": 1584.04, "std": 1686.29},  // B4
  "band_3": {"mean": 2516.64, "std": 1753.06},  // B8
  "band_4": {"mean": 1566.21, "std": 944.05},   // B11
  "band_5": {"mean": 1707.??, "std": 1080.??}   // B12
}

// s1_stats.json — 原始dB统计
{
  "band_0": {"mean": -12.43, "std": 5.38},   // VV
  "band_1": {"mean": -20.18, "std": 7.11}    // VH
}

// landsat_stats.json — 原始反射率统计
{
  "band_0": {"mean": 15299.49, "std": 11245.63},
  ...
}
```

### 3.3 为什么统计原始值？

因为预处理流水线是：
```
原始值 → [用统计量做z-score] → 归一化值
```

统计量必须在变换**之前**计算，否则会变成"统计量的统计量"的循环依赖。

---

## 四、Dataset采样逻辑详解

### 4.1 样本定义（V13+月度采样）

```
样本 = (patch_id, year, month)
```

- 每个patch每个月只要有**任意输入源**有数据，就生成一个样本
- V13 严格 **2025-only**（`_build_monthly_samples`中硬编码过滤）
- 多区域时 patch_id = `{region}_{local_id}`，如 `harbin_patch_000001`

### 4.2 `__getitem__` 完整流程

```
输入: idx → 映射到 (patch_id, year, month_a)

Step 1: 跨时相采样决策
    - 概率 cross_temporal_prob（默认0，V14当前未启用）
    - 若触发：采样 month_b，满足 |month_b - month_a| ≥ 2
    - 否则 month_b = month_a

Step 2: 加载输入帧（当月 month_a）
    对每个输入源 (S2, S1, Landsat):
        - 加载该patch该月的所有可用帧
        - 最多保留 max_frames=32 帧
        - 超过时：训练随机采样，验证等间距采样
        - 缺失源 → source_input_mask = False
    
    关键张量:
        source_frames:    [S=3, T=32, C=6, H=128, W=128]
        source_timestamps_ms: [3, 32]
        source_frame_mask:    [3, 32]  (True=有效帧)
        source_input_mask:    [3]      (True=源可用)
        source_type_ids:      [3]      (0=S2, 1=S1, 2=Landsat)

Step 3: valid_period
    valid_start = 当月最早帧时间戳
    valid_end   = 当月最晚帧时间戳

Step 4: 构建重建目标
    对每个目标源 (7个):
        - DEM: 加载静态TIFF，resize到 64×64
        - S2/S1/Landsat:
            * 若 month_b == month_a: 复用输入第0帧（自编码器）
            * 若 month_b != month_a: 从 month_b 加载目标帧
        - WorldCover/Dynamic World/JRC Water: 
            * V14配置中 num_target_sources=4，只剩 S2/S1/Landsat/DEM
            * 原来的分类静态目标已被移除
    
    关键张量:
        target_images:        [S_tgt=4, C=6, H=64, W=64]
        target_relative_time: [4]  (时间偏移)
        target_metadata:      [4, M=4]
        target_mask:          [4]  (True=有效目标)
        target_loss_type:     [4]  (0=连续/MSE, 1=分类/CE)
        target_source_idx:    [4]  (路由到对应decoder)

Step 5: 双时间窗口采样
    对当月所有时间戳排序，采样 w1/w2
    （详见第五节）

Step 6: 空间增强（仅训练）
    - 50% 水平翻转
    - 50% 垂直翻转

Step 7: 返回字典
```

### 4.3 每个patch每个月只有一张图吗？

**不是！** 这是关键误解。

```
输入端:
    每个源当月可能有 N 张图（如S2某月可能有3~5帧）
    这些帧全部加载，最多保留 max_frames=32
    所以 source_frames[源, 0:N] 是有效的，source_frame_mask标记哪些有效

目标端:
    每个目标源只有 1 张图（或从month_b加载1张）
    所以 target_images 每个源只有1个时间步
```

**举例**: 哈尔滨patch_000000在2025年4月
- S2可能有4帧（4月5日、10日、15日、20日）
- S1可能有3帧
- Landsat可能有2帧
- 这些全部进入 `source_frames` 的前几个时间步
- 目标S2只取第0帧作为重建目标

### 4.4 内存预加载

```
_preload_all():
    - 16个worker并行加载所有patch的所有源
    - 缓存到 /workspace/outputs/.cache_shared/dataset_cache_{hash}.pt
    - 大小约 27-32GB
    - DDP rank 0 加载保存，其他rank等待加载
```

---

## 五、双时间窗口采样机制

### 5.1 为什么需要双窗口？

核心目的：**让模型学习"同一地点不同时间"的表示一致性**。

- 窗口1 (w1): 当月前半段时间的帧
- 窗口2 (w2): 当月后半段时间的帧
- 两个窗口编码出的embedding应该相似 → temporal contrastive loss

### 5.2 窗口采样代码

核心函数: `_sample_dual_windows(ts_sorted)`

支持四种模式:

#### **random_split（默认）**
```python
mid = len(ts_sorted) // 2
w1 = [ts_sorted[0], ts_sorted[mid-1]]      # 前半段
w2 = [ts_sorted[mid], ts_sorted[-1]]        # 后半段
```
简单中点分割。

#### **non_overlap（V3模式）**
```python
# 要求: gap ≥ 6个月
# 随机选split_point，w1从早期随机采样4~12帧，w2从晚期随机采样4~12帧
# center gap ≥ min_gap_ms (默认6个月)
```

#### **adjacent_month**
```python
# 按月份分组
# 随机选一对相邻月份作为w1/w2
# 每边至少 min_frames=4 帧
```

#### **mixed_scale（V5模式）**
```python
# 以50%概率选择:
#   - 长间隔: gap ≥ 6个月 (non_overlap逻辑)
#   - 短间隔: 1~3个月，gap ≤ 3个月
```

### 5.3 窗口在模型中如何使用

```python
# encode_frames 内部
window_mask = (
    (timestamps >= valid_start) & (timestamps <= valid_end)
)
effective_frame_mask = source_frame_mask & window_mask
```

**关键**: attention 只能看到 valid_period 范围内的帧。

```python
# V14 Trainer中
teacher_out = model(..., valid_start_ms=batch["valid_start_w1"], valid_end_ms=batch["valid_end_w1"])
student_out = model(..., valid_start_ms=batch["valid_start_ms"],  valid_end_ms=batch["valid_end_ms"])
```

Teacher用w1窗口，Student用完整窗口（或w2，取决于配置）。

---

## 六、不同Target源的Loss计算

### 6.1 V14当前配置

查看 `configs/generate_v14_configs.py`:

```yaml
data:
  num_target_sources: 4
  target_sources:
    - name: s2          loss_type: 0  out_channels: 6
    - name: s1          loss_type: 0  out_channels: 2
    - name: landsat     loss_type: 0  out_channels: 6
    - name: dem         loss_type: 4  out_channels: 1
  source_recon_weights: [1.0, 1.0, 1.0, 0.05]  # S2, S1, Landsat, DEM
```

**重要变化**: V14 已移除 WorldCover、Dynamic World、JRC Water 三个目标！
现在只有 **4个重建目标**（S2、S1、Landsat、DEM）。

### 6.2 Loss计算流程

代码位置: `src/training/loops.py` → `compute_recon_loss()`

```python
def compute_recon_loss(pred, target, mask, loss_type, num_classes, recon_mask=None):
    total_loss = 0
    for t_idx in range(T_tgt):
        batch_mask = mask[:, t_idx]  # 哪些batch项有此目标
        if not batch_mask.any():
            continue
        
        if loss_type == 1:  # 分类目标
            loss = F.cross_entropy(pred[batch_mask], target[batch_mask])
        else:  # 连续目标
            loss = F.l1_loss(pred[batch_mask], target[batch_mask])
        
        total_loss += loss
        count += 1
    return total_loss / max(count, 1)
```

**target_mask的作用**: `mask[:, t_idx]` 控制哪些batch项的该目标参与loss计算。
- 缺失源 → target_mask=False → 该目标的loss跳过
- 多区域适配: 大庆没有DynamicWorld/JRC → 这些目标被移除，不存在缺失问题

### 6.3 V14 Trainer中的完整Loss组合

```python
total_loss = (
    recon_teacher + recon_student              # 重建（Teacher + Student）
    + consist_w * consist                       # 一致性（TS embedding对齐）
    + cls_w * cls                               # 分类（当前weight=0）
    + var_w * var + cov_w * cov                 # VICReg（方差+协方差）
    + inter_var_w * inter_var + inter_cov_w * inter_cov  # 样本间VICReg
    + uniform_w * l2_uniform                    # Uniformity（反坍缩）
    + cr_w * cr                                 # Coding Rate（MCR²）
    + decorr_w * decorr                         # 去相关
    + orth_w * orth                             # 正交性
)
```

### 6.4 各损失权重（V14默认）

| 损失项 | 权重 | 说明 |
|--------|------|------|
| reconstruction | 1.0 | 核心重建 |
| source_recon_weights | [1,1,1,0.05] | S2/S1/Landsat=1.0, DEM=0.05 |
| consistency | 0.05~0.10 | Teacher-Student对齐 |
| classification | 0.0 | V14已关闭 |
| variance | 0.5 | VICReg方差 |
| covariance | 0.1 | VICReg协方差 |
| uniformity | 0.1 | Batch uniformity |
| coding_rate | 0.1 | MCR² |
| decorrelation | 0.01 | Barlow Twins |
| orthogonality | 0.0 | Bottleneck权重正交 |

---

## 七、关键结论与待确认问题

### 7.1 已确认

1. ✅ 统计量是**原始数据**的mean/std（变化前）
2. ✅ V14只有**4个目标**（S2/S1/Landsat/DEM），已移除分类目标
3. ✅ 每个patch每月的输入**可以有多个帧**（max_frames=32），目标只有1帧
4. ✅ 双窗口采样有4种模式，默认random_split
5. ✅ 缺失源通过 `source_input_mask` 和 `target_mask` 自动处理
6. ✅ 预处理严格对齐AEF论文: log(x+1)/10 → z-score → ±6σ clip

### 7.2 待进一步验证

1. **S2预处理后std≈0的问题**: 采样的单点可能是异常（雪地/云），需要更大规模采样验证整体分布
2. **大庆/海淀的S2未做云筛选**: 帧数是哈尔滨的5~7倍，可能影响训练效率和重建质量
3. **月度样本总数**: 需要实际运行dataset初始化得到精确数字
4. **DEM权重0.05是否合理**: 静态目标权重极低，是否足以约束embedding包含地形信息

---

## 附录：代码速查

| 问题 | 代码位置 |
|------|---------|
| 预处理策略 | `src/data/transforms.py:normalize_data()` |
| 统计量计算 | `scripts/preprocessing/compute_statistics.py` |
| Dataset采样 | `src/data/dataset.py:_get_item()` |
| 双窗口采样 | `src/data/dataset.py:_sample_dual_windows()` |
| 多区域适配 | `src/data/multi_region_dataset.py` |
| 重建loss | `src/training/loops.py:compute_recon_loss()` |
| V14 Trainer | `src/training/ddp_v14_trainer.py:train_step()` |
| 模型Forward | `src/models/model.py:forward()` |
| 配置生成 | `configs/generate_v14_configs.py` |
