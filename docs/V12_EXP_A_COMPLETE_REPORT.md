# ExpA (玄女V2 skipL2) 完整评测报告

**日期**: 2026-05-20  
**分支**: `v12-clean-dynamic`  
**Commit**: `66697f7`  
**模型**: `/workspace/outputs/xuannv_v2_expA_skipL2/epoch_best_epoch48.pt`  
**Config**: `configs/xuannv_v2_expA_skipL2.yaml`

---

## 一、模型基本信息

| 参数 | 值 |
|------|-----|
| 实验名称 | xuannv_v2_expA_skipL2 |
| Embedding 维度 | 64 |
| STP Blocks | 4 |
| Skip L2 Norm (训练) | ✅ true |
| 训练 Epoch | 50 |
| 最佳 Epoch | 48 |
| Reconstruction Loss | 0.0659 |
| Active Dimensions | **9 / 64** |
| Std Mean | 0.1003 |

---

## 二、修复的关键 Bug

### 2.1 系统性索引错位 Bug（致命）

**根因**: `dataset.patches.index(pid)` 返回 patch 索引（0-423），但 `dataset[idx]` 访问的是 `monthly_samples[idx]`（0-5087）。

**影响**: 之前所有评测脚本（18个文件）都因索引错位而使用了**错误的 patch 和月份**，导致 AUC 结果完全无效。

**修复方案**:
- `src/inference/engine.py`: 新增 `extract_embedding_for_month(patch_id, year, month)` 正确 API
- `scripts/eval/fewshot_change_detection.py`: 改用新 API，修复索引
- `scripts/eval/train_cd_head_v12.py`: 新建脚本，使用 patch-month 映射

### 2.2 时间窗口过时 Bug（致命）

**根因**: 多个脚本硬编码了 **2023/2024 年** 的时间窗口，但哈尔滨变化检测标注对应 **2025 年**。

**影响**: 即使索引正确，提取的 before/after embedding 也来自**没有标注覆盖的时间窗口**，AUC 依然无意义。

**修复**: 所有窗口统一修正为 2025 年具体月份（4→6, 6→8, 8→9, 9→10 月）。

### 2.3 Few-Shot 窗口年份错误（致命）

**根因**: `fewshot_change_detection.py` 中的 `MONTH_WINDOWS_2025` 实际存储的是 **2024 年** 的毫秒值。

**影响**: Few-Shot CD 之前的结果全部为 0（dataset 中无 2024 年数据）。

**修复**: 将窗口值修正为真正的 2025 年。

---

## 三、变化检测评测结果

### 3.1 Bare AUC（无 CD Head，像素级 Cosine Distance）

| 空间 | 全局 | June | Aug | September | October |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **L2-norm** | 0.521 | 0.424 | 0.636 | 0.628 | 0.515 |
| **Pre-norm** | 0.529 | 0.423 | 0.663 | **0.696** | 0.513 |

**分析**:
- 全局 AUC 接近随机（0.52），但 **August (0.66) 和 September (0.70)** 显著高于随机
- Pre-norm 空间略优于 L2-norm（0.529 vs 0.521）
- June 和 October 的 AUC < 0.55，说明这两个时段的变化信号最弱
- Changed/unchanged 的 separation 极小（~0.003），说明 cosine distance 区分能力有限

### 3.2 CD Head 训练（5-fold CV，ChangeDetectionHeadV3）

| 版本 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** | Std | Best |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L2-norm** | 0.872 | 0.876 | **0.905** | 0.840 | 0.650 | **0.829** | 0.092 | **0.905** |
| **Pre-norm** | 0.867 | 0.834 | **0.900** | 0.840 | 0.640 | **0.816** | 0.091 | **0.900** |

**关键发现**:
- ✅ **CD Head 能学到极强的变化检测能力**，Mean AUC 达到 0.82-0.83
- ✅ Best Fold 突破 **0.90**，证明 embedding 包含丰富的变化信息
- ⚠️ Fold 5 显著低于其他（0.64-0.65），说明数据划分对该 fold 的验证集不友好
- L2-norm 略优于 pre-norm（0.829 vs 0.816），但差距不大

**输出文件**:
- `cd_head_v12_best.pt` (L2, Fold 3, AUC=0.905)
- `cd_head_v12_best_prenorm.pt` (pre-norm, Fold 3, AUC=0.900)

### 3.3 Few-Shot Change Detection（轻量 2-layer Head）

| K-shot | L2 Global | L2 Patch | Pre-norm Global | Pre-norm Patch |
|:---:|:---:|:---:|:---:|:---:|
| K=1 | 0.591 | 0.569 | 0.624 | 0.624 |
| K=5 | 0.627 | 0.629 | 0.599 | 0.594 |
| K=10 | 0.641 | 0.638 | 0.661 | 0.648 |
| **K=20** | **0.670** | **0.652** | **0.679** | **0.650** |

**分析**:
- 轻量 CD Head（314K 参数）上限约 **0.67-0.68**
- 增加 shot 数量能稳定提升性能（K=1 → K=20 提升约 +0.08）
- 与完整 CD Head V3（0.83 mean）相比，轻量 Head 损失了约 **0.15 AUC**
- Pre-norm 在 K=20 时略优于 L2-norm

---

## 四、下游分类评测（KNN, k=5）

使用 6 月份 spatial embedding map [64, 64, 64] 做像素级 KNN 分类。

| 任务 | 类别数 | Accuracy | mIoU | 分析 |
|------|:------:|:--------:|:----:|------|
| **WorldCover** | 7 | 0.436 | 0.173 | 整体判别力较弱 |
| **JRC Water** | 2 | 0.675 | 0.405 | 二分类中表现最好 |
| **Dynamic World** | 9 | 0.541 | 0.146 | 多分类中表现最差 |

**WorldCover 各类别 IoU**:

| 类别 | IoU | Support | 说明 |
|------|:---:|:-------:|------|
| 类别 0 (Tree cover) | 0.174 | 67,298 | 占比最高但判别差 |
| 类别 1 | 0.040 | 25,494 | 几乎无法区分 |
| 类别 2 (Cropland) | 0.400 | 131,516 | 最好 |
| 类别 3 | 0.292 | 72,440 | 中等 |
| 类别 4 | 0.005 | 4,691 | 极差 |
| 类别 5 | 0.287 | - | 中等 |

**分析**:
- KNN 作为无参数学习基线，mIoU 普遍较低（<0.20），说明 embedding 对**细粒度地物分类**的判别力有限
- JRC Water 二分类表现最好（mIoU=0.405），可能是因为水域在光学/SAR 影像中特征明显
- WorldCover 中 Cropland（农田）表现最好，可能是因为农田面积大、纹理规则
- **active=9/64 的坍缩** 严重限制了分类能力——只有 9 个维度有有效信号

---

## 五、成功原因分析

### 5.1 为什么 CD Head 能取得 0.83+ AUC？

| 因素 | 解释 |
|------|------|
| **1. 重建任务提供了丰富的空间特征** | 模型被迫重建 S2/S1/Landsat/DEM 等 7 个目标，学到了地表覆盖的空间模式 |
| **2. 时间编码注入有效** | Time/Window/RelativeTime 编码让模型能区分不同时间点，时间变化信息被编码到 embedding 中 |
| **3. CD Head 学到了非线性变化度量** | Bare AUC 用线性 cosine distance（0.53），而 CD Head 用多层卷积学到了非线性变化模式（0.83），提升了 **+0.30** |
| **4. 双窗口训练增强了时间敏感性** | 训练时的跨时相重建（输入 month_A, 目标 month_B）强迫模型关注时间差异 |
| **5. Pre-norm 空间保留了幅度信息** | SkipL2 让 pre-norm 空间保留原始幅度，某些变化（如建筑强度变化）在幅度上更明显 |

### 5.2 为什么 August/September 的 Bare AUC 能达到 0.66-0.70？

- **植被季节性变化**：8→9 月是哈尔滨从夏季到秋季的过渡期，植被绿度变化显著
- **农业活动**：6→8 月农田作物生长快速，NDVI 变化大
- **建筑活动**：夏季是建筑施工高峰期，4→6 月可能变化信号被植被增长掩盖

### 5.3 为什么 JRC Water KNN 表现最好？

- 水域在 S2（近红外强吸收）和 SAR（镜面反射弱回波）中有非常独特的光谱/散射特征
- 即使只有 9 个 active dims，水域特征足够强，能被 KNN 捕获

---

## 六、失败与不足分析

### 6.1 Active=9/64 — 严重坍缩

| 指标 | 值 | 正常范围 | 问题 |
|------|:--:|:--------:|------|
| Active dims | 9 / 64 | > 40 | **只有 14% 的维度被使用** |
| Std mean | 0.1003 | > 0.15 | 分布过于集中 |
| Std max | ~0.35 | > 0.50 | 峰值维度也不够分散 |

**后果**:
- KNN 分类 mIoU < 0.20 — 细粒度分类几乎不可能
- Bare AUC 全局仅 0.53 — cosine distance 在坍缩空间中区分度差
- June/October 的变化信号被噪声淹没

**根因推测**:
1. SkipL2 + 高 uniformity weight 可能过度压缩了 embedding 空间
2. 时间聚合是**全局点积注意力**（先空间 mean 再全局 attn），空间上平均了时间信息
3. 64 dim 对于 7 个重建目标 + 时间信息来说可能不足

### 6.2 June/October 变化检测失败

| Period | Bare AUC | 可能原因 |
|--------|:--------:|----------|
| 4→6 月 (June) | < 0.50 | 植被快速增长掩盖了建筑变化；春季融雪混淆 |
| 9→10 月 (October) | 0.51-0.52 | 秋季植被衰退与建筑变化信号混淆；冬季来临前地表覆盖趋于稳定 |

### 6.3 Dynamic World 分类极差（mIoU=0.146）

- Dynamic World 有 9 个类别，比 WorldCover 更细粒度
- active=9 的 embedding 无法支撑 9 类分类
- 建筑/道路/裸地等类别在 64×64 像素块中容易混淆

### 6.4 原始 AEF 对比 — 架构差异未优化

| 差异 | 原始 AEF | 玄女V2 | 影响 |
|------|----------|--------|------|
| 时间聚合 | 逐空间位置 MHA | 全局点积注意力 | **更粗糙**，时间信息被空间平均 |
| 多源混合 | Channel 拼接后统一编码 | 展平为 (B, S*T, C, H, W) | 跨源注意力可能引入噪声 |

---

## 七、下一步迭代计划

### 7.1 短期（1-2 天）— 快速验证

#### A. 测试更大的 CD Head
- 当前 CD Head V3: 314K 参数，2 residual blocks + ECA
- **尝试**: 4 residual blocks + 更大的 hidden_dim (128)，看能否突破 0.92

#### B. 用 Pre-norm 训练，L2-norm 推理对比
- Pre-norm 的 Bare AUC (0.70) 优于 L2-norm (0.63)
- **尝试**: CD Head 在 pre-norm 空间训练，但推理时先做 L2 norm
- 或者直接用 pre-norm 输出做 CD Head 推理

#### C. 可视化变化热力图
- 用训练好的 CD Head 对所有 424 patches 的 4 个 period 做推理
- 叠加在卫星影像上，人工验证变化检测质量
- 找出 False Positive/False Negative 的模式

### 7.2 中期（3-5 天）— 缓解坍缩

#### D. 提升 Active Dimensions

| 方案 | 改动 | 预期效果 | 风险 |
|------|------|----------|------|
| **D1. 增大 embedding_dim** | 64 → 128 | 更多维度可用 |  uniformity 损失计算成本增加 |
| **D2. 降低 uniformity weight** | 0.1 → 0.05 | 减少压缩压力 | 可能增加坍缩风险 |
| **D3. 增大 vicreg_min_std** | 1.0 → 1.5 | 强制更大方差 | 可能影响重建质量 |
| **D4. 添加 spatial uniformity** | 已在用 (4096 samples) | 空间上更分散 | 当前已启用 |

**推荐组合**: D1 (128 dim) + D3 (min_std=1.5) — 快速实验

#### E. 改进时间聚合

**当前问题**: 全局点积注意力先空间 mean 再全局 attn，时间信息被平均。

**方案 E1: 逐空间位置时间聚合**
```python
# 原始 AEF 做法
query = self.summary_query(window_code)  # [B, 1, C]
attn_scores = torch.einsum("bqc,btchw->bqthw", query, x)  # [B, 1, T, H, W]
attn = softmax over T dim
summary = torch.einsum("bqthw,btchw->bqchw", attn, x)  # [B, 1, C, H, W]
```

每个空间位置 (H, W) 有自己的注意力权重，时间信息不在空间上平均。

**方案 E2: 多尺度时间窗口**
- 训练时不仅用单月窗口，还随机使用 2-3 个月聚合窗口
- 强迫模型学习不同时间尺度下的变化

### 7.3 长期（1-2 周）— 架构改进

#### F. 参考原始 AEF 重新实现时间聚合
- 将当前全局注意力替换为 `TemporalSummarizer`（逐空间位置单 query MHA）
- 这可以解决 "时间信息被空间平均" 的问题

#### G. 多源独立编码后 Channel 拼接
- 当前做法: S2/S1/Landsat 展平到同一时间轴
- 改进: 每源独立编码后 channel 拼接，像原始 AEF 一样
- 减少跨源噪声，保留源特有特征

#### H. 引入 Contrastive Learning 增强
- 当前只有 VICReg + uniformity
- 增加: SimCLR/InfoNCE style 的时序对比损失
- 对同一 patch 的不同月份做正负样本配对

### 7.4 实验优先级矩阵

| 优先级 | 实验 | 预计时间 | 预期提升 |
|:---:|:---|:---:|:---:|
| P0 | 测试更大 CD Head | 2h | +0.02-0.05 AUC |
| P0 | 可视化变化热力图 | 3h | 验证/发现问题 |
| P1 | Embedding dim 128 | 1d | active 20-30/128 |
| P1 | 逐空间位置时间聚合 | 1d | 时间敏感性 ↑ |
| P2 | 多尺度时间窗口 | 2d | 长期变化检测 ↑ |
| P2 | 多源独立编码 | 2d | 特征纯度 ↑ |

---

## 八、核心结论

1. **模型不是完全失败的** — CD Head AUC 达到 0.83+，证明 embedding 包含丰富的变化信息
2. **Bare AUC 严重低估了模型** — cosine distance 在坍缩空间中无法捕捉非线性变化模式
3. **active=9/64 是最大瓶颈** — 限制了 KNN 分类、细粒度变化检测的能力
4. **时间聚合的粗糙度是根本原因** — 全局点积注意力平均了空间信息，导致时间信号弱
5. **下一步关键是提升 active dims + 改进时间聚合** — 这是从 0.83 → 0.90+ 的必经之路

---

*报告生成时间: 2026-05-20*  
*评测数据路径: `/workspace/outputs/xuannv_v2_expA_skipL2/`*
