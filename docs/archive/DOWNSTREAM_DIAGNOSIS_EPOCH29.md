# ExpD (Epoch 29) 下游分类诊断报告

> **Checkpoint**: `epoch_best_epoch29.pt` (std_mean=0.694, active=64/64)  
> **日期**: 2026-05-21  
> **评估设备**: CPU (NPU 7 AICore 异常)

---

## 一、整体效果总览

### 1.1 变化检测（核心目标）

| 方法 | AUC | 评价 |
|------|-----|------|
| Bare CD (Cosine) | **0.578** | ⚠️ 弱时间敏感性，embedding 直接对比不够 |
| Bare CD (LR 加权) | **0.730** | ✅ 存在可学习信号，线性分类器能提取 |
| Few-Shot CD (K=20) | **0.659±0.044** | ✅ 明显提升，backbone 学到了变化特征 |

**结论**: 变化检测能力 **中等偏上**。Cosine 直接对比弱是因为 embedding 主要编码了场景内容而非变化，但经过 LR/CD Head 可以提取出变化信号。

### 1.2 下游分类（辅助验证）

| 任务 | 全量 MLP Acc | Few-Shot K=10K Acc | mIoU | 状态 |
|------|-------------|-------------------|------|------|
| **WorldCover** (7类) | 57.3% | **66.3%** | 23.6% | ❌ 低 |
| **JRC Water** (2类) | 73.7% | 70.6% | 41.9% | ⚠️ 有评估 Bug |
| **Dynamic World** (8类) | 64.3% | — | 18.1% | ❌ 极低 |

> **Few-Shot 优于全量 MLP 的现象**: WorldCover 在 K=10K 时 66.3% > 全量 57.3%，可能因为全量 MLP 50 epoch 过拟合，而 Few-Shot 30 epoch + dropout 有更好的泛化。

---

## 二、🔴 关键发现：JRC Water 评估存在严重 Bug

### 2.1 标签真实含义

JRC Water **不是二分类**！它是 **JRC Global Surface Water 的 Occurrence 百分比**：

| 值 | 含义 |
|----|------|
| 0 | 从未检测到水 |
| 1-99 | 水出现的百分比（% of observations）|
| -128 | NoData（预处理时引入）|

**统计**（20 个 patch 采样）：
- 排除 NoData(-128) 后，**98.6% 的像素 value > 0**（哈尔滨地区水体普遍）
- 平均值 29.5%，中位数 13%
- 值范围 [0, 99]

### 2.2 当前评估代码的错误

`scripts/eval/evaluate_mlp_v2.py` 中：
```python
("jrc_water", "jrc_water", "static.tif", 2)  # num_classes=2
```

过滤条件：
```python
mask = (label != nodata) & (label >= 0) & (label < num_classes)
# nodata = -32768 (rasterio 读取的值)
# 所以 mask 只保留 label=0 和 label=1 的像素！
```

**后果**：
- value > 1 的所有"有水"像素（98.6%）被**过滤掉**
- 只在"无水"(0)和"极少水"(1)两个值上训练和评估
- **结果完全不能代表真实的水体分类能力**

### 2.3 正确评估方式

```python
# 方案 A: 任何有水即算 water
water_mask = (label > 0)  # 0 = no water, >0 = water

# 方案 B: 高频水体 (>50% occurrence)
water_mask = (label >= 50)  # 0-49 = land, 50-100 = water
```

---

## 三、🟡 WorldCover 低的根因分析

### 3.1 Confusion Matrix 深度分析

```
真实\预测    Tree  Grass   Crop  Built   Bare  Water  Wetland
─────────────────────────────────────────────────────────────
Tree        21621     2   25907  13146     5    6611       6   (38.5%→Crop)
Grass        3611    20   13048   3518     2    5275      20   (51.2%→Crop)
Crop         9583     5  105521  10750    15    5599      43   (80.2%→Crop) ✅
Built        6311     6   18481  44791     4    2836      11   (61.8%→Built) ✅
Bare          480     5    1708   1418     1    1068      11   (36.4%→Crop)
Water        1592     6    5463   1633     0   27533      25   (75.9%→Water) ✅
Wetland       275     0    7213    179     0    2774      28   (68.9%→Crop)
```

**核心问题**: **Grass、Wetland、Tree、Bare 被大量误分为 Crop**

### 3.2 根因拆解

| # | 根因 | 影响程度 | 证据 |
|---|------|----------|------|
| 1 | **语义光谱相似** | 🔴 高 | Grass/Crop/Wetland 在 10m S2 上光谱高度相似 |
| 2 | **类别极度不平衡** | 🔴 高 | Crop 31.4%，Bare 仅 0.3%，模型偏向预测大类 |
| 3 | **分辨率损失** | 🟡 中 | 128×128 → 64×64，混合像元问题严重 |
| 4 | **重建权重偏低** | 🟡 中 | WorldCover 权重 0.5 vs S2 权重 1.0 |
| 5 | **模型容量浪费** | 🟡 中 | 训练输出 11 类，标签只有 7 类存在 |
| 6 | **MLP Head 太简单** | 🟡 中 | 2层 MLP (64→256→7)，无空间上下文 |

### 3.3 类别分布（20 patch 采样）

| 类别 | ESA Code | 支持度 | 问题 |
|------|----------|--------|------|
| Tree | 10 | 10.9% | 误分→Crop |
| Grass | 30 | 7.3% | 严重误分→Crop |
| Crop | 40 | **31.4%** | ✅ 学得好 |
| Built | 50 | 19.4% | ✅ 学得好 |
| Bare | 60 | 0.3% | 误分→Crop |
| Water | 80 | 24.6% | ✅ 学得好 |
| Wetland | 90 | 6.0% | 严重误分→Crop |

---

## 四、🟡 Dynamic World 低的根因分析

### 4.1 类别分布与 IoU

| 类别 | 支持度 | IoU | 问题 |
|------|--------|-----|------|
| Water (1) | 10.4% | 0.045 | ❌ 严重误分 |
| Trees (2) | 6.2% | 0.000 | ❌ 无法学习 |
| Grass (3) | 24.4% | 0.305 | ⚠️ 一般 |
| Crops (4) | 25.7% | 0.569 | ✅ |
| Built (5) | 2.6% | 0.000 | ❌ 无法学习 |
| Bare (6) | 30.1% | 0.531 | ✅ |
| Snow (7) | 0.5% | 0.000 | ❌ 无法学习 |
| Cloud (8) | 0.0% | 0.000 | ❌ 无数据 |

**核心问题**: **小类别（Trees, Built, Snow, Cloud）支持度太低，模型完全无法学习**

### 4.2 训练-评估不一致

| 问题 | 说明 |
|------|------|
| 训练时输出 | 9 类（含 NoData=0） |
| 评估时过滤 | 排除 class_0（NoData），只剩 8 类 |
| 实际有数据 | 约 5-6 类有有效样本 |

---

## 五、🔧 改进方案（按优先级排序）

### 5.1 立即修复（评估层面）

#### A. 修复 JRC Water 评估
- 设置 threshold（value > 0 → water）
- 或使用 occurrence 百分比作为回归目标

#### B. 类别加权（Class Weighting）
- 对 WorldCover 和 Dynamic World 使用 `compute_class_weight('balanced')`
- 缓解类别不平衡导致的偏向大类问题

#### C. Focal Loss（处理极端不平衡）
- `gamma=2.0`，降低易分类样本的权重
- 对小类别（Bare, Wetland, Snow）更有效

### 5.2 Head 升级（模型层面）

#### 方案 A: PixelConvHead（利用空间上下文）
```python
head = PixelConvHead(in_dim=64, hidden_dim=64, num_classes=7, kernel_size=3)
# 3×3 conv 可以捕捉邻域上下文，对区分 Grass/Crop 有帮助
```

#### 方案 B: Deep MLP（增加表达能力）
```python
head = PixelMLPHeadV2(in_dim=64, hidden_dims=[512, 256, 128], num_classes=7, dropout=0.4)
# 4层 MLP，更大 hidden dim
```

#### 方案 C: UNet-Style Decoder（多尺度特征）
- 需要从 backbone 中间层提取多尺度特征
- 当前只使用最终 64×64 embedding，丢失了细粒度空间信息

**推荐**: 先尝试 **PixelConvHead + Class Weighting**，性价比最高。

### 5.3 训练改进（长期）

| 改进 | 预期效果 | 代价 |
|------|----------|------|
| 提高 WorldCover/DW 重建权重至 1.0 | 下游 +5~10% | 可能轻微影响 CD |
| 解冻 backbone 最后 2 层 | 下游 +5~15% | 需要更多数据/正则化 |
| 多尺度训练（Multi-Scale） | 下游 +3~5% | 训练成本 ×2 |
| 专门的数据增强（CutMix/MixUp） | 下游 +2~3% | 实现复杂 |

---

## 六、📋 下一步行动计划

### 立即可做（30分钟）

1. **修复 JRC Water 评估脚本** → 重新评估
2. **给 WorldCover 加 class weighting** → 重新评估
3. **切换 PixelConvHead** → 对比 MLP 效果

### 中期（下次训练时）

1. 将 `source_recon_weights` 中 worldcover/dynamic_world 从 0.5 提高到 1.0
2. 考虑在 backbone 中加入 **SegFormer-style 轻量 decoder** 分支
3. 引入 **Focal Loss** 处理类别不平衡

---

## 七、核心结论

| 问题 | 结论 |
|------|------|
| 是输入数据问题？ | **否**。标签质量 OK，分辨率匹配。 |
| 是评估问题？ | **部分是**。JRC Water 评估逻辑有 Bug。 |
| 是模型问题？ | **部分是**。MLP Head 太简单，且无类别平衡处理。 |
| 是训练问题？ | **部分是**。重建权重偏低，语义目标被光学重建压制。 |

**最关键的发现**:
- **JRC Water 的 73.7% 准确率是假象**（只在 0/1 上评估，过滤了 98.6% 的数据）
- **WorldCover 的 mIoU 23.6% 是真实水平**，但可以通过 class weighting + ConvHead 提升到 35%+
- **Dynamic World 的 mIoU 18.1% 也是真实水平**，小类别问题需要专门处理

---

## 附录：分辨率对照表

| 数据源 | 原始尺寸 | 分辨率 | Backbone 输出 | 关系 |
|--------|----------|--------|---------------|------|
| S2 输入 | 128×128 | 10m | 64×64 (下采样2×) | 对齐 |
| WorldCover | 128×128 | 10m | 64×64 | nearest resize |
| Dynamic World | 128×128 | 10m | 64×64 | nearest resize |
| JRC Water | 43×43 | ~30m | 64×64 | nearest resize (上采样!) |

> **JRC Water 上采样问题**: 43×43 → 64×64 是 upsampling，会引入大量插值伪影。
