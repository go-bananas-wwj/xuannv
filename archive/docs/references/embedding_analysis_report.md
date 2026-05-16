# AEF Embedding 质量深度分析报告

> 分析日期: 2026-04-18  
> 分析对象: AlphaEarth Foundations 官方 embedding vs 当前复现模型 (aef_qwen_v4, epoch 113)  
> 数据来源: `/workspace/outputs/alphaearth_harbin/` (官方), `/workspace/outputs/aef_qwen_v4_cd_upgrade/` (复现)

---

## 一、论文附录/补充材料中的关键训练细节

论文正文多次引用补充材料 (Supplemental Materials)，其中与训练直接相关的核心细节如下：

### S16 — 模型训练与架构细节
- **教师-学生一致性 (Teacher-Student Consistency)**: 教师分支看完整输入，学生分支输入被扰动（随机丢帧、丢源、截断前后段）。要求两者 embedding 一致。这是论文减少拼接缝、传感器覆盖差异和缺测噪声的核心机制。
- **条件解码 (Conditional Decoding)**: 每个目标源解码器接收 embedding 和该源的条件变量（目标时间在有效期内的归一化位置、轨道几何元数据）。目的是把"测量过程相关信息"从共享 embedding 中分离。
- **Batch Uniformity 目标 (S16.2.4)**: 论文使用 batch uniformity 损失鼓励球面嵌入均匀覆盖单位球。这是防止 embedding 坍缩到局部区域的关键约束。
- **VMF 瓶颈**: 论文使用 von Mises-Fisher 分布建模 embedding，浓度参数固定。embedding 解释为 VMF 分布的均值方向，允许引入可控噪声。

### S18 — 评估方法
- **Linear Probe**: 冻结 embedding，训练线性分类头
- **kNN (k=3)**: 基于 embedding 空间最近邻分类
- **变化检测两种范式**:
  - 有监督: 双时相 embedding 拼接 → Linear Probe → 二分类
  - 无监督: embedding 余弦距离阈值 → BA 评估
- **核心指标**: Balanced Accuracy (BA) 用于分类和变化检测；R²/MAE 用于回归

### S22 — 推理与量化
- 32-bit float embedding 量化到 8-bit，存储减少 4 倍，性能损失可忽略
- 量化方案: min-max 线性映射到 [0, 255]
- 全球 embedding field 按 UTM 分区和固定 tile 生成

### 关键结论
论文**没有使用 temporal contrastive loss**。时间敏感性是通过以下条件化机制隐式学习的：
1. **Valid period 条件摘要**: embedding 与有效期强绑定
2. **教师-学生一致性**: 扰动鲁棒性迫使模型学习时间无关的语义
3. **多源重建**: 不同时间窗口的输入重建相同目标，迫使 embedding 编码时间变化
4. **Batch uniformity**: 防止坍缩，保持空间区分度

---

## 二、如何定量评价 Embedding 的好坏

### 2.1 直接度量（无需下游标签）

| 指标 | 计算方法 | 好 embedding 的标准 | 当前模型值 | 官方值 |
|------|---------|-------------------|-----------|--------|
| **时间区分度 (Temporal Discrimination)** | 同像素不同时间的 cosine distance | 有变化区域应 > 0.1，无变化区域应 < 0.05 | pixel: **0.0374±0.007** | pixel: **0.0326±0.022** |
| **空间区分度 (Spatial Discrimination)** | 同时间不同像素的 cosine distance | 应显著大于时间区分度 | — | **0.278±0.184** |
| **Uniformity Loss** | RBF-based batch uniformity | 64维约 -4，128维约 -4.5 | 未记录 | 未记录 |
| **重建误差 (Reconstruction Loss)** | L1 + CE | 越低越好 | ~0.29 | 未公开 |
| **Embedding 范数分布** | 统计 ||x||₂ | 应接近 1.0 (球面约束) | — | — |

### 2.2 下游任务度量（需要标签）

| 指标 | 适用场景 | 好 embedding 的标准 |
|------|---------|-------------------|
| **变化检测 AUC** | 双时相 embedding 对比 | > 0.7 有实用价值，> 0.9 优秀 |
| **Linear Probe BA** | 分类任务 | > 70% 表示 embedding 语义丰富 |
| **kNN BA (k=3)** | 少样本分类 | 与 Linear Probe 差距小表示流形良好 |
| **Few-shot F1** | 极端稀疏数据 | 1-shot/10-shot 性能下降少 |

### 2.3 关键洞察：时间区分度的"正确理解"

**年度 embedding 的时间区分度本应很低**——这是预期行为，不是 bug。

原因：
- 年度汇总 embedding 的目标是编码"该年度内地表的平均语义状态"
- 大多数地表在相邻年份间没有显著变化（建筑、道路、森林等稳定地物）
- 真正有价值的时间信号存在于**更细粒度的时间窗口**（季度、月度）

因此，评价时间敏感性应该：
1. **用季度/月度窗口对比**，而非年度对比
2. **聚焦已知变化区域**，计算变化 vs 未变化的 distance 分离度
3. **结合空间区分度**：如果时间区分度 ≈ 空间区分度，说明模型没有学到时间

---

## 三、旧版本为什么能做到高 AUC

### 3.1 旧版本 (V6/V7) 核心配置

```yaml
# V7 配置关键参数
model:
  embedding_dim: 64          # 更小维度，信息更密集
  num_blocks: 6              # 更轻量主干
  vmf_kappa: 100.0           # 渐进 50→500

training:
  reconstruction_weight: 1.0
  uniformity_weight: 1.0     # 高权重！
  variance_weight: 0.25      # VICReg 方差
  consistency_weight: 0.05   # 教师-学生一致性
  classification_weight: 0.05 # 分类头
```

### 3.2 高 AUC 的关键因素

| 因素 | 旧版本 (V7) | 当前版本 (V4) | 影响 |
|------|------------|--------------|------|
| **教师-学生一致性** | ✅ weight=0.05-0.1 | ❌ 完全移除 | 教师-学生一致性是论文核心机制，移除导致鲁棒性下降 |
| **分类头训练** | ✅ weight=0.05-0.3 | ❌ 移除 | 分类头迫使 embedding 编码语义信息 |
| **Uniformity 权重** | ✅ 0.3-1.0 | ⚠️ koleo=0.15 | Uniformity 权重太低，embedding 可能坍缩 |
| **Variance + Decorrelation** | ✅ VICReg+Barlow Twins | ⚠️ VICReg 已禁用 | 反坍缩机制被削弱 |
| **数据预处理** | ✅ log变换+±6σ+分类编码 | ⚠️ 部分简化 | 预处理不一致导致输入分布偏移 |
| **Temporal Loss** | ❌ 无 | ✅ weight=1.0 | 理论上帮助时间敏感性，但实际未生效 |
| **VMF Kappa 渐进** | ✅ 50→500 | ❌ 固定 2000 | 渐进 kappa 让模型先学自由表示再约束 |

### 3.3 核心结论

旧版本能做到较高 AUC（文档中提到建筑物提取 IoU > 50%，变化检测 F1 > 50%），**不是因为 temporal loss，而是因为**：

1. **教师-学生一致性** 强迫 embedding 对输入扰动不变 → 提取稳定语义
2. **高 uniformity 权重 + variance + decorrelation** 防止坍缩 → 保持空间区分度
3. **分类头** 提供语义监督信号 → 迫使 embedding 编码可分类的特征
4. **完善的数据预处理** → 输入分布与论文一致

当前版本虽然加入了 temporal loss，但：
- 移除了教师-学生一致性（论文核心）
- uniformity 权重从 1.0 降到 0.15
- 移除了分类头
- VICReg 被禁用
- temporal loss 本身设计有问题（见第四部分）

---

## 四、官方 AEF 哈尔滨 Embedding 区分度分析

### 4.1 基本统计

```
文件: alphaearth_harbin_2023.tif / alphaearth_harbin_2024.tif
形状: [64 bands, 3280 rows, 3519 cols]
数据类型: float64 (实际为 uint8 量化后映射)
每 band 唯一值: ~116 (确认是 8-bit 量化)
```

### 4.2 时间区分度 (2023 vs 2024)

```
Cosine Similarity:
  mean:  0.9348
  std:   0.0437
  min:   0.4741
  max:   0.9976
  median: 0.9438

Cosine Distance (=(1-cos_sim)/2):
  mean:   0.0326
  std:    0.0219
  min:    0.0012
  max:    0.2630
  median: 0.0281
  
  p25:    0.0159
  p50:    0.0281
  p75:    0.0441
  p90:    0.0612
  p95:    0.0735
  p99:    0.1041
```

### 4.3 空间区分度 (同一年不同像素)

```
Same-year different-pixel cosine distance:
  mean:   0.2779
  std:    0.1839
  median: 0.2322
```

### 4.4 关键发现

| 发现 | 含义 |
|------|------|
| **时间区分度 (0.033) << 空间区分度 (0.278)** | 年度 embedding 主要编码语义，时间变化信息被高度压缩 |
| **99% 像素的 distance < 0.105** | 绝大多数区域年度间几乎无变化 |
| **change_score 均值 0.033, max 0.263** | 官方也认识到年度变化信号非常微弱 |
| **量化导致每 band 仅 ~116 个唯一值** | 8-bit 量化严重限制了 embedding 的表达能力 |

### 4.5 对变化检测的启示

官方年度 embedding **不适合直接用于无监督变化检测**（仅靠余弦距离阈值）：
- 变化信号淹没在大量未变化像素中
- 需要**监督学习**（拼接+Linear Probe）或**更细粒度时间窗口**

论文中 71.3% BA 的无监督变化检测结果，是在**LCMAP 年度 land cover change 标注**上评估的，这些标注覆盖的是**已知有较大变化**的区域，而非全局随机采样。

---

## 五、当前模型 (V4, Epoch 113) 与官方对比

### 5.1 时间区分度对比

| 指标 | 当前模型 (V4) | 官方 AEF | 评估 |
|------|--------------|---------|------|
| Pixel-level cosine distance | **0.0374 ± 0.007** | **0.0326 ± 0.022** | ✅ 接近 |
| Global-level cosine distance | **0.0075 ± 0.0078** | **~0.0325** | ⚠️ 显著偏低 |
| 空间区分度 | 未测量 | 0.278 ± 0.184 | — |

### 5.2 训练动态对比 (Epoch 101-114)

```
当前模型 V4:
  total loss:   15.54 → 15.39 (几乎停滞)
  recon loss:   0.288 → 0.289 (持平)
  temporal loss: 9.77 → 9.74 (完全没降)
  koleo loss:    3.20 → 2.46 (缓慢下降)
  pre_sim:       ~0.98 (持续极高)
```

### 5.3 诊断：当前模型的核心问题

**问题 1: Temporal Loss 没有驱动实际学习**
- temporal loss 数值虽大 (~9.8)，但 12 个 epoch 完全不下降
- 模型学到的"策略"：保持全局语义一致 (pre_sim=0.98)，在像素级添加微小随机扰动
- 这些扰动满足 temporal loss 的数学目标，但对变化检测无用

**问题 2: 移除了论文核心机制**
- 教师-学生一致性被完全移除
- 分类头被移除
- uniformity 权重从 1.0 降到 0.15
- 反坍缩机制（VICReg + decorrelation）被削弱

**问题 3: Temporal Loss 与 Reconstruction 的根本冲突**
- recon 需要保留语义 → 推使 embeddings 相似
- temporal 需要区分时间 → 推使 embeddings 不同
- 模型折中：全局语义一致，像素级添加噪声

---

## 六、改进建议

### 6.1 短期（立即实施）

1. **恢复教师-学生一致性损失**
   - 这是论文核心机制，不能移除
   - weight = 0.05-0.1，与 reconstruction 协同而非对抗

2. **恢复分类头训练**
   - 即使是弱监督（WorldCover 6 类），也能提供语义信号
   - weight = 0.05-0.1

3. **提升 uniformity 权重至 0.5-1.0**
   - 当前 koleo=0.15 太弱，embedding 容易坍缩
   - 或恢复 raw_uniformity_loss + decorrelation + variance 组合

4. **重新设计 temporal loss**
   - 当前对所有像素一视同仁，与 recon 冲突
   - 建议：只对"可能变化"的像素（如 NDVI 变化 > 阈值）计算 temporal loss
   - 或改用"时间预测"目标：让模型预测时间差，而非强行推远 embedding

### 6.2 中期（下一轮训练）

5. **使用更细粒度时间窗口验证**
   - 年度对比的区分度天然很低
   - 用季度/月度窗口验证时间敏感性

6. **渐进式 VMF kappa**
   - 从 50 开始，逐步增加到 500-2000
   - 让模型先学自由表示，再约束到球面

7. **增加数据量**
   - 当前 424 patches 远远不够
   - 论文使用 30 亿观测（覆盖 1.1% 陆地面积）

### 6.3 评价 embedding 的标准流程

建议建立以下评价流水线：

```
1. 无监督度量:
   - 时间区分度 (季度/月度窗口)
   - 空间区分度
   - uniformity loss
   - 重建误差

2. 半监督度量:
   - WorldCover 6 类 Linear Probe BA
   - OSM 建筑物 KNN F1

3. 变化检测度量:
   - 有监督: 拼接+Linear Probe → AUC/BA
   - 无监督: cosine distance + 阈值搜索 → BA
   - 在已知变化区域上单独评估
```

---

## 七、总结

| 维度 | 结论 |
|------|------|
| **官方 embedding** | 年度汇总的时间区分度本身就很低 (0.033)，这是预期行为。空间区分度 (0.278) 远大于时间区分度。变化检测需要监督学习或更细粒度时间窗口。 |
| **当前模型问题** | 不是"时间区分度不够"，而是**训练目标设计不当**。Temporal loss 与 recon 冲突且未生效，同时移除了论文核心的教师-学生一致性和分类头。 |
| **旧版本优势** | 旧版本虽然没有 temporal loss，但靠**教师-学生一致性 + 高 uniformity + 分类头 + 完善预处理**获得了更好的语义编码和下游性能。 |
| **下一步** | 恢复教师-学生一致性和分类头，提升 uniformity 权重，重新设计 temporal loss（只对变化像素计算），建立标准评价流水线。 |

---

*报告生成完毕。如需进一步分析特定 patch 的可视化对比或运行完整 AUC 验证，请告知。*
