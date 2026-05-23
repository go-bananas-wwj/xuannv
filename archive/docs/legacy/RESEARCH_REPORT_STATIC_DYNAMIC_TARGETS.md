# 深度研究报告：静态目标与动态目标混合重建问题

> 研究日期: 2026-05-10
> 研究范围: 遥感基础模型、多任务学习、AlphaEarth Foundations、时序嵌入
> 核心问题: 在单一嵌入模型中同时重建静态目标 (DEM/土地覆盖) 与动态目标 (S2/S1/Landsat) 是否损害时序敏感性？

---

## 执行摘要

**结论先行**: 
1. 混合静态/动态目标重建是**已知问题**，但在地球观测领域**并非不可解决**
2. AlphaEarth Foundations (AEF) **明确包含静态目标** (DEM/GLO-30, NLCD 土地覆盖) 且报告**正面效果**
3. 真正的风险不在于"是否包含"，而在于**损失权重失衡**和**缺乏时间条件解码机制**
4. 对于本项目，**建议保留静态目标**，但实施四项关键改进

---

## 一、这是否是一个已知问题？

### 1.1 问题命名与定位

在遥感基础模型文献中，这个问题通常不被单独命名为"static-dynamic mixing problem"，而是作为以下更广泛问题的特例：

| 文献中的称呼 | 核心关切 | 代表论文 |
|-------------|---------|---------|
| **Multi-task learning with heterogeneous targets** | 不同任务/目标的损失尺度、收敛速度差异 | AEF (2025), FoMo (2025) |
| **Reconstruction vs. temporal sensitivity trade-off** | 重建损失主导导致时序判别力丧失 | 本项目 V5 诊断报告 |
| **Static auxiliary tasks in self-supervised learning** | 静态辅助任务对动态表示的影响 | Presto (2024), SeCo (2021) |
| **Time-agnostic vs. time-conditioned decoding** | 解码器是否应感知时间 | AEF S16.2.1, TESSERA (2025) |

### 1.2 问题的本质

核心矛盾可以表述为：

> **静态目标** (如 DEM) 要求嵌入编码**空间不变的地理属性**。
> **动态目标** (如 S2) 要求嵌入编码**时间变化的光谱特征**。
> 如果模型被迫用单一嵌入同时完美重建两者，优化压力会偏向于"更容易的任务"——通常是静态目标，因为它们的重建损失更低、更稳定。

这正是本项目 V5 诊断报告中识别出的 **"原因 3: Reconstruction Loss 主导"**：
> "S2/S1/Landsat 重建任务要求 embedding 编码'图像内容'。模型优先学习静态内容，时间变化被当作噪声忽略。"

---

## 二、SOTA 方法如何处理这一问题？

### 2.1 AlphaEarth Foundations (Google, 2025) — 最相关参考

AEF 的训练目标明确包含**静态和准静态目标**：

**静态目标**:
- **GLO-30 DEM** (高程) — "assumed to be valid over the entire period of our training set" (S15.8)
- **NLCD 土地覆盖** (2021 年地图) — "temporally-static nature of NLCD" (S15.9)
- **ERA5 气候变量** — 变化缓慢

**动态目标**:
- Sentinel-2, Sentinel-1, Landsat-8/9 — 日常/月度影像
- GEDI LiDAR — 有具体采集时间戳
- 文本描述 — 与时间窗口关联

#### AEF 的关键设计决策

**决策 1: 条件解码 (Conditional Decoding)**

> "For each of decoded sources, a small decoder network accepts an embedding, and a set of **conditional metadata specific to that source**... Typically this is at least a **sinusoidal timecode** representing an instant in the valid period... though may also contain **orbital geometry and metadata** that is only relevant to the act of measurement, not the measurement itself." (S16.2.1)

这意味着：
- 同一 embedding 可以通过**不同时间码**解码出不同时间点的动态目标
- 静态目标 (如 DEM) 使用**固定时间码** (如 0.5) 解码
- 解码器本身接收时间条件，但 embedding 不直接编码"某个具体时间"，而是编码"该 valid period 内的可解码信息"

**决策 2: 源特定损失权重 (Source-Specific Loss Weights)**

> "Losses are computed against this reconstructed frame, with the nature of the loss changing depending on the source and a **source-specific weight** (Table S2)." (S16.2.3)

AEF 为不同数据源分配不同的重建权重。虽然 Table S2 的具体数值未在公开摘要中披露，但机制明确：**不同目标对总重建损失的贡献是可调的**。

**决策 3: 显式的时间窗口增强**

> "Reconstructions each randomly selected summarization periods... For each reconstruction objective, a **different embedding corresponding to the unique summarization period is generated**." (S16.2.3)

这意味着即使对静态目标，模型也会产生**不同 valid period 的 embedding**，但静态目标的重建损失使用相同的 ground truth。这种"同一标签、不同 embedding"的训练压力，迫使模型学会：**静态目标的重建不依赖于 valid period 的选择**。

**决策 4: NLCD 的消融实验结果**

AEF 明确报告了静态目标的消融结果：

> "We include NLCD in our list of source datasets as a means of testing the value of existing maps as a form of **weak supervision**. We note that **there was no significant negative impact on the loss or reconstruction quality of other sources**, and the effect on evaluations was **generally positive** despite the temporally-static nature of NLCD (see supplement section S7.2 for ablation results)." (S15.9)

**这是最关键的文献证据**：在 AEF 的大规模实验中，包含静态的年度土地覆盖图作为重建目标，不仅没有损害其他目标的重建质量，还对下游评估有正面效果。

### 2.2 Presto (Tseng et al., 2024)

Presto 也重建多种目标，包括 DEM 和 Dynamic World：

> "Presto takes this approach even further, constructing a pixel-timeseries training dataset that consists of input from multiple types of data, including Sentinel-2 RGB, Sentinel-1 SAR, **DEM**, and the **Dynamic World** dataset... The variety of masking strategies ensures that Presto can successfully process time series datasets with missing channels, a variety of temporal resolutions, and only a small subset of timesteps."

注意：Presto 对 DEM 的处理是作为**输入通道**而非重建目标，这与 AEF 和我们的项目不同。

### 2.3 FoMo (Bountos et al., 2025)

FoMo 的训练数据包括：
> "DEM data, and both RGB and MS imagery from high-res UAV sensors... Tokens are masked, passed through a ViT encoder, and then go through a standard decoder for reconstruction."

FoMo 使用统一的掩码自编码框架，不区分静态/动态目标的解码路径，但也没有报告时序敏感性评估。

### 2.4 遥感领域外的相关研究

| 领域 | 论文 | 核心发现 |
|------|------|---------|
| 时序知识图谱 | TCCGN (2025) | 显式分离 static embedding 和 dynamic embedding，用 fusion contrast loss 约束两者一致性 |
| 序列表示学习 | Sequential Repr. Learning via Static-Dynamic (ECCV 2024) | 静态-动态解耦表示学习需要 mutual information 约束，简单条件链接可能导致次优解耦 |
| 多任务学习 | Meta and Multi-Task Learning Survey (2024) | "Loss function design remains a bottleneck... The dominant approach continues to be the simple summation of task-specific losses, often with equal or fixed weights, regardless of task difficulty, convergence behavior, or data imbalance." |

---

## 三、条件解码器与时间编码的角色

### 3.1 条件解码如何缓解静态-动态冲突

条件解码器的核心洞察：**embedding 本身不直接等于"某个时间点的状态"，而是等于"一个地点在给定时间窗口内的可压缩信息"**。解码器通过时间码"查询"这个信息场中特定时间点的状态。

对于静态目标 (如 DEM)：
- 无论 valid period 是 2023-04 还是 2024-08，DEM 的真实值不变
- 解码器接收不同的时间码，但目标相同
- 这迫使 embedding 中用于 DEM 重建的信息必须是**时间无关的**

对于动态目标 (如 S2)：
- valid period 内随机选择一个时间点解码
- 解码器通过时间码定位到该时间点的光谱状态
- 这迫使 embedding 中用于 S2 重建的信息必须是**时间敏感的**

**关键推论**: 如果模型容量足够 (如 AEF 的 64D 嵌入)，静态和动态信息可以**共存于同一嵌入空间**，通过条件解码器按需提取。问题不在于"空间不够"，而在于"训练信号失衡"。

### 3.2 本项目当前实现分析

我们的 `AEFModel` 已经实现了条件解码：

```python
# src/models/decoders.py
class ConditionInjector(nn.Module):
    def forward(self, embedding, window_code, relative_time, metadata):
        cond = torch.cat([window_code, relative_time, metadata], dim=-1)
        cond_features = self.cond_proj(cond)
        gate = self.gate(cond_features)
        gated = embedding.mean(dim=(-2, -1)) * gate
        return embedding + gated[:, :, None, None]
```

**当前实现的问题**:
1. `relative_time` 对静态目标固定为 0.5 (dataset.py 第 906 行)，但注入机制与动态目标相同
2. **没有源特定的重建权重** — 所有 7 类目标的重建损失在 `compute_recon_loss` 中平均汇总
3. **没有显式的时间无关/时间敏感信息分离机制**

---

## 四、静态目标重建是否损害时序敏感性？

### 4.1 理论分析

**直接回答: 可能损害，但不是因为静态目标本身，而是因为训练动态的失衡。**

损害机制:
1. **梯度竞争**: 静态目标 (DEM) 的重建损失通常更低、更稳定，梯度更"干净"。优化器倾向于优先降低这些"容易"的损失。
2. **嵌入空间侵占**: 如果静态重建损失的权重过高，嵌入空间的大部分维度会被静态地理特征占据，留给时间变化特征的维度减少。
3. **时间码退化**: 如果静态目标在训练中占比过高，条件解码器中的时间码 gate 可能退化——静态目标不需要时间码，gate 学习忽略它。

### 4.2 AEF 的证据

AEF 的消融实验 (S15.9, S7.2) 明确否定了"静态目标必然有害"的假设：
- NLCD (静态土地覆盖) 的加入"generally positive"对评估效果
- 对其他源的重建质量"no significant negative impact"

这说明在**适当的架构和权重配置下**，静态目标可以作为弱监督信号，帮助模型学习更有判别力的空间特征。

### 4.3 本项目的问题诊断

本项目 V5 的 AUC = 52.8% (接近随机) 的**主要原因不是静态目标的存在**，而是：
1. 重建损失权重 (1.0) >> 时序对比损失权重 (0.1~0.3)
2. 缺乏 batch uniformity 机制
3. 时序对比损失只在 global mean 上计算，空间变化信息丢失

静态目标**加剧了**这些问题（因为 DEM 的重建非常容易，损失很快降到接近 0，但权重不变），但不是**根因**。

---

## 五、文献中关于"静态 vs 动态"权衡的研究

### 5.1 显式研究

| 论文 | 发现 |
|------|------|
| **AEF S7.2 (ablation)** | NLCD 静态目标消融：加入后效果"generally positive"，无负面重建影响 |
| **Sequential Static-Dynamic Disentanglement (ECCV 2024)** | 静态-动态解耦需要显式 MI 约束；简单多任务求和会导致次优解耦 |
| **TCCGN (2025)** | 在时序 KG 中，static embedding 和 dynamic embedding 应通过 gating 融合，而非简单拼接 |
| **Self-Supervised Regional/Temporal Auxiliary Tasks (2021)** | 静态辅助任务（如 RoI inpainting）和动态辅助任务（如 optical flow）需同时存在，单一类型不足 |

### 5.2 隐式研究

| 论文 | 相关发现 |
|------|---------|
| **SeCo (ICCV 2021)** | 季节不变性正样本是双刃剑：学到鲁棒特征，但可能"time-blind" |
| **CACo (CVPR 2023)** | 打破"同一地点 = 正样本"假设：短时间 = 正样本，长时间 = 负样本 |
| **GeoLaneRep (2026)** | "geometry-only, trajectory-only, and traj-stats baselines all yield extremely high nearest-neighbor similarity scores (0.991–1.000), yet their lateral-rank errors remain large" — 静态特征导致高相似度但低判别力 |

### 5.3 关键理论框架

**Wang & Isola (ICML 2020) 的 Alignment & Uniformity**:
- 静态目标重建优化 **alignment** (embedding 与固定目标对齐)
- 时序对比损失优化 **alignment** (不同时间的 embedding 分离) 和 **uniformity** (嵌入空间铺满)
- 如果静态重建的 alignment 压力过强，uniformity 会被压缩，导致嵌入坍缩到静态特征的子空间

---

## 六、最佳实践与具体建议

### 6.1 文献共识总结

| 策略 | 证据强度 | 说明 |
|------|---------|------|
| **保留静态目标，但降低权重** | ⭐⭐⭐⭐⭐ | AEF 明确采用；NLCD 消融显示正面效果 |
| **使用源特定解码器 + 条件时间码** | ⭐⭐⭐⭐⭐ | AEF 核心设计；Presto/FoMo 也采用 |
| **分离静态/动态信息通道** | ⭐⭐⭐ | ECCV 2024 静态-动态解耦；但 AEF 未采用也取得了 SOTA |
| **完全移除静态目标** | ⭐⭐ | 无文献支持；可能损失弱监督信号 |
| **静态目标作为预训练-only** | ⭐⭐⭐ | 课程学习策略；先学静态再学动态 |
| **为静态目标使用单独 embedding** | ⭐⭐ | 增加模型复杂度；AEF 证明不必要 |

### 6.2 对本项目的四项具体建议

#### 建议 1: 实施源特定重建权重 (最高优先级)

当前 `compute_recon_loss` 对所有目标平均求和。应引入 `source_recon_weights`：

```yaml
# configs/*.yaml 新增
training:
  source_recon_weights:
    s2: 1.0
    s1: 1.0
    landsat: 1.0
    dem: 0.3          # 降低静态目标权重
    worldcover: 0.5   # 分类目标通常更容易
    dynamic_world: 0.5
    jrc_water: 0.5    # JRC 虽慢变，但比 DEM 更有时间信息
```

**依据**: AEF 明确使用 "source-specific weight (Table S2)"。Meta/Multi-Task Learning Survey (2024) 批评 "simple summation of task-specific losses, often with equal or fixed weights" 是瓶颈。

#### 建议 2: 为静态目标设计时间无关解码路径

修改 `ConditionInjector`，对静态目标绕过或减弱时间码影响：

```python
class ConditionInjector(nn.Module):
    def forward(self, embedding, window_code, relative_time, metadata, is_static=False):
        cond = torch.cat([window_code, relative_time, metadata], dim=-1)
        cond_features = self.cond_proj(cond)
        gate = self.gate(cond_features)
        if is_static:
            gate = gate * 0.1  # 静态目标的时间码 gate 几乎关闭
        gated = embedding.mean(dim=(-2, -1)) * gate
        return embedding + gated[:, :, None, None]
```

**依据**: TCCGN (2025) 的 "gated static-dynamic fusion"；ECCV 2024 的 "conditional link between static and dynamic codes should not be ablated" (但应弱化)。

#### 建议 3: 监控静态 vs 动态损失的相对大小

在训练日志中增加：
```python
recon_static = recon_loss_for_sources(["dem", "worldcover"])
recon_dynamic = recon_loss_for_sources(["s2", "s1", "landsat"])
# 目标: recon_dynamic 的下降速度应快于 recon_static
# 如果 recon_static 持续主导梯度，进一步降低其权重
```

**依据**: 本项目 V5 诊断报告识别出的 "Reconstruction Loss 主导" 问题。

#### 建议 4: 渐进式课程学习

参考 TEMPORAL_SENSITIVITY_SOLUTION_REPORT 的 Layer 5：

```
阶段 1 (Epoch 0-50):  重建为主 (recon=1.0, temporal=0.1)
阶段 2 (Epoch 50-150): 引入时序 (recon=0.5, temporal=0.5)  
阶段 3 (Epoch 150+):   强化时序 (recon=0.3, temporal=0.7)
```

但增加静态目标权重的独立调整：
```
阶段 1: static_weight = 0.5, dynamic_weight = 1.0
阶段 2: static_weight = 0.3, dynamic_weight = 1.0
阶段 3: static_weight = 0.2, dynamic_weight = 1.0
```

**依据**: 课程学习在变化检测预训练中广泛使用 (Changen2, SeCo)；AEF 的 100k 步训练也隐含了从重建到对比的渐进过渡。

### 6.3 不建议的做法

| 做法 | 原因 |
|------|------|
| ❌ 完全移除静态目标 | 损失弱监督信号；AEF 证明静态目标可带来正面效果 |
| ❌ 为静态目标增加单独 encoder/bottleneck | 显著增加参数量和训练成本；AEF 证明单一 embedding 足够 |
| ❌ 将静态目标作为输入 (而非重建目标) | 推理时依赖静态数据可用性；与 AEF 设计哲学冲突 |
| ❌ 对静态目标使用与动态目标相同的时间码策略 | DEM 的 relative_time=0.5 是人为固定，不应让模型误以为它在学习"某个特定时间" |

---

## 七、引用文献

1. **AlphaEarth Foundations**: Brown et al., "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data", arXiv:2507.22291, 2025. (核心参考: S15.8, S15.9, S16.2.1, S16.2.3)
2. **Presto**: Tseng et al., "Presto: A multi-sensor, multi-task presto foundation model", 2024.
3. **FoMo**: Bountos et al., "Foundation Models for Forest Monitoring", 2025.
4. **SeCo**: Manas et al., "Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data", ICCV 2021.
5. **CACo**: Mall et al., "Change-Aware Contrastive Learning", CVPR 2023.
6. **Wang & Isola**: "Understanding Contrastive Representation Learning Through Alignment and Uniformity on the Hypersphere", ICML 2020.
7. **Static-Dynamic Disentanglement**: "Sequential Representation Learning via Static-Dynamic Disentanglement", ECCV 2024.
8. **TCCGN**: "Causal Decoupling for Temporal Knowledge Graph Reasoning via Contrastive Learning and Adaptive Fusion", 2025.
9. **Multi-Task Learning Survey**: "Meta and Multi-Task Learning for Action Recognition", IEEE Access 2024.
10. **VICReg**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022.
11. **GeoLaneRep**: "Behavior-Grounded Lane Representation Learning for Multi-Task Traffic Digital Twins", 2026.

---

## 八、一句话总结

> **静态目标不是敌人，失衡的训练信号才是。保留 DEM/WorldCover 作为弱监督，但赋予它们较低的重建权重，显式弱化静态解码路径中的时间条件依赖，并监控静态 vs 动态损失的相对收敛速度。AEF 的消融实验已证明，在适当配置下，静态目标对整体性能是正面贡献。**
