# 基于 Embedding / 特征的变化检测方法调研报告

> 调研时间：2026-05-15  
> 调研范围：遥感（Remote Sensing）领域基于 embedding 的变化检测方法，涵盖 2021-2025 年经典方法与最新 SOTA  
> 关键词：embedding based change detection, CACo, ChangeStar, BIT, temporal contrastive learning, CD head, cosine distance

---

## 目录

1. [Bare 方法：直接使用 Cosine / L2 Distance 的变化检测](#1-bare-方法直接使用-cosine--l2-distance-的变化检测)
2. [CD Head（变化检测头）的设计](#2-cd-head变化检测头的设计)
3. [时序 Embedding 对比的经典方法](#3-时序-embedding-对比的经典方法)
4. [如何训练使 Embedding 对变化敏感](#4-如何训练使-embedding-对变化敏感)
5. [评估指标与 Benchmark 数据集](#5-评估指标与-benchmark-数据集)
6. [2024-2025 最新 SOTA 方法](#6-2024-2025-最新-sota-方法)
7. [关键参考文献汇总](#7-关键参考文献汇总)

---

## 1. Bare 方法：直接使用 Cosine / L2 Distance 的变化检测

### 1.1 核心思想

Bare 方法（又称 "裸方法"）是指**不使用任何额外训练的变化检测头**，直接利用预训练模型提取的双时相 embedding，通过距离度量来判断变化。其优势在于：
- **零额外训练成本**：无需标注的变化检测数据
- **即插即用**：任何预训练 backbone 均可直接应用
- **可解释性强**：变化强度直接由距离量化

### 1.2 常用距离度量

| 距离度量 | 公式 | 适用场景 | 来源 |
|---------|------|---------|------|
| **Cosine Distance** | `1 - cos(θ) = 1 - (a·b)/(||a||·||b||)` | 归一化 embedding 空间，对幅度变化不敏感 | [Rodrigues et al., ICASSP 2014](https://homepages.dcc.ufmg.br/~erickson/publications/rodrigues_icassp2014.pdf) |
| **L2 (Euclidean) Distance** | `||a - b||₂` | 原始特征空间，保留幅度信息 | 通用基准 |
| **Correlation Coefficient** | `ρ = Cov(a,b)/(σₐσᵦ)` | 统计相关性度量 | [Rodrigues et al., 2014](https://homepages.dcc.ufmg.br/~erickson/publications/rodrigues_icassp2014.pdf) |
| **Spectral Angle Distance** | `arccos((a·b)/(||a||·||b||))` | 高光谱/多光谱遥感 | [SJAN, Remote Sensing 2022](https://www.mdpi.com/2072-4292/14/14/3394) |

> **重要发现**：Rodrigues 等人在 OFFICE 数据集上的对比实验表明，**Cosine Distance 和 Correlation Coefficient 的 AUC 达到 0.9985 和 0.9982**，而 L2 Distance 仅 0.6959。这证明在 embedding 空间中，**角度/方向信息比幅度信息对变化更敏感**。

### 1.3 AEF (AlphaEarth Foundations) 的 Bare 方法实践

Google DeepMind 的 **AEF** 模型（Brown et al., 2025）是 Bare 方法的典型工业级应用：

- **模型输出**：64 维单位长度 embedding（L2 normalized）
- **变化检测方式**：
  - **有监督**：将双时相 embedding 拼接后训练线性分类器或 kNN (k=3)
  - **无监督**：直接计算 embedding 间的 dot product（即 cosine similarity），设定阈值生成变化掩膜
- **性能**：在 LCMAP 变化检测基准上，AEF 无监督方法达到 **71.3% ± 1.14 Balanced Accuracy**，有监督达到 **78.4% ± 1.11**
- **关键洞察**：AEF embedding 被约束在单位超球面上（von Mises-Fisher 分布），使得 cosine similarity 成为自然的变化度量

### 1.4 Element 84 的 Sentinel-2 Embedding 实验

Element 84 的研究展示了基于 SSL4EO-S12 预训练模型的 Bare 方法：
- 使用自监督预训练模型将卫星图像转换为高质量 vector embedding
- 通过 **outlier detection**（如 Isolation Forest、k-NN）在 embedding 空间中检测异常
- 优势：可处理季节性变化，无需人工选择对比时间点
- 参考：[Element 84 Blog, 2023](https://element84.com/machine-learning/exploring-unsupervised-change-detection-with-sentinel-2-vector-embeddings/)

### 1.5 Bare 方法的优缺点

| 优点 | 缺点 |
|------|------|
| 无需变化检测训练数据 | 性能上限受限于预训练 embedding 的质量 |
| 计算简单，推理快速 | 无法区分变化类型（仅二分类） |
| 对域偏移有一定鲁棒性 | 对伪变化（光照、季节）敏感 |
| 适合 few-shot / zero-shot 场景 | 阈值选择困难 |

---

## 2. CD Head（变化检测头）的设计

### 2.1 什么是 CD Head

CD Head 是指在双时相特征提取器之上，**专门设计用于变化检测任务的可训练模块**。它将两个时间点的特征融合后输出像素级的变化概率图。

### 2.2 经典 CD Head 设计范式

#### 2.2.1 ChangeMixin / ChangeMixin2（ChangeStar 系列）

由 Zheng 等人（Wuhan University, 2021-2024）提出：

**ChangeMixin（v1）核心设计**：
- **Temporal Swap Network**：将双时相特征 `X_t` 和 `X_{t+1}` 沿通道维度以两种顺序拼接：
  ```
  TSM(X_t, X_{t+1}) = [cat(X_t, X_{t+1}), cat(X_{t+1}, X_t)]
  ```
- 通过 small FCN（N 层 3×3 Conv + BN + ReLU）处理
- 输出：binary change map
- **Inductive Bias**：Temporal Symmetry（时间对称性）

**ChangeMixin2（v2，2024）改进**：
- 引入 **Temporal Difference Network (TDN)** 解决早期收敛慢的问题
- 融合公式：
  ```
  X_{t→t+1} = X^{tdn}_{t↔t+1} + X^{tsn}_{t→t+1}
  ```
  - `tdn`：时序差分网络（Temporal Difference Network）
  - `tsn`：时序交换网络（Temporal Swap Network）
- 支持三种变化检测任务：
  - Binary Change Detection (BCD)
  - Object Change Detection (OCD)
  - Semantic Change Detection (SCD)

> **性能**：ChangeStar2 在 CDD 数据集上达到 **98.0% F1**（Swin-T backbone），在 S2Looking 上达到 **68.8% F1**
> 
> 参考：[ChangeStar, ICCV 2021](https://arxiv.org/abs/2108.07002) | [ChangeStar2, 2024](https://arxiv.org/abs/2406.15694)

#### 2.2.2 BIT（Bitemporal Image Transformer）

由 Chen, Qi & Shi（2021）提出，是首个将 Transformer 引入遥感变化检测的工作：

**BIT 核心架构**（三个组件）：
1. **Siamese Semantic Tokenizer**：将像素分组为语义 token（visual words），实现空间压缩
2. **Transformer Encoder**：在紧凑的 token 空间中建模时空上下文
3. **Siamese Transformer Decoder**：将上下文丰富的 token 投影回像素空间，精炼原始特征

**工作流程**：
```
Input: I_t, I_{t+1}
  → CNN Backbone → Feature Maps F_t, F_{t+1}
  → Semantic Tokenizer → Tokens T_t, T_{t+1}
  → Transformer Encoder → Enhanced Tokens
  → Transformer Decoder → Refined Features F'_t, F'_{t+1}
  → Feature Differencing → Change Map
```

**关键洞察**：BIT 的核心假设是——变化的高层概念可以用少量语义 token 表示，这使得计算效率大幅提升。

> **性能**：BIT 在 LEVIR-CD 上达到 **89.94% F1**，仅用 ResNet18 backbone 就超越了多个 SOTA 方法，且计算成本仅为纯 CNN 基线的 1/3
>
> 参考：[BIT, IEEE TGRS 2021](https://arxiv.org/abs/2103.00208)

#### 2.2.3 轻量化 CD Head 设计（1M-CDNet / 3M-CDNet）

由 Li 等人（2021）提出的高效设计：

**3M-CDNet Classifier**：
```
Conv1×1 (768→256) → Bilinear Upsample ×2 → Conv3×3 (256→256) 
→ Conv3×3 (256→256) → Conv1×1 (256→2) → Bilinear Upsample ×2 → Sigmoid
```

**关键设计原则**：
- 使用 **高分辨率特征图**（1/2 输入分辨率）进行检测，保留小目标变化细节
- 两级特征融合策略（two-level feature fusion）
- Dropout 正则化提升泛化能力
- **1M-CDNet** 仅 1.26M 参数，在 LEVIR-CD 上达到 **91.18% F1**

> 参考：[1M-CDNet / 3M-CDNet, Remote Sensing 2021](https://www.mdpi.com/2072-4292/13/24/5152)

#### 2.2.4 MTF（Merge Temporal Features）模块

Wang 等人（2022）提出的 C3PO 方法中，MTF 模块将变化检测解耦为语义分割 + 变化检测：

**核心洞察**：变化检测中存在三种变化类型：
- **Appear**（出现）：t1 无 → t2 有
- **Disappear**（消失）：t1 有 → t2 无
- **Exchange**（替换）：t1 有 A → t2 有 B

MTF 采用 **多分支结构**，不同分支分别学习这三种变化类型，实验证明区分变化类型能显著提升性能。

> 参考：[C3PO / MTF, 2022](https://arxiv.org/abs/2206.07557)

### 2.3 CD Head 设计的关键原则总结

| 设计要素 | 说明 | 代表方法 |
|---------|------|---------|
| **Temporal Symmetry** | 强制模型对时间顺序不敏感 | ChangeMixin2 |
| **Feature Differencing** | 显式计算双时相特征差 | BIT, 多数方法 |
| **Token-based Context** | 在压缩 token 空间建模长程依赖 | BIT |
| **Multi-scale Fusion** | 融合不同尺度特征 | 3M-CDNet, SAM-MSCD |
| **Attention Mechanism** | 跨时相注意力对齐 | BiFA, SCANet |
| **Lightweight Design** | 参数量 < 5M，适合部署 | 1M-CDNet, LightCDNet |

---

## 3. 时序 Embedding 对比的经典方法

### 3.1 CACo（Change-Aware Contrastive Learning）

**作者**：Utkarsh Mall, Bharath Hariharan, Kavita Bala（Cornell University）  
**发表**：CVPR 2023  
**核心贡献**：首次将"变化感知"引入遥感图像的自监督对比学习

#### 3.1.1 核心思想

遥感图像具有独特的时间结构：
- **短期时间差**（<1 年）：主要反映季节性变化，应被视为**正样本**（pull closer）
- **长期时间差**（~4 年）：可能包含真实地表变化，应被视为**负样本**（push apart）——**但仅在确实发生变化时**

CACo 的关键洞察：**并非所有地点在长期都会变化**。城市区域可能变化剧烈，而森林区域可能多年不变。

#### 3.1.2 Change-Aware Contrastive Loss

```
标准对比损失（如 SeCo）:
  - 短期对 → 正样本
  - 长期对 → 负样本（无条件）

CACo 损失:
  - 短期对 → 正样本（同 SeCo）
  - 长期对 → 负样本（仅当估计存在变化时）
```

**Bootstrapping 机制**：
1. 用当前 representation 估计变化区域
2. 用变化估计改进 representation
3. 迭代优化

#### 3.1.3 地理采样策略

- 使用 **σ=5 km 的高斯采样器** 围绕城市区域采样
- 拒绝海洋样本并重新采样
- 城市区域变化概率更高，因此信息更丰富

#### 3.1.4 性能

| 方法 | Backbone | EuroSat (100k) | EuroSat (1M) |
|------|----------|----------------|--------------|
| SeCo | ResNet-50 | 93.12 | 95.63 |
| **CACo** | ResNet-50 | **94.48** | **95.90** |

下游任务提升：
- 语义分割：相对提升 **6.5%**
- 变化检测：相对提升 **8.5%**

> 参考：[CACo, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf) | [项目主页](https://research.cs.cornell.edu/caco/)

### 3.2 SeCo（Seasonal Contrast）

**核心思想**：利用遥感数据的季节性结构进行自监督预训练。

- **正样本**：同一地点、不同季节但同年份的图像对
- **负样本**：不同地点的图像
- 基于 MoCo v2 框架
- 学习对季节变化不变、对地理位置敏感的 representation

> 参考：[SeCo, ICCV 2021](https://arxiv.org/abs/2103.16607)

### 3.3 ChangeStar / STAR（Single-Temporal Supervised Learning）

**核心创新**：仅用单时相标注即可训练变化检测器。

**STAR 学习范式**：
- 传统方法需要成对的双时相标注图像（bitemporal supervision）
- STAR 从**未配对的单时相标注图像**中挖掘变化监督信号
- 变化标签通过语义标签的集合运算构造：`C_{i→j} = A(S_i, S_j)`

**关键优势**：
- 避免昂贵的人工成对标注
- 假设空间更大，泛化性更强
- 利用 Temporal Symmetry 作为归纳偏置防止过拟合

> 参考：[ChangeStar, ICCV 2021](https://arxiv.org/abs/2108.07002) | [ChangeStar2, 2024](https://arxiv.org/abs/2406.15694)

### 3.4 Deep Metric Learning for Unsupervised CD（WACV 2025）

Bandara & Patel 提出的**无监督深度度量学习方法**：

**核心设计**：
- 为每对双时相图像**单独优化**网络参数（非预训练后固定）
- 图像域损失 + 特征域损失 + 上下文一致性约束
- 使用 VGG-16 前两个尺度的深度特征

**损失组成**：
| 损失项 | 作用 | 贡献 |
|--------|------|------|
| `L_img` | 图像域相似度 | OA=0.911, F1=0.222, AUC=0.821 |
| `L_feat` | 特征域相似/不相似度 | 显著提升 |
| `L_ext` | 上下文一致性 | OA=0.958, F1=0.325, AUC=0.937 |
| **全部** | 联合优化 | **OA=0.958, F1=0.325, AUC=0.937** |

> 参考：[Deep Metric Learning CD, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Bandara_Deep_Metric_Learning_for_Unsupervised_Remote_Sensing_Change_Detection_WACV_2025_paper.pdf)

---

## 4. 如何训练使 Embedding 对变化敏感

### 4.1 核心训练策略汇总

| 策略 | 原理 | 代表方法 |
|------|------|---------|
| **Temporal Contrastive Learning** | 短期拉远/拉近，长期条件推远 | CACo, SeCo |
| **Reconstruction + Contrastive** | 重建保证信息量，对比学习保证判别性 | SatMAE, MA3E |
| **Feature Differencing Loss** | 显式优化特征差异 | Deep Metric Learning CD |
| **Anti-Collapse Regularization** | 防止 embedding 坍缩 | AEF (VMF + raw uniformity) |
| **Pixel-level Contrastive** | 像素级时序对比 | temporal_cosine_pixel_loss |
| **Multi-scale Feature Fusion** | 融合多层次变化信息 | ChangeStar2, SAM-MSCD |

### 4.2 防止 Embedding 坍缩的关键技术

Embedding 坍缩（collapse）是变化检测的核心挑战——如果模型将所有输入映射到相似向量，则无法检测变化。

#### 4.2.1 AEF 的反坍缩设计

AEF 采用多重机制防止坍缩：
1. **VMF Bottleneck**：embedding 作为 von Mises-Fisher 分布的均值方向，通过 L2 norm + 噪声采样保持球面结构
2. **Raw Uniformity Loss**：在 pre-norm 空间计算 uniformity，自适应温度 `t=2/D`
3. **Decorrelation Loss**：Barlow Twins 风格的去相关约束
4. **Bottleneck Orthogonality Loss**：Conv1×1 压缩层的正交约束
5. **Batch Rotation Consistency**：embedding 与 batch-rotated 版本比较，最小化 dot product 绝对值

#### 4.2.2 时序对比损失设计

**全局时序对比（Temporal Contrastive）**：
```python
# Hinge-style loss
loss = max(0, margin - sim(emb_t1, emb_t2_positive) + sim(emb_t1, emb_t2_negative))
```

**像素级时序对比（Pixel-level）**：
- **Temporal Cosine Pixel Loss**：逐像素 cosine distance，使变化区域 embedding 差异大
- **Anti-Diagonal InfoNCE**：像素级 InfoNCE，对角线为正样本
- **Gap-Aware Temporal Cosine**：根据时间间隔动态调整 target similarity

#### 4.2.3 训练监控指标

| 指标 | 正常范围 | 异常信号 | 处理建议 |
|------|---------|---------|---------|
| `raw_unif` | -4.0 ~ -1.0 | > -0.5 持续 5 epoch | 报告坍缩，检查 loss weights |
| `pre_unif` | 接近 raw_unif | 差距 > 0.5 | pre-norm 与 norm 空间不一致 |
| `recon` | < 0.3 | warmup 后 > 0.5 | 检查数据质量 |
| `var_reg` | 接近 0 | > 0.5 | 方差坍缩，增加正则化 |
| `orth` | < 0.3 | > 0.5 | 权重不正交 |
| `decorr` | < 1.0 | > 2.0 | 特征强相关 |

### 4.3 提升时间敏感性的训练技巧

1. **双窗口训练（Dual Window）**：
   - 从同一样本中抽取两个不重叠的时间窗口
   - 最大化窗口间 embedding 差异（若存在变化）
   - 最小化窗口内差异（保证一致性）

2. **时间编码注入**：
   - 使用 sinusoidal timecode 编码绝对时间
   - 使用 relative time encoding 编码相对时间差
   - 使模型感知时间距离

3. **多尺度时序建模**：
   - 短期（日/周）：捕捉快速变化
   - 中期（月/季）：捕捉季节性模式
   - 长期（年）：捕捉趋势性变化

---

## 5. 评估指标与 Benchmark 数据集

### 5.1 常用评估指标

#### 5.1.1 像素级二分类指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Precision (P)** | TP / (TP + FP) | 预测为变化中真正变化的比例 |
| **Recall (R)** | TP / (TP + FN) | 真实变化中被检测出的比例 |
| **F1-Score** | 2·P·R / (P + R) | Precision 和 Recall 的调和平均 |
| **IoU (Intersection over Union)** | TP / (TP + FP + FN) | 预测与真实变化区域的重叠度 |
| **OA (Overall Accuracy)** | (TP + TN) / Total | 整体像素正确率 |
| **Kappa** | (pₒ - pₑ) / (1 - pₑ) | 考虑偶然一致性的修正准确率 |

#### 5.1.2 ROC-AUC

- **AUC（Area Under ROC Curve）**：衡量模型在不同阈值下的综合判别能力
- 特别适合 **Bare 方法**（无固定阈值）
- AUC > 0.8 为良好，> 0.9 为优秀
- 在 OSCD 数据集上，Deep Metric Learning 方法达到 **AUC ≈ 0.937**

#### 5.1.3 语义变化检测指标

| 指标 | 说明 |
|------|------|
| **SeK (Separated Kappa)** | 分离 Kappa，评估语义变化与二值变化的联合性能 |
| **mIoU** | 多类别平均 IoU |
| **F_scd** | 语义变化检测专用 F1 |

### 5.2 主要 Benchmark 数据集

#### 5.2.1 二值变化检测（BCD）

| 数据集 | 图像对数 | 分辨率 | 场景 | 特点 |
|--------|---------|--------|------|------|
| **LEVIR-CD** | 637 | 1024×1024, 0.5m | 美国德州城市建设 | 最常用基准，31K 变化实例 |
| **WHU-CD** | 1 (大图像切分) | 32507×15354, 0.2m→256×256 | 新西兰基督城震后重建 | 建筑变化为主，12,796→16,077 建筑 |
| **S2Looking** | 5000 | 1024×1024, 0.5-0.8m | 全球农村地区 | 多角度拍摄，存在空间错位 |
| **CDD** | 16000 (切分后) | 256×256 | 季节变化 | 强调季节不变性 |
| **CLCD** | 600 | 512×512, 0.5-2m | 中国广东耕地 | 多类型变化（建筑、道路、湖泊） |
| **SYSU-CD** | 20000 | 256×256, 0.5m | 香港 | 大规模、类别无关 |
| **ChangeNet** | 31000 | 1900×1200, 0.3m | 多场景 | 6 个时相，5 类变化标注 |

#### 5.2.2 语义变化检测（SCD）

| 数据集 | 图像对数 | 类别数 | 特点 |
|--------|---------|--------|------|
| **SECOND** | 4662 | 6 | 512×512，建筑、道路等 |
| **Hi-UCD** | ~1000 | - | 高分辨率语义变化 |
| **CNAM-CD** | - | - | 含建筑物子类 |

#### 5.2.3 多光谱/异构变化检测

| 数据集 | 模态 | 特点 |
|--------|------|------|
| **OSCD** | Sentinel-2, 13-band | 24 对全球多光谱图像 |
| **SZTAKI** | 航空图像 | 23 年时间差 |
| **QuickBird** | 3-band VHR | 季节性变化显著 |
| **SAR-CD** | Sentinel-1 SAR | 10,000 对，含模拟变化 |

### 5.3 SOTA 性能参考（LEVIR-CD & WHU-CD）

#### LEVIR-CD 最新 SOTA（2024-2026）

| 方法 | 类型 | P | R | F1 | IoU | OA |
|------|------|---|---|----|-----|----|
| BIT | Transformer | 90.33 | 89.56 | 89.94 | 81.72 | 98.98 |
| Changeformer | Transformer | 92.05 | 88.80 | 90.40 | 82.48 | 99.04 |
| BAN | VFM+Adapter | 93.55 | 90.70 | 92.10 | 85.36 | 99.21 |
| SFCD | CNN | 92.97 | 91.69 | 92.33 | 85.75 | 99.22 |
| **SAM-MSCD** | **VFM** | **93.64** | 91.26 | **92.54** | **85.94** | **99.24** |
| **UniCDv2** | **Unified** | - | - | **92.10** | - | - |
| **RoCD** | **VFM** | - | - | **92.11** | **85.38** | - |

#### WHU-CD 最新 SOTA

| 方法 | F1 | IoU | OA |
|------|----|-----|----|
| BIT | 81.21 | 68.46 | 98.12 |
| MDIPNet | 89.08 | 80.31 | 99.02 |
| TTP | 91.36 | 84.10 | 99.21 |
| **SAM-MSCD** | **92.33** | **85.75** | **99.29** |
| **RoCD** | - | - | - |
| **UniCDv2** | **93.94** | **88.57** | - |

---

## 6. 2024-2025 最新 SOTA 方法

### 6.1 基于 Vision Foundation Model 的方法

#### 6.1.1 SAM-MSCD（2026）

- **核心**：将 Segment Anything Model (SAM) 适配到遥感变化检测
- **创新**：多尺度变化检测框架，有效利用 SAM 的预训练知识
- **性能**：在 4 个数据集上达到 SOTA，LEVIR-CD F1=**92.54%**，WHU-CD F1=**92.33%**

> 参考：[SAM-MSCD, Remote Sensing 2026](https://www.mdpi.com/2072-4292/18/3/506)

#### 6.1.2 RoCD（2025）

- **核心**：Refine-and-Fuse 框架，利用预训练视觉模型
- **模块**：
  - RAF（Refine-and-Fuse）：精炼并融合预训练特征
  - FusionR-Decoder：融合解码器
- **性能**：LEVIR-CD F1=**92.11%**，WHU-CD 领先

> 参考：[RoCD, Frontiers in Remote Sensing 2025](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1744950/full)

#### 6.1.3 BAN（Bi-Temporal Adapter Network, 2024）

- **核心**：复用 CLIP 的 ViT，通过 bridging modules 将通用特征注入 CD 分支
- **特点**：在 BCD 和 SCD 设置下均可训练

### 6.2 基于 Mamba 的方法

#### 6.2.1 ChangeMamba（2024）

- **核心**：使用 Spatiotemporal State Space Model 进行遥感变化检测
- **优势**：长程依赖建模效率高于 Transformer
- 参考：[ChangeMamba, IEEE TGRS 2024](https://ieeexplore.ieee.org/document/10533653)

### 6.3 基于扩散模型 / 生成式方法

#### 6.3.1 RCDNet（Referring Change Detection, 2025）

- **核心**：结合文本 prompt 的指代变化检测
- **数据生成**：使用 InstructPix2Pix 生成合成变化数据
- **架构**：VMamba-based Siamese Encoder + Fusion Module + Mask Decoder
- **文本嵌入**：CLIP text encoder 生成类别文本嵌入，通过 cross-attention 指导视觉查询
- **性能**：WHU-CD IoU 超越 M-CD **0.8**，LEVIR-CD 同样领先

> 参考：[RCDNet, arXiv 2025](https://arxiv.org/abs/2512.11719)

### 6.4 统一框架方法

#### 6.4.1 UniCD / UniCDv2（2026）

- **核心**：统一框架支持全监督、弱监督、半监督变化检测
- **模块**：STAM（Spatial-Temporal Attention Module）
- **性能**：
  - LEVIR-CD：F1=**92.10%**
  - WHU-CD：F1=**93.94%**，IoU=**88.57%**
  - 弱监督 LEVIR-CD：F1=**77.80%**（超越 S2C 12.72%）

> 参考：[UniCD, arXiv 2026](https://arxiv.org/abs/2601.17747)

#### 6.4.2 DDLNet（2024）

- **核心**：Dual-Domain Learning，频域 + 空域联合学习
- **模块**：FEM（Feature Enhancement Module）+ SRM（Spatial Refinement Module）
- **性能**：WHU-CD F1=**90.2%**，LEVIR-CD F1=**90.6%**

### 6.5 基于 Embedding 的 Few-Shot / 无监督方法

| 方法 | 范式 | 特点 |
|------|------|------|
| **AEF + kNN** | 无监督 | 64-dim embedding，cosine/dot product，BA=71.3% |
| **SSL4EO-S12 + Outlier** | 无监督 | 预训练 + 异常检测，处理季节性变化 |
| **Deep Metric Learning** | 无监督 | 每对图像单独优化，OSCD AUC=0.937 |
| **SeFi-CD** | Few-Shot | 文本 prompt 生成 visual prompt |
| **CDChat** | VLM | 生成变化文本描述 |

---

## 7. 关键参考文献汇总

### 基础方法

1. **AEF (AlphaEarth Foundations)** - Brown et al., 2025. "An embedding field model for accurate and efficient global mapping from sparse label data." [arXiv:2507.22291](https://arxiv.org/abs/2507.22291)

2. **BIT** - Chen, Qi & Shi, 2021. "Remote Sensing Image Change Detection with Transformers." IEEE TGRS. [arXiv:2103.00208](https://arxiv.org/abs/2103.00208)

3. **ChangeStar** - Zheng et al., 2021. "Change is Everywhere: Single-Temporal Supervised Object Change Detection." ICCV. [arXiv:2108.07002](https://arxiv.org/abs/2108.07002)

4. **ChangeStar2** - Zheng et al., 2024. "Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection." [arXiv:2406.15694](https://arxiv.org/abs/2406.15694)

### 对比学习与自监督

5. **CACo** - Mall, Hariharan & Bala, 2023. "Change-Aware Sampling and Contrastive Learning for Satellite Images." CVPR. [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)

6. **SeCo** - Manas et al., 2021. "Seasonal Contrast: Unsupervised Pre-training from Uncurated Remote Sensing Data." ICCV.

7. **SatMAE** - Cong et al., 2022. "SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery." [arXiv:2207.08051](https://arxiv.org/abs/2207.08051)

8. **MA3E** - 2024. "Masked Angle-Aware Autoencoder for Remote Sensing Images." [arXiv:2408.01946](https://arxiv.org/abs/2408.01946)

### CD Head 与架构

9. **3M-CDNet / 1M-CDNet** - 2021. "An Efficient Lightweight Neural Network for Remote Sensing Image Change Detection." Remote Sensing. [MDPI](https://www.mdpi.com/2072-4292/13/24/5152)

10. **C3PO / MTF** - Wang et al., 2022. "How to Reduce Change Detection to Semantic Segmentation." [arXiv:2206.07557](https://arxiv.org/abs/2206.07557)

11. **ChangeMamba** - Chen et al., 2024. "Remote Sensing Change Detection With Spatiotemporal State Space Model." IEEE TGRS.

### 2024-2025 SOTA

12. **SAM-MSCD** - 2026. "A Multi-Scale Remote Sensing Image Change Detection Network Based on Vision Foundation Model." Remote Sensing. [MDPI](https://www.mdpi.com/2072-4292/18/3/506)

13. **RoCD** - 2025. "Leveraging foundation vision models with refine-and-fuse framework for robust change detection." Frontiers. [Link](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1744950/full)

14. **UniCD** - 2026. "A Unified Framework for Remote Sensing Change Detection." [arXiv:2601.17747](https://arxiv.org/abs/2601.17747)

15. **RCDNet** - 2025. "Referring Change Detection in Remote Sensing Imagery." [arXiv:2512.11719](https://arxiv.org/abs/2512.11719)

16. **DDLNet** - 2024. "Boosting Remote Sensing Change Detection with Dual-Domain Learning." [arXiv:2406.13606](https://arxiv.org/abs/2406.13606)

### 无监督 / Bare 方法

17. **Deep Metric Learning CD** - Bandara & Patel, WACV 2025. "Deep Metric Learning for Unsupervised Remote Sensing Change Detection." [PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Bandara_Deep_Metric_Learning_for_Unsupervised_Remote_Sensing_Change_Detection_WACV_2025_paper.pdf)

18. **Element 84 Blog** - 2023. "Exploring unsupervised change detection with Sentinel-2 vector embeddings." [Link](https://element84.com/machine-learning/exploring-unsupervised-change-detection-with-sentinel-2-vector-embeddings/)

### 综述

19. **Foundation Models for RS Survey** - 2024. "Foundation Models for Remote Sensing and Earth Observation: A Survey." [arXiv:2410.16602](https://arxiv.org/abs/2410.16602)

20. **RSFM Benchmarking** - 2025. "Towards Efficient Benchmarking of Foundation Models in Remote Sensing." [arXiv:2505.03299](https://arxiv.org/abs/2505.03299)

---

## 附录：对本项目的直接启示

### A. Bare 方法在本项目的应用

本项目（xuannv_embdding）当前已有验证脚本计算 **cosine distance** 和 **CD Head** 两种路径：

| 方法 | 当前状态 | 优化建议 |
|------|---------|---------|
| Cosine Distance | 已实现，AUC ~0.65 | 检查时间窗口正确性；确认 CRS 处理无误 |
| CD Head (PixelMLP) | 已实现，需 few-shot 训练 | 尝试更复杂的 CD Head 设计（如 ChangeMixin 风格） |

### B. 可引入的改进方向

1. **参考 ChangeMixin2 的 Temporal Difference Network**：显式建模时序差分特征
2. **参考 CACo 的变化感知采样**：在对比学习中引入条件推远机制
3. **参考 BIT 的 Token-based Context**：在 STPBlock 后增加 semantic tokenization
4. **参考 AEF 的量化策略**：8-bit 量化后性能几乎无损，可减少存储
5. **参考 UniCD 的统一框架**：统一 bare + CD Head + 弱监督的训练流程

### C. 关键性能基准

| 任务 | 及格 | 良好 | 优秀 |
|------|------|------|------|
| Few-Shot CD AUC | > 0.70 | > 0.80 | > 0.85 |
| Bare AUC | > 0.60 | > 0.70 | > 0.75 |
| LEVIR-CD F1 | > 85% | > 90% | > 92% |
| WHU-CD F1 | > 85% | > 90% | > 93% |
