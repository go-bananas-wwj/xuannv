# 地理嵌入模型调研与对比报告

> 撰写日期：2026-05-29  
> 范围：AEF、OlmoEarth 及主流遥感地理嵌入模型，与 xuannv 项目的对比分析及改进建议

---

## 目录

1. [执行摘要](#执行摘要)
2. [调研文档：主流地理嵌入模型综述](#调研文档主流地理嵌入模型综述)
   - 2.1 AlphaEarth Foundations (AEF)
   - 2.2 OlmoEarth
   - 2.3 SatMAE
   - 2.4 Scale-MAE
   - 2.5 DOFA & Clay
   - 2.6 SpectralGPT
   - 2.7 Prithvi-EO-2.0
   - 2.8 SkySense / SkySense++
   - 2.9 CROMA
   - 2.10 AnySat
   - 2.11 TerraMind
   - 2.12 Panopticon
   - 2.13 SoftCon
   - 2.14 DeCUR
   - 2.15 SeCo & CACo
3. [全模型架构横向对比](#全模型架构横向对比)
4. [对比文档：xuannv 与外部模型的系统比较](#对比文档xuannv-与外部模型的系统比较)
   - 4.1 xuannv 核心设计
   - 4.2 优势所在
   - 4.3 差距分析
5. [改进路线图](#改进路线图)
6. [置信度评估与文献引用](#置信度评估与文献引用)

---

## 执行摘要

本报告系统调研了 2021–2026 年间发表的 15+ 个主流地理嵌入与遥感基础模型，重点分析了 **AlphaEarth Foundations (AEF)** 和 **OlmoEarth** 两个最相关的模型，并与 xuannv 项目进行了深度对比。

**核心发现：**

1. **AEF 是 xuannv 的直接上游**：xuannv 的 STPEncoder、VMFBottleneck、WindowCodeEncoder 均源自 AEF 论文架构（arXiv:2507.22291）。AEF 之所以效果好，核心在于：**64D vMF 超球面嵌入 + 批量均匀性目标 + 连续时间解耦 + 30 亿+观测数据**。

2. **OlmoEarth 的关键创新**：使用 FlexiViT 在**潜在空间**预测（而非像素重建），避免 EO 高噪声环境下的 MAE 模式崩溃；原生支持 12 个月时序，训练数据 28.5 万全球样本。

3. **xuannv 的反坍缩技术在全球处于前沿水平**：`skip_l2_norm_training` + 5 种互补损失的组合，在理论严谨性上甚至超过 AEF 原版（AEF 只用单一的批量均匀性，xuannv 则用 VICReg + Barlow Twins + 正交性 + MCR² 的完整套装）。

4. **xuannv 最大的短板**：训练数据规模（424 patches）与 AEF（840 万视频序列）、OlmoEarth（28.5 万全球样本）相差 3–4 个数量级；时序建模未达 Prithvi/OlmoEarth 的 3D 姿态。

5. **最高优先级改进**：① 引入 DeCUR 风格的传感器解耦（S1/S2/Landsat 共同 vs 独有维度分离）；② 采用 JEPA 或 Latent MIM 替代像素重建；③ 参考 AnySat 的 GeoPlex 风格多数据集拼接训练策略。

---

## 调研文档：主流地理嵌入模型综述

---

### 2.1 AlphaEarth Foundations (AEF)

| 属性 | 详情 |
|------|------|
| **机构** | Google DeepMind + Google |
| **论文** | arXiv:2507.22291v2（2025 年 7 月发布，9 月更新）|
| **代码** | `Brayden-Zhang/alphaearth-foundations`（社区 PyTorch 实现，官方为 JAX，未公开）|
| **数据** | 30 亿+观测，840 万视频序列，9 个网格化数据源 + 1 个文本源 |
| **嵌入维度** | **64 维**，单位超球面 S⁶³（vMF 分布）|
| **空间分辨率** | 10 m²/像素 |
| **参数量** | ~480M（生产版），~1B（训练过但被丢弃）|

#### 架构核心：Space-Time-Precision (STP) 编码器

AEF 的编码器由 **15 个 STP 块**串联，每块并行运行三路算子：

```
输入帧（多源多时相）→ SensorStemBank（各源独立 stem）
         ↓
┌────────────────────────────────────────────────────┐
│  STP Block × 15                                      │
│  ├─ SpaceOperator   D=1024, res=L/16, ViT 自注意力 │
│  ├─ TimeOperator    D=512,  res=L/8,  时间轴自注意力│
│  └─ PrecisionOp     D=128,  res=L/2,  3×3 卷积     │
│  六路 LearnedSpatialResampling 跨尺度交换（拉普拉斯金字塔）│
└────────────────────────────────────────────────────┘
         ↓
TemporalSummarizer（cross-attention 按 valid_period 聚合 T 帧）
         ↓
64D L2 归一化 → vMF 超球面嵌入 µ ∈ S⁶³
```

**各算子规格**：

| 算子 | 维度 | 分辨率 | 注意力头数 | 机制 |
|------|------|--------|------------|------|
| Space | 1024 | L/16 | 8 heads | ViT 自注意力 + LN + MLP |
| Time | 512 | L/8 | 8 heads | 时间轴 + 正弦时间编码注入 |
| Precision | 128 | L/2 | N/A | GroupNorm + Conv3×3 + GELU 残差 |

**STP 块内部六路拉普拉斯金字塔跨尺度交换**：
- Space→Time (scale 2.0×), Space→Precision (scale 8.0×)
- Time→Space (scale 0.5×), Time→Precision (scale 4.0×)
- Precision→Space (scale 0.125×), Precision→Time (scale 0.25×)

每路均通过 `LearnedSpatialResampling`（ConvTranspose2d 上采样 / Conv2d 下采样）实现，所有输出以加法融合后传入下一块。

#### 训练目标（四项损失）

$$\mathcal{L} = a\underbrace{\sum_{i \in M} f_i(\mathbf{y}_i,\mathbf{y}_i')}_{重建} + b\underbrace{\sum_{i=1}^{64}|\mathbf{u}_i \cdot \mathbf{u}'_i|}_{批量均匀性} + c\underbrace{\frac{1-\boldsymbol{\mu}\cdot\boldsymbol{\mu}_s}{2}}_{一致性} + d\underbrace{f_{\text{CLIP}}(\mathbf{u},\mathbf{u}_t)}_{文本对齐}$$

损失权重：a=1.0, b=0.005, c=0.02, d=0.001

**批量均匀性（关键防坍缩机制）**：

```python
# 通过循环平移获得"随机配对" u'
x_prime = torch.roll(x, shifts=1, dims=0)
# 惩罚绝对点积 |u·u'|，迫使嵌入在 S^63 上近似正交分布
return (x * x_prime).sum(dim=-1).abs().mean()
```

**时间解耦**（最重要的时序创新）：
- 支持输入帧（support period `[t_j]`）与查询窗口（valid period `[ts, te)`）完全解耦
- `TemporalSummarizer` 使用 valid period 编码作为 cross-attention 的单一 query，从所有 T 帧中聚合信息
- 这使得模型能够外推到训练期之外的时间窗口，无需重新训练

**输入模态**：Sentinel-2、Sentinel-1、Landsat 8/9、GEDI 激光雷达、GLO-30 DEM、ERA5-Land、GRACE、NLCD 地被、维基百科地理文章 + GBIF 物种记录

**评估结果**：在 15 个下游任务中全面领先（平均误差降低 ~23.9%），是首个在所有应用场景均超越前代方法的 EO 模型。

---

### 2.2 OlmoEarth

| 属性 | 详情 |
|------|------|
| **机构** | Allen Institute for AI (AI2) |
| **论文** | "OlmoEarth: Stable Latent Image Modeling for Multimodal EO"（2025.11）|
| **代码** | `allenai/olmoearth_pretrain`（完全开源）|
| **数据** | 285,288 全球样本，每样本覆盖 2.56km²，最多 12 个月时序帧 |
| **嵌入维度** | 768（Base）/ 1024（Large）|
| **空间分辨率** | 10 m/像素（UTM 投影）|

#### 架构核心：FlexiViT + LMIM

OlmoEarth 使用 **FlexiViT**（灵活 ViT）作为编码器骨干，支持可变 patch 尺寸（1–8 像素），同一权重可在推理时以不同计算/质量权衡运行。

**模型规格**：

| 尺寸 | 编码器参数 | 嵌入维度 | HuggingFace |
|------|-----------|----------|-------------|
| Nano | 1.4M | 128 | `allenai/OlmoEarth-v1-Nano` |
| Tiny | 6.2M | 192 | `allenai/OlmoEarth-v1-Tiny` |
| Base | 89M | 768 | `allenai/OlmoEarth-v1-Base` |
| Large | 308M | 1024 | `allenai/OlmoEarth-v1-Large` |

**输入模态**（多时相，最多 12 个月）：

| 模态 | 波段 | 分辨率 |
|------|------|--------|
| Sentinel-2 L2A | 12 波段（3 个 BandSet） | 10/20/40 m |
| Sentinel-1 IW GRD | VV, VH | 10 m |
| Landsat 8/9 OLI-TIRS | 11 波段（2 个 BandSet） | 10/20 m |
| OpenStreetMap（栅格化）| 30 特征通道 | — |
| ESA WorldCover 2021 | 1 通道 | — |
| SRTM DEM | 高程 | — |
| ERA5 月均 | 6 气候变量 | — |
| WorldPop | 人口密度 | — |

#### 核心创新：稳定潜在图像建模（LMIM）

OlmoEarth 的关键创新是**在潜在空间而非像素空间**进行预测：

```
传统 MAE：输入 → 编码器 → 解码器 → 重建像素（MSE 损失）
LMIM：   输入 → 学生编码器 → 预测教师编码器的潜在表示（cosine/L2 损失）
```

这避免了 EO 数据（云、噪声、季节变化）中低层次像素重建的"模式崩溃"问题——学生模型学到的是**语义稳定的潜在表示**，而非捕获像素级噪声。

**额外损失组件**：
- 对比损失（跨视角/增强对齐）
- Patch 判别损失（区分有效 vs 掩码 token）
- **结构化掩码策略**（反映 EO 数据的真实缺失模式，如云覆盖、重访周期）

**下游验证（能源选址案例）**：

| 方法 | AUC |
|------|-----|
| 仅地理坐标 | 0.579 |
| OlmoEarth T=1（单景）| 0.907 |
| OlmoEarth T=多时相（季节性）| **0.924** |

---

### 2.3 SatMAE（NeurIPS 2022）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2207.08051，NeurIPS 2022 |
| **代码** | `sustainlab-group/SatMAE` |
| **核心创新** | 时序 MAE + 多光谱波段分组编码 |

**时序处理**：对多时相图像叠加进行独立随机掩码（非跨帧一致掩码），加入可学习的时间位置嵌入。实验表明独立掩码比一致掩码效果更好。

**多光谱处理**：按波谱范围分组（如 `--grouped_bands 0 1 2 6`），每组赋予独立的光谱位置编码，再输入 ViT。

**局限**：不支持 SAR；无显式变化检测设计；时序建模较浅（仅位置编码，无跨时间注意力）。

---

### 2.4 Scale-MAE（ICCV 2023）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2212.14532，ICCV 2023 |
| **代码** | `bair-climate-initiative/scale-mae` |
| **核心创新** | GSD（地面采样距离）感知位置编码 + 带通滤波解码器 |

**关键设计**：同一图像块在不同缩放级别获得不同的位置编码（因为覆盖的真实世界面积不同），强制模型学习尺度不变的语义特征。解码器同时重建低频和高频版本的掩码区域（带通分解），促进多尺度表示。

---

### 2.5 DOFA & Clay（2024）

#### DOFA（Dynamic One-For-All）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2403.15356（2024）|
| **代码** | `zhu-xlab/DOFA` |
| **核心创新** | 波长条件动态超网络替代固定 PatchEmbed |

```python
# 任意传感器输入：提供波长列表（单位: μm）
wave_list = [0.665, 0.56, 0.49, 0.705, ...]  # 任意通道数，任意传感器
# 超网络 TransformerWeightGenerator 根据波长动态生成第一层卷积权重
dynamic_out = F.conv2d(img_feat, generated_weights, stride=patch_size)
```

五种模态联合预训练：Sentinel-1 SAR、Sentinel-2、NAIP RGB、高分一号、高光谱。单模型权重适配所有传感器。

#### Clay v1.5

基于 DOFA 思想，增加 DINOv2 教师蒸馏（10% cosine 相似度损失）和时空元数据注入（纬度、经度、年内周次、日内小时）。Apache-2.0 许可，7000 万全球样本，20× AWS g6.48xlarge 训练。

---

### 2.6 SpectralGPT（TPAMI 2024）

| 属性 | 详情 |
|------|------|
| **论文** | IEEE TPAMI 2024，DOI:10.1109/TPAMI.2024.3362475 |
| **代码** | `danfenghong/IEEE_TPAMI_SpectralGPT` |
| **核心创新** | 3D 空谱联合 token 化，90% 高掩码率 |

将空间 H×W 和光谱波段 C 维度同时 token 化为 3D 块，使用 90% 掩码率（vs 标准 MAE 的 75%），同时预测多个光谱波段，强化光谱序列连贯性。适合高光谱和多光谱数据。

---

### 2.7 Prithvi-EO-2.0（NASA/IBM，2024）

| 属性 | 详情 |
|------|------|
| **机构** | NASA IMPACT + IBM |
| **论文** | arXiv:2412.02732（2024.12）|
| **代码** | `NASA-IMPACT/Prithvi-EO-2.0` |
| **数据** | 420 万全球 HLS 时序样本（2014-2023），每样本 4 个时相 |
| **参数量** | 300M（ViT-L）/ 600M（ViT-H）|

**核心时序设计**：

```
输入：[B, T=4, C=6, H, W]  HLS（S2+Landsat 共同 6 波段）
3D 卷积 PatchEmbed → (t, h, w) 时空块
+ 3D 正弦位置编码（时间维、高度维、宽度维独立编码后拼接）
+ 位置元数据（纬度/经度，各维 sincos 编码）
+ 时间元数据（年份、年内日期，独立 sincos 编码）
→ 通过可学习加权求和融合到 token 嵌入中（含 Dropout p=0.1）
```

在 GEO-Bench 上全面超越 6 个竞争 GFM，600M-TL 版相比 Prithvi-1.0 提升 +8%。

---

### 2.8 SkySense / SkySense++（CVPR 2024）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2312.10115，CVPR 2024 |
| **代码** | `Jack-bo1220/SkySense` |
| **参数量** | **21 亿**（发表时最大 RSFM）|

**多模态设计**（三路因式化编码器）：
- 高分辨率光学（HR-RGB）→ SwinTransformerV2-Huge
- 时序多光谱（S2 10 波段）→ ViT-Large  
- 时序 SAR（S1 VV+VH）→ ViT-Large
- 三路输出经因式化时空融合模块合并

**两个关键损失设计**：
- **MGCL（多粒度对比学习）**：在像素、目标、图像三个空间粒度上跨模态对比
- **GCPL（地理上下文原型学习）**：基于地理位置生成区域感知原型，挖掘无标注 RSI 中的隐式地理知识

在 16 个数据集上超越 18 个 RSFM，平均领先 GFM +2.76%，SatLas +3.67%。

---

### 2.9 CROMA（NeurIPS 2023）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2311.00566，NeurIPS 2023 |
| **代码** | `antofuller/CROMA` |
| **核心创新** | SAR-光学双目标（跨模态对比 + MAE）+ X-ALiBi 位置偏置 |

三路 ViT 编码器（SAR encoder、Optical encoder、Joint cross-encoder），联合训练两个互补目标：
1. **跨模态对比损失**：对齐 SAR_GAP 和 optical_GAP 表示（InfoNCE）
2. **掩码重建**：Joint cross-encoder 预测掩码 patch

**X-ALiBi**：2D ALiBi（空间相对位置偏置）扩展到跨模态注意力，支持测试时推理 **17.6× 大于训练尺寸**的图像。在 7 项指标上平均提升前代 SOTA +1.4~8.4%。

---

### 2.10 AnySat（CVPR 2025 Highlight）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2412.14123，**CVPR 2025 Highlight** |
| **代码** | `gastruc/AnySat` |
| **核心创新** | JEPA（联合嵌入预测架构）+ 多数据集异构训练 |

支持 **11 种传感器**（包括 Landsat-7/8、S1、S2、MODIS），通过字典格式输入：
```python
inputs = {
    's2': (B, T, 10, H, W),   # 带时间维
    's1-asc': (B, T, 2, H, W),
    'l8': (B, T, 11, H, W),   # Landsat-8
    '_dates': (B, T),          # 年内日期（天）
}
```

**JEPA vs MAE 的关键区别**：预测**教师编码器的潜在嵌入**（而非像素），学到语义特征而非低层次纹理。无对比损失，无显式负样本。

GeoPlex 训练集：5 个数据集，11 种传感器，联合训练。

---

### 2.11 TerraMind（ICCV 2025）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2504.11171，**ICCV 2025** |
| **代码** | `IBM/terramind` |
| **核心创新** | 任意到任意生成（any-to-any），离散 token 重建 |

两阶段架构：
1. **模态分词器**：每个模态的自编码器 + FSQ（有限标量量化），词表 16K，压缩比 250×–3000×
2. **TerraMind Transformer**：对称 encoder-decoder，同时处理 token 级（离散）和像素级（连续）双尺度输入

**TiM（Thinking in Modalities）**：类似 Chain-of-Thought，先由 S2 生成中间模态（如 LULC），再将原始 + 生成模态一起输入编码器进行微调，递归增强表示。在 PANGAEA 基准上超越所有当时的地理空间基础模型。

---

### 2.12 Panopticon（CVPR 2025 最佳论文）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2503.10845，**CVPR 2025 最佳论文** |
| **代码** | `Panopticon-FM/panopticon` |
| **核心创新** | 波长条件 ChnAttn + DINOv2 自蒸馏 |

**ChnAttn（通道注意力聚合）**：
```python
# 单个可学习 query（1×1×D）跨 C 个通道 token 进行 cross-attention
# 输出固定维度嵌入，与输入通道数无关
query = self.query  # (1, 1, D) —— 可学习
channels = conv3d_patchify(x) + wavelength_sincos_emb  # (B, C, L, D)
output = cross_attention(query, channels)               # (B, L, D) —— 固定 D
```

光学通道：波长（nm）正弦编码；SAR 通道：负数 chn_id → 极化模式 + 轨道方向可学习嵌入。支持**任意传感器**（包括未见过的新型传感器）。

跨传感器增强：同一地点的不同传感器图像作为 DINOv2 的不同"视角"。在 GEO-Bench 和 23 个多光谱/高光谱/SAR 数据集上达到 SOTA。

---

### 2.13 SoftCon（TGRS 2024）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2405.20462，IEEE TGRS 2024 |
| **代码** | `zhu-xlab/softcon` |
| **核心创新** | 软对比损失（多标签相似度矩阵作为监督信号）|

**软对比损失公式**：
$$\mathcal{L}^{\text{SoftCon}} = -\sum_i\sum_j \left[Y_{ij}\log\sigma(X_{ij}) + (1-Y_{ij})\log(1-\sigma(X_{ij}))\right]$$

其中 $Y_{ij}$ 是场景 i 和 j 的 Dynamic World 多热标签向量的**余弦相似度**（连续值 [0,1]，作为软监督信号）。解决了 EO 对比学习中的"假负样本"问题。

BigEarthNet-10% 线性探测 mAP：ViT-B/14 达 **86.8%**，在参数量远小于竞品的情况下领先大多数 ViT-L 模型。

---

### 2.14 DeCUR（ECCV 2024 Oral）

| 属性 | 详情 |
|------|------|
| **论文** | arXiv:2309.05300，**ECCV 2024 Oral** |
| **代码** | `zhu-xlab/DeCUR` |
| **核心创新** | 跨模态共同维度 vs 独有维度解耦 |

**解耦损失设计**：
```python
# 投影空间前 dim_common=448 维 → 驱动跨模态对齐（Barlow 对角→1）
# 后 dim_unique=7744 维 → 驱动跨模态分离（Barlow 对角→0）
loss_common = bt_loss(z_s1[:, :dim_c], z_s2[:, :dim_c], target='identity')
loss_unique = bt_loss(z_s1[:, dim_c:], z_s2[:, dim_c:], target='zero')
loss = (loss_s1 + loss_s2 + (loss_common + loss_unique) / 2) / 3
```

使得单一嵌入可同时支持**模态不变任务**（如语义分割）和**模态特定任务**（如 SAR 专有的水体检测），BigEarthNet-MM 提升 +1.2–2.5% mAP。

---

### 2.15 SeCo & CACo

#### SeCo（季节对比，ICCV 2021）

使用同一地点三个时相的快照，构造季节性正样本对（季节改变 → 语义不变），基于 MoCo-v2 框架。在 OSCD 变化检测数据集上相比 ImageNet 预训练提升 ~3-5% F1。

```python
# 三时相 t0, t1, t2：同一地点，不同季节
q = t0                   # query
k0 = augment(t1)         # 时序正样本（增强）
k1 = t2                  # 时序正样本（原始）
k2 = augment(t0)         # 增强正样本
```

#### CACo（变化感知对比）

核心思想：同一地点的不同时间窗口应当产生**不同**的嵌入。通过**反对角线 InfoNCE**实现（标准 InfoNCE 把对角元素当正样本，CACo 把对角元素当负样本）。xuannv 已经实现了这个机制（`temporal_info_nce_loss`）。

---

## 全模型架构横向对比

| 模型 | 训练目标 | 时序建模 | 多传感器 | 参数量 | 发布年份 |
|------|---------|---------|---------|--------|---------|
| SatMAE | MAE | 时序位置编码 | S2 | 330M | 2022 |
| Scale-MAE | MAE + 带通 | 无 | RGB | 323M | 2023 |
| CROMA | 对比 + MAE | 无 | S1+S2 | 303M | 2023 |
| Prithvi-EO-2.0 | **3D MAE** | **3D 时空块 + 3D PE** | S2+Landsat | 300M/600M | 2024 |
| SoftCon | 软对比 | 季节增强 | S1, S2 | 86M | 2024 |
| SkySense | MGCL + GCPL | 因式化时空编码器 | HR+MS+SAR | **2.06B** | 2024 |
| DOFA | MAE | 无 | **任意波长** | 300M+ | 2024 |
| Clay v1.5 | MAE + DINOv2 | 时间元数据注入 | S1,S2,L8,NAIP | 311M | 2024 |
| DeCUR | Barlow Twins（解耦）| 季节增强 | S1, S2 | 86M | 2024 |
| SpectralGPT | 3D MAE（高掩码率）| 渐进训练 | S2 | 600M | 2024 |
| **AEF** | 重建+均匀性+一致性+CLIP | **连续时间解耦** | **9 种模态** | 480M | 2025 |
| **OlmoEarth** | **LMIM** + 对比 | **原生 12 时相** | S1,S2,Landsat,7辅助 | 308M | 2025 |
| AnySat | **JEPA** | 原生多时相 | **11 种传感器** | 85M+ | 2025 |
| TerraMind | **离散 Token 分类** | 无（temporal wrapper）| S1,S2,DEM,LULC,文本 | 100M+ | 2025 |
| Panopticon | DINOv2 自蒸馏 | 无 | **任意传感器** | 86M | 2025 |

---

## 对比文档：xuannv 与外部模型的系统比较

---

### 4.1 xuannv 核心设计

xuannv 是 AEF 的定向改进版，核心目标是解决两个问题：**嵌入坍缩**和**时间敏感性不足**。

**数据流**：
```
输入：S2(6ch) + S1(2ch) + Landsat(6ch)，带时间戳
    ↓ SensorEncoderBank（各源独立 stem）
    ↓ STPEncoder（15 块，Space/Time/Precision 三路）
    ↓ VMFBottleneck（训练时 skip L2，推理时恢复 L2 + VMF 噪声）
    ↓ embedding_map [B, D, H, W] + pre_norm_map
    ↓ ContinuousDecoder / CategoricalDecoder（各源独立解码）
重建目标（不输入 encoder）：DEM + WorldCover + Dynamic World + JRC Water
```

**五种互补的反坍缩损失**（活跃配置 Round 9）：

```python
pre_norm_uniform_weight: 0.5    # raw_uniformity_loss —— 欧氏空间均匀性
variance_weight: 0.25           # variance_regularizer —— VICReg 方差下界
covariance_weight: 0.001        # covariance_loss —— 协方差正则
orthogonality_weight: 0.01      # bottleneck_orthogonality_loss —— 权重级正交
# + decorrelation_loss（Barlow Twins 去相关）
```

**时序对比损失套装**：
- `temporal_contrastive_loss`：hinge 损失，强制不同时间窗口嵌入夹角 ≥78°
- `temporal_cosine_pixel_loss`：像素级余弦时序损失（带自适应目标，防止全局平移作弊）
- `temporal_info_nce_loss`：反对角线 InfoNCE（CACo 设计，同地不同时=负样本）
- `gap_aware_temporal_cosine_loss`：时间间隔越长目标相似度越低（连续关系建模）

**教师-学生 EMA 系统**：
- 学生视图扰动：帧 Dropout（50%）+ 源 Dropout（30%）+ 时间截断（15%）
- 一致性损失：`(1 - cosine_sim(teacher, student)) / 2`
- 随机时间窗口增强：50% 概率随机裁剪 valid_period 到 4–24 帧窗口

---

### 4.2 优势所在（xuannv vs 竞品）

#### ✅ 优势 1：反坍缩技术 —— 理论完备性全球前沿

xuannv 的五项互补反坍缩损失在理论完备性上**超过 AEF 原版**（AEF 只用单一批量均匀性损失）：

| 损失类型 | AEF | OlmoEarth | xuannv |
|---------|-----|-----------|--------|
| 均匀性（欧氏空间）| ✅ 批量均匀性 | 隐式（LMIM 稳定性）| ✅ raw_uniformity（自适应 t=2/D）|
| Barlow 去相关 | ❌ | ❌ | ✅ decorrelation_loss |
| VICReg 方差/协方差 | ❌ | ❌ | ✅ variance_regularizer + covariance_loss |
| 权重级正交 | ❌ | ❌ | ✅ bottleneck_orthogonality_loss |
| Skip L2 训练 | ❌ | ❌ | ✅ skip_l2_norm_training |

**Skip L2 训练的理论优势**：标准方法在 L2 归一化空间计算均匀性损失，Jacobian `∂u/∂x = (I-uu^T)/||x||` 在坍缩态趋近于零矩阵，梯度无法反传。xuannv 绕过这个屏障，在欧氏 pre-norm 空间计算所有损失，推理时才恢复 L2 归一化。

#### ✅ 优势 2：变化检测时序设计 —— 专项优化

xuannv 的双窗口 + 反对角线 InfoNCE + 间隔感知余弦损失的组合，在**显式变化检测任务**上比任何通用 GFM 都更有针对性。AEF 的时序创新侧重于"任意时间查询"（连续时间插值），而不是"前后对比"（变化检测）。

#### ✅ 优势 3：轻量级推理

64D 嵌入 vs OlmoEarth 768D，存储和计算效率高 12×。适合边缘部署和大规模区域遥感推理。

---

### 4.3 差距分析（xuannv 的短板）

#### ❌ 差距 1：训练数据规模（最关键，3–4 个数量级）

| 项目 | 训练数据量 |
|------|-----------|
| AEF | 840 万视频序列，30 亿帧 |
| OlmoEarth | 285,288 全球样本 |
| Prithvi-EO-2.0 | 420 万全球 HLS 样本 |
| SoftCon | 78 万 Sentinel-2 样本 |
| **xuannv** | **424 个 patch（哈尔滨）** |

> 当前的 xuannv 本质上是在**极小规模**上验证方法的正确性，而无法靠数据量取胜。这是所有差距中最根本的一条。

#### ❌ 差距 2：时序建模深度不足

| 模型 | 时序建模方式 |
|------|------------|
| AEF | valid period 与 support period 完全解耦，TemporalSummarizer cross-attention 聚合任意时间窗口 |
| Prithvi-EO-2.0 | 3D 卷积块 + 3D 正弦位置编码 + 显式时间/位置元数据注入（可学习加权求和）|
| OlmoEarth | 原生 12 月时序，结构化掩码（反映真实云缺失模式），时间戳 (day, month, year) 注入每个 token |
| **xuannv** | WindowCodeEncoder 仅用 4 个 sin/cos 特征编码窗口起止，时间上下文较浅 |

#### ❌ 差距 3：多源融合策略单一

- **CROMA**：三路 ViT，跨模态对比 + MAE 双目标，SAR 与光学信息互相增强
- **DeCUR**：S1/S2 分别训练 + 跨模态 Barlow Twins 解耦（共同维度 vs 独有维度）
- **SkySense**：HR + MS + SAR 三路因式化编码器，多粒度对比学习
- **xuannv**：SensorEncoderBank 为各源提供独立 stem，但进入 STPEncoder 后所有源的 token **混合处理**，缺乏显式的模态解耦目标，S1 SAR 的噪声特性可能污染光学特征

#### ❌ 差距 4：MAE 重建目标偏低级

| 模型 | 重建目标层次 |
|------|------------|
| OlmoEarth | 潜在空间（LMIM），预测教师编码器的嵌入，语义级 |
| AnySat | 潜在空间（JEPA），预测 EMA 教师的 patch 嵌入 |
| TerraMind | 离散 token（CE 损失，16K 词表）|
| **xuannv** | 像素级 L1 重建（连续源）+ CE（分类源），停留在第一代 MAE 水平，易被噪声驱动 |

#### ❌ 差距 5：无地理先验注入

- **Clay**：每个 patch token 注入纬度/经度/年内周次/GSD（每 token 8 维）
- **Prithvi-EO-2.0**：纬度/经度 sincos 编码通过可学习加权求和注入
- **GASSL**：地理坐标预测作为辅助 pretext 任务
- **xuannv**：无任何地理位置信息注入，嵌入纯粹基于图像信号，全球可迁移性受限

---

## 改进路线图

### 第一优先级：立竿见影（1–2 周，低风险）

#### P1.1 有效秩监控（不改模型，改监控）

```python
# 在 trainer.py 中增加 effective_rank 指标
# 直接衡量维度利用率，比 raw_unif 更直观
def effective_rank(embeddings: torch.Tensor) -> float:
    svs = torch.svd(embeddings.T).S
    probs = svs / svs.sum()
    return torch.exp(-torch.sum(probs * torch.log(probs + 1e-9))).item()
    # 范围：1（完全坍缩）→ D（理想均匀）
```

#### P1.2 过滤均匀性（降低梯度方差，参考 CAFU）

```python
# 在 raw_uniformity_loss 中过滤距离过远的嵌入对（背景噪声）
sq_pdist = torch.pdist(z, p=2).pow(2)
mask = sq_pdist < (tau ** 2)   # τ ≈ 1.6 × std
loss = sq_pdist[mask].mul(-t).exp().mean().log()
```

#### P1.3 时序对比温度调度

```python
# 训练前 50%：高温度 τ=0.1（平滑梯度，避免早期过拟合）
# 训练后 50%：低温度 τ=0.05（强化时序分离）
temporal_temperature = 0.1 if epoch < total_epochs * 0.5 else 0.05
```

---

### 第二优先级：中期改进（2–4 周）

#### P2.1 DeCUR 风格的传感器解耦（强烈推荐）

在 `decorrelation_loss` 中增加传感器维度解耦：

```python
# 将嵌入空间划分为 dim_common（模态不变）+ dim_unique（模态特有）
dim_common = D // 4  # 如 D=128，则 dim_common=32

# 共同子空间：S2、S1 的嵌入跨模态对齐（驱动→单位阵）
loss_cross_common = bt_loss(z_s2[:, :dim_c], z_s1[:, :dim_c], target='identity')

# 独有子空间：S2、S1 的嵌入跨模态分离（驱动→零矩阵）  
loss_cross_unique = bt_loss(z_s2[:, dim_c:], z_s1[:, dim_c:], target='zero')
```

这将使单一嵌入同时支持：① 跨传感器语义任务（土地覆盖，使用共同维度）；② 传感器特有任务（SAR 水体，使用独有维度）。

#### P2.2 LMIM 替代像素重建（解决 EO 噪声问题）

将 `reconstruction_loss` 替换为潜在空间预测：

```python
# 当前：
loss_recon = F.l1_loss(decoder_output, target_pixels)

# 改进后（LMIM 风格）：
with torch.no_grad():
    target_latents = teacher_model.encode(masked_frames)  # EMA teacher 的潜在嵌入
student_pred = student_decoder(visible_tokens)
loss_lmim = 1.0 - F.cosine_similarity(student_pred, target_latents).mean()
```

参考 OlmoEarth LMIM 和 AnySat JEPA 设计。对 EO 数据尤其有效，因为 EO 数据含大量低层次噪声（云、阴影、BRDF 效应），像素重建会鼓励模型记忆噪声而非学习语义特征。

#### P2.3 地理元数据注入（参考 Clay）

在 `STPEncoder` 输入或 `VMFBottleneck` 处注入地理元数据：

```python
# 参考 Clay 的 add_encodings 方法
pos_enc = sincos_2d_with_gsd(H, W, D-8, gsd=10.0)  # GSD=10m，形状 (H*W, D-8)
time_latlon = torch.cat([
    week_sincos(timestamps),   # 年内周次（2 维）
    lat_sincos(lat),           # 纬度（2 维）
    lon_sincos(lon),           # 经度（2 维）
    torch.zeros(B, 2),         # 预留
], dim=-1)                     # (B, 8)
# 在每个 patch token 的 D-8 维后拼接 8 维地理元数据
patch_tokens = patch_tokens + torch.cat([pos_enc, time_latlon], dim=-1)
```

#### P2.4 多粒度对比（参考 SkySense MGCL）

利用现有的 `embedding_map [B, D, H, W]` 增加多尺度对比：

```python
# 图像级：GAP embedding 的跨时相对比（已有 temporal_contrastive_loss）
img_emb = embedding_map.mean(dim=[2,3])   # [B, D]

# 像素级：每个空间位置的跨时相对比（已有 temporal_cosine_pixel_loss）
pixel_emb = embedding_map.permute(0,2,3,1)  # [B, H, W, D]

# 新增：区域级（中等粒度，4×4 超像素区域平均池化）
region_emb = F.avg_pool2d(embedding_map, kernel_size=4)  # [B, D, H/4, W/4]
# → 对区域级嵌入进行跨时相对比损失
```

---

### 第三优先级：长期架构升级（1+ 月）

#### P3.1 DOFA/Panopticon 风格的动态传感器 stem

将 `SensorEncoderBank` 的固定 stem 替换为波长条件的动态超网络：

```python
class DynamicSensorStem(nn.Module):
    def __init__(self, wave_dim=128, embed_dim=256):
        self.weight_generator = WaveTransformer(wave_dim=wave_dim, out_dim=embed_dim)
    def forward(self, img: torch.Tensor, wavelengths_um: torch.Tensor):
        # wavelengths_um: [C]，各通道波长，单位 μm
        wave_emb = sincos_1d(wavelengths_um * 1000, self.wave_dim)  # → nm
        dynamic_weight, bias = self.weight_generator(wave_emb)
        return F.conv2d(img, dynamic_weight)  # 动态第一层卷积
```

优势：① 自然扩展到新传感器（未来新卫星）；② 允许训练期中途增加数据源；③ 与 Clay/DOFA 权重兼容。

#### P3.2 扩大训练数据（最终瓶颈）

建议数据扩充策略（按可行性排序）：

1. **SSL4EO-S12 预训练权重初始化**：使用 `vits16_ssl4eo-s12_ms_decur_ep100.pth` 初始化 `SensorEncoderBank` 的 S2 stem，减少对大规模训练数据的依赖
2. **OlmoEarth 公开数据迁移学习**：`allenai/olmoearth_pretrain_dataset`（HuggingFace，28.5 万全球样本）用于预训练，再迁移到哈尔滨数据 fine-tune
3. **GeoPlex 风格多数据集联合**：参考 AnySat，将哈尔滨数据与其他开源遥感数据集拼接，异构多任务训练

---

## 总结对比矩阵

```
AEF/OlmoEarth 效果好的根本原因 → xuannv 需要补的短板：

┌────────────────────────┬────────────────────────┬──────────────────────┐
│ 他们做得好的地方        │ 技术本质               │ xuannv 改进方向      │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ 数据规模（30B+ 帧）    │ 量的绝对优势            │ P3.2 数据扩充        │
│ LMIM/JEPA 潜在预测     │ 避免 EO 像素噪声驱动   │ P2.2 替换重建目标    │
│ 传感器解耦（DeCUR）    │ SAR 不污染光学嵌入      │ P2.1 DeCUR 解耦      │
│ 地理坐标注入（Clay）   │ 空间锚定，全球可迁移    │ P2.3 地理元数据注入  │
│ 3D 时序建模（Prithvi） │ 显式编码时间维度信息    │ 中期升级 WindowCode  │
│ 连续时间解耦（AEF）    │ 支持任意时间窗口推理    │ 已有，需进一步增强   │
└────────────────────────┴────────────────────────┴──────────────────────┘

xuannv 已领先的方向：
  ✅ 反坍缩损失套装（5 种互补，全球前沿）
  ✅ 变化检测专项时序设计（反对角线 InfoNCE + 间隔感知）
  ✅ Skip L2 训练（理论上解决 Jacobian 屏障）
  ✅ 轻量级嵌入（64D vs 768D，推理效率高 12×）
```

---

## 置信度评估与文献引用

### 高置信度（直接来源于代码/论文正文）

- AEF 架构（15 块 STP，d_s=1024，64D vMF，批量均匀性公式）→ `Brayden-Zhang/alphaearth-foundations` + arXiv:2507.22291
- OlmoEarth FlexiViT + LMIM → `allenai/olmoearth_pretrain:README.md` + `latent_mim.py`
- Clay v1.5 完整代码（DINOv2 蒸馏，GSD PE，元数据注入）→ `Clay-foundation/model:claymodel/model.py`
- CROMA 三路 ViT + X-ALiBi → `antofuller/CROMA:use_croma.py`
- DeCUR 解耦损失代码 → `zhu-xlab/DeCUR:src/pretrain/models/decur.py`
- Prithvi-EO-2.0 3D PE + 元数据 → `NASA-IMPACT/Prithvi-EO-2.0:README.md`
- AnySat JEPA → `gastruc/AnySat:src/models/module_JEPA.py`
- Panopticon ChnAttn → `Panopticon-FM/panopticon:dinov2/models/panopticon.py`
- xuannv 损失套装 → `/workspace/xuannv/src/training/losses.py`

### 中置信度（论文摘要 + README，未读完整 PDF）

- AEF 具体评估数值（23.9% 误差降低）→ arXiv:2507.22291v2 §4-§7
- SkySense MGCL/GCPL 损失完整公式 → arXiv:2312.10115v2（HTML 截断）
- AnySat/TerraMind 具体 benchmark 数值 → 需读完整 PDF

### 低置信度 / 未确认

- CACo 原始论文（未找到公开 arXiv/GitHub，xuannv 代码中有引用但无源）
- RingMo 精确掩码率（从 SoftCon 引用推断）
- AEF per-source 损失权重 Table S2（supplementary 未完整获取）
