# AlphaEarth Foundations (AEF) 原始论文架构深度调研报告

> **论文**: AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data  
> **作者**: Christopher F. Brown*, Michal R. Kazmierski*, Valerie J. Pasquarella* 等 (Google DeepMind)  
> **arXiv**: [2507.22291](https://arxiv.org/abs/2507.22291) (v1: 2025-07-29, v2: 2025-09-08)  
> **调研日期**: 2026-05-15

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [VMF Bottleneck 设计细节](#2-vmf-bottleneck-设计细节)
3. [传感器编码器 (SensorEncoderBank)](#3-传感器编码器-sensorencoderbank)
4. [STP Block (Space-Time-Precision)](#4-stp-block-space-time-precision)
5. [时间编码方式](#5-时间编码方式)
6. [解码器 (Decoders / Heads)](#6-解码器-decoders--heads)
7. [损失函数组合](#7-损失函数组合)
8. [训练策略](#8-训练策略)
9. [关键补充: 推理与量化](#9-关键补充-推理与量化)
10. [参考文献](#10-参考文献)

---

## 1. 总体架构概览

AEF 是一个**多源、多时相、自监督的嵌入场模型 (embedding field model)**，核心目标是将异构地球观测数据压缩为时空连续的 64 维单位球面嵌入向量。其架构由三大网络协同组成 [^1][^2][^4]：

- **教师网络 (Teacher Video Embedding Model)**：处理完整、未扰动的输入影像序列，通过 STP 编码器 + VMF 瓶颈生成嵌入，并配备隐式解码器进行多源重建。
- **学生网络 (Student Video Embedding Model)**：与教师共享**完全相同的参数和架构**，但输入经过随机扰动（随机丢弃时间步或整个数据源），用于学习对缺失数据的鲁棒表示。
- **文本对齐网络 (Text Alignment Model)**：使用冻结的 Gemini 语言模型，将地理编码的 Wikipedia / GBIF 文本描述与影像嵌入进行 CLIP 式对比学习。

模型最终选用了 **~480M 参数**的变体（另有一个 ~1B 参数版本），在推理效率与性能之间取得平衡 [^1][^2]。

**输入输出规范** [^1][^3]：
- **输入**: 多源时序影像帧序列 \(N_i\)（如 Sentinel-2、Sentinel-1、Landsat），每帧带有精确的时间戳（毫秒级 epoch time）和空间分辨率。
- **支持期 (Support Period)**: 输入数据的时间范围。
- **有效期 (Valid Period)**: \( [t_s, t_e) \)，用户查询的摘要时间窗口，可以与支持期不完全重叠（支持**时间插值与外推**）。
- **输出**: \( L \times L \times 64 \) 的 embedding field，每个空间位置对应一个 64 维单位球面 \( S^{63} \) 上的向量，空间分辨率 10m。

---

## 2. VMF Bottleneck 设计细节

### 2.1 核心创新：空间密集的变分瓶颈

传统方法通常将特征图全局平均池化（GAP）为单一向量，导致空间信息完全丢失。AEF 的关键创新在于：

> **不在空间上折叠特征，而是在 \( L \times L \) 的网格上，为每个空间位置都预测一个 von Mises-Fisher (VMF) 分布的均值方向向量** [^1][^2][^5]。

- VMF 分布是定义在超球面上的高斯分布等价物。
- 最终输出一个 \( L \times L \times 64 \) 的张量，称为 **"嵌入场" (Embedding Field)**。
- 推理时取其中一个像素的嵌入向量作为该点的最终表示。

### 2.2 VMF 参数化

- **嵌入维度**: \( D = 64 \)，约束在单位超球面 \( S^{63} \) 上 [^1][^2]。
- **浓度参数 (Concentration, \(\kappa\))**: 固定为 **\( \kappa = 8 \times 10^3 = 8000 \)** [^1][^2]。
  - \(\kappa\) 控制嵌入流形的"平滑度"：\(\kappa\) 越低，分布越分散（噪声越大，流形越平滑）；\(\kappa\) 越高，分布越集中（信息容量越大）。
  - 论文在 S7.3 中对 \(\kappa\) 和嵌入维度进行了消融实验（Figure S22），最终选定 \( D=64, \kappa=8000 \) [^1]。
- **训练 vs 推理差异**:
  - **训练时**: 从 VMF 分布中**采样**得到嵌入（noisy bottleneck），作为正则化手段，迫使解码器对噪声鲁棒，提高泛化能力 [^1][^2]。
  - **推理时**: 直接取 VMF 分布的**均值方向 (mean direction)** 作为确定性嵌入输出，不进行随机采样 [^2]。

### 2.3 Batch Uniformity 目标

为防止嵌入坍缩 (collapse) 并促进嵌入在 \( S^{63} \) 上均匀分布，论文引入 batch uniformity 损失：

\[
\text{BatchUniformity} = \sum_{i=1}^{64} |u_i \cdot u'_i|
\]

其中 \( u' \) 是将 batch 维度旋转后得到的向量。由于球面上随机向量平均正交 [^6]，最小化该点积的绝对值可促进均匀分布。该损失权重最终设为 **\( b = 0.05 \)** [^1][^2]。

---

## 3. 传感器编码器 (SensorEncoderBank)

### 3.1 输入数据源与预处理

AEF 将数据源分为 **输入源 (Encoded / Input)** 和 **目标源 (Decoded / Target)**。论文发现，最小化的有效输入集为 [^1][^2]：

| 类型 | 数据源 | 波段/通道 | 用途 |
|------|--------|-----------|------|
| 光学 | Sentinel-2 L1C | B2, B3, B4, B8, B11 (5 bands) | 输入+目标 |
| 光学/热红外 | Landsat-8/9 C2 T1 TOA | B2, B3, B4, B5, B6, B8, B10 (7 bands) | 输入+目标 |
| C波段 SAR | Sentinel-1 GRD | VV, VH, HH, HV, angle (5 channels) | 输入+目标 |
| L波段 SAR | ALOS PALSAR-2 ScanSAR L2.2 | HH, HV, lin (3 channels) | 仅目标 |
| 高程 | Copernicus DEM GLO-30 | DEM (1 channel) | 仅目标 |
| 气候 | ERA5-Land Monthly | 降水、气温、露点、气压等 | 仅目标 |
| LiDAR | GEDI L2A | RH[0-100] (relative height) | 仅目标 |
| 重力 | GRACE Monthly Mass Grids | 等效液态水厚度 | 仅目标 |
| 土地覆盖 | NLCD 2019/2021 | 分类标签 | 仅目标 |
| 文本 | Wikipedia / GBIF | 文本嵌入 | 仅目标 (对比学习) |

### 3.2 预处理与归一化

- **Sentinel-2 & Landsat**: 像素强度先经过 \( s(x) = \log(x+1)/10 \) 变换，再基于全局统计量进行标准化 [^1]。
- **Sentinel-1**: 将 DN 值转换为 dB: \( \gamma = 10 \cdot \log_{10}(DN^2) - 83 \)；局部入射角 (lin) 转换为弧度 [^1]。
- **Cloud Score+**: Sentinel-2 使用 Cloud Score+ 的 cloud score 波段，以 0.5 为阈值进行二值化掩膜 [^1]。
- **所有栅格数据**: 重投影到 UTM 坐标系，再统一重采样到共同网格。

### 3.3 输入投影器 (Input Projectors)

每个数据源有独立的输入投影器（图 2A 中标记为 "H"），负责将各源的通道和空间分辨率映射到模型统一的潜在空间，并执行初始下采样到 **1/2 L**（即 Precision Path 的分辨率）[^1][^2]。

---

## 4. STP Block (Space-Time-Precision)

### 4.1 设计动机

AEF 的视频摘要架构必须同时满足两个矛盾需求 [^1][^2]：
1. **高度局部化的表示**（保持空间精度）
2. **建模时空长距离依赖关系**（全局上下文）

为此设计了 **Space-Time-Precision (STP) 编码器**，由重复的三路并行算子块组成，块间通过空间金字塔 "交换" (exchange) 进行信息融合。

### 4.2 三路并行算子

给定边长为 \( L \) 的方形输入，每个 STP block 包含三个同时运行的算子 [^1][^2][^4]：

| 路径 | 分辨率 | 机制 | 维度 | 目的 |
|------|--------|------|------|------|
| **Space (空间路)** | \( \frac{1}{16}L \) | ViT-like Spatial Self-Attention | \( D_S = 1024 \) | 捕捉局部与非局部的空间长距离依赖 |
| **Time (时间路)** | \( \frac{1}{8}L \) | Time-axial Self-Attention | \( D_T = 512 \) | 沿时间轴的自注意力，建模时序动态 |
| **Precision (精度路)** | \( \frac{1}{2}L \) | 3×3 卷积 | \( D_P = 128 \) | 保留高空间分辨率的局部细节 |

- **时间路的时间编码**: 每个序列元素（时间步）通过与正弦时间码 (sinusoidal timecode) 关联后进行条件化 [^1][^2]。
- **块间交互**: STP blocks 以**学习的拉普拉斯金字塔重采样 (learned Laplacian pyramid rescaling)** 终止，使得三个算子的状态可以传递到下一个 block 的对应算子中 [^1][^2]。
- **最终输出**: STP 编码器以最终学习的空间重采样终止，输出分辨率恢复到 **Precision 路径的 1/2 L**，对每个输入源产生 \( \sum N_i \) 个输出特征图 [^1][^2]。

### 4.3 网络深度

- **STP blocks 总数**: **15 个** [^1][^2]。
- **模型总参数量**: ~480M（选用版本）/ ~1B（大版本）[^1][^2]。

---

## 5. 时间编码方式

### 5.1 时间戳编码

- 原始时间戳（毫秒级 epoch time）被转换为 **正弦时间码 (sinusoidal timecodes)** [^1][^2]。
- 时间路 (Time Operator) 中的每个序列元素都通过与该时间码关联后进行条件化 [^2]。

### 5.2 支持期与有效期分离

这是 AEF 支持**连续时间**的核心机制 [^1][^2]：

- **支持期 (Support Period)**: 输入观测数据实际覆盖的时间范围。
- **有效期 (Valid Period)**: \( [t_s, t_e) \)，用户希望获取地表摘要的时间窗口。
- 二者无需完全重叠：
  - **插值**: 有效期内没有输入观测。
  - **外推**: 有效期完全在支持期之前或之后。

### 5.3 时间条件摘要 (Temporal Summarization)

在 STP 编码器之后，模型进行时间条件摘要 [^1][^2]：

1. 基于有效期 \( [t_s, t_e) \) 生成一个**学习的查询特征 (learned query feature)**。
2. 该查询特征与正弦时间码结合，通过**时间轴注意力池化 (time-axial attention pooling)** 聚合多时相信息。
3. 产生的时序摘要用一个**学习的核 (learned kernel)** 上采样到尺寸 \( L \)。
4. 随后进入 VMF 瓶颈生成嵌入。

### 5.4 解码时的时间条件

解码器接收的元数据包含 [^1][^2]：
- 一个正弦时间码，表示在有效期内的相对位置（归一化到 \( [0, 1) \)）。
- 传感器几何元数据（仅与测量行为相关，与测量内容无关）。
- 这使得解码器能够为任意时间戳生成空间连续的预测（例如，从 GEDI 生成密集的、超分辨率的 LiDAR 剖面）。

---

## 6. 解码器 (Decoders / Heads)

### 6.1 隐式解码器 (Implicit Decoders)

AEF 为每个目标源 \( i \in M_D \) 配备了一个**源特定的隐式解码器** [^1][^2]：

- **架构**: 两个隐藏层的 MLP，宽度为 **512** [^1][^2]。
- **输入**: 
  - VMF 瓶颈输出的嵌入向量（64维）
  - 该源特定的条件元数据（时间码、传感器几何等）
- **输出**: \( L \times L \) 的重建图 \( y'_i \)，带有 \( C_i \) 个通道。
- **应用方式**: 该解码器网络被应用到输出网格的**每一个空间位置**上。

### 6.2 解码器特性

- **空间连续性**: 由于解码器在每个像素上独立运行并以时间码为条件，它能为任意时间戳生成空间连续的预测 [^1][^2]。
- **超分辨率能力**: 即使嵌入的名义分辨率为 10m，解码器也能重建原始分辨率不同于 10m 的目标（如 30m 的 DEM、25m 的 GEDI、0.5° 的 GRACE），通过重采样损失 (re-gridding loss) 处理分辨率不匹配 [^1][^2]。

---

## 7. 损失函数组合

AEF 的训练目标是一个四项损失的加权和 [^1][^2][^4]：

\[
l = \underbrace{\frac{a}{M} \sum_{i \in M} f_i(y_i, y'_i) w_i}_{\text{(a) 重建损失}} + \underbrace{b \sum_{i=1}^{64} |u_i \cdot u'_i|}_{\text{(b) Batch Uniformity}} + \underbrace{c \frac{1 - \boldsymbol{\mu} \cdot \boldsymbol{\mu}_s}{2}}_{\text{(c) Consistency}} + \underbrace{d \cdot f_{CLIP}(\boldsymbol{\mu}, \boldsymbol{\mu}_t)}_{\text{(d) Text Contrastive}}
\]

### 7.1 损失权重

| 损失项 | 权重符号 | 论文设定值 | 说明 |
|--------|----------|-----------|------|
| 重建损失 (Reconstruction) | \( a \) | 1.0 (基准，各源再乘 \( w_i \)) | 整体重建目标权重 |
| Batch Uniformity | \( b \) | **0.05** | 最终选用值。消融显示 \( b=0.005 \) 在某些评估上最优，但 0.05 是最终训练设定 [^1][^2] |
| Consistency (Teacher-Student) | \( c \) | **0.02** | 平衡重建视觉质量与学生/教师一致性 [^1][^2] |
| Text Contrastive (CLIP) | \( d \) | **0.001** | 文本-影像对齐权重 [^1][^2] |

> 注意：论文提到 "Loss weights were normalized prior to training" [^1][^2]。

### 7.2 各损失项详解

#### (a) 重建损失 (Reconstruction Loss)

- **目标**: 对每个解码源，模型从输入序列中随机选择一帧（或被保留的帧）作为目标 \( y_i \)。该目标帧可能从输入序列中移除（即被隐藏），强迫模型从嵌入中重建它 [^1][^2]。
- **损失函数**:
  - **连续数据源**: L1 损失
  - **分类数据源 (如 NLCD)**: 交叉熵损失 (Cross-Entropy)
- **源特定权重 \( w_i \)** [^1]：

| 数据源 | Shift-invariant Loss | Re-gridding Loss | 误差度量 | 损失权重 \( w_i \) |
|--------|---------------------|------------------|----------|-------------------|
| Sentinel-2 L1C | 20m | – | L1 | 1.0 |
| Sentinel-1 GRD | 20m | – | L1 | 1.0 |
| Landsat Group | – | 30m | L1 | 1.0 |
| PALSAR-2 ScanSAR L2.2 | – | 30m | L1 | 1.0 |
| ERA5-Land Monthly | – | – | L1 | 1.0 |
| GEDI L2A | – | 20m | L1 | 1.0 |
| GRACE Monthly Mass Grids V4 | – | 1280m | L1 | 0.5 |
| Copernicus DEM GLO-30 | – | 30m | L1 | 1.0 |
| NLCD Group | – | 30m | Cross-Entropy | 0.5 |

- **特殊处理**:
  - **Shift-invariant loss**: 在指定平面偏移距离内取最小误差，处理仪器空间配准误差。
  - **Re-gridding loss**: 通过面积加权平均将重建和目标重采样到给定名义分辨率后再计算误差。
  - 所有损失都使用**每帧每像素权重**来处理条带边缘和无效像素 [^1][^2]。

#### (b) Batch Uniformity Loss

- **公式**: \( \sum_{i=1}^{64} |u_i \cdot u'_i| \)
- **原理**: 在 batch 维度旋转嵌入向量，最小化对应维度点积的绝对值。由于球面上随机向量期望正交，这是均匀分布的必要条件 [^1][^2]。
- **作用**: 防止嵌入坍缩，鼓励嵌入在 \( S^{63} \) 上均匀分布。
- **消融**: 对 \( b \in [0, 0.001, 0.005, 0.01, 0.1] \) 进行扫描，发现 \( b=0 \) 和 \( b=0.1 \) 表现最差，\( b=0.005 \) 在某些评估上最优。但论文最终训练采用 **\( b=0.05 \)** [^1][^2]。

#### (c) Consistency Loss (Teacher-Student)

- **公式**: \( \text{ConsistencyLoss} = \frac{1 - \boldsymbol{\mu} \cdot \boldsymbol{\mu}_s}{2} \)
  - \( \boldsymbol{\mu} \): 教师嵌入（完整输入）
  - \( \boldsymbol{\mu}_s \): 学生嵌入（扰动输入）
- **学生输入扰动策略** [^1][^2]：
  1. **随机丢弃整个数据源**: Landsat Group 30% 概率丢弃；Sentinel-1 GRD 30% 概率丢弃；Sentinel-2 L1C **从不丢弃**。
  2. **三种时间步扰动策略（等概率选择）**:
     - (a) 随机丢弃时间步：Landsat 30%、S1 30%、S2 50%。
     - (b) 丢弃输入序列的后六个月（类预测）。
     - (c) 丢弃输入序列的前六个月（类回溯）。
- **有效期对齐**: 若采用策略 (a)，选择与未扰动输入年度相交的随机摘要期；若策略 (b)/(c)，则分别选择序列后/前六个月内的摘要期。
- **权重**: **\( c = 0.02 \)**。论文指出，虽然该权重已大幅减少瓦片伪影 (tile artifacts)，但在不规则输入导致的 embedding field 中仍可见，未来可通过更激进的 consistency 项进一步消除 [^1][^2]。

#### (d) Text Contrastive Loss (CLIP-style)

- **机制**: 使用冻结的 Gemini 语言模型生成文本嵌入，通过一个 MLP 解码器将其与教师模型的影像嵌入对齐，使用标准 CLIP 损失 [^1][^2]。
- **采样**: 若训练样本有对应的 Wikipedia / GBIF 文本点，则随机选取一个文本点，并选择一个与教师输入年度相交且长度 > 4 天的唯一随机摘要期。
- **权重**: **\( d = 0.001 \)** [^1][^2]。

---

## 8. 训练策略

### 8.1 训练数据规模

- **总观测帧数**: 超过 **30 亿**帧（来自 5M+ 全球分布站点，覆盖地球陆地面积约 **1.1%**）[^1][^2]
- **视频序列数**: **8,412,511** 条
- **输入序列子采样**: 每条序列采样至 **103 帧**，包括 65 帧 Sentinel-2 L1C、17 帧 Sentinel-1 GRD、21 帧 Landsat Group [^1][^2]
- **不可用/扰动帧**: 用掩码 (mask) 替代 [^1]

### 8.2 硬件与计算资源

- **训练设备**: **512 块 TPU v4** [^1][^2]
- **训练时长**: **56 小时**
- **总步数**: **100,000 steps**
- **Batch Size**: **256 条视频序列**
- **并行策略**: 按 batch 分片，且每个 batch 元素进一步分配到 **2 块 TPU v4** 上 [^1][^2]

### 8.3 优化器与学习率

- **优化器**: **Adam** [^1][^2]
- **学习率调度**: 分段线性 (piecewise linear)
  - 阶段 1: 从 **0 → 1e-4**，跨越步数 **[0, 1e3)**
  - 阶段 2: 从 **1e-4 → 0**，跨越步数 **[1e3, 1e5]**
- **超参数选择标准**: 最小化训练损失，同时保持满意的重建视觉质量、对比学习目标与 batch uniformity 目标的稳定性，以及诊断评估上的期望性能（嵌入是否能区分特定输入源的存在）[^1][^2]。

### 8.4 Teacher-Student 训练细节

- **参数共享**: 教师与学生模型**共享全部参数** [^1][^2]。
- **前向传播**: 每次训练迭代运行两次前向传播——一次教师（完整输入），一次学生（扰动输入）。
- **目标**: 教师生成高质量嵌入，学生从有限/扰动输入中模仿教师，一致性损失强制二者在相同有效期下的嵌入对齐 [^1][^2]。

### 8.5 重建目标采样

- 在每条训练序列中，从每个源的序列中**随机选择一帧**作为重建目标。
- 若该帧属于模型输入序列，则直接从输入序列中移除。
- 该帧及其元数据和时间码用于计算重建损失。
- 每个重建目标使用**不同的嵌入**（对应唯一的摘要期），目标时间戳被归一化到该有效期内 \( [0, 1) \) [^1][^2]。

---

## 9. 关键补充: 推理与量化

### 9.1 推理流程

1. 按 UTM 区域对 1.28 km × 1.28 km 的缓冲影像芯片进行推理。
2. 合并结果以实现无缝的年度全球覆盖。
3. 输出 10m 分辨率的 embedding field。

### 9.2 量化策略

- 为减少存储和计算开销，将 32-bit float 嵌入量化到 **8-bit (uint8)**。
- 量化方法：非线性幂变换 + 缩放 + 取整 + 截断（power=2）。
- 结果：**4 倍压缩**，且下游任务性能损失可忽略不计 [^1]。
- 最终发布的 Satellite Embedding 数据集：2017–2024/2025 年度全球层，存储于 Google Earth Engine。

---

## 10. 参考文献

[^1]: Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., et al. (2025). *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data*. arXiv:2507.22291. 原始论文及补充材料 (S1-S8, S16)。

[^2]: Brown et al. (2025). AlphaEarth Foundations 补充材料 S16 (Modeling) 与 S2 (Training algorithm)。

[^3]: Ma, Y., et al. (2025). *Harvesting AlphaEarth: Benchmarking the Geospatial Foundation Model for Agricultural Downstream Tasks*. arXiv:2601.00857.

[^4]: Houriez et al. (2025). *Scalable Geospatial Data Generation Using AlphaEarth Foundations Model*. arXiv:2508.11739.

[^5]: 知乎专栏 (2025). 《如何看待AlphaEarth对遥感图像解译的贡献？》—— 模型架构解析。

[^6]: Cai, T., Fan, J., & Jiang, T. (2013). Distributions of angles in random packing on spheres. *Journal of Machine Learning Research*, 14(1), 1837–1864.

---

> **本报告结束**。所有关键架构参数、损失权重、训练配置均直接引用自原始论文及其补充材料，确保与 AEF 官方设计一致。
