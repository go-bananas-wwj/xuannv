# AlphaEarth Foundations (AEF) GitHub 复现与 Earth Foundation Model 调研报告

> 调研日期：2026-05-15  
> 调研范围：GitHub 仓库、arXiv 论文、技术博客、社区项目  
> 关键词：`AlphaEarth Foundations`, `earth foundation model remote sensing`, `AEF remote sensing github`, `earth foundation model embedding collapse`

---

## 1. 官方与社区复现概况

### 1.1 官方发布（Google DeepMind）

AlphaEarth Foundations (AEF) 由 Google DeepMind 于 2025 年 7 月发布，论文为 **arXiv:2507.22291**[^1]。官方已公开：

- **预计算 Embedding 数据集**：2017–2024 年全球年度 64 维像素级 embedding field，10m 分辨率，通过 Google Earth Engine (GEE) 和 Source Cooperative 分发[^2][^3]。
- **论文与补充材料**：详细描述了 STP 架构、VMF Bottleneck、Batch Uniformity Loss、Teacher-Student 一致性训练等技术细节[^1]。
- **推理 API**：可通过 Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` 访问。

**官方未公开的内容**：
- 完整训练代码
- 原始 6PiB 训练数据集（8,412,511 个时空坐标，约 30 亿帧）
- 模型权重（仅提供预计算 embedding，不提供 backbone checkpoint）

### 1.2 社区复现项目

| 项目 | 作者/组织 | GitHub 链接 | 状态 | 关键说明 |
|------|----------|------------|------|---------|
| **alphaearth-foundations** (unofficial PyTorch) | Brayden-Zhang | [github.com/Brayden-Zhang/alphaearth-foundations](https://github.com/Brayden-Zhang/alphaearth-foundations) | 早期复现 | 非官方 PyTorch 实现，使用 OlmoEarth 预训练数据集的 **1/40 Landsat 子集**，batch_size=16（论文 256），max_steps=20k（论文 100k）[^4] |
| **Beta-Earth** | Asterisk Labs (Miko Czerkawski) | [github.com/asterisk-labs/beta-earth](https://github.com/asterisk-labs/beta-earth) | 活跃 | 基于 AEF 公开 embedding 训练的**轻量级 emulator**，使用 SegFormer-B2 + DINOv3 backbone，可从 Sentinel-1/2 + DEM 输入近似 AEF 输出[^5] |
| **deltabit** | Caleb Robinson | [github.com/calebrob6/deltabit](https://github.com/calebrob6/deltabit) | 工具型 | 交互式变化检测 workbench，直接调用 AEF embedding 进行相似度计算[^6] |
| **earth-embedding-sandbox** | Lucas Kruitwagen | [github.com/Lkruitwagen/earth-embedding-sandbox](https://github.com/Lkruitwagen/earth-embedding-sandbox) | Demo | AlphaEarth embedding 探索工具，支持相似度搜索与查询[^7] |
| **Final-Results-Alpha-Earth-Embedding-Papers** | Gabriel Ireland | [github.com/gabrielireland/Final-Results-Alpha-Earth-Embedding-Papers](https://github.com/gabrielireland/Final-Results-Alpha-Earth-Embedding-Papers) | 研究型 | 控制变量对比：AlphaEarth embedding vs. Sentinel 季节合成，在作物分类任务上的系统评估[^8] |
| **AlphaEarthHack** | UT Austin | GitHub Topic: alphaearth | Hackathon | 2026 年地质科学 Hackathon 项目，探索 AEF 在地质应用中的使用 |
| **Awesome-Remote-Sensing-Foundation-Models** | Jack-bo1220 / jeffaudi / xiaoaoran | 多个 awesome 列表 | 汇总 | 系统整理遥感基础模型，将 AEF 列为重要参考[^9] |

### 1.3 复现现状评估

**结论：目前无完整功能等价的开源复现。**

- **最接近架构复现**的是 Brayden-Zhang 的项目，但受限于数据规模和算力，训练规模仅为官方的约 **1/500**（数据量、batch size、步数均大幅缩减）。
- **最实用的社区项目**是 Beta-Earth，其策略是"绕过复现、直接蒸馏"——利用 AEF 公开的预计算 embedding 作为监督信号，训练轻量级本地推理模型。
- 大多数项目将 AEF 视为**黑盒特征提取器**，在其之上构建下游应用（变化检测、分类、相似度搜索），而非复现模型本身。

---

## 2. 复现中遇到的常见问题

### 2.1 数据与算力壁垒

| 问题 | 官方配置 | 社区复现限制 | 影响 |
|------|---------|-------------|------|
| 训练数据规模 | ~6PiB，30 亿帧，840 万个坐标 | 公开数据集仅 OlmoEarth 等少量替代 | 数据分布差异大，复现性能天花板低 |
| Batch Size | 256 | 通常 16–32 | 大 batch 对 uniformity loss 至关重要 |
| 训练步数 | 100,000 | 通常 20,000 | 收敛不充分 |
| 多源对齐 | 同时处理 S2/S1/Landsat/PALSAR/GEDI/ERA5/GRACE 等 | 多数项目仅用 1–2 种数据源 | 丢失了多模态融合的关键能力 |

### 2.2 时间分辨率瓶颈

社区反馈强烈指出：**目前公开可获取的 AEF embedding 多为年度合成**，这导致亚年度（sub-annual）变化检测能力严重受限：

> "A field that rotates from winter wheat to summer corn? Both crops get averaged into the same embedding. Phenology and flooding have the same problem. Until embeddings ship at native temporal resolution, sub-annual change detection stays mostly manual."[^10]

这正是 xuannv_embdding 项目致力于解决的核心问题之一——**提升时间敏感性（temporal sensitivity）**。

### 2.3 Embedding 解释性与工程化问题

- **黑盒性**：64 维向量难以人工解释，需通过降维（PCA/UMAP）或下游任务探针理解语义[^7]。
- **数据格式碎片化**：不同 Earth Embedding 产品（Clay、Major TOM、AlphaEarth、Presto）使用不同的 tile scheme、CRS 假设、文件格式，缺乏统一标准[^10]。
- **缺乏模型权重**：只能获取 embedding，无法微调 backbone，限制了领域自适应能力。

### 2.4 评估与基准挑战

- NeuCo-Bench（CVPR 2025 EarthVision）指出：现有 EO 基础模型评估"碎片化"，缺乏对**紧凑 embedding** 的标准化、多任务、黑盒评估[^11]。
- 私有基准和不可复现的评估脚本降低了社区信任度。

---

## 3. 对 Embedding 质量的改进方法

### 3.1 AEF 官方方案（论文核心）

AEF 在设计上即针对 embedding 质量和空间利用率进行了多项创新：

1. **Noisy von Mises-Fisher (VMF) Bottleneck**
   - 将时空特征压缩到单位球面 S⁶³ 上，每个空间位置估计一个 VMF 分布的 mean direction
   - VMF concentration 参数（κ）可参数化控制 embedding 流形的"平滑度"
   - 相比高斯先验，VMF 更适合 L2 归一化后的超球面几何，避免潜在坍缩[^12]

2. **Batch Uniformity Objective**
   - 核心思想：在单位球面上，均匀分布的随机向量平均正交
   - 实现：对 batch 内 embedding 沿 batch 维度旋转得到 u'，最小化 `Σ|u_i · u'_i|`
   - 官方设定权重 **b = 0.05**（ablation 显示 b=0.005 在部分任务上最优）
   - 作者明确指出："setting the weight > 0 prevented collapse scenarios where this term would tend to 1 otherwise"[^1]

3. **Teacher-Student Consistency**
   - Teacher 观察完整输入，Student 接收 mask/降采样后的输入
   - 一致性损失确保表示对缺失数据不敏感，提升鲁棒性

4. **Text Alignment (CLIP-style)**
   - 将地理标记文本（Wikipedia 等）与图像 embedding 对齐
   - 增强语义一致性和零样本迁移能力

### 3.2 自监督学习领域的反坍缩方法（可直接借鉴）

| 方法 | 核心机制 | 与 AEF/xuannv 的关联 | 来源 |
|------|---------|---------------------|------|
| **VICReg** | Variance Hinge Loss + Covariance Decorrelation | xuannv 已采用 `variance_regularizer` 和 `decorrelation_loss` | Bardes et al., ICLR 2022[^13] |
| **Barlow Twins** | 归一化互相关矩阵逼近单位矩阵 | xuannv 的 `decorrelation_loss` 即基于此思想 | Zbontar et al., 2021[^13] |
| **Hyperspherical Uniformity Gap (HUG)** | 显式优化类间/类内超球面均匀性，产生更平坦的损失 landscape | 可进一步增强 uniformity loss 的理论基础 | Liu et al., CVPR 2023[^14] |
| **Coding Rate Loss** | 最大化 `log det(I + d/(Nε²) EEᵀ)`，鼓励高秩占用和体积扩张 | 可作为 `raw_uniformity_loss` 的补充或替代 | Jiang et al., 2024[^15] |
| **DirectSpec** | 直接平坦化奇异值谱（Gram 矩阵减均值） | 防止完全和不完全坍缩 | Peng et al., 2024[^15] |
| **Kernel VICReg** | 在 RKHS 中重新定义方差/协方差惩罚，捕捉非线性流形结构 | 适合遥感数据的高维非线性特性 | Li et al., 2026[^16] |

### 3.3 Earth Observation 领域的特殊改进

- **NeuCo-Bench 洞察**：embedding 尺寸并非越大越好。在 1,024 维约束下，**post-encoding 时序融合**（对季节视图编码后融合）在时序敏感任务上显著优于 pre-encoding 融合[^11]。
- **Beta-Earth 洞察**：即使从简单 RGB 输入，也能对 AEF 输出实现"合理强的近似"，说明 AEF embedding 的相当一部分信息可由低维光谱输入捕获，但也暗示多模态（SAR + DEM）仍有不可替代的价值[^5]。
- **云掩码与数据完整性**：社区普遍认为应在预训练阶段保留云层、阴影等"混乱"样本，而非过度筛选。"Pretrain on the full distribution"[^10]。

---

## 4. 训练配置的关键参数

### 4.1 AEF 官方配置（来自论文补充材料）

| 参数 | 取值 | 说明 |
|------|------|------|
| Embedding 维度 | 64 | 单位球面 S⁶³，每像素 64 字节 |
| 空间分辨率 | 10m | 最佳 in-class 分辨率 |
| Batch Size | 256 | 大 batch 对 uniformity loss 重要 |
| 训练步数 | 100,000 | — |
| 训练坐标数 | 8,412,511 | 排除南极洲后 |
| 总帧数 | 3,047,520,515 | 约 30 亿帧 |
| 数据总量 | ~6 PiB | 含复制冗余 |
| 基本采样块 | 1.28km × 1.28km | 含 160m 缓冲 overlap |
| 归一化 | Z-score，±6σ clip | 全局统计量 |
| 重建损失权重 (a) | 1.0 | 整体重建目标 |
| Batch Uniformity 权重 (b) | **0.05** | 官方最终取值；ablation 最优约 0.005 |

### 4.2 各数据源重建损失配置

| 数据源 | Shift-Invariant Loss | Re-gridding Loss | 误差度量 | Loss Weight |
|--------|---------------------|------------------|---------|-------------|
| Sentinel-2 L1C | 20m | — | L1 | 1.0 |
| Sentinel-1 GRD | 20m | — | L1 | 1.0 |
| Landsat Group | — | 30m | L1 | 1.0 |
| PALSAR-2 ScanSAR | — | 30m | L1 | 1.0 |
| ERA5-Land Monthly | — | — | L1 | 1.0 |
| GEDI L2A | — | 20m | L1 | 1.0 |
| GRACE Monthly Mass | — | 1280m | L1 | 0.5 |
| Copernicus DEM GLO-30 | — | 30m | L1 | 1.0 |
| NLCD Group | — | 30m | Cross Entropy | 0.5 |

### 4.3 社区复现的缩减配置

以 Brayden-Zhang 的复现为例[^4]：

```bash
python -m alphaearth.run_train_olmoearth \
    --data_dir ./data/olmoearth_pretrain_dataset/10_landsat_monthly \
    --batch_size 32 \
    --num_workers 4 \
    --patch_size 256 \
    --max_steps 20000 \
    --output_dir ./outputs_olmoearth
```

**与官方的关键差距**：
- 仅使用 Landsat 单一数据源（官方 8+ 源）
- batch_size 仅为官方的 1/8
- 训练步数仅为 1/5
- 数据量约为官方的 1/40

---

## 5. Embedding Collapse 的解决方案

### 5.1 AEF 官方的坍缩预防机制

AEF 论文明确承认 embedding collapse 是核心风险，并采用了**多层防御**：

1. **VMF 分布约束**：将 embedding 限制在超球面上，避免欧氏空间中的零点收缩
2. **Batch Uniformity Loss**：显式鼓励 batch 内 embedding 正交，权重 b>0 时"防止坍缩场景"
3. **多源重建损失**：强重建目标（a=1.0）迫使 embedding 保留足够信息以重建多种传感器数据
4. **Teacher-Student 一致性**：Student 必须在输入退化时仍匹配 Teacher，防止捷径学习

### 5.2 自监督学习社区的反坍缩技术全景

根据近期综述和研究[^15][^13][^14][^16]，可归纳为以下策略：

| 策略类别 | 具体方法 | 适用场景 |
|---------|---------|---------|
| **显式方差保持** | VICReg Variance Hinge Loss | 防止所有 embedding 收缩到零 |
| **去相关/白化** | Barlow Twins、VICReg Covariance、W-MSE | 防止信息冗余（informational collapse） |
| **超球面均匀性** | HUG、Batch Uniformity、DirectSpec | 在归一化空间中强制分散 |
| **谱平坦化** | Direct All-Pass Filtering、Coding Rate | 提升有效秩（effective rank） |
| **非最大移除** | nmrGCL | 在图对比学习中重分配信息 |
| **温度缩放** | TempScale | 缓解 Transformer 中的过度低通滤波 |
| **核方法提升** | Kernel VICReg (RBF/Laplacian/RQ) | 捕捉非线性流形，避免欧氏假设失效 |
| **一致性匹配** | CM-loss (Diffusion Codebook) | 跨噪声级别稳定 embedding |

### 5.3 对 xuannv_embdding 项目的直接建议

基于本次调研，以下方法可作为 xuannv_embdding 现有反坍缩体系（raw_uniformity + decorrelation + variance_regularizer + orthogonality）的补充或升级：

1. **Batch Uniformity 的改进**：AEF 的 batch 旋转正交损失简单有效，可考虑将其整合进现有 `raw_uniformity_loss`，或作为独立项加入损失组合。
2. **Kernel VICReg**：若当前欧氏空间 uniformity 遇到瓶颈，可尝试在 RKHS 中计算方差/协方差，尤其适合遥感数据的高维非线性特性。
3. **Coding Rate Loss**：作为 `raw_uniformity` 的理论增强，显式最大化 embedding 矩阵的体积。
4. **HUG (Hyperspherical Uniformity Gap)**：如果未来引入监督信号或聚类结构，HUG 框架可提供更强的几何约束。
5. **保持强重建目标**：AEF 的经验表明，a=1.0 的多源重建损失是防止坍缩的重要锚点——重建损失权重不宜过度降低。

---

## 6. 相关基准与评估框架

| 框架 | 说明 | 链接 |
|------|------|------|
| **NeuCo-Bench** | 2025 CVPR EarthVision 挑战使用的 neural embedding 评估框架，支持隐藏任务评估，防止预训练偏置 | [github.com/embed2scale/NeuCo-Bench](https://github.com/embed2scale/NeuCo-Bench)[^11] |
| **GEO-Bench** | 地理基础模型基准，侧重 segmentation | — |
| **PANGAEA** | 地理基础模型基准，需访问 backbone | — |
| **Awesome-RSFMs** | 遥感基础模型全面综述与资源列表 | [github.com/xiaoaoran/awesome-RSFMs](https://github.com/xiaoaoran/awesome-RSFMs)[^9] |

---

## 参考文献与来源

[^1]: Brown, C. F., et al. "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data." arXiv:2507.22291, 2025.  
 链接：https://arxiv.org/abs/2507.22291

[^2]: Source Cooperative. "AlphaEarth Foundations Satellite Embedding Dataset."  
 链接：https://source.coop/tge-labs/aef

[^3]: Element84. "Exploring AlphaEarth Embeddings." 2025-12-03.  
 链接：https://element84.com/machine-learning/exploring-alphaearth-embeddings/

[^4]: Brayden-Zhang. "alphaearth-foundations (unofficial PyTorch implementation)." GitHub, 2025.  
 链接：https://github.com/Brayden-Zhang/alphaearth-foundations

[^5]: Satellite Image Deep Learning. "BetaEarth: Open Embeddings of Sentinel-2 and Sentinel-1 with a Little Help of AlphaEarth." 2026-04-29.  
 链接：https://www.satellite-image-deep-learning.com/p/betaearth-open-embeddings-of-sentinel

[^6]: Robinson, C. "deltabit — An interactive change-detection workbench for satellite imagery using AlphaEarth Foundations embeddings." GitHub.  
 链接：https://github.com/calebrob6/deltabit

[^7]: Kruitwagen, L. "earth-embedding-sandbox." GitHub, 2025.  
 链接：https://github.com/Lkruitwagen/earth-embedding-sandbox

[^8]: Ireland, G. "Final-Results-Alpha-Earth-Embedding-Papers." GitHub, 2026.  
 链接：https://github.com/gabrielireland/Final-Results-Alpha-Earth-Embedding-Papers

[^9]: Xiao, A., et al. "Foundation Models for Remote Sensing and Earth Observation: A Survey." arXiv:2410.16602, 2024.  
 链接：https://github.com/xiaoaoran/awesome-RSFMs

[^10]: Cloud Native Geo. "Earth Embedding Products." 2026-02-28.  
 链接：https://github.com/cloudnativegeo/cloudnativegeo.org/blob/main/content/blog/260228-earth-embedding-products.md

[^11]: Vinge, R., et al. "NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation." arXiv:2510.17914, 2025.  
 链接：https://github.com/embed2scale/NeuCo-Bench

[^12]: Davidson, T. R., et al. 相关 vMF VAE 工作；以及 AEF 论文补充材料 S16.2.4。

[^13]: Bardes, A., Ponce, J., & LeCun, Y. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022.  
 链接：https://arxiv.org/abs/2105.04906

[^14]: Liu, W., et al. "Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap." CVPR 2023.  
 链接：https://arxiv.org/abs/2303.06484

[^15]: Emergent Mind. "Embedding Dimensional Collapse." 2025-12-17. 综合综述。  
 链接：https://www.emergentmind.com/topics/embedding-dimensional-collapse

[^16]: Li, et al. "Kernel VICReg for Self-Supervised Learning in Reproducing Kernel Hilbert Space." arXiv:2509.07289, 2026.

---

*本报告由 Agent 基于公开网络资源调研生成，所有引用均已标注来源。*
