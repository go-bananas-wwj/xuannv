# Translation Notes / 翻译备注

> **论文**: Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications  
> **生成日期**: 2026-05-27

---

## 一、术语表 / Glossary

| 英文术语 | 中文译名 | 备注 |
|---------|---------|------|
| Foundation Model | 基础模型 | 大规模预训练后可通过微调适配下游任务的通用模型 |
| Geospatial Foundation Model (GFM) | 地理空间基础模型 | 专门针对地理空间/遥感数据的基础模型 |
| Earth Observation (EO) | 地球观测 | 通过卫星等遥感手段对地球表面进行观测 |
| Remote Sensing | 遥感 | 不接触目标物而获取其信息的技术 |
| Temporal | 时序的 / 时间的 | 强调时间维度上的变化 |
| Multi-Temporal | 多时序的 | 包含多个时间点的数据或建模 |
| Self-Supervised Learning | 自监督学习 | 利用数据本身结构生成监督信号的训练方式 |
| Masked Autoencoder (MAE) | 掩码自编码器 | He et al. 提出的基于掩码重建的预训练方法 |
| Vision Transformer (ViT) | 视觉 Transformer | 将 Transformer 架构应用于图像的模型 |
| Embedding | 嵌入 / 嵌入向量 | 将高维数据映射到低维稠密向量的表示 |
| Positional Embedding | 位置嵌入 | 为模型提供位置信息的可学习或固定编码 |
| Patch Embedding | Patch 嵌入 | 将图像分块后映射为向量序列的操作 |
| Fine-tuning | 微调 | 在预训练模型基础上用少量标注数据继续训练 |
| Pretraining | 预训练 | 在大规模无标注数据上进行的初始训练 |
| Backbone | 骨干网络 | 模型中用于特征提取的主体部分 |
| Decoder | 解码器 | 将特征表示恢复为原始数据或预测输出的模块 |
| Token | Token | Transformer 处理的基本单元（此处为图像 patch 的嵌入） |
| Attention | 注意力机制 | Transformer 核心的权重计算机制 |
| Metadata | 元数据 | 描述数据的数据，如时间、位置信息等 |
| Geospatial Embedding | 地理空间嵌入 | 编码地理位置信息的嵌入向量 |
| Harmonized Landsat Sentinel-2 (HLS) | 协调型 Landsat-Sentinel-2 | NASA 统一处理的两颗卫星融合产品 |
| Land Use / Land Cover (LULC) | 土地利用/土地覆盖 | 描述地表覆盖类型的分类体系 |
| Ecoregion | 生态区 | 具有相似生态特征的区域 |
| mIoU (mean Intersection over Union) | 平均交并比 | 语义分割任务的核心评估指标 |
| Accuracy | 准确率 | 分类任务的基本评估指标 |
| Precision / Recall / F1-score | 精确率 / 召回率 / F1 分数 | 分类任务的综合评估指标 |
| RMSE (Root Mean Square Error) | 均方根误差 | 回归任务的评估指标 |
| R² (Coefficient of Determination) | 决定系数 | 回归模型拟合优度的指标 |
| LoRA (Low-Rank Adaptation) | 低秩适配 | 一种参数高效的微调方法 |
| Subject Matter Expert (SME) | 领域专家 | 具有特定领域专业知识的专家 |
| Benchmarking | 基准测试 | 在标准化任务集上比较不同模型性能 |
| Downstream Task | 下游任务 | 利用预训练模型解决的具体应用任务 |
| Transferability | 可迁移性 | 模型知识迁移到新任务的能力 |
| TerraTorch | TerraTorch | IBM 开发的地理空间深度学习工具包 |
| Hugging Face | Hugging Face | 主流模型托管与分享平台 |
| PyTorch Lightning | PyTorch Lightning | PyTorch 的高层封装框架 |
| TorchGeo | TorchGeo | 用于地理空间数据的 PyTorch 扩展库 |
| Stratified Sampling | 分层抽样 | 按类别比例进行抽样的方法 |
| Oversampling | 过采样 | 对少数类样本增加采样量的策略 |
| Nearest-Neighbor Interpolation | 最近邻插值 | 用最近已知像素值填充缺失像素的简单方法 |
| Fmask | Fmask | 用于识别云和云阴影的算法/波段 |
| Sentinel-1 / Sentinel-2 | 哨兵一号 / 哨兵二号 | ESA 的 SAR 和光学卫星系列 |
| Landsat 8 / 9 | 陆地卫星 8/9 | NASA/USGS 的光学卫星 |
| MERRA-2 | MERRA-2 | NASA 的再分析气象数据集 |
| FLUXNET | FLUXNET | 全球涡动相关通量观测网络 |
| GPP (Gross Primary Productivity) | 总初级生产力 | 生态系统通过光合作用固定的碳总量 |
| AGB (Above Ground Biomass) | 地上生物量 | 地表以上植物有机物的总质量 |
| Biome | 生物群落 | 具有相似植被和气候特征的大尺度生态区域 |
| Hypernetwork | 超网络 | 生成其他网络权重的网络 |
| Contrastive Learning | 对比学习 | 通过拉近正样本、推远负样本进行表征学习 |
| Distillation | 蒸馏 | 将大模型知识迁移到小模型的方法 |
| Band-Pass Filter | 带通滤波器 | 允许特定频率范围通过的滤波器 |
| Spectral Band | 光谱波段 | 传感器在特定波长范围内记录的数据通道 |
| Reflectance | 反射率 | 地表反射太阳辐射的比例 |
| Cloud Shadow | 云阴影 | 云遮挡阳光在地表形成的阴影 |
| MGRS (Military Grid Reference System) | 军事格网参考系统 | 基于 UTM 的全球格网系统 |

---

## 二、翻译策略 / Translation Strategy

1. **保留不译**: 模型名称（Prithvi-EO-2.0, SatMAE, DOFA, DeCUR 等）、机构名称（NASA, IBM, ESA）、软件/平台名称（Hugging Face, TerraTorch, PyTorch Lightning, TorchGeo）、数据集名称（HLS, GEO-Bench, BioMassters, Sen4Map, PASTIS, L4S, Sen1Floods11, MTBS, FLUXNET, MERRA-2）、卫星名称（Sentinel-1/2, Landsat 8/9）。

2. **首次出现附英文**: 对于重要术语，首次出现时中文译名后附英文原文，如"掩码自编码器（Masked Autoencoder, MAE）"。

3. **公式保留原样**: 所有数学公式（Accuracy, mIoU, PE 等）保留原始 LaTeX 风格表示，不进行中文化。

4. **引用标记保留**: 所有 `[数字]` 引用标记保持原样，指向原文参考文献列表。

5. **表格处理**: 由于 PDF 提取的表格文本高度压缩且难以完美还原，本文件采用"保留原文关键数值 + 中文说明"的策略。完整精确数值请参考原始 PDF 或论文原文。

6. **图表就近放置**: 每个 Figure 的 Markdown 中通过 `![caption](assets/page_XX.png)` 引用了对应页面的渲染图，方便对照阅读。

---

## 三、不确定处标注 / Ambiguities

| 位置 | 原文 | 说明 |
|------|------|------|
| C007 | "-band-pass filter" | 确认 Scale-MAE 使用图像处理中的带通滤波进行多尺度分解，译法无误。 |
| C011 | "3D patch embeddings... kernel size (t, p, p)" | 时间维度 t 和空间维度 p 的大小在原文中未给出具体数值，属于模型实现细节。 |
| C012 | "convolutional embeddings" | 依据 Xu et al. [21] 的工作，指用轻量卷积层替代原始的线性 patch 投影层，译文采用"卷积嵌入"。 |
| C020 | "Linear classification head" | 指在冻结的骨干网络后接一层线性层进行分类，译文采用"线性分类头"。 |
| C024 | "leave-one-year-out cross-validation" | 标准的交叉验证策略，每年轮流作为测试集，其余年份作为训练集。 |

---

## 四、特殊说明 / Special Notes

- **Prithvi 含义**: Prithvi 在梵语中意为"地球"，是印度神话中的大地女神。论文脚注中明确说明。
- **TL 后缀含义**: 模型名称中 "-TL" 表示使用了 Temporal（时间）和 Location（位置）嵌入的版本。
- **HLS 数据特点**: HLS 是 NASA 将 Landsat 8/9 和 Sentinel-2A/2B 数据融合后的产品，实现了 30 米分辨率下约 2-3 天的重访周期。
- **论文核心创新点**: (1) 全球范围 420 万样本的时序预训练数据集；(2) 3D Patch 嵌入 + 时空嵌入 + 地理空间嵌入；(3) 600M 参数的大规模 ViT-H 骨干；(4) 在 GEO-Bench 和多样化下游任务上的全面验证。
