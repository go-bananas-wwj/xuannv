# Translation Notes | 翻译备注

> **Paper**: TerraMind: Large-Scale Generative Multimodality for Earth Observation  
> **Reader Date**: 2026-05-27

---

## 1. Terminology Glossary | 术语表

| English Term | Chinese Translation | Notes |
|-------------|---------------------|-------|
| any-to-any | 任意到任意 | 指模型可在任意输入模态和任意输出模态之间转换 |
| Earth observation (EO) | 地球观测 | 遥感领域标准译法 |
| foundation model | 基础模型 | 大模型领域通用术语 |
| generative | 生成式 | 区别于判别式模型 |
| multimodal / multimodality | 多模态 | 同时处理多种数据类型 |
| token-level | token 级 | NLP 领域术语，保留英文 token |
| pixel-level | 像素级 | 图像处理领域术语 |
| dual-scale | 双尺度 | 本文核心概念，指 pixel + token 两个尺度 |
| Thinking-in-Modalities (TiM) | 模态思维 | 本文提出的新概念，类比 chain-of-thought |
| pretraining | 预训练 | 深度学习标准术语 |
| finetuning / fine-tuning | 微调 | 两种拼写统一为"微调" |
| zero-shot | 零样本 | 无需任务特定训练即可推理 |
| few-shot | 少样本 | 仅需少量标注样本 |
| embedding space | 嵌入空间 | 表征学习中的潜在空间 |
| latent space | 潜在空间 | 与 embedding space 同义使用 |
| cross-modal | 跨模态 | 在不同模态之间建立联系 |
| correlation learning | 相关性学习 | 学习模态间关联 |
| tokenizer | tokenizer / 分词器 | 保留英文或中文均可，本文混合使用 |
| Finite Scalar Quantization (FSQ) | 有限标量量化 | 量化技术名称 |
| Vector Quantization (VQ) | 矢量量化 | 经典量化方法 |
| diffusion model / decoding | 扩散模型 / 扩散解码 | 生成模型类别 |
| encoder-decoder | 编码器-解码器 | 经典架构 |
| transformer backbone | Transformer 主干 | 模型基础架构 |
| modality-agnostic | 模态无关的 | 不依赖特定模态类型 |
| patch embedding | patch 嵌入 | ViT 中的图像块嵌入 |
| autoregressive | 自回归 | 序列生成方式 |
| cross-entropy loss | 交叉熵损失 | 分类任务标准损失 |
| masked target tokens | 被掩码的目标 token | MAE/BERT 风格预训练 |
| synthetic modalities | 合成模态 | 人工生成的模态数据 |
| recursive augmentation | 递归增强 | TiM 中的核心机制 |
| chain-of-thought | 思维链 | NLP 领域概念，TiM 受其启发 |
| mIoU | 平均交并比 | mean Intersection over Union，分割指标 |
| PSNR | 峰值信噪比 | Peak Signal-to-Noise Ratio，图像质量指标 |
| SSIM | 结构相似性指数 | Structural Similarity Index，图像质量指标 |
| F1 score | F1 分数 | 精确率和召回率的调和平均 |
| t-SNE | t-分布随机邻域嵌入 | 降维可视化方法 |
| cosine similarity | 余弦相似度 | 向量相似性度量 |
| land cover | 土地覆盖 | 遥感标准术语 |
| land use / land cover (LULC) | 土地利用/土地覆盖 | 常用缩写 LULC |
| digital elevation model (DEM) | 数字高程模型 | 地形数据标准格式 |
| normalized difference vegetation index (NDVI) | 归一化植被指数 | 植被监测指标 |
| synthetic aperture radar (SAR) | 合成孔径雷达 | Sentinel-1 使用的雷达技术 |
| Ground Range Detected (GRD) | 地距检测 | Sentinel-1 产品级别 |
| Radiometric Terrain Corrected (RTC) | 辐射地形校正 | Sentinel-1 产品级别 |
| surface reflectance | 地表反射率 | L2A 产品提供 |
| top-of-atmosphere reflectance | 大气层顶反射率 | L1C 产品提供 |
| speckle filtering | 斑点滤波 | SAR 数据预处理步骤 |
| cloud masking | 云掩膜 | 光学影像预处理步骤 |
| spatial leakage | 空间泄漏 | 地理空间数据划分中的常见问题 |
| spatial cross-validation | 空间交叉验证 | 防止空间泄漏的验证策略 |
| geospatial heterogeneity | 地理空间异质性 | EO 数据的空间分布不均 |
| self-supervised learning (SSL) | 自监督学习 | 无监督预训练范式 |
| geospatial foundation model (GFM) | 地理空间基础模型 | EO 领域的 foundation model |
| PANGAEA benchmark | PANGAEA 基准测试 | 地理空间 FM 评估基准 |
| BigEarthNet | BigEarthNet 数据集 | 多标签土地覆盖数据集 |
| PASTIS | PASTIS 数据集 | 农业语义分割数据集 |
| MADOS | MADOS 数据集 | 海洋语义分割数据集 |
| OSCD | OSCD 数据集 | 卫星变化检测数据集 |
| CARPK | CARPK 数据集 | 停车场车辆计数数据集 |
| AGB | AGB 数据集 | 地上生物量数据集 |
| ESA WorldCover | 欧空局世界覆盖 | 全球土地覆盖产品 |
| OpenStreetMap | 开放街图 | 众包地理数据 |
| Copernicus | 哥白尼计划 | 欧洲对地观测计划 |
| Sentinel-1 / Sentinel-2 | 哨兵-1 / 哨兵-2 | 欧空局卫星系列 |
| WordPiece | WordPiece | Google 开发的子词 tokenizer |
| bfloat16 | bfloat16 | 16位浮点格式 |
| AdamW | AdamW | 优化器算法 |
| gradient clipping | 梯度裁剪 | 训练稳定技术 |
| mixed-precision training | 混合精度训练 | 加速训练技术 |
| U-Net | U-Net | 医学/遥感图像分割网络 |
| codebook | 码本 | VQ-VAE 中的离散表征集合 |
| codebook utilization | 码本利用率 | 实际使用的码本条目比例 |
| reconstruction quality | 重建质量 | 自编码器输出与输入的相似度 |
| visual plausibility | 视觉合理性 | 生成结果的直观质量 |
| structural alignment | 结构对齐 | 生成结果与真值的结构一致性 |
| perceptual degradation | 感知退化 | 人眼可察觉的质量下降 |

---

## 2. Translation Decisions | 翻译决策

### 2.1 保留英文的术语

以下术语在中文翻译中保留英文原文（首次出现时附中文注释）：

- **token / tokenizer**: 深度学习和 NLP 领域的核心概念，保留英文更精确
- **ViT / CNN / Transformer**: 架构缩写，保留英文
- **Sentinel-2 L1C / L2A / GRD / RTC**: 卫星产品级别，保留英文代码
- **FSQ / VQ**: 量化方法缩写
- **TiM**: 本文提出的新概念缩写，保留英文
- **mIoU / PSNR / SSIM / F1**: 指标缩写

### 2.2 处理不确定性的地方

以下翻译存在不确定性，已在译文中以自然方式处理：

1. **"thinking in modalities"**: 直译为"模态思维"，类比"思维链"(chain-of-thought)。原文使用了引号，译文保留引号。
2. **"dual-scale early fusion"**: 译为"双尺度早期融合"，其中 early fusion 是相对于 late fusion 而言的。
3. **"image-like modalities" vs "sequence-like modalities"**: 分别译为"图像类模态"和"序列类模态"，这是作者对模态的二分法。
4. **"modality-agnostic"**: 译为"模态无关的"，也可译为"模态不可知的"，选择前者更通顺。
5. **"spatial leakage"**: 译为"空间泄漏"，地理空间 ML 中的标准概念。

### 2.3 格式和风格

- 中英对照格式：**英文原文段落在前，中文翻译在后**，以 `[XXX-zh]` 锚点标记
- 图表说明：保留 Figure/Table 编号，中文翻译紧随其后
- 引用标记：保留原文的 `[N]` 引用格式，未展开为完整引用
- 公式：本文中公式较少，若有则以 LaTeX 格式保留
- 数值和单位：保留原始格式（如 500 billion, 10m, 256×256）

---

## 3. Known Issues | 已知问题

1. **PDF 文本提取的两栏交错**: 原论文为双栏排版，`pdftotext -layout` 输出存在左右栏文本交错现象。已通过列分离脚本处理，但个别段落仍可能存在轻微的句子顺序问题。核心章节已人工校对。

2. **图注与正文的对应**: 部分图注（Figure caption）在 PDF 中位于页面底部，与正文中引用的位置有距离。在 paper.md 中已将图注放置于首次提及的段落附近。

3. **附录章节概述性翻译**: 第 7-11 节（附录）采取了概述性翻译策略，保留了核心信息但压缩了细节。如需完整翻译，可基于 `pdftotext_output.txt` 进一步展开。

4. **参考文献简化**: 参考文献列表仅保留了主要引用，完整列表请参见原文。

---

## 4. Source Text Processing Log | 源文本处理日志

| Step | Tool | Output | Notes |
|------|------|--------|-------|
| Text extraction | `pdftotext -layout` | `pdftotext_output.txt` | 两栏布局，存在列交错 |
| Column separation | Custom Python script | `cleaned_text_v2.txt` | 基于大空格分割左右栏 |
| Section splitting | Regex + heuristics | `sections_final.json` | 识别主要章节边界 |
| Paragraph splitting | Custom Python script | `paragraphs.json` | 按段落和图表拆分 |
| Translation | Manual (AI-assisted) | `paper.md` | 逐段中英对照翻译 |
| Asset extraction | `pdfimages -j` | `assets/fig-*.jpg` | 共 225 个图像文件 |

---

*Generated by Paper Reader Agent | 2026-05-27*
