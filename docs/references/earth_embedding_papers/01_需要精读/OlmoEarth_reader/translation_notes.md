# Translation Notes | 翻译备注

## Terminology | 术语表

| English | 中文 | Notes |
|---------|------|-------|
| foundation model | 基础模型 | 通用术语，保留英文亦可 |
| Earth observation | 地球观测 | 遥感领域标准译法 |
| multimodal | 多模态 | 保留英文亦可 |
| spatio-temporal | 时空的 | 形容词形式 |
| masking | 掩码/掩蔽 | 本文统一使用"掩码" |
| bandset | 波段组 | 作者自定义术语，指模态的子分区 |
| latent space | 潜在空间 | 深度学习中标准译法 |
| embedding | 嵌入/嵌入向量 | 保留英文亦可 |
| contrastive loss | 对比损失 | 自监督学习标准术语 |
| patch discrimination | 块判别 | 本文特定损失函数 |
| instance contrastive loss | 实例对比损失 | SimCLR 风格 |
| self-supervised learning (SSL) | 自监督学习 | 标准译法 |
| linear probing (LP) | 线性探测 | 迁移学习评估方法 |
| kNN | k近邻 | 保留英文缩写 |
| fine-tuning | 微调 | 标准译法 |
| encoder-decoder | 编码器-解码器 | 标准架构译法 |
| Vision Transformer (ViT) | 视觉Transformer | 保留英文缩写 |
| FlexiViT | FlexiViT | 专有方法名，保留英文 |
| Latent MIM Lite | Latent MIM Lite | 本文提出的方法，保留英文 |
| reconstruction loss | 重建损失 | 标准译法 |
| representation collapse | 表征坍缩 | 自监督学习中常见现象 |
| Sentinel-1 / Sentinel-2 / Landsat | Sentinel-1 / Sentinel-2 / Landsat | 卫星名称保留英文 |
| OpenStreetMap | OpenStreetMap | 保留英文 |
| WorldCover / WorldCereal / SRTM | WorldCover / WorldCereal / SRTM | 数据集名称保留英文 |
| m-bigearthnet / m-so2sat / m-eurosat | m-bigearthnet 等 | 基准测试名称保留英文 |
| PASTIS / MADOS / Sen1Floods11 | PASTIS / MADOS / Sen1Floods11 | 保留英文 |
| BreizhCrops / CropHarvest | BreizhCrops / CropHarvest | 保留英文 |
| non-profit / NGO | 非营利组织 / 非政府组织 | 根据上下文选择 |
| platform | 平台 | OlmoEarth Platform 译为 OlmoEarth 平台 |

## Translation Principles | 翻译原则

1. **技术术语保留英文**：首次出现时附中文，后续可保留英文，确保学术准确性。
2. **公式、数值、引用不变**：所有 `[N]` 引用、`1 × 10⁻⁴` 等科学计数法、百分比、模型名称均保留原样。
3. **段落结构对齐**：每个源文段落对应一个中文段落，不压缩为 bullet points。
4. **不确定处标注 [?]**：本文件暂无标注，所有翻译经上下文确认。
5. **图表就近放置**：Figure/Table 的 caption 紧随图表引用路径插入。

## Known Issues | 已知问题

- 部分 Figure/Table 的提取图像中混有图表文本，已尽量过滤。
- 参考文献列表极长，为控制文件体积，部分引用标题翻译后保留了英文原文结构。
- Page 20 的 Figure 5 caption 与附录 E 的文本在原始 PDF 中位置接近，可能存在轻微交叉。

## Source Mapping | 源映射规则

- `S001+`：章节标题 (Section headers)
- `C001+`：正文段落 (Content paragraphs)
- `F001+`：图表题注 (Figure captions)
- `T001+`：表格题注 (Table captions)
