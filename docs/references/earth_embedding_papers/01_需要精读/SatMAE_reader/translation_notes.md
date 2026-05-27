# 术语表与翻译备注 (Translation Notes)

## 一、核心术语表

| 英文术语 | 中文翻译 | 备注 |
|----------|----------|------|
| Masked Autoencoder (MAE) | 掩码自编码器 | 核心方法，保留英文缩写 |
| self-supervised learning (SSL) | 自监督学习 | 全文统一 |
| pre-training | 预训练 | 与 fine-tuning（微调）相对 |
| fine-tuning | 微调 | 下游任务适配 |
| temporal | 时序的/时间的 | 根据上下文译为“时序” |
| multi-spectral | 多光谱的 | 遥感领域标准术语 |
| satellite imagery | 卫星影像 | 也可用“遥感影像” |
| remote sensing | 遥感 | 学科领域名 |
| positional encoding | 位置编码 | Transformer 标准术语 |
| patch | 图像块/块 | 视觉 Transformer 中的最小单元 |
| masking | 掩码 | 动词/名词统一 |
| consistent masking | 一致掩码 | 所有时序帧掩码同一空间位置 |
| independent masking | 独立掩码 | 各时序帧独立随机掩码 |
| token | token / 令牌 | 视上下文保留英文或译中文 |
| transformer | Transformer | 保留英文，不翻译 |
| Vision Transformer (ViT) | 视觉 Transformer | 保留缩写 |
| backbone | 主干网络 | 模型骨干结构 |
| embedding | 嵌入/表征 | 根据语境选择 |
| representation | 表征/表示 | 与 embedding 近义 |
| pretext task | 代理任务 | 自监督中的前置任务 |
| downstream task | 下游任务 | 预训练后的应用任务 |
| land cover classification | 土地覆盖分类 | 遥感标准任务名 |
| semantic segmentation | 语义分割 | 像素级分类任务 |
| Sentinel-2 | Sentinel-2 | ESA 卫星名，保留英文 |
| fMoW | fMoW | Functional Map of the World 缩写 |
| BigEarthNet | BigEarthNet | 数据集名，保留 |
| EuroSAT | EuroSAT | 数据集名，保留 |
| SpaceNet | SpaceNet | 数据集名，保留 |
| NAIP | NAIP | National Agricultural Imagery Program |
| spectral band | 光谱波段 | 简称“波段” |
| RGB+NIR | RGB+近红外 | 可见光+近红外波段 |
| Red Edge | 红边 | 植被敏感波段 |
| SWIR | 短波红外 | Short-Wave Infrared |
| channel | 通道 | 与 band 在图像语境下近义 |
| crop (v.) | 裁剪 | 数据预处理操作 |
| data augmentation | 数据增强 | 标准术语 |
| test-time augmentation | 测试时增强 | 推理阶段的数据增强 |
| ablation study | 消融研究 | 验证各组件贡献 |
| reconstruction | 重建/重构 | MAE 的核心目标 |
| top-1 accuracy | Top 1 准确率 | 分类指标 |
| top-5 accuracy | Top 5 准确率 | 分类指标 |
| mIoU | 平均交并比 | mean Intersection over Union |
| mAP | 平均精度均值 | mean Average Precision |

## 二、翻译策略

1. **保留英文的术语**：所有模型名（MAE、ViT、SatMAE）、数据集名（fMoW、Sentinel-2）、会议名（NeurIPS）、指标缩写（mIoU、mAP）均保留英文原文。
2. **公式与数值**：所有数学公式、百分比、实验数值均保留原样，不做翻译或转换。
3. **引用格式**：方括号引用 [1], [34, 35] 等保留原样，对应参考文献列表。
4. **不确定处**：如原文存在 LaTeX 残留或语义不完整，标注为 `[?]` 或保留原文。
5. **段落对应**：每个原文段落（S###）均有唯一标识，中英严格对照，不合并、不拆分。

## 三、图表映射

| 图/表 ID | 页码 | 对应资源文件 |
|----------|------|--------------|
| F001 | p2 | fig1_architecture_*.png (Figure 1: 模型架构) |
| F002 | p4 | fig2_encoding.png (Figure 2: 编码方式) |
| F003 | p5 | fig3_temporal_masking.png, fig3_spectral_masking.png (Figure 3: 掩码策略) |
| F004 | p9 | fig6_results.png (Figure 6: 实验结果可视化) |
| F005 | p22 | fig8_temporal_recon.png (Figure 8: 时序重建质量) |
| F006-F009 | p23-p24 | fig9_*, fig10_* (Figure 9-10: 多光谱重建质量) |
| T006 | p16 | table_sentinel_bands.png (Sentinel-2 波段统计) |

## 四、已知问题

- 部分段落的 LaTeX 数学符号在 PDF 提取时格式有所损失，翻译时已尽量还原语义。
- 表格内容由于 PDF 文本提取的限制，部分以纯文本形式嵌入段落中，未单独提取为图像。
- 参考文献列表（S051-S053）因篇幅过长，未逐条翻译，保留英文原文。

## 五、版本信息

- 源文件：SatMAE_时序多光谱MAE_NeurIPS2022.pdf
- 页数：24 页
- 生成日期：2026-05-27
