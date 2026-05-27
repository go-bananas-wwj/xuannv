# Translation Notes / 翻译备注

## 术语表 / Glossary

| 英文术语 | 中文翻译 | 备注 |
|---------|---------|------|
| masked autoencoder (MAE) | 掩码自编码器 | 自监督学习方法，掩码部分输入并重建 |
| pixel-level | 像素级 | 以单个像素为处理单位 |
| earth observation | 地球观测 | 遥感领域标准术语 |
| remote sensing | 遥感 | 通过卫星/航空器观测地球表面 |
| timeseries / time series | 时间序列 | 按时间顺序排列的数据点 |
| pixel-timeseries | 像素时间序列 | 单个像素随时间变化的观测序列 |
| multi-sensor | 多传感器 | 融合来自多个传感器的数据 |
| self-supervised learning | 自监督学习 | 无需人工标注的训练范式 |
| transfer learning | 迁移学习 | 将预训练知识迁移到下游任务 |
| feature extractor | 特征提取器 | 提取数据高层表示的模型组件 |
| encoder-decoder | 编码器-解码器 | 先压缩再重建的模型架构 |
| embedding | 嵌入 / 表征向量 | 高维数据的低维稠密表示 |
| token | 标记 | Transformer 处理的基本单元 |
| channel group | 通道组 | 将相关波段分组为单个标记 |
| structured masking | 结构性掩码 | 按时间/通道/空间结构进行掩码 |
| fine-tuning | 微调 | 在预训练模型基础上适应特定任务 |
| linear probing | 线性探测 | 冻结主干仅训练线性分类头 |
| FLOPs | 浮点运算次数 | 模型计算复杂度指标 |
| receptive field | 感受野 | 模型能看到的输入空间范围 |
| Sentinel-1 / S1 | Sentinel-1（哨兵1号） | ESA SAR 卫星 |
| Sentinel-2 / S2 | Sentinel-2（哨兵2号） | ESA 多光谱光学卫星 |
| DEM | 数字高程模型 | Digital Elevation Model |
| RGB | 红绿蓝 | 可见光三波段 |
| MS / multispectral | 多光谱 | 包含多个光谱波段 |
| SAR | 合成孔径雷达 | Synthetic Aperture Radar |
| AUC ROC | ROC曲线下面积 | 分类性能指标 |
| RMSE | 均方根误差 | 回归性能指标 |
| F1 score | F1 分数 | 精确率和召回率的调和平均 |
| kNN@5 | 5-近邻 | k-Nearest Neighbors with k=5 |

## 翻译策略

1. **技术术语保留英文**：首次出现时附中文翻译，后续可保留英文（如 masked autoencoder）。
2. **模型名保留原文**：Presto, TIML, MOSAIKS, SatMAE, ScaleMAE 等。
3. **引用保留原文**：(Brown et al., 2022) 等引用格式不变。
4. **公式与数值不变**：如 $F_1$, 128, 10m/px 等。
5. **表格数据**：表头翻译，数据单元格保留原文。
6. **不确定处标注 [?]**：本文档中暂未发现需标注的不确定翻译。

## 待处理项 / Pending

- 部分表格数据段落（C017, C020, C026, C027, C035, C036, C041, C045, C066, C073, C078）以 "[表格数据保留原文]" 形式处理，因纯数据无需翻译。
- 参考文献列表（C050-C053 等）保留英文原文。

## 图表资产

共提取 21 张图片（fig-000.png ~ fig-020.png），来源于 pdfimages。
部分图片为论文内图表的组成部分，已按 Figure 1~8 进行映射。
