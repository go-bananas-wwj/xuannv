# Translation Notes

## 技术术语对照表

| 英文术语 | 中文翻译 | 备注 |
|----------|----------|------|
| Geospatial Foundation Model (GFM) | 地理空间基础模型 | 保留英文缩写 GFM |
| AlphaEarth Foundation (AEF) | AlphaEarth Foundation (AEF) | 保留原名及缩写 |
| Earth Observation (EO) | 地球观测 (EO) | 保留缩写 |
| Remote Sensing (RS) | 遥感 (RS) | 保留缩写 |
| embedding | 嵌入 / embedding | 技术术语，保留英文或中文 |
| featurization | 特征化 | 指将数据转换为特征表示的过程 |
| yield prediction | 产量预测 | 农业领域核心任务 |
| tillage mapping | 耕作制图 | 区分 high-intensity / low-intensity tillage |
| cover crop mapping | 覆盖作物制图 | conservation agricultural practice |
| cross validation (CV) | 交叉验证 (CV) | 如 State-Year CV, County-Year CV |
| spatial transferability | 空间迁移能力 | 模型跨地理区域泛化能力 |
| temporal sensitivity | 时间敏感性 | 论文核心发现：AEF 时间敏感性有限 |
| harmonic regression | 谐波回归 | 时间序列拟合方法 |
| vegetation index (VI) | 植被指数 (VI) | NDVI, GCVI 等 |
| Random Forest (RF) | 随机森林 (RF) | 机器学习模型 |
| XGBoosting (XGB) | XGBoosting (XGB) | 梯度提升树模型 |
| spatial autocorrelation | 空间自相关 | 地学统计概念 |
| scale-transfer | 尺度迁移 | 跨空间尺度（县→field）的模型迁移 |
| space-transfer | 空间迁移 | 跨地理生态区的模型迁移 |
| ecoregion | 生态区 | 如 ETF (Eastern Temperate Forests), GP (Great Plains) |
| Google Earth Engine (GEE) | Google Earth Engine (GEE) | 云平台，保留原名 |
| field-level | field 尺度 | 指单个农田/地块级别，保留 field |
| county-level | 县级 | 行政区划级别 |
| ground truth | ground truth | 通常保留英文，或译为"真值" |
| data harmonization | 数据协调 / 数据 harmonization | 多源数据融合对齐 |

## 不确定或保留的翻译

1. **"featurization"**：译为"特征化"，但该词在遥感/ML领域无统一中文译法，部分文献保留英文。
2. **"field-level"**：译为"field 尺度"，field 在此指农业中的"田块"，非一般意义上的"领域"。
3. **"EO tasks"**：译为"EO 任务"或"地球观测任务"，根据上下文选择。
4. **"space-time encoder"**：译为"空间-时间编码器"，AEF 核心组件。
5. **"teacher-student framework"**：译为"教师-学生框架"，即知识蒸馏框架。
6. **公式中的符号**：保留原文数学符号，如 GDD_j, PPT_d, T_max,d 等。
7. **参考文献**：References 部分保留原文，未逐条翻译，仅添加 [参考文献延续 - 保留原文] 标注。

## 跳过的内容

- 参考文献（Reference）部分共约5页，为保持学术准确性，保留英文原文，未逐条翻译。
- 部分纯数据表格（如表3-表7的数值内容）在翻译中以结构化文本呈现，保留了所有数值和指标名称。

## 布局提取中的问题

1. **公式提取**：pdfplumber 对 PDF 中的数学公式提取效果不佳，部分公式出现符号错位或换行断裂。在 paper.md 中已根据上下文手动修正公式呈现（如公式1-14）。
2. **表格布局**：原始 PDF 中的表格在文本提取后变为线性文本，行列关系需手动推断和重构。翻译中以 Markdown 表格或结构化列表形式重新组织。
3. **图表引用**：部分正文段落中内嵌了图表引用（如 "Figure 1 (a))"），在块分割时可能产生不自然的断句。已尽量在翻译中保持语义连贯。
4. **图片提取**：使用 PyMuPDF 共提取 12 张图片，对应论文中的 Figure 1–9 及 Figure S1。部分图片分辨率受原始 PDF 限制。
5. **页眉/页脚**：每页顶部的 "Under Review" 页眉在提取时已自动过滤，未纳入正文内容。

## 阅读建议

- **重点关注**：第1页 Abstract、第3页 Introduction 结尾、第18–20页 Discussion（尤其是 5.2 Limitations）是本文核心。
- **核心结论**：AEF 在本地实验中有竞争力，但存在 **limited spatial transferability（空间迁移能力有限）**、**low interpretability（可解释性低）** 和 **limited time sensitivity（时间敏感性有限）** 三大局限。
- **公式阅读**：方法部分（第7–8页）包含较多公式，建议结合原文 PDF 对照阅读以确保符号理解准确。
