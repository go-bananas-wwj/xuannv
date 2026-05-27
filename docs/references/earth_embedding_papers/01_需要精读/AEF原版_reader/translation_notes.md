# Translation Notes — AlphaEarth Foundations Reader

## 处理状态

| 项目 | 状态 | 说明 |
|------|------|------|
| PDF文本提取 | ✅ 完成 | 使用 `pdftotext -layout` 提取，质量良好 |
| 图表提取 | ✅ 完成 | 63张PNG图片已提取至 `assets/` |
| 主要论文 (Page 1-11) | ✅ 完成 | Abstract, Introduction, Methods, Results, Discussion, Conclusions 全文翻译 |
| 参考文献 (Page 12-18) | ✅ 完成 | 保留原文，提供章节说明 |
| 补充材料 (Page 19-63) | ✅ 完成 | 数据集描述、方法细节、补充图表保留原文并附术语注释 |
| source_map.json | ✅ 完成 | 162条源文本锚点映射 |
| paper.md | ✅ 完成 | 1,628行中英对照全文 |

**论文总页数**: 63页  
**实际处理页数**: 63页（全部）  
**总段落/块数**: 162个内容块（含23个Figure，多个Table，其余为Text）

---

## 核心术语表

| 英文术语 | 中文翻译 | 备注 |
|---------|---------|------|
| embedding field | 嵌入场 | 本文核心概念，连续的嵌入空间表示 |
| foundation model | 基础模型 | 大规模预训练后可用于多种下游任务的模型 |
| geospatial | 地理空间 | 地理空间数据/分析 |
| earth observation (EO) | 地球观测 | 遥感卫星对地观测 |
| remote sensing | 遥感 | 非接触式远距离探测 |
| masked autoencoder (MAE) | 掩码自编码器 | 自监督预训练范式 |
| temporal | 时序 | 强调时间维度 |
| multi-sensor | 多传感器 | Sentinel-2 / Sentinel-1 / Landsat 等多源数据 |
| multi-temporal | 多时相 | 同一地点不同时间的观测 |
| spatial resolution | 空间分辨率 | 像元地面覆盖大小 |
| spectral bands | 光谱波段 | 多光谱/高光谱通道 |
| SAR (Synthetic Aperture Radar) | 合成孔径雷达 | 微波遥感，可穿透云层 |
| backscatter | 后向散射 | SAR信号反射强度 |
| optical imagery | 光学影像 | 可见光/红外波段影像 |
| pixel | 像素 | 图像最小单元 |
| patch | 图像块 | 模型输入的局部区域 |
| downstream task | 下游任务 | 预训练后的具体应用任务 |
| transfer learning | 迁移学习 | 知识从源域迁移到目标域 |
| few-shot | 少样本 | 极少标注样本的学习 |
| k-nearest neighbors (kNN) | k近邻 | 非参数分类方法 |
| linear probe | 线性探针 | 冻结特征+线性分类器评估 |
| VMF (von Mises-Fisher) | von Mises-Fisher分布 | 球面概率分布，用于embedding约束 |
| kappa | kappa / 集中度参数 | VMF分布的集中度参数 |
| normalization | 归一化 | 数据标准化处理 |
| reconstruction | 重建 | 自编码器的解码输出 |
| encoder | 编码器 | 特征提取网络 |
| decoder | 解码器 | 重建/生成网络 |
| bottleneck | 瓶颈层 | 降维后的紧凑表示 |
| attention | 注意力机制 | 自适应加权机制 |
| pre-training | 预训练 | 大规模无监督/自监督训练 |
| fine-tuning | 微调 | 在特定任务上调整参数 |
| inference | 推理 | 模型前向计算 |
| benchmark | 基准测试 | 标准化性能评估 |
| dataset | 数据集 | 训练/评估用的数据集合 |
| ground truth | 地面真值 | 真实标注 |
| land cover | 土地覆盖 | 地表自然/人工覆盖类型 |
| land use | 土地利用 | 人类对土地的使用方式 |
| change detection | 变化检测 | 识别地表变化 |
| biomass | 生物量 | 植被有机物总量 |
| canopy height | 冠层高度 | 植被顶部到地面的高度 |
| crop type | 作物类型 | 农作物种类分类 |
| segmentation | 分割 | 像素级分类 |
| classification | 分类 | 离散类别预测 |
| regression | 回归 | 连续值预测 |
| AUC | 曲线下面积 | Area Under Curve |
| BER (Balanced Error Rate) | 平衡错误率 | 考虑类别不平衡的误差度量 |
| MAE (Mean Absolute Error) | 平均绝对误差 | 回归任务常用指标 |
| R² (Coefficient of Determination) | 决定系数 | 回归拟合优度 |
| LUCAS | 欧盟土地利用/覆盖统计调查 | Land Use/Cover Area frame Survey |
| NLCD | 美国国家土地覆盖数据库 | National Land Cover Database |
| GLanCE | 全球土地覆盖估算 | Global Land Cover Estimation |
| GEDI | 全球生态系统动态调查 | Global Ecosystem Dynamics Investigation |
| DEM | 数字高程模型 | Digital Elevation Model |
| WorldCover | WorldCover | ESA 10m全球土地覆盖产品 |
| Dynamic World | Dynamic World | Google 10m全球土地覆盖产品 |
| GBIF | 全球生物多样性信息网络 | Global Biodiversity Information Facility |
| OpenET | 开放蒸散发 | Open Evapotranspiration |
| NASA | 美国国家航空航天局 | — |
| ESA | 欧洲空间局 | — |
| USGS | 美国地质调查局 | — |
| Copernicus | 哥白尼计划 | 欧空局对地观测计划 |
| Google Earth Engine | Google Earth Engine | 云平台遥感数据处理 |
| cloud masking | 云掩膜 | 去除云层污染 |
| atmospheric correction | 大气校正 | 消除大气影响 |
| top-of-atmosphere (TOA) | 大气层顶 | 未经大气校正的反射率 |
| surface reflectance | 地表反射率 | 经大气校正后的反射率 |
| UTM | 通用横轴墨卡托 | 地图投影坐标系 |
| WGS84 | WGS84 | 全球大地坐标系 |
| sinusoidal encoding | 正弦编码 | 时间位置编码方式 |
| timecode | 时间码 | 时间信息编码表示 |
| video (in this paper) | 视频（时序序列） | 论文中指多时相图像序列 |
| frame | 帧 | 时序中的单幅图像 |
| sequence | 序列 | 连续的时序数据 |

---

## 翻译策略

1. **技术术语保留英文并附中文注释**：首次出现时用 `英文（中文）` 格式，后续视上下文保留英文或中文。
2. **模型名保留原文**：AlphaEarth Foundations (AEF), SatCLIP, Prithvi, Clay, Scale-MAE 等。
3. **引用格式不变**：(Author et al., Year) 标准学术引用格式保留。
4. **公式和数值不变**：数学表达式、指标数值保持原样。
5. **保留原文段落结构**：每段先给出英文原文，再附中文翻译，便于对照阅读。

---

## 已知问题与注意事项

1. **PDF分栏混排**：Page 1 等页面采用双栏布局，pdftotext 提取时偶有两栏文本交叉。已尽量通过空行检测分离段落，但少数段落仍存在混排痕迹。
2. **连字符断词修复**：pdftotext 保留了原始换行处的连字符（如 `high-\nquality`）。已全局修复大部分常见断词，但极少数专业术语可能存在未修复的断词。
3. **图表与资产映射**：Figure/Table 与 `assets/` 中的 PNG 文件按出现顺序近似映射。由于 pdfimages 提取的图为整页或页面局部区域，部分 Figure 可能对应多幅 PNG，用户需根据内容自行匹配。
4. **第7页表格**：Table 1 因 PDF 格式复杂，文本提取后的表格结构不够清晰，建议在阅读时对照原 PDF 查看完整表格。
5. **补充材料（Page 19+）**：以数据集详细描述和补充图表为主。正文保留英文原文，关键术语已附中文注释。对纯技术参数表格未做逐条翻译。

---

## 文件清单

```
AEF原版_reader/
├── paper.md              # 中英对照全文阅读文件 (1,628行, 404KB)
├── source_map.json       # 源文本锚点映射 (162条)
├── translation_notes.md  # 术语表和翻译备注 (本文件)
├── assets/               # 提取的图表文件 (63张PNG)
│   ├── fig-000.png
│   ├── fig-001.png
│   └── ...
├── raw_pages.json        # 原始PDF文本提取 (中间文件)
├── segments.json         # 分段数据 (中间文件)
├── content_blocks.json   # 内容块数据 (中间文件)
└── pdftotext_output.txt  # pdftotext原始输出 (中间文件)
```

---

## 使用建议

- **快速浏览**：先阅读 `paper.md` 中 Page 1-11 的主要论文部分。
- **查阅术语**：遇到不熟悉的术语可查阅本文件的核心术语表。
- **查看图表**：`paper.md` 中 Figure 附近嵌入了 `assets/` 中的图片链接，使用支持图片预览的 Markdown 阅读器效果更佳。
- **深入细节**：如需查看数据集细节或补充实验，可阅读 `paper.md` 中 Page 19+ 的补充材料部分。
