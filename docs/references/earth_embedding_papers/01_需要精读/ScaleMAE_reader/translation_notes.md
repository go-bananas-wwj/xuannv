# Translation Notes | 翻译备注

> 生成日期: 2026-05-27  
> 论文: Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning (ICCV 2023)

---

## Terminology Glossary | 术语表

| 英文术语 | 中文译法 | 说明 |
|----------|----------|------|
| Scale-MAE | Scale-MAE | 论文提出的方法名，保留原名 |
| Masked Autoencoder (MAE) | 掩码自编码器 | 自监督学习方法，由He et al. 2021提出 |
| scale-aware | 尺度感知的 | 核心概念，指模型显式利用尺度信息 |
| multiscale | 多尺度的 | 涉及多个空间尺度 |
| geospatial | 地理空间的 | 与地球表面空间数据相关 |
| Ground Sample Distance (GSD) | 地面采样距离 | 遥感核心概念，像素中心间的地面物理距离 |
| positional encoding | 位置编码 | Transformer架构中的位置信息嵌入 |
| GSD Positional Encoding (GSDPE) | GSD位置编码 | 本文核心创新，基于GSD缩放的位置编码 |
| Laplacian pyramid | Laplacian金字塔 | 图像多尺度表示方法，由Burt & Adelson 1983提出 |
| Laplacian Block (LB) | Laplacian块 | Scale-MAE解码器中的核心组件 |
| bandpass filter | 带通滤波器 | 保留特定频率范围、滤除其他频率的滤波器 |
| low/high frequency | 低频/高频 | 低频=平滑/平均信息，高频=边缘/细节信息 |
| ViT (Vision Transformer) | Vision Transformer | 视觉Transformer，由Dosovitskiy et al. 2020提出 |
| patchify | 分块/patch化 | 将图像切分为固定大小的patch序列 |
| mask token | 掩码令牌 | MAE中代表被掩码patch的可学习向量 |
| remote sensing | 遥感 | 通过卫星/飞机等传感器获取地球表面信息 |
| fine-tune / finetuning | 微调 | 在预训练模型基础上针对特定任务进行训练 |
| linear probing | 线性探测 | 冻结主干网络，仅训练线性分类头 |
| kNN classification | k近邻分类 | 非参数分类方法，用于评估表示质量 |
| mIoU (mean Intersection over Union) | 平均交并比 | 语义分割评估指标 |
| transfer learning | 迁移学习 | 将预训练知识迁移到下游任务 |
| self-supervised learning | 自监督学习 | 无需人工标注的监督信号学习方式 |
| representation learning | 表示学习 | 从数据中学习有意义特征的过程 |
| super-resolution | 超分辨率 | 从低分辨率图像恢复高分辨率图像 |
| deconvolution / transpose convolution | 反卷积/转置卷积 | 上采样操作 |
| depth-wise convolution | 深度可分离卷积 | 逐通道卷积，减少参数量 |
| orthorectified | 正射纠正的 | 消除地形和传感器几何畸变的图像处理 |
| pansharpened | 全色锐化的 | 融合全色高分辨率图像与多光谱图像 |
| electro-optical (EO) | 电光的 | 利用可见光/近红外等波段成像的传感器 |
| synthetic aperture radar (SAR) | 合成孔径雷达 | 主动微波遥感成像系统 |
| Functional Map of the World (FMoW) | 全球功能图 | 大规模遥感图像分类数据集 |
| SpaceNet | SpaceNet | 建筑物分割遥感数据集系列 |

---

## Translation Conventions | 翻译规范

1. **技术术语保留英文**: 方法名（Scale-MAE, SatMAE, ConvMAE）、架构名（ViT, UperNet, PSANet）、指标名（mIoU, GSD）保留英文原词。
2. **公式保留原样**: 所有数学公式和方程编号均保留原文。
3. **引用标注保留**: 文中引用如 [13], [26] 等保留原样，指代原文参考文献。
4. **段落结构对齐**: 中英段落一一对应，不压缩为bullet points。
5. **不确定处标记**: 本翻译中无标记为 `[?]` 的不确定处，所有术语均经过交叉验证。

---

## Key Translation Decisions | 关键翻译决策

| 原文表达 | 译文 | 理由 |
|----------|------|------|
| "the area of the Earth covered by the image determines the scale of the ViT positional encoding, not the image resolution" | "图像覆盖的地球区域决定了ViT位置编码的尺度，而非图像分辨率" | 准确传达GSDPE的核心思想：尺度由地面覆盖面积决定 |
| "bandpass filter" | "带通滤波器" | 信号处理标准术语 |
| "tasking the network with reconstructing both low/high frequency images" | "让网络同时重建低频和高频图像" | "tasking"在此语境下指使网络承担的任务目标 |
| "scale-inclusive positional encodings" | "包含尺度的位置编码" | 强调位置编码中融入了尺度信息 |
| "progressive multi-frequency feature extraction" | "渐进式多频特征提取" | "progressive"指Laplacian金字塔的逐层渐进结构 |
| "learned mask token" | "学习到的mask token" | MAE中的可学习参数，保留token一词 |

---

## Notes on Figures and Tables | 图表备注

- **Figure 2** 中的 "LB" 为 Laplacian Block 的缩写，已在图注中注明。
- **Table 5** 中地名采用中文通用译法：Rio(里约), Shanghai(上海), Vegas(拉斯维加斯), Paris(巴黎), Khartoum(喀土穆)。
- **Figure 5** 中的 "Relative GSD" 指相对于数据集原生GSD的比例尺度。
- **Table 9** 中的 "Combined" 指将低/高分辨率重建结果合并为单一损失，而非独立优化。

---

## Source Map Legend | 源映射图例

- **S###**: Source paragraph — 源文本段落
- **C###**: Chapter/section heading — 章节标题
- **F###**: Figure caption — 图表标题
- **T###**: Table caption — 表格标题

源映射文件: `source_map.json`
