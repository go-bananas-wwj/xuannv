# Translation Notes & Glossary

## Terminology

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Dynamic Weight Factorization | 动态权重分解 | DOFA 核心机制 |
| Earth Observation (EO) | 地球观测 | 卫星遥感领域 |
| Foundation Model (FM) | 基础模型 | 大模型/预训练模型 |
| Neural Plasticity | 神经可塑性 | 生物学启发概念 |
| Hypernetwork | 超网络 | 元网络，生成其他网络权重 |
| Wavelength-conditioned | 波长条件化的 | 以波长为条件 |
| Dynamic Weight Generator | 动态权重生成器 | DOFA 核心组件 |
| Masked Image Modeling (MIM) | 掩码图像建模 | 自监督预训练方法 |
| Patch Embedding | Patch 嵌入 / 块嵌入 | ViT 中的图像分块嵌入 |
| Self-Attention | 自注意力 | Transformer 核心机制 |
| Transformer | Transformer | 保持原词 |
| Spectral Band | 光谱波段 | 遥感术语 |
| Ground Sample Distance (GSD) | 地面采样距离 | 空间分辨率指标 |
| Synthetic Aperture Radar (SAR) | 合成孔径雷达 | 雷达遥感 |
| Multispectral | 多光谱 | 少量波段（如 Sentinel-2） |
| Hyperspectral | 高光谱 | 大量连续波段（如 EnMAP） |
| Linear Probing | 线性探测 | 评估预训练表示的方法 |
| Fine-tuning | 微调 | 迁移学习技术 |
| Continual Pretraining | 持续预训练 | 连续多模态预训练 |
| Knowledge Distillation | 知识蒸馏 | 模型压缩/迁移技术 |
| Feature Distillation | 特征蒸馏 | 中间层特征对齐 |
| mIoU | 平均交并比 | 分割任务评估指标 |
| mAP50 | 平均精度均值 (IoU=0.5) | 检测任务评估指标 |
| OA (Overall Accuracy) | 总体准确率 | 分类任务评估指标 |
| SOTA | 最先进 (State of the Art) | 当前最佳性能 |
| DINOv2 | DINOv2 | Meta 自监督视觉模型 |
| ViT | Vision Transformer | 视觉 Transformer |
| Sentinel-1 / Sentinel-2 | Sentinel-1 / Sentinel-2 | ESA 卫星系列，保持原名 |
| NAIP | NAIP | 美国航拍影像计划 |
| EnMAP | EnMAP | 德国高光谱卫星 |

## Translation Choices

1. **DOFA** 全称 "Dynamic One-For-All" 译为 "动态万能模型" 或保留英文缩写。
2. **Neuroplasticity** 统一译为 "神经可塑性"，引用生物神经科学概念。
3. **Wavelength** 在上下文中指 "中心波长" (central wavelength)，是光谱波段的标识参数。
4. **Hypernetwork** 译为 "超网络"，指一种元网络结构，其输出作为另一网络的权重。
5. 公式和数学符号保持原样，仅翻译 surrounding text。
6. 数据集名称（如 m-bigearthnet, DIOR, RESISC-45）保持原名。
7. 模型名称（如 GFM, ScaleMAE, CROMA, SatMAE）保持原名。

## Uncertain Translations

- [?] "proxy-based distillation" 译为 "基于代理的蒸馏"
- [?] "weight space interpolation" 译为 "权重空间插值"
