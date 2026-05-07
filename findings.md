# 研究发现与素材库

## AEF论文关键发现（已精读正文+补充材料S6/S7/S8）

### 技术架构
- STP编码器: Space(1/16 L) + Time(1/8 L) + Precision(1/2 L)
- 参数量: ~480M (选用小版本，推理效率优先)
- 嵌入维度: 64维 → int8 = 64 bytes/embedding
- 空间分辨率: 10米
- 防止坍缩: Batch Uniformity Objective (S63球面分布，与batch-rotated embedding的dot product最小化)
- 时序支持: valid period与support period分离，支持interpolation/extrapolation
- 训练数据: 30亿观测，9个数据源 + 1个文本源，覆盖1.1%陆地
- 模型版本: 论文用v2.0，生产版本v2.1

### 评估方法论（补充材料S6/S7详细描述）
- **15个评估集，11个数据集**
- **样本策略**: 1-shot / 10-shot / max-trial（max-trial为数百样本级别）
- **迁移方法**: kNN (k=1, k=3) + Linear probe
- **分类指标**: Balanced Accuracy (BA)，Balanced Error Rate kappa (BERκ)
- **回归指标**: R², MAE, MAE⁻¹
- **置信区间**: 1σ (68.27%)，通过bootstrapping + k-folds计算
- **空间采样**: 最小1.28km间距，保证样本独立性
- **公平性**: 所有基线使用相同输入，超参数在评估集上调优

### 变化检测评估结果
**直接监督（训练分类器）**:
- LCMAP land cover: BA = 78.4%±1.11 (linear)
- LCMAP land use: BA = 79.3%±1.67 (kNN k=3)

**无监督阈值化（embedding距离+阈值）**:
- LCMAP land cover: BA = 71.3%±1.14
- LCMAP land use: BA = 71.4%±2.08

**对比基线**:
- MOSAIKS: 72.0% BA (kNN k=3)
- Composite: 71.5% BA (kNN k=3)
- ViT: 67.0%-72.9% BA
- Prithvi & MOSAIKS linear: 接近随机基线
- **SatCLIP/XY/XYZ被排除**（无时间处理能力）

### AEF关键技术边界（玄女底座的差异化突破口）
1. **时序敏感性非独立优化目标**: batch uniformity解决的是**空间分布均匀性**，不是**时间变化区分度**。变化检测是"事后对比两个period的embedding"，训练时并未强制模型学习时间变化。
2. **无监督变化检测精度有限**: BA仅71%，说明embedding本身对时间变化的区分能力不足，需要依赖监督信号补足。
3. **未提供交互式探索工具**: 论文未涉及相似性检索、交互式可视化等下游应用工具。

### AEF的"未解决问题"=我们的机会
> "While the mean gain was in AEF's favor for both settings, we consider the extreme degree of variability indicative that adequate general 1-shot or 10-shot performance remains an unsolved research frontier." —— 补充材料S6.1

这意味着即使是AEF，在极端低样本场景下表现也不稳定，这是一个明确的改进方向。

---

## EarthEmbeddingExplorer项目分析

### 定位
跨模态卫星图像检索工具，支持**文本搜图、图像搜图、位置搜图**。

### 核心技术
- **数据集**: MajorTOM Core-S2L2A（ESA发布），全球Sentinel-2 L2A，10m分辨率
- **采样**: 从224万张中均匀采样约25万张（1%），384×384像素
- **检索模型**: SigLIP（图文）, FarSLIP（遥感图文）, SatCLIP（位置图像）, DINOv2（纯视觉）, Clay, OlmoEarth
- **存储**: Parquet分片 + HTTP Range请求，23TB原始数据按需拉取
- **部署**: ModelScope Studio xGPU环境

### 与玄女底座的结合点
- EarthEmbeddingExplorer目前**未集成玄女底座的embedding**，但架构完全兼容
- 汇报叙事: "基于嵌入向量的检索，行业已有探索（如EarthEmbeddingExplorer支持6种模型），但玄女底座的64维向量带有时序信息，可以做'找相似地点+看时间变化'——这是现有工具做不到的"
- 视频素材: `/workspace/EarthEmbddingExplorer-演示视频.mp4`（101MB）

---

## 竞品数据

| 模型 | 机构 | 维度 | 参数量 | 空间分辨率 | 时序支持 | 多模态 | 变化检测 | 检索工具 | 开源 |
|------|------|------|--------|-----------|----------|--------|----------|----------|------|
| **AEF** | Google DeepMind | 64 | ~480M | 10m | ✅ valid period | ✅ S2/S1/Landsat | BA 71.3%(无监督) | ❌ | 部分 |
| **Clay v1.5** | Clay | 1024 | 大 | 10m | ✅ | ✅ 多传感器 | 未重点报道 | ✅ TerraBit | ✅ |
| **Tessera** | AI2 | 512(128产品) | 中 | 10m | ✅ | ✅ S1/S2 | 有 | 有 | 部分 |
| **OlmoEarth-nano** | AI2 | 128 | 1.4M | - | ✅ | ✅ S1/S2/Landsat | 有 | 未重点报道 | ✅ |
| **DINOv3** | Meta | 1024 | ViT-L | 0.6m | ❌ | ❌ RGB only | 需微调 | 需适配 | ✅ |
| **SatCLIP** | Microsoft | - | - | - | ❌ | ❌ | ❌（无时间能力） | 有 | ✅ |
| **Prithvi** | NASA/IBM | - | - | - | ✅ | ✅ SAR+光学 | 有 | 有 | ✅ |
| **玄女底座** | 自研 | 64 | 自研 | 10m | ✅ **双窗口时序对比** | ✅ S2/S1/Landsat | **AUC 0.903** | ✅ EarthEmbeddingExplorer | 自研 |

---

## 玄女底座已知指标
- 哈尔滨变化检测AUC: 0.903 (69-patch验证，2025年4-10月)
- 反坍缩raw_uniformity: ~-4.0 (正常范围)
- 时序敏感性: 双窗口时序对比损失 + gap-aware (V6.5)
- 嵌入维度: 64维
- 输入源: S2 + S1 + Landsat
- 目标源: 7类（输入3类 + DEM + WorldCover + DynamicWorld + JRC Water）
