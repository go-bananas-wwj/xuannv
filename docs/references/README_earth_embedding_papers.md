# 地球嵌入 (Earth Embedding) 相关论文搜集

> 搜集日期: 2026-05-27  
> 目标: 寻找与 xuannv_embdding (AlphaEarth Foundations 改进版) 相似的工作

---

## 📌 核心相关论文（与 xuannv 直接可比）

### 1. AlphaEarth Foundations (AEF) — Google DeepMind
| 属性 | 信息 |
|------|------|
| **论文** | AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data |
| **arXiv** | [2507.22291](https://arxiv.org/abs/2507.22291) |
| **机构** | Google DeepMind |
| **核心特点** | 64维像素级嵌入场，多源时序融合 (S2/S1/Landsat/ERA5/GEDI/GRACE/DEM/Wikipedia)，von Mises-Fisher 球面嵌入，时空Precision编码器，教师-学生框架 |
| **与 xuannv 关系** | ⭐⭐⭐ **直接基线/原版**。xuannv 是 AEF 的独立改进版，核心使命是解决 AEF 的嵌入坍缩和时间敏感性问题 |
| **下载文件** | `02_AlphaEarth_Foundations_arxiv_2507.22291.pdf` |

### 2. OlmoEarth — Allen AI / University of Washington 等
| 属性 | 信息 |
|------|------|
| **论文** | OlmoEarth: A Multimodal Spatio-Temporal Foundation Model for Earth Observation |
| **arXiv** | [2511.13655](https://arxiv.org/abs/2511.13655) |
| **机构** | Allen Institute for AI, University of Washington 等 |
| **核心特点** | 多模态时空基础模型，新颖的自监督学习、掩码策略和损失函数，在24个任务中15个SOTA |
| **与 xuannv 关系** | ⭐⭐⭐ **高度相似**。同样是多模态时空基础模型，关注 embedding 质量和下游任务泛化 |
| **下载文件** | `01_OlmoEarth_arxiv_2511.13655.pdf` |

### 3. CLAY Foundation Model — DevelopmentSeed / Radiant Earth
| 属性 | 信息 |
|------|------|
| **项目** | [clay-foundation.github.io/model](https://clay-foundation.github.io/model) |
| **论文** | 无单独 arXiv 论文（技术报告/项目文档） |
| **机构** | DevelopmentSeed, Radiant Earth Foundation, Renaissance Philanthropy |
| **核心特点** | ViT 架构，MAE 自监督学习，多传感器输入 (S1/S2/Landsat/NAIP/LINZ)，输出 patch-level embeddings，~500M 参数，Apache 2.0 |
| **与 xuannv 关系** | ⭐⭐⭐ **高度相似**。同样是地球基础模型，多传感器输入，embedding 驱动下游任务。V1.5 已发布 |
| ** HuggingFace** | [made-with-clay/Clay](https://huggingface.co/made-with-clay/Clay) |

---

## 🔬 基础模型家族（相同技术路线）

### 4. Prithvi / Prithvi-EO-2.0 — NASA + IBM
| 属性 | 信息 |
|------|------|
| **论文 v1** | Foundation Models for Generalist Geospatial Artificial Intelligence ([arXiv:2310.18660](https://arxiv.org/abs/2310.18660)) |
| **论文 v2** | Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model ([arXiv:2412.02732](https://arxiv.org/abs/2412.02732)) |
| **机构** | NASA + IBM |
| **核心特点** | 时序 ViT，MAE 预训练，HLS (Landsat+Sentinel-2) 数据，时空注意力，300M/600M 参数 |
| **与 xuannv 关系** | ⭐⭐⭐ **技术路线最接近**。同样基于时空 Transformer + MAE，多传感器时序输入。Prithvi-EO-2.0 引入时间和位置嵌入增强 |
| **下载文件** | `04_Prithvi_EO_2.0_arxiv_2412.02732.pdf`, `05_Prithvi_v1_arxiv_2310.18660.pdf` |

### 5. Presto — NASA Harvest / Mila / McGill
| 属性 | 信息 |
|------|------|
| **论文** | Lightweight, Pre-trained Transformers for Remote Sensing Timeseries ([arXiv:2304.14065](https://arxiv.org/abs/2304.14065)) |
| **机构** | NASA Harvest, Mila, McGill, ASU |
| **核心特点** | **像素级**时间序列 Transformer，仅 0.4M 参数，多传感器 (S1/S2/ERA5/DEM/Dynamic World)，MAE 掩码自编码，极轻量 |
| **与 xuannv 关系** | ⭐⭐⭐ **高度相似**。同样是像素级 embedding，多传感器时序输入，MAE 自监督。Presto 在 2023 年就实现了极轻量的像素级表示学习 |
| **下载文件** | `06_Presto_arxiv_2304.14065.pdf` |

### 6. DOFA (Dynamic One-For-All) — TUM
| 属性 | 信息 |
|------|------|
| **论文** | Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation ([arXiv:2403.15356](https://arxiv.org/abs/2403.15356)) |
| **机构** | Technical University of Munich (TUM) |
| **核心特点** | 神经可塑性启发，**波长条件动态超网络**，单一 Transformer 处理任意传感器/任意波段数，5种传感器联合训练 |
| **与 xuannv 关系** | ⭐⭐⭐ **高度相似**。多传感器统一表示学习，动态权重生成。与 xuannv 的 SensorEncoderBank + STPBlocks 思想相通 |
| **下载文件** | `07_DOFA_arxiv_2403.15356.pdf` |

### 7. TerraMind — IBM + ESA Φ-lab
| 属性 | 信息 |
|------|------|
| **论文** | TerraMind: Large-Scale Generative Multimodality for Earth Observation ([arXiv:2504.11171](https://arxiv.org/abs/2504.11171)) |
| **会议** | ICCV 2025 |
| **机构** | IBM Research + ESA Φ-lab |
| **核心特点** | **首个任意模态到任意模态生成式**基础模型，双尺度表示 (token-level + pixel-level)，Thinking-in-Modalities (TiM)，9种地理空间模态 |
| **与 xuannv 关系** | ⭐⭐⭐ **高度相似**。多模态融合 + 生成式重建，与 xuannv 的多源重建目标类似。TerraMind 强调模态间生成能力 |
| **下载文件** | `08_TerraMind_arxiv_2504.11171.pdf` |

---

## 📊 其他重要相关论文

### 8. Earth AI — Google
| 属性 | 信息 |
|------|------|
| **论文** | Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning ([arXiv:2510.18318](https://arxiv.org/abs/2510.18318)) |
| **机构** | Google |
| **核心特点** | 遥感预训练骨干 (RS-Global MTP)，VLM + OVD，MAE + 多任务预训练，300M 图像 |
| **与 xuannv 关系** | ⭐⭐ 多任务预训练 + 表示学习，但侧重 VLM 和检测 |
| **下载文件** | `09_Earth_AI_arxiv_2510.18318.pdf` |

### 9. FUSAR-GPT (使用 AEF)
| 属性 | 信息 |
|------|------|
| **论文** | FUSAR-GPT: ... ([arXiv:2602.19190](https://arxiv.org/abs/2602.19190)) |
| **核心特点** | 引入 AEF 作为多源时序特征补偿，SAR 图像理解 |
| **与 xuannv 关系** | ⭐⭐ 展示了 AEF embedding 的下游应用 |
| **下载文件** | `10_FUSAR_GPT_arxiv_2602.19190.pdf` |

### 10. Harvesting AlphaEarth — 农业下游评估
| 属性 | 信息 |
|------|------|
| **论文** | Harvesting AlphaEarth: Benchmarking the Geospatial Foundation Model for Agricultural Downstream Tasks ([arXiv:2601.00857](https://arxiv.org/abs/2601.00857)) |
| **核心特点** | 系统评估 AEF embedding 在农业下游任务，指出 AEF 的**时间敏感性有限、空间迁移性有限、可解释性低** |
| **与 xuannv 关系** | ⭐⭐⭐ **xuannv 的改进动机直接来源**！该论文明确指出了 AEF 的时间敏感性问题，正是 xuannv 要解决的核心问题 |
| **下载文件** | `03_Harvesting_AlphaEarth_arxiv_2601.00857.pdf` |

---

## 🏗️ 早期/里程碑基础模型

### 11. ScaleMAE — BAIR / UC Berkeley
| 属性 | 信息 |
|------|------|
| **论文** | Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning ([arXiv:2212.14532](https://arxiv.org/abs/2212.14532)) |
| **会议** | ICCV 2023 |
| **核心特点** | 尺度感知 MAE，拉普拉斯金字塔解码器，处理 0.1m-30m 多分辨率 |
| **下载文件** | `11_ScaleMAE_arxiv_2212.14532.pdf` |

### 12. SatMAE — Stanford
| 属性 | 信息 |
|------|------|
| **论文** | SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery ([arXiv:2207.08051](https://arxiv.org/abs/2207.08051)) |
| **会议** | NeurIPS 2022 |
| **核心特点** | 时序+多光谱分组掩码策略，fMoW Sentinel-2 |
| **下载文件** | `12_SatMAE_arxiv_2207.08051.pdf` |

### 13. SpectralGPT — 多光谱专用
| 属性 | 信息 |
|------|------|
| **论文** | SpectralGPT: Spectral Remote Sensing Foundation Model ([arXiv:2311.07113](https://arxiv.org/abs/2311.07113)) |
| **期刊** | IEEE TPAMI 2024 |
| **核心特点** | **首个光谱专用基础模型**，3D 张量掩码 (90%掩码率)，渐进式预训练，空间-光谱耦合建模 |
| **下载文件** | `13_SpectralGPT_arxiv_2311.07113.pdf` |

### 14. CROMA — 雷达-光学对比学习
| 属性 | 信息 |
|------|------|
| **论文** | CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders ([arXiv:2311.00566](https://arxiv.org/abs/2311.00566)) |
| **会议** | NeurIPS 2023 |
| **核心特点** | 对比学习 + MAE 结合，S1 SAR + S2 光学联合预训练 |
| **下载文件** | `19_CROMA_arxiv_2311.00566.pdf` |

### 15. BFM (Billion-scale Foundation Model)
| 属性 | 信息 |
|------|------|
| **论文** | A Billion-scale Foundation Model for Remote Sensing Images ([arXiv:2304.05215](https://arxiv.org/abs/2304.05215)) |
| **核心特点** | 十亿级参数，大规模遥感图像预训练 |
| **下载文件** | `20_BFM_arxiv_2304.05215.pdf` |

### 16. EarthPT — 时序基础模型
| 属性 | 信息 |
|------|------|
| **论文** | EarthPT: a time series foundation model for Earth Observation ([arXiv:2309.07207](https://arxiv.org/abs/2309.07207)) |
| **会议** | NeurIPS 2023 CCAI Workshop |
| **核心特点** | 时间序列专用，地球观测数据 |
| **下载文件** | `21_EarthPT_arxiv_2309.07207.pdf` |

---

## 🌍 Earth Embedding 概念与综述

### 17. Earth Embeddings: Towards AI-centric Representations of our Planet
| 属性 | 信息 |
|------|------|
| **作者** | Klemmer, Rolf, Russwurm 等 (20+ 位作者，包括 Hannah Kerner, Lester Mackey, Xiaoxiang Zhu 等) |
| **发表** | EarthArXiv 2025 |
| **DOI** | [10.31223/X5HX9S](https://doi.org/10.31223/X5HX9S) |
| **核心观点** | 提出 **Earth Embeddings** 统一概念：融合不同地理空间数据源，压缩高度相关的原始数据为密集表示，可作为基础模型的通用位置 token |
| **与 xuannv 关系** | ⭐⭐⭐ **概念框架**。这篇论文系统阐述了 Earth Embedding 的概念、分类法、生态系统，是理解该领域的必读综述 |
| **注意** | 发布在 EarthArXiv 而非标准 arXiv，无法通过 arxiv.org/pdf 直接下载 |

### 18. Earth Embeddings as Products: Taxonomy, Ecosystem, and Standardized Access
| 属性 | 信息 |
|------|------|
| **论文** | ([arXiv:2601.13134](https://arxiv.org/abs/2601.13134)) |
| **核心特点** | Earth Embedding 的产品化、分类法、标准化访问 |
| **下载文件** | `14_Earth_Embeddings_Products_arxiv_2601.13134.pdf` |

### 19. Major TOM: Global and Dense Embeddings of Earth
| 属性 | 信息 |
|------|------|
| **论文** | ([arXiv:2412.05600](https://arxiv.org/abs/2412.05600)) |
| **核心特点** | 全球密集 Earth Embedding |
| **下载文件** | `15_Major_TOM_arxiv_2412.05600.pdf` |

### 20. Foundation Models for Remote Sensing and Earth Observation: A Survey
| 属性 | 信息 |
|------|------|
| **论文** | ([arXiv:2410.16602](https://arxiv.org/abs/2410.16602)) |
| **核心特点** | 遥感基础模型全面综述 |
| **下载文件** | `17_RS_Foundation_Models_Survey_arxiv_2410.16602.pdf` |

---

## 📈 Benchmarks & Evaluation

### 21. PANGAEA — 地理空间基础模型全球基准
| 属性 | 信息 |
|------|------|
| **论文** | ([arXiv:2412.04204](https://arxiv.org/abs/2412.04204)) |
| **核心特点** | 全球包容性基准测试，多任务评估 |
| **下载文件** | `16_PANGAEA_Benchmark_arxiv_2412.04204.pdf` |

### 22. SatCLIP — 位置嵌入
| 属性 | 信息 |
|------|------|
| **论文** | SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery ([arXiv:2311.17179](https://arxiv.org/abs/2311.17179)) |
| **会议** | AAAI 2025 |
| **核心特点** | 卫星图像 + 位置联合嵌入，通用位置表示 |
| **下载文件** | `18_SatCLIP_arxiv_2311.17179.pdf` |

---

## 🔍 AEF/xuannv 下游应用论文

### 23. Earth Embeddings Reveal Diverse Urban Signals from Space
| 属性 | 信息 |
|------|------|
| **论文** | ([arXiv:2604.03456](https://arxiv.org/abs/2604.03456)) |
| **核心特点** | 对比 AEF / Prithvi-EO-2.0 / Clay-v1.5 三种 Earth Embedding 在城市信号预测中的表现 |
| **下载文件** | `22_Earth_Embeddings_Urban_Signals_arxiv_2604.03456.pdf` |

### 24-26. 其他 AEF/基础模型应用
- **Rapid Adaptation of EO FMs for Segmentation** (`23_Rapid_Adaptation_EO_FM_arxiv_2409.09907.pdf`) — Clay + LoRA 洪水检测
- **Geospatial FMs for SDGs** (`24_Geospatial_FM_SDGs_arxiv_2505.24528.pdf`) — 基础模型在可持续发展目标上的评估
- **Landslide Hazard Mapping with GFMs** (`25_Landslide_Hazard_arxiv_2511.04474.pdf`) — Prithvi-EO-2.0 滑坡制图

---

## 📝 关键发现与 xuannv 定位

### 技术生态图谱

```
Earth Embedding 领域
│
├── 【像素级嵌入】 ← xuannv 所在赛道
│   ├── AlphaEarth Foundations (AEF) — 原版，Google DeepMind
│   ├── **xuannv_embdding** — AEF 改进版，解决坍缩+时间敏感性
│   ├── Presto — 0.4M 参数像素级时序 Transformer
│   └── Major TOM — 全球密集嵌入
│
├── 【Patch 级嵌入】
│   ├── CLAY — ViT+MAE，多传感器 patch embedding
│   ├── Prithvi — 时序 ViT，HLS 数据
│   ├── ScaleMAE — 多尺度感知 MAE
│   ├── SatMAE — 时序+多光谱分组掩码
│   └── DOFA — 动态权重，任意传感器
│
├── 【生成式/多模态】
│   ├── TerraMind — 任意模态到任意模态生成 (IBM+ESA)
│   ├── Earth AI — Google 多任务预训练
│   └── OlmoEarth — 多模态时空 (Allen AI)
│
├── 【光谱专用】
│   ├── SpectralGPT — 3D 掩码光谱基础模型
│   └── HyperSIGMA — 双流光谱-空间
│
└── 【综述/概念】
    ├── Earth Embeddings (EarthArXiv 2025) — 概念框架
    ├── RS Foundation Models Survey — 全面综述
    └── Earth Embeddings as Products — 产品化分类
```

### xuannv 的独特定位

与上述工作相比，**xuannv_embdding 的核心差异化**在于：

1. **解决嵌入坍缩** — 训练时跳过 L2 Norm，在 pre-norm 空间计算反坍缩损失 (raw_uniformity, decorrelation, variance_regularizer)
2. **提升时间敏感性** — 不重叠双窗口 + 时序对比损失 (temporal contrastive loss)
3. **变化检测能力** — 专为变化检测优化的 embedding 设计
4. **输入严格对齐论文** — 仅 S2/S1/Landsat 作为输入，静态数据只参与重建

这些改进直接回应了 [Harvesting AlphaEarth](#10-harvesting-alphaearth--农业下游评估) 论文中指出的 AEF **"limited time sensitivity"** 问题。

---

## 📂 文件清单

| 编号 | 文件名 | 论文 | 大小 | 页数 | 状态 |
|------|--------|------|------|------|------|
| 01 | `01_OlmoEarth_arxiv_2511.13655.pdf` | OlmoEarth | 6.1M | 11 | ✅ |
| 02 | `02_AlphaEarth_Foundations_arxiv_2507.22291.pdf` | AEF 原版 | 25M | 22 | ✅ |
| 03 | `03_Harvesting_AlphaEarth_arxiv_2601.00857.pdf` | AEF 农业评估（指出时间敏感性问题） | 1.6M | 8 | ✅ |
| 04 | `04_Prithvi_EO_2.0_arxiv_2412.02732.pdf` | Prithvi-EO-2.0 | 16M | 7 | ✅ |
| 05 | `05_Prithvi_v1_arxiv_2310.18660.pdf` | Prithvi v1 | 8.3M | - | ✅ |
| 06 | `06_Presto_arxiv_2304.14065.pdf` | Presto (像素级轻量) | 3.8M | 3 | ✅ |
| 07 | `07_DOFA_arxiv_2403.15356.pdf` | DOFA (动态多传感器) | 8.8M | 7 | ✅ |
| 08 | `08_TerraMind_arxiv_2504.11171.pdf` | TerraMind (生成式多模态) | 43M | 11 | ✅ |
| 09 | `09_Earth_AI_arxiv_2510.18318.pdf` | Earth AI (Google) | 4.8M | 6 | ✅ |
| 10 | `10_FUSAR_GPT_arxiv_2602.19190.pdf` | FUSAR-GPT (使用 AEF) | 26M | 10 | ✅ |
| 11 | `11_ScaleMAE_arxiv_2212.14532.pdf` | ScaleMAE | 2.1M | 6 | ✅ |
| 12 | `12_SatMAE_arxiv_2207.08051.pdf` | SatMAE | 14M | - | ✅ |
| 13 | `13_SpectralGPT_arxiv_2311.07113.pdf` | SpectralGPT | 30M | - | ✅ |
| 14 | `14_Earth_Embeddings_Products_arxiv_2601.13134.pdf` | Earth Embeddings as Products | 215K | 5 | ✅ |
| 15 | `15_Major_TOM_arxiv_2412.05600.pdf` | Major TOM | 10M | - | ✅ |
| 16 | `16_PANGAEA_Benchmark_arxiv_2412.04204.pdf` | PANGAEA Benchmark | 19M | - | ✅ |
| 17 | `17_RS_Foundation_Models_Survey_arxiv_2410.16602.pdf` | RS FM Survey | 2.6M | 12 | ✅ |
| 18 | `18_SatCLIP_arxiv_2311.17179.pdf` | SatCLIP | 33M | - | ✅ |
| 19 | `19_CROMA_arxiv_2311.00566.pdf` | CROMA | 6.6M | 2 | ✅ |
| 20 | `20_BFM_arxiv_2304.05215.pdf` | BFM | 34M | - | ✅ |
| 21 | `21_EarthPT_arxiv_2309.07207.pdf` | EarthPT | 2.8M | 6 | ✅ |
| 22 | `22_Earth_Embeddings_Urban_Signals_arxiv_2604.03456.pdf` | Earth Embeddings Urban Signals | 25M | 7 | ✅ |
| 23 | `23_Rapid_Adaptation_EO_FM_arxiv_2409.09907.pdf` | Rapid Adaptation EO FMs | 2.2M | 2 | ✅ |
| 24 | `24_Geospatial_FM_SDGs_arxiv_2505.24528.pdf` | Geospatial FMs for SDGs | 4.2M | 6 | ✅ |
| 25 | `25_Landslide_Hazard_arxiv_2511.04474.pdf` | Landslide Hazard Mapping | 2.5M | 6 | ✅ |

---

## 🔗 重要资源链接

- **Awesome-Remote-Sensing-Foundation-Models**: https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models
- **Awesome-Geospatial-Embeddings**: https://github.com/hfangcat/Awesome-Geospatial-Embeddings
- **TerraTorch** (地理空间 FM 微调工具包): https://github.com/terrastackai/terratorch
- **TorchGeo**: https://github.com/microsoft/torchgeo
- **rs-embed** (一键获取任意 RS FM embedding): https://github.com/cybergis/rs-embed
- **PANGAEA Benchmark**: https://github.com/VMarsocci/pangaea
- **GEO-Bench**: https://github.com/ServiceNow/geo-bench

---

## ⚠️ 未能下载的论文

| 论文 | 原因 | 替代获取方式 |
|------|------|-------------|
| **Earth Embeddings: Towards AI-centric Representations of our Planet** (Klemmer et al., 2025) | 发布在 EarthArXiv，非标准 arXiv | [EarthArXiv 下载页](https://eartharxiv.org/repository/view/11083/) |
| **Clay Foundation Model** | 无单独 arXiv 论文，仅项目文档 | [项目官网](https://clay-foundation.github.io/model/) |
| **SkySense** (CVPR 2024) | 需要找 arXiv 链接 | 搜索 "SkySense CVPR 2024 arxiv" |
| **msGFM** (CVPR 2024) | 需要找 arXiv 链接 | 搜索 "msGFM CVPR 2024 arxiv" |
| **MMEarth** (ECCV 2024) | 需要找 arXiv 链接 | 搜索 "MMEarth ECCV 2024 arxiv" |
| **SatMAE++** (CVPR 2024) | 需要找 arXiv 链接 | 搜索 "SatMAE++ CVPR 2024 arxiv" |
| **Cross-Scale MAE** (NeurIPS 2023) | 需要找 arXiv 链接 | 搜索 "Cross-Scale MAE NeurIPS 2023 arxiv" |
| **RingMo** (TGRS 2022) | 需要找 arXiv 链接 | 搜索 "RingMo remote sensing foundation model arxiv" |
| **USat** (arXiv 2023) | 需要找 arXiv 链接 | 搜索 "USat unified self-supervised encoder arxiv" |
| **GFM** (ICCV 2023, continual pretraining) | 需要找 arXiv 链接 | 搜索 "GFM geospatial foundation model continual pretraining arxiv" |
| **SSL4EO-S12** | 需要找 arXiv 链接 | 搜索 "SSL4EO-S12 arxiv" |
| **SatLasPretrain** (ICCV 2023) | 需要找 arXiv 链接 | 搜索 "Satlas pretrain arxiv" |
| **Earth Embeddings (SIGGRAPH Frontiers 2025)** | 会议论文，可能无预印本 | 通过 ACM Digital Library 获取 |
