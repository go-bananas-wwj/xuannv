# Presto: A Pixel-Level Pre-trained Multi-Sensor Masked Autoencoder for Earth Observation
# Presto：面向地球观测的像素级预训练多传感器掩码自编码器

---

## Metadata

| 属性 | 值 |
|------|-----|
| 标题 (EN) | Presto: A Pixel-Level Pre-trained Multi-Sensor Masked Autoencoder for Earth Observation |
| 标题 (ZH) | Presto：面向地球观测的像素级预训练多传感器掩码自编码器 |
| 作者 | Gabriel Tseng, Ruben Cartuyvels, Ivan Zvonkov, Mirali Purohit, David Rolnick, Hannah Kerner |
| 机构 | Mila, McGill University, KU Leuven, University of Maryland, Arizona State University |
| 发表 | arXiv:2304.14065v4 [cs.CV] 5 Feb 2024 |
| 页数 | 21 |
| 生成日期 | 2026-05-27 |

---

## Page Index

| 页码 | 内容 |
|------|------|
| 1 | paragraph, section_heading |
| 2 | figure_caption, paragraph |
| 3 | paragraph, section_heading |
| 4 | figure_caption, paragraph, section_heading |
| 5 | paragraph, section_heading, table_caption |
| 6 | figure_caption, paragraph, section_heading, table_caption |
| 7 | bullet, figure_caption, paragraph, section_heading |
| 8 | bullet, paragraph, table_caption |
| 9 | figure_caption, paragraph, section_heading |
| 10 | bullet, paragraph, section_heading, table_caption |
| 11 | paragraph, section_heading, table_caption |
| 12 | paragraph, section_heading, table_caption |
| 13 | paragraph |
| 14 | paragraph |
| 15 | paragraph |
| 16 | figure_caption, paragraph, section_heading |
| 17 | bullet, paragraph, table_caption |
| 18 | paragraph, table_caption |
| 19 | paragraph, table_caption |
| 20 | table_caption |
| 21 | figure_caption, paragraph, table_caption |

---

<!-- Page 1 -->
## Page 1

**[C001]**
EN: Lightweight, Pre-trained Transformers for Remote Sensing Timeseries
ZH: 面向遥感时间序列的轻量级预训练Transformer

**[C002]**
EN: Gabriel Tseng1,2      Ruben Cartuyvels1,3 Ivan Zvonkov4             Mirali Purohit5 David Rolnick1,2 Hannah Kerner5 1 Mila – Quebec AI Institute 2 McGill University arXiv:2304.14065v4 [cs.CV] 5 Feb 2024
ZH: Gabriel Tseng¹²  Ruben Cartuyvels¹³  Ivan Zvonkov⁴  Mirali Purohit⁵  David Rolnick¹²  Hannah Kerner⁵\n¹ Mila – 魁北克人工智能研究所  ² 麦吉尔大学  ³ 鲁汶大学  ⁴ 马里兰大学学院市分校  ⁵ 亚利桑那州立大学

### [S001] 3 KU Leuven 4 University of Maryland, College Park 5 Arizona State University
**3 鲁汶大学  4 马里兰大学学院市分校  5 亚利桑那州立大学**

### [S002] Abstract
**摘要**

**[C003]**
EN: Machine learning methods for satellite data have a range of societally relevant applications, but labels used to train models can be difficult or impossible to acquire. Self-supervision is a natural solution in settings with limited labeled data, but current self-supervised models for satellite data fail to take advantage of the characteristics of that data, including the temporal dimension (which is critical for many applications, such as monitoring crop growth) and availability of data from many complementary sensors (which can significantly improve a model’s predictive performance). We present Presto (the Pretrained Remote Sensing Transformer), a model pre-trained on remote sensing pixel-timeseries data. By designing Presto specifically for remote sensing data, we can create a significantly smaller but performant model. Presto excels at a wide variety of globally distributed remote sensing tasks and performs competitively with much larger models while requiring far less compute. Presto can be used for transfer learning or as a feature extractor for simple models, enabling efficient deployment at scale.
ZH: 面向卫星数据的机器学习方法具有广泛的社会相关应用，但用于训练模型的标签往往难以获取甚至无法获取。在标注数据有限的场景中，自监督学习是一种自然的解决方案，但当前面向卫星数据的自监督模型未能充分利用数据的特性，包括时间维度（这对许多应用至关重要，例如监测作物生长）以及来自多种互补传感器的数据可用性（这能显著提升模型的预测性能）。本文提出 Presto（Pretrained Remote Sensing Transformer，预训练遥感Transformer），一种在遥感像素时间序列数据上进行预训练的模型。通过专为遥感数据设计Presto，我们能够创建一个显著更小但性能出色的模型。Presto 在多种全球分布的遥感任务上表现优异，与规模大得多的模型相比具有竞争力，同时所需计算资源远少得多。Presto 可用于迁移学习或作为简单模型的特征提取器，从而实现大规模高效部署。

### [S003] 1    Introduction
**1 引言**

**[C004]**
EN: Machine learning is increasingly being applied to the remote sensing domain, in particular to under- stand the evolution of the Earth’s surface over time (Brown et al., 2022; Voosen, 2020; Abys et al., 2024; Wang et al., 2020b). These applications can have important societally beneficial outcomes, ranging from tracking progress on sustainable development goals (Ferreira et al., 2020) to improved weather forecasting (English et al., 2013; Voosen, 2020) to disaster management (Kansakar and Hossain, 2016). However, labeled datasets often contain labels that are few, sparse, and unreliable (Bressan et al., 2022), especially for under-resourced geographies, leading to poor global gener- alization (Yifang et al., 2015; Kerner et al., 2020; Nakalembe et al., 2021). This has spurred the investigation of self-supervised learning algorithms for remote sensing data. Current self-supervised approaches for remote sensing data have drawn from methods in computer vision, yielding models that treat remote sensing data as single-timestep images (Jean et al., 2019; Manas et al., 2021; Ayush et al., 2021). Such models (i) cannot benefit from patterns that emerge when an area is monitored over time, which is especially important for agriculture and other seasonal landcover, (ii) typically only consider a single satellite product (such as Sentinel-2 multispectral data), despite there being hundreds of publicly available satellite data products (GEE), (iii) are typically large and computationally expensive (Reed et al., 2022; Cong et al., 2022; Fuller et al., 2023), making the deployment of these models at scale challenging, and (iv) cannot natively handle the labels for
ZH: 机器学习正越来越多地应用于遥感领域，特别是用于理解地球表面随时间的演变（Brown 等，2022；Voosen，2020；Abys 等，2024；Wang 等，2020b）。这些应用可以产生重要的社会效益，范围涵盖追踪可持续发展目标的进展（Ferreira 等，2020）、改进天气预报（English 等，2013；Voosen，2020）到灾害管理（Kansakar 和 Hossain，2016）。然而，标注数据集往往存在标签稀少、稀疏且不可靠的问题（Bressan 等，2022），尤其在资源匮乏的地理区域，导致全球泛化能力差（Yifang 等，2015；Kerner 等，2020；Nakalembe 等，2021）。这推动了面向遥感数据的自监督学习算法的探索。

**[C005]**
EN: Preprint.
ZH: 预印本


---

<!-- Page 2 -->
## Page 2

**[F001] Figure / 图**
> **EN**: Figure 1: Presto learns from structurally-masked remote sensing pixel-timeseries. We construct a multi-sensor remote sensing pixel-timeseries, and randomly select one of the four masking strategies described in Section 3.3. The encoder-decoder model is trained to reconstruct the original timeseries. At fine-tuning time, we discard the decoder and only use the encoder’s output. The downstream task may have incomplete inputs (missing timesteps or sensors) since the encoder is specifically trained on such inputs. Presto receives both static-in-time and dynamic-in-time inputs and the location metadata of each pixel timeseries.
> **ZH**: 图1：Presto 从结构性掩码的遥感像素时间序列中学习。我们构建了一个多传感器遥感像素时间序列，并随机选择第3.3节描述的四种掩码策略之一。编码器-解码器模型被训练来重建原始时间序列。在微调阶段，我们丢弃解码器，仅使用编码器的输出。下游任务可能具有不完整的输入（缺失时间步或传感器），因为编码器专门针对此类输入进行了训练。Presto 接收静态时间输入和动态时间输入，以及每个像素时间序列的位置元数据。

**[C006]**
EN: many remote sensing datasets, which are points or irregularly shaped polygons (Rao et al., 2020; Batjes et al., 2017), requiring additional methods to handle these labels (Wang et al., 2020a). We introduce the Pretrained Remote Sensing Transformer (Presto), a lightweight model designed to ingest pixel-timeseries inputs from a variety of Earth observation sensors and data products. Presto operates on individual pixels, using the temporal and multimodal structure of the data instead of the image structure. To learn powerful representations of remote sensing data that can be adapted to a wide range of tasks, Presto leverages a self-supervised masked autoencoding approach, reconstructing unobserved timepoints and sensory modalities. This allows Presto to be robust to missing data and to flexibly accommodate diverse input formats. We find Presto excels even in image-based tasks where the temporal dimension is completely absent. Presto addresses the following requirements, which are critical to the useful deployment of pre-trained models in the remote sensing context: • Computational efficiency: When deployed, models built for remote sensing data are typically used to make contiguous geospatial predictions over millions (or billions) of samples to form a predicted map. The computational performance of models is therefore one of the primary considerations at deployment time. Van Tricht (2021), Hengl et al. (2017) and Robinson et al. (2019) are all global- or large- scale map making efforts that prioritized efficiency over accuracy when deploying remote sensing models at scale. Presto is competitive with ViT or ResNet based models, despite having up to 1000× fewer trainable parameters and requiring orders of magnitude fewer FLOPs at inference time. • Ability to process inputs of varying shapes: Different downstream tasks may require very different remote sensing inputs. For example, for crop mapping and yield estimation, Sainte Fare Garnot et al. (2020) and You et al. (2017) discarded all spatial information in the inputs in favor of emphasizing temporal patterns. We test Presto on a wide range of downstream inputs (for example, with spatial information present or absent, and with single or multiple timesteps of data), and find it is competitive with models designed specifically for those inputs. • Ability to process a range of remote sensing datasets: For fuel moisture estimation, Rao et al. (2020) found that the inclusion of derived products in addition to raw inputs significantly improved performance. Presto can ingest a range of static-in-time and dynamic-in-time raw input data as well as derived product inputs widely used in Earth observation (such as NDVI (Rouse et al., 1974)). • Ability to handle missing data: The coverage of remote sensing products is often spatially and temporally incomplete. For example, certain regions experience very high (> 90%) cloud coverage, reducing the utility of optical measurements such as Sentinel-2 imagery (Sudmanns et al., 2019). Because Presto ingests a variety of remote sensing inputs, it can leverage alternative data sources if
ZH: 当前面向遥感数据的自监督方法借鉴了计算机视觉中的方法，产生了将遥感数据视为单时间步图像的模型（Jean 等，2019；Manas 等，2021；Ayush 等，2021）。这类模型（i）无法从随时间监测某区域时出现的模式中获益，而这对于农业和其他季节性土地覆盖尤为重要；（ii）通常只考虑单一卫星产品（如 Sentinel-2 多光谱数据），尽管有数百种公开可用的卫星数据产品（GEE）；（iii）通常规模庞大且计算昂贵（Reed 等，2022；Cong 等，2022；Fuller 等，2023），使得这些模型的大规模部署具有挑战性；（iv）无法原生处理许多遥感数据集的标注，这些数据集是点或不规则多边形（Rao 等，2020；Batjes 等，2017），需要额外方法来处理这些标注（Wang 等，2020a）。


---

<!-- Page 3 -->
## Page 3

**[C007]**
EN: one is missing (for instance, relying on Sentinel-1, which sees through clouds, if Sentinel-2 images are cloudy). Our results support the surprising conclusion that a pixel-based approach can in some cases match or outperform sophisticated computer vision-based approaches. We hypothesize that this is possible because (i) Presto learns from many semantically dense data sources, allowing it to extract informative patterns from pixel-timeseries, and (ii) many remote sensing tasks require significantly smaller receptive fields than those provided by computer vision-based models. Brown et al. (2022) leveraged such properties to train a model 100× smaller than standard models while achieving state-of-the-art land-cover segmentation results.
ZH: 其中一种缺失（例如，依赖能穿透云层的 Sentinel-1，当 Sentinel-2 图像被云层遮挡时）。我们的结果支持一个令人惊讶的结论：在某些情况下，基于像素的方法可以匹敌甚至超越复杂的基于计算机视觉的方法。我们假设这是可能的，因为（i）Presto 从许多语义密集的数据源中学习，使其能够从像素时间序列中提取信息丰富的模式，以及（ii）许多遥感任务所需的感受野显著小于基于计算机视觉模型所提供的感受野。Brown 等（2022）利用这些特性训练了一个比标准模型小100倍的模型，同时实现了最先进的土地覆盖分割结果。

### [C008] 2     Related Work
**2 相关工作**

**[C009]**
EN: Architectures for Remote Sensing When processing remote sensing timeseries, transformers have been extensively investigated either as unmodified architectures (Rußwurm and Körner, 2020) or as architectures designed for specific tasks (Sainte Fare Garnot et al., 2020; Tarasiou et al., 2023). Recurrent networks have also been investigated (Kerner et al., 2020; Rußwurm and Körner, 2020). When treating remote sensing data as single or few (up to 3) timestep images, architectures from computer vision are commonly used, ranging from ResNets (Manas et al., 2021; Ayush et al., 2021; Rußwurm et al., 2020) to Vision Transformers (Cong et al., 2022; Reed et al., 2022; Fuller et al., 2023).
ZH: 遥感架构 在处理遥感时间序列时，Transformer 已被广泛研究，无论是作为未经修改的架构（Rußwurm 和 Körner，2020）还是为特定任务设计的架构（Sainte Fare Garnot 等，2020；Tarasiou 等，2023）。循环网络也已被研究（Kerner 等，2020；Rußwurm 和 Körner，2020）。当将遥感数据视为单张或少张（最多3张）时间步图像时，通常使用来自计算机视觉的架构，范围从 ResNet（Manas 等，2021；Ayush 等，2021；Rußwurm 等，2020）到 Vision Transformer（Cong 等，2022；Reed 等，2022；Fuller 等，2023）。

### [C010] Self-supervised learning for Remote Sensing
**遥感的自监督学习**

**[C011]**
EN: While contrastive learning has been investigated for remote sensing (Manas et al., 2021), recent self-supervised learning research has focused on masked autoencoders (Yuan et al., 2022; Cong et al., 2022; Reed et al., 2022; Fuller et al., 2023). However, these approaches (i) focus on learning from raw satellite data products (ignoring derived products such as elevation) and typically only ingest data from a single sensor (the exception being the CROMA model of Fuller et al. (2023), which ingests both Sentinel-1 and Sentinel-2 data), (ii) ingest very few or no timesteps (Reed et al. (2022) and Fuller et al. (2023) ingest only one timestep while Cong et al. (2022) ingest up to three timesteps), (iii) expect data in a certain size (for instance, ViT based models require spatial dimensions to be present), so that missing data is not handled natively, and (iv) generally yield larger models ranging from 2.5 million parameters (Yuan and Lin, 2020) to over 300 million parameters for ViT-based methods, making their deployment in compute-constrained settings challenging.
ZH: 虽然对比学习已被用于遥感研究（Manas 等，2021），但近期自监督学习研究聚焦于掩码自编码器（Yuan 等，2022；Cong 等，2022；Reed 等，2022），其中输入的一部分被掩码，模型被训练来重建该输入。在我们的工作中，我们利用遥感数据的时间特性，不仅掩码输入的空间维度，还掩码时间维度和传感器模态。

### [C012] 3     Method
**3 方法**

**[C013]**
EN: We aim to learn a model, f , which can learn useful representations in a self-supervised manner given unlabelled remote sensing pixel-timeseries data while meeting the usability requirements outlined in Section 1. This model can then be applied to a wide variety of downstream remote sensing tasks. These downstream tasks may contain input data from a range of sensors with differing numbers of timesteps. Our approach is based on the masked autoencoding framework (He et al., 2022), in which the network architecture includes both an encoder (f ) and a decoder (g). During pre-training, part of the input is masked out and the encoder embeds the remaining (non-masked) part of the input. The decoder aims to reconstruct the masked-out part of the input, given the encoder’s output. At fine-tuning time, we discard g and only use f (either as a feature extractor or a fine-tuneable model) for downstream tasks. In the sections below, we discuss how Presto customizes this general framework for multi-sensor remote sensing timeseries data. An overview of the Presto pre-training methodology is shown in Figure 1, and full pre-training details are in Section A.1.
ZH: 我们的目标是学习一个模型 f，使其能够在给定未标注数据的情况下以自监督方式学习有用的表示。我们将像素时间序列 x 转换为多个标记（每个由嵌入 e 表示），供 Transformer 编码器处理。然后，Transformer 解码器将这些标记映射回原始输入空间，重建原始像素时间序列。我们采用掩码自编码器方法：对输入的某些部分进行掩码，模型被训练来重建这些被掩码的部分。

### [S004] 3.1    Pre-training Data
**3.1 预训练数据**

**[C014]**
EN: Self-supervised models for remote sensing must generalize to a wide range of geographies and tasks (Lacoste et al., 2023). We therefore aimed to collect a globally representative pre-training dataset. We followed the sampling strategy of Brown et al. (2022) to construct a dataset of 21.5M pixel samples, each with a resolution of 10m per pixel. Appendix A.1.1 describes the pre-training dataset construction process in detail. Presto was trained on pixel-timeseries of 12-month contiguous
ZH: 面向遥感的自监督模型必须泛化到广泛的地理区域和任务（Lacoste 等，2023）。为实现这一点，预训练数据应尽可能多样化。我们使用来自全球不同地理位置的像素时间序列，采样自2020年初至2021年底的两年期间，每个月由一个时间步表示。


---

<!-- Page 4 -->
## Page 4

**[F002] Figure / 图**
> **EN**: Figure 2: Presto learns to reconstruct channels that are completely masked in a spatially cohesive manner. In this experiment, we masked only the Sentinel-2 RGB channels; Presto was able to reconstruct these channels even when they were absent from the input. The reconstructions are spatially consistent even though Presto only receives single pixel inputs.
> **ZH**: 图2：Presto 学会以空间连贯的方式重建完全掩码的通道。在此实验中，我们仅掩码 Sentinel-2 的 RGB 通道；即使这些通道在输入中完全缺失，Presto 也能重建它们。尽管 Presto 仅接收单个像素输入，重建结果在空间上仍然保持一致。

**[C015]**
EN: intervals, sampled from a 2-year period from the beginning of 2020 until the end of 2021, with each month represented by one timestep (similar to the approach adopted by Tseng et al. (2021)). Derived data products that result from the analysis of lower level data (e.g., Parkinson et al. (2006)) can significantly improve model performance (Rao et al., 2020; Hengl et al., 2017). We therefore pre-trained Presto on a diverse set of directly-sensed and derived Earth observation products which we pre-processed and exported using Google Earth Engine (Gorelick et al., 2017). A pre-training batch contained several pixel-timeseries samples, each of which is a concatenation of dynamic-in-time datapoints with each timestep representing a month (yielding T = 12 timesteps in total). The following dynamic-in-time data products were used, yielding 15 channels: (i) Sentinel-2 (S2) multispectral data, (ii) Sentinel-1 (S1) radar data, (iii) ERA5 climate reanalysis data, (iv) NDVI (Rouse et al., 1974) derived from Sentinel-2 data and (v) land cover classes V from Dynamic World. To every pixel-timeseries we appended two static-in-time products: (i) topography data from the SRTM digital elevation model (90m Digital Elevation Data, 2003) and (ii) location coordinates of each pixel. Hence, one pre-training sample x, comprising a pixel-timeseries t ∈ [RT ×15 ; V T ×1 ] and static variables s ∈ R1×5 , is summarized as follows: h                                                              i x = tS1       S2   ERA5 i ; ti ; ti    ; tNDVI i    ; tDW i  | i = 1, ..., 12 ; sTG ; sLoc              (1) From now on, we use “pixel-timeseries” to refer to both the dynamic and the static variables.
ZH: [待翻译 / pending]

### [S005] 3.2   Encoding and tokenization
**3.2 编码与标记化**

**[C016]**
EN: We transformed the pixel-timeseries x into a number of tokens (each represented by an embedding e) to be processed by the Presto transformer. Per timestep 0 ≤ i < T , we split the input variables into channel groups C according to their type of sensor or source: e.g., the S1 bands form one channel group. We describe these groups in more detail in Appendix A.1.3. Each real-valued channel group represents a different sensor, native spatial resolution or (in the case of Sentinel-2 channel-groups) region of the electromagnetic spectrum. We projected each channel group to a common latent space of dimension de by separate learned linear projections hC : e.g., eS1      S1 S1 i = h (ti ). The Dynamic World classes are categorical, so we embedded them by indexing them into an embedding matrix. Unlike natural images in which the data and its label are self-contained, remote sensing labels are inherently associated to a place and time on Earth (i.e., a latitude/longitude and timestamp). In addition, while natural images contain RGB channels from the same camera sensor, Presto’s pixel- timeseries input contains channels from multiple remote sensing instruments and data products. We therefore wanted to communicate to the model: (i) the location of the datapoint (already present in
ZH: 我们将像素时间序列 x 转换为多个标记（每个由嵌入 e 表示），供 Transformer 处理。每个标记携带关于（i）变量值的信息，（ii）其时间戳，以及（iii）其通道组。我们将位置坐标 sLoc 作为静态变量输入。


---

<!-- Page 5 -->
## Page 5

**[T001] Table / 表**
> **EN**: Table 1: We evaluated Presto on a wide variety of downstream tasks, including segmentation (seg.), multi-label (ml) scene classification (class.) and regression (reg.) tasks. There is diversity in terms of data composition, geographic area and training set size. Input shape describes the shape of a single sample, in terms of [Height, Width, Timesteps, Channels]. We bold the temporal dimension, to highlight time-series versus single-timestep inputs.
> **ZH**: 表1：我们在多种下游任务上评估了 Presto，包括分割（seg.）、多标签（ml）场景分类（class.）和回归（reg.）任务。这些任务在数据组成、地理区域和训练集大小方面具有多样性。输入形状描述了单个样本的形状，以 [高度, 宽度, 时间步, 通道] 表示。我们加粗时间维度，以突出时间序列与单时间步输入的区别。

**[C017]**
EN: Input shape       Train Dataset            Task        Region [H, W, T, C]      samples Kenya                           1,345 CropHarvest        Seg.        Brazil      [1, 1, 12, 18]        203 Togo                           1,319 S2-Agri100         Class.      France      [5, 5, 24, 10]      1,500 ML                         [6, 6, 1, 2] TreeSat                     Germany                           45,337 Class.                   [6, 6, 1, 11] [64, 64, 1, 3] EuroSat            Class.      Europe                         21,600 [64, 64, 1, 11] Fuel Moisture      Reg.          USA        [1, 1, 3, 19]      1,578 Algae Blooms       Reg.          USA       [1, 1, 12, 19]        777
ZH: [表格数据保留原文，见源文件]

**[C018]**
EN: the input as static variable through coordinates sLoc ) and a variable’s (ii) timestamp and (iii) channel group. We did this by adding encodings to the previously described embeddings e. The complete encoding has dimension de and contains a concatenation of positional, month, and learned channel encodings described below. • Positional: We used the sinusoidal positional encoding originally used by Vaswani et al. (2017). • Month: We added an encoding representing the month being captured by each token, because we expect timesteps from similar months to have similar features even if they are from different years. We assign an integer to each month ranging from 0 to 11, yielding: pmonth,2i = sin ((2π × month)/12)                                   (2) pmonth,2i+1 = cos ((2π × month)/12)                                   (3) For static-in-time variables, the positional and month encodings were set to zero. • Channel Group: Each token is associated with a set of input channels. In multispectral SatMAE (Cong et al., 2022), a fixed encoding was used to communicate input-band information with different channels representing different wavelengths, which is possible because only input data from one sensor (Sentinel-2) is used. However, since Presto’s input data includes multiple remote sensing products, we applied a learnable encoding for each channel group from the set of possible channel groups C = {S1, S2 RGB, ..., ERA5, TG, Loc}. The transformer input E ∈ R(T ·|Cdynamic |+|Cstatic |)×de (for encoder dimension de ) is a concatenation of: • Dynamic variables, for timesteps i < T and channel groups c ∈ C: eci = hc (tci ) + [pcchannel ; psin(i) ; pmonth(i) ] • Topographical data: eTG = hTG (sTG ) + [pTG channel ; 0; 0] • Coordinates: eLoc = hLoc (sLoc )
ZH: 通过坐标 sLoc 将位置作为静态变量输入，以及变量的（ii）时间戳和（iii）通道组。

### [S006] 3.3   Pre-training via Structured Masking
**3.3 通过结构性掩码进行预训练**

**[C019]**
EN: A key requirement for Presto was to perform well even with incomplete inputs (i.e., when there are missing timesteps, channels, or both). When masking out part of the input x, we therefore tailored the masking strategies to encourage the model to learn representations that perform well when given a subset of bands or timesteps for downstream tasks. For a T × D input of T timesteps and D total input channels, we used the following masking techniques (illustrated in Figure 1), where Presto considers a token to be a 1 × d input (a single timestep of d grouped channels). The coordinates were never masked but the static topological tokens can be. 1. Random: (t × d) masked values, with t < T and d < D
ZH: Presto 的一个关键要求是即使在输入不完整的情况下（即存在缺失时间步、通道或传感器时）也能表现良好。为确保模型学习处理此类输入，我们在预训练期间对输入进行结构性掩码。我们使用四种掩码策略：1. 随机空间掩码；2. 通道组掩码；3. 连续时间步掩码；4. 完整时间步掩码。


---

<!-- Page 6 -->
## Page 6

**[T002] Table / 表**
> **EN**: Table 2: Mean F1 score across all CropHarvest tasks. Presto outpeforms TIML (Tseng et al., 2022) and MOSAIKS-1D while requiring the adaptation of far fewer parameters. The TIML and MOSAIKS-1D model did not receive Dynamic World as input, so we measured Presto’s performance both with and without it.
> **ZH**: 表2：所有 CropHarvest 任务的平均 F1 分数。Presto 优于 TIML（Tseng 等，2022）和 MOSAIKS-1D，同时仅需少量可训练参数。

**[C020]**
EN: #. parameters Model              Total Adapted     Mean F1 Random Forest                           0.441 MOSAIKS-1DR       418K        8193      0.738 TIML               91K         91K      0.802 PrestoR                                 0.835 402K         129 no DW                               0.836
ZH: [表格数据保留原文]

**[F003] Figure / 图**
> **EN**: Figure 3: Presto is robust to incomplete inputs. We measured the AUC ROC score of Presto with Linear probing (PrestoR ) on the CropHarvest dataset when no Dynamic World input is passed, and with a subset of input months (the x-axis). We plot the performance of MOSAIKS-1D and TIML when they receive the full 12 months of input (dashed horizontal lines) - PrestoR recovered the performance of these models given only a subset of input months.
> **ZH**: 图3：Presto 对不完整输入具有鲁棒性。我们测量了 Presto 在线性探测（PrestoR）下的 AUC ROC 分数，同时随机移除输入时间步或通道组。

**[C021]**
EN: 2. Channel-groups: (T × d) masked values, with d < D 3. Contiguous timesteps: (t × D) masked values, t < T 4. Timesteps: (t × D) masked values, with t < T For each training instance, we randomly sampled from the above strategies to construct a mask. To handle both the categorical and continuous inputs we used the following loss function, which balances the continuous and categorical losses for every batch so that each reconstructed value receives the same weighting in the final loss: Ltotal = LMSE + λ NNcont cat LCE . LMSE is the mean squared error reconstruction loss used for the continuous values, LCE is the cross entropy loss used for the categorical values, Ncont is the number of masked continuous values and Ncat is the number of masked categorical values in the batch. λ is a hyperparameter, which we set to 2.
ZH: [待翻译 / pending]

### [S007] 4   Experiments
**4 实验**

**[C022]**
EN: In all experiments described below, we use a Presto model with identical encoder and decoder configurations (2 attention layers with 8 heads, an embedding size of 128 and an MLP ratio of 4). We investigated the effect of different encoder configurations in Table 8. For downstream evaluation, we took the encoder-decoder model learned during pre-training and discarded the decoder. As in He et al. (2022), we passed a global pool of all the encoder’s output tokens to a downstream classifier. We evaluated the performance of three different models: PrestoR , PrestoRF , and PrestoF T , defined below.
ZH: 在下面描述的所有实验中，我们使用编码器和解码器配置相同的 Presto 模型（2个注意力层，8个注意力头，宽度128）。


---

<!-- Page 7 -->
## Page 7

**[F004] Figure / 图**
> **EN**: Figure 4: We obtained per-image predictions using Presto by computing a mean and standard deviation of Presto’s per-pixel outputs, and passing this concatenated vector to a downstream classifier. We illustrate this for the EuroSat task.
> **ZH**: 图4：我们通过计算 Presto 逐像素输出的均值和标准差，并将该拼接向量传递给下游分类器，从而获得每张图像的预测。

- [C023] **EN**: • Feature extraction. Rolf et al. (2021) demonstrated the utility of neural networks as feature- extractors on top of which computationally efficient classifiers could be trained. PrestoR and PrestoRF consist respectively of linear or logistic regressions and random forests trained on Presto’s embeddings. Since only the regression/random forest is trained, this a computationally efficient method for adapting Presto to a wide range of tasks. • Fine-tuning. PrestoF T consists of the Presto encoder, followed by a linear transformation of the pooled tokens to the desired outputs. This entire model (the encoder and the linear transformation) is fine-tuned on the training data from each evaluation task. We used a subset of the (downstream) training data for validation. During pre-training, we used a validation task consisting of classifying all points in the CropHarvest dataset (Tseng et al., 2021) according to their FAO indicative crop classifications. For this validation task, we excluded points used for evaluation (Section 5.1). For evaluation, we compared Presto to state-of-the-art task-specific baselines (Section 5). Because there are no other global self-supervised models for pixel-timeseries, we adapted MOSAIKS (Rolf et al., 2021) for timeseries data by performing convolutions over the temporal rather than spatial dimension (MOSAIKS-1D). We used the output features with random forests (MOSAIKS-1DRF ) and regressions (MOSAIKS-1DR ).

### [S008] 5     Evaluation Tasks & Results
**5 评估任务与结果**

**[C024]**
EN: We evaluated Presto using six evaluation tasks spanning diverse task types, geographic locations (4 continents and 38 countries), input data modalities, and fine-tuning dataset sizes (Table 1). Whenever possible, we benchmarked Presto against the state-of-the-art model for that task. Applying Presto to downstream tasks is computationally efficient. While other methods require a cluster of GPUs for fine-tuning (Cong et al., 2022), we fine-tuned Presto on a single GPU or CPU. For the fuel moisture task described in Section 5.1, fine-tuning Presto took under 6 minutes on a 2017 MacBook Pro’s CPU. When Presto is used as a feature extractor, simple models can be trained which require few parameters to be learned, as we show in Table 2. Even when fully fine-tuned, Presto’s small size meant that relatively few parameters needed to be trained (Tables 5 and 6). This makes Presto accessible to practitioners, especially those lacking significant computational resources. Below, we describe the tasks used to evaluate Presto and discuss Presto’s performance on these tasks.
ZH: 我们使用六个评估任务评估 Presto，涵盖多样的任务类型、地理位置（4个大洲和38个国家）和数据集大小。

### [S009] 5.1   Timeseries Tasks
**5.1 时间序列任务**

- [C025] **EN**: • Crop type Segmentation: The CropHarvest (Tseng et al., 2021) evaluation datasets consist of binary pixel classification of (i) maize in Kenya, (ii) coffee in Brazil and (iii) cropland in Togo. We compared Presto to the baselines provided by CropHarvest and to Task-Informed Meta-Learning (TIML, Tseng et al., 2022), which achieved state-of-the-art results on these datasets.
  **ZH**: • 作物类型分割：CropHarvest（Tseng 等，2021）评估数据集由肯尼亚、巴西、多哥和布基纳法索的二元像素分类任务组成。


---

<!-- Page 8 -->
## Page 8

**[T003] Table / 表**
> **EN**: Table 3: RMSE results on the regression tasks. The literature baselines are not directly comparable, since they use different input datasets or private test data (or both). Rao et al. (2020) reported an RMSE of 25 on the fuel moisture dataset with a physics-assisted neural network and the algae bloom competition winner reported an RMSE of 0.761, indicating our results are within the scope of utility. Best results are highlighted blue, with second best results in bold. Models have a high variance in performance across tasks, so we calculated the mean difference in RMSE from the linear regression baseline across both tasks. Presto performed most consistently, both when used as a feature-extractor and when fine-tuned.
> **ZH**: 表3：回归任务的 RMSE 结果。文献基线不直接可比，因为它们使用不同的输入数据。

**[C026]**
EN: Fuel        Algae          Mean Moisture    Blooms        difference Linear Regression             28.20       0.850            0% Random Forest                23.84         1.249        15.7% MOSAIKS-1DRF                  28.75        0.972        8.15% PrestoF T (random init.)      26.07        0.955        2.40% PrestoF T                    25.28        0.815      −7.24% PrestoRF                      25.98        0.884     −1.94%
ZH: [表格数据保留原文]

**[T004] Table / 表**
> **EN**: Table 4: Results on the TreeSatAI dataset. We compared Presto to the dataset’s benchmark models. The MLPs contain 3 layers (with 563K-723K parameters respectively) and are tuned for this task. We froze the Presto encoder’s 402k parameters and trained a random forest on its outputs with default scikit-learn hyperparameters.
> **ZH**: 表4：TreeSatAI 数据集的结果。我们将 Presto 与数据集的基准模型进行比较。MLP 包含3个隐藏层。

**[C027]**
EN: Weighted             Micro Model         Data        F1     mAP          F1     mAP MLP                    10.09     29.42      12.82      33.09 LightGBM      S1       11.86     32.79      14.07      35.11 PrestoRF              38.34     35.45      40.79      38.64 MLP                    51.97    64.19       54.59     65.83 LightGBM      S2       48.17     61.99      52.52      61.66 PrestoRF              55.29      61.53     58.29       63.31
ZH: [表格数据保留原文]

- [C028] **EN**: • Fuel Moisture: The live fuel moisture dataset (Rao et al., 2020) measures live fuel moisture content in the Western U.S. Rao et al. (2020)’s baseline used 5-fold cross validation to evaluate model performance; for future comparability, we used a single geographically separated test set (a test set covering a different geographic area than the training set). • Algae Blooms: The algae blooms dataset (alg, 2023) measures the severity of cyanobacterial algal blooms in different parts of the U.S. We used the subset of the dataset in the Midwestern U.S. The dataset was originally released as part of a competition, so the test data is not available. In addition, competitors could download many Earth observation datasets to train their models, making direct comparisons to competition results difficult. Since the competition’s winning solution used a tree- based method, we benchmarked against a regression and a random forest using a geographically separated test set.
  **ZH**: • 燃料湿度：活燃料湿度数据集（Rao 等，2020）测量活燃料湿度含量，这是野火风险的关键指标。

**[C029]**
EN: 5.1.1   Timeseries Results
ZH: 5.1.1 时间序列结果

**[C030]**
EN: Presto excels at timeseries tasks, significantly outperforming the state-of-the-art for CropHarvest (Table 2) and outperforming all baselines for the regression tasks (Table 3). We found that Presto is performant when passed only a subset of timesteps compared to the 12 timesteps used for pre-training. Presto remained performant when receiving only 3 input timesteps for the fuel moisture task (Table 3). We also evaluated Presto when a subset of input months are passed for the CropHarvest dataset (Figure 3). Using a subset of the 12 months, Presto surpassed the performance of TIML and MOSAIKS-1D which used all input months.
ZH: Presto 在时间序列任务上表现出色，显著优于 CropHarvest 的现有最先进技术，同时需要更少的参数和计算。


---

<!-- Page 9 -->
## Page 9

**[F005] Figure / 图**
> **EN**: Figure 5: EuroSat accuracy of a kNN@5 classifier given pre-trained model embeddings at a variety of input resolutions (following Reed et al. (2022)) as a function of FLOPs required to encode an image (note the log scale on the x-axes). All image-based models resized images to 224 × 224, so the FLOPs required to encode an image do not change with image resolution. Presto achieved competitive results with image-based models while requiring up to four orders of magnitude less FLOPs to encode an image.
> **ZH**: 图5：给定预训练模型嵌入时，kNN@5 分类器在 EuroSat 上的准确率随输入分辨率（遵循 Reed 等（2022））和编码单个图像所需 FLOPs 的变化关系。

**[C031]**
EN: Presto is also robust to the removal of input channels. On the CropHarvest dataset (Table 2), Presto remained performant without the Dynamic World input, showing a negligible difference in mean F1 score compared to the full input.
ZH: [待翻译 / pending]

### [S010] 5.2     Image-based Tasks
**5.2 基于图像的任务**

**[C032]**
EN: Presto is designed to ingest single pixel-timeseries. When one prediction is required for a set of pixels (as for image-based tasks and the Image-Timeseries tasks in Section 5.3), we used the following approach to obtain per-image predictions from Presto’s pixel outputs (Figure 4): (i) we encoded the pixels in an image individually, yielding N output tokens, (ii) we calculated the mean and standard deviation of these N output tokens per dimension and concatenated the result, yielding a 2de -dimensional vector (where de is Presto’s output token size, or 128), and (iii) we passed this mean and standard deviation vector to a downstream classifier. • TreeSatAI: The TreeSatAI dataset consists of detecting the presence of one or more tree species (out of 20 possible species) in forestry images in Germany (Ahlswede et al., 2023). We used the train and test splits provided by Ahlswede et al. (2023) and compared Presto to the deep learning and tree-based baselines provided with the dataset. As done for the baselines, we evaluated models using only Sentinel-2 (S2) or Sentinel-1 (S1) data. • EuroSAT: The EuroSAT dataset classifies Sentinel-2 multispectral images in Europe with one of 10 landcover classes (Helber et al., 2019). We used the train and test splits provided by Neumann et al. (2019). We compared Presto to SatMAE, ConvMAE and ScaleMAE using a k Nearest Neighbors (kNN) classifier at a variety of input resolutions, as was done by Reed et al. (2022). We also compared fine-tuned Presto against Seasonal Contrast (SeCo) (Manas et al., 2021) and Geography-Aware Self-Supervised Learning (GASSL) (Ayush et al., 2021). EuroSAT provides all multispectral Sentinel-2 bands, but most other models ingest only RGB images. We evaluated Presto both when it received all multispectral bands as input (MS) and when it only received the RGB bands.
ZH: Presto 设计用于摄取单像素时间序列。当需要对一组像素进行单一预测时（例如在图像分类任务中），我们独立处理每个像素，然后聚合结果。

### [C033] 5.2.1    Image-based Results
**5.2.1 基于图像的结果**

**[C034]**
EN: Despite being pre-trained on pixel-timeseries data, Presto is competitive on single-timestep image datasets against much larger models. We followed the setup of Reed et al. (2022) in measuring the performance of a kNN-classifier on Presto’s output embeddings for the EuroSat dataset at varying resolutions. Presto achieved comparable average accuracy (over all image resolutions) to larger ViT-based models with RGB data and significantly outperformed these models with multispectral (MS) data (Figure 5), while requiring orders of magnitude less compute to encode the images in both cases and for any resolution. Presto is performant even when only a small subset of input channels are available compared to the pre-training channels. For the EuroSAT task (Table 5), Presto received either the full Sentinel-2
ZH: 尽管仅在像素时间序列数据上预训练，Presto 在单时间步图像数据集上仍具有竞争力。


---

<!-- Page 10 -->
## Page 10

**[T005] Table / 表**
> **EN**: Table 5: EuroSAT finetuning accuracy. Presto is the only backbone that can handle both MS and RGB inputs (separate SatMAE models are trained for RGB and MS inputs). We reported Presto results for full resolution; results at reduced resolutions are in Table 11.
> **ZH**: 表5：EuroSAT 微调准确率。Presto 是唯一能同时处理多光谱（MS）和 RGB 输入的主干网络（SatMAE 的单独模型针对 MS 和 RGB 训练）。

**[C035]**
EN: Params Backbone        Inputs               Accuracy (M) GASSL       ResNet-18       RGB         11.69        0.895 SeCo        ResNet-18       RGB         11.69        0.931 SatMAE      ViT-Large       RGB        303.10        0.955 SatMAE      ViT-Large       MS         305.96        0.990 Random                      RGB                      0.745 Presto                       0.40 init.                     MS                       0.924 RGB                      0.849 Presto      Presto                       0.40 MS                       0.953
ZH: [表格数据保留原文]

**[T006] Table / 表**
> **EN**: Table 6: Results on the S2-Agri100 dataset. We followd (Yuan et al., 2022) in reporting overall accuracy (OA), Kappa Cohen score (κ) and macro-F1 score. All results are an average of 3 runs - standard errors are reported in Table 16.
> **ZH**: 表6：S2-Agri100 数据集的结果。我们遵循（Yuan 等，2022）报告总体准确率（OA）、Kappa 系数和 F1 分数。

**[C036]**
EN: Params       Pre OA        κ       F1 (M)       Trained? SITS                                  65.13   0.55     42.12 2.5 Former                    ✓            67.03   0.56    42.83 45.98      0.35    27.45 Presto       0.4 ✓         68.89      0.58     40.41
ZH: [表格数据保留原文]

**[C037]**
EN: input or only RGB bands (which represent only a single token, since only one timestep is available). Similarly, we evaluated Presto when it receives either Sentinel-2 or Sentinel-1 data for the TreeSatAI task (Table 4). In both cases, Presto was competitive with methods designed to ingest single-timestep, single-sensor data.
ZH: 输入或仅 RGB 波段（由于只有一个时间步可用，仅表示单个标记）。

### [S011] 5.3     Image-Timeseries Tasks
**5.3 图像-时间序列任务**

- [C038] **EN**: • S2-Agri100 : The S2-Agri dataset (Sainte Fare Garnot et al., 2020) classifies crop types in agricul- tural parcels. We used a variant of S2-Agri (S2-Agri100 ) developed by Yuan et al. (2022) for the SITS-Former model in which 100 parcels for each crop type are used for training and validation respectively (all other parcels are used for testing), and a 5 × 5 pixel patch from each parcel is used for input. We benchmarked Presto against both the pre-trained and randomly initialized SITS-Former model.
  **ZH**: • S2-Agri100：S2-Agri 数据集（Sainte Fare Garnot 等，2020）对法国农业区的作物类型进行分类。

**[C039]**
EN: 5.3.1    Image-Timeseries Results
ZH: 5.3.1 图像-时间序列结果

**[C040]**
EN: The S2-Agri100 dataset consists of 24 timesteps at 10 to 30 day intervals (compared to Presto’s pre-training data, which consists of 12-month timeseries). Presto remained performant on this dataset, achieving comparable results with SITS-Former despite having 6× fewer parameters (shown in Table 6). This shows that Presto can ingest timeseries at different temporal resolutions and at varying intervals. In addition, the S2-Agri dataset is missing pixel location metadata, which is always passed to Presto during pre-training. S2-Agri was sampled from a single S2-tile, so we used the location of the central pixel of this tile for all pixels in the dataset. Even with this much less accurate location metadata, Presto remained performant.
ZH: S2-Agri100 数据集包含24个时间步，间隔10至30天（相比 Presto 的预训练数据每月一个时间步）。


---

<!-- Page 11 -->
## Page 11

**[T007] Table / 表**
> **EN**: Table 7: Structured masking strategies yield the best downstream performance. We measured PrestoR ’s F1 score on the CropHarvest validation task. Combining structured strategies outperformed the “Random” masking employed by (He et al., 2022).
> **ZH**: 表7：结构性掩码策略产生最佳下游性能。我们在 CropHarvest 验证集上测量 PrestoR 的 F1 分数。

**[C041]**
EN: Channel                             Contiguous       F1 Random     Timesteps Groups                              Timesteps       Score ✓                                               0.646 ✓                                    0.653 ✓                        0.664 ✓          0.649 ✓          ✓           ✓             ✓         0.665
ZH: [表格数据保留原文]

### [S012] 5.4   Ablations
**5.4 消融实验**

**[C042]**
EN: We conducted three ablations to better understand Presto’s performance: • Structured masking strategies perform best: Table 7 shows results from ablating the masking strategies. Unlike other masked autoencoder methods (He et al., 2022), we found that combining structured masking with random masking outperforms random masking alone. • Pre-training Presto is critical to achieve strong performance: In Tables 3, 5 and Table 6, we compared the performance of a randomly-initialized Presto architecture with the pre-trained model. Pre-training yielded a significant increase in performance (a 50% increase in accuracy on the S2-Agri100 dataset). Even when the downstream training dataset size was large (EuroSat has 21,600 training samples), pre-training yielded a 14% increase in accuracy given RGB inputs and up to 22% increase in accuracy at lower resolutions (Table 11). For TreeSatAI with S1 data (Table 15), a randomly initialized model slightly outperformed the pre-trained model. We hypothesize that this is due to the difference in input relative to the pre-training data, since the TreetSatAI input consists of a single image from only one timestep and one channel group. • Presto’s performance scales with model size: To measure how different model sizes affect Presto’s performance, we pre-trained two larger Presto variants: a deeper variant with 4 encoder layers instead of 2, and a wider variant with a doubled encoder size (Table 8). Performance improved as model size increased, suggesting that practitioners who can afford greater computational costs could obtain better results by training a larger Presto model.
ZH: 我们进行了三项消融实验以更好理解 Presto 的性能：• 结构性掩码策略的效果；• 模型大小的影响；• 预训练与从头训练的比较。

### [S013] 6     Discussion & Conclusion
**6 讨论与结论**

**[C043]**
EN: Limitations Presto is designed to ingest 10m/px resolution imagery and is pre-trained on products at this scale. This decision is motivated by the free, global availability over time of products at this scale (such as Sentinel-1 and Sentinel-2). Presto does not natively process very-high resolution imagery such as < 1 m/px imagery from commercial satellites or drones, which can be costly and often lack complete coverage globally and temporally. In addition, Presto is a pixel-timeseries model. While we demonstrated Presto’s flexibility on single-timestep image datasets, image-based models may be preferred if a user’s goal is to process entire images to make a prediction. We observed that Presto’s performance on the EuroSAT dataset plateaued as the input resolution increased (Table 5), due to images from classes where the relevant pixels for the class are a minority of the pixels in the image (e.g., highways). In such scene classification challenges, image-based models which can learn the shape of the relevant pixels may be better suited. We discuss this further in Section A.6.
ZH: 局限性 Presto 设计用于摄取 10m/px 分辨率影像，并在该分辨率的产品上预训练。虽然 10m 分辨率对许多遥感应用来说是合理的默认选择，但某些任务需要更高或更低的分辨率。

**[C044]**
EN: Conclusion We present Presto: a lightweight, pre-trained timeseries transformer for remote sensing. By leveraging structure unique to remote sensing data—specifically, (i) an important temporal dimension, (ii) associated metadata and (iii) a diversity of sensors, we are able to train an extremely lightweight model which achieves state-of-the-art results in a wide variety of globally distributed evaluation tasks. Computational efficiency is of paramount importance in remote sensing settings and often determines which models ultimately get selected for deployment. We demonstrated that strong performance can be achieved while meeting this constraint, and that self-supervised learning can provide significant benefits even for small models.
ZH: 结论 我们提出 Presto：一种面向遥感的轻量级预训练时间序列 Transformer。Presto 专为遥感数据设计，比计算机视觉模型小得多，同时保持竞争力。Presto 可灵活处理缺失数据，并可用于迁移学习或特征提取。


---

<!-- Page 12 -->
## Page 12

**[T008] Table / 表**
> **EN**: Table 8: Effect of model size on validation performance. To understand the effect of model size on performance, we pre-train two larger variants of Presto. As in Table 7, we measure PrestoR ’s performance on the CropHarvest validation task. The number of parameters includes both the encoder and decoder parameters. The FLOPS are computed for a “full” input (12 timesteps, with no missing channels), when passed through the encoder and decoder.
> **ZH**: 表8：模型大小对验证性能的影响。为理解模型大小对性能的影响，我们预训练了两个更大的变体。

**[C045]**
EN: # params    FLOPs      F1 Depth   Width (M)       (M)      score 2      128       0.81     88.94   0.665 2      256       2.02    220.81   0.687 4      128       1.21    132.42   0.669
ZH: [表格数据保留原文]

**[C046]**
EN: Impact statement
ZH: 影响声明

**[C047]**
EN: Machine learning applications to remote sensing have a wide range of societally beneficial outcomes, ranging from tracking progress on sustainable development goals (Ferreira et al., 2020) to improved weather forecasting (English et al., 2013; Voosen, 2020) to disaster management (Kansakar and Hossain, 2016). Presto is designed to be accessible to a wide range of practitioners; we achieve this by only training Presto on publicly available data and by keeping the model size small enough so it can be leveraged in compute-constrained environments. In addition to increasing Presto’s accessibility, its small size also lowers its carbon footprint (Strubell et al., 2019). As described by Tuia et al. (2023), a natural concern when applying machine learning algorithms to remote sensing data is its use to collect information about individuals who are unaware that data is being collected, and therefore cannot consent to this practice. We therefore encourage deployment of Presto in collaboration with local communities and stakeholders (Krafft; Kshirsagar et al., 2021; Nakalembe and Kerner, 2023).
ZH: 面向遥感的机器学习应用具有广泛的社会效益，包括改善粮食安全、灾害响应和气候监测。通过开发更小、更高效的模型，我们使这些技术更易于获取和部署，包括在资源受限的环境中。

**[C048]**
EN: Acknowledgements
ZH: 致谢

**[C049]**
EN: This work was supported by NASA under the NASA Harvest Consortium on Food Security and Agriculture (Award #80NSSC18M0039). This research was enabled in part by compute resources provided by Mila (mila.quebec); in addition, we acknowledge material support from NVIDIA Corporation in the form of computational resources. We thank Esther Rolf and Caleb Robinson for reviewing drafts of this manuscript.
ZH: 本工作得到 NASA Harvest 粮食安全与农业联盟（NASA 应用科学项目，编号 80NSSC18K0654）的支持。

### [S014] References
**参考文献**

**[C050]**
EN: Earth engine data catalogue. https://developers.google.com/earth-engine/datasets/ catalog. Accessed: 2023-01-31. Tick      tick    bloom:          Harmful        algal     bloom     detection             challenge. https://www.drivendata.org/competitions/143/tick-tick-bloom/page/649/, 2023.             Accessed: 2023-03-10. S. 90m Digital Elevation Data. The CGIAR consortium for spatial information, 2003. C. Abys, S. Skakun, and I. Becker-Reshef. Two decades of winter wheat expansion and intensification in russia. Remote Sensing Applications: Society and Environment, 2024. S. Ahlswede, C. Schulz, C. Gava, P. Helber, B. Bischke, M. Förster, F. Arias, J. Hees, B. Demir, and B. Kleinschmit. Treesatai benchmark archive: A multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data, 2023. K. Ayush, B. Uzkent, C. Meng, K. Tanmay, M. Burke, D. Lobell, and S. Ermon. Geography-aware self-supervised learning. In CVPR, 2021. N. H. Batjes, E. Ribeiro, A. Van Oostrum, J. Leenaars, T. Hengl, and J. Mendes de Jesus. Wosis: providing standardised soil profile data for the world. Earth System Science Data, 2017.
ZH: [参考文献保留原文]


---

<!-- Page 13 -->
## Page 13

**[C051]**
EN: V. Böhm, W. J. Leong, R. B. Mahesh, I. Prapas, E. Nemni, F. Kalaitzis, S. Ganju, and R. Ramos- Pollan. Sar-based landslide classification pretraining leads to better segmentation. In Artificial Intelligence for Humanitarian Assistance and Disaster Response Workshop at NeurIPS, 2022. P. O. Bressan, J. M. Junior, J. A. C. Martins, M. J. de Melo, D. N. Gonçalves, D. M. Freitas, A. P. M. Ramos, M. T. G. Furuya, L. P. Osco, J. de Andrade Silva, et al. Semantic segmentation with labeling uncertainty and class imbalance applied to vegetation mapping. International Journal of Applied Earth Observation and Geoinformation, 2022. C. F. Brown, S. P. Brumby, B. Guzder-Williams, T. Birch, S. B. Hyde, J. Mazzariello, W. Czerwinski, V. J. Pasquarella, R. Haertel, S. Ilyushchenko, K. Schwehr, M. Weisse, F. Stolle, C. Hanson, O. Guinan, R. Moore, and A. M. Tait. Dynamic world, near real-time global 10 m land use land cover mapping. Scientific Data, Jun 2022. Y. Cong, S. Khanna, C. Meng, P. Liu, E. Rozi, Y. He, M. Burke, D. B. Lobell, and S. Ermon. SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, NeurIPS, 2022. URL https://openreview. net/forum?id=WBhqzpF6KYH. S. Di Tommaso, S. Wang, V. Vajipey, N. Gorelick, R. Strey, and D. B. Lobell. Annual field- scale maps of tall and short crops at the global scale using gedi and sentinel-2. arXiv preprint arXiv:2212.09681, 2022. S. English, T. McNally, N. Bormann, K. Salonen, M. Matricardi, A. Moranyi, M. Rennie, M. Janisková, S. Di Michele, A. Geer, et al. Impact of satellite data, 2013. B. Ferreira, M. Iten, and R. G. Silva. Monitoring sustainable development by means of earth observation data and machine learning: A review. Environmental Sciences Europe, 2020. A. Fuller, K. Millard, and J. R. Green. CROMA: Remote sensing representations with contrastive radar-optical masked autoencoders. In Thirty-seventh Conference on Neural Information Process- ing Systems, 2023. URL https://openreview.net/forum?id=ezqI5WgGvY. P. Gao, T. Ma, H. Li, Z. Lin, J. Dai, and Y. Qiao. Convmae: Masked convolution meets masked autoencoders. arXiv preprint arXiv:2205.03892, 2022. N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore. Google earth engine: Planetary-scale geospatial analysis for everyone. Remote sensing of Environment, 2017. M. C. Hansen, P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland, et al. High-resolution global maps of 21st-century forest cover change. Science, 2013. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. P. Helber, B. Bischke, A. Dengel, and D. Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019. T. Hengl, J. Mendes de Jesus, G. B. Heuvelink, M. Ruiperez Gonzalez, M. Kilibarda, A. Blagotić, W. Shangguan, M. N. Wright, X. Geng, B. Bauer-Marschallinger, et al. Soilgrids250m: Global gridded soil information based on machine learning. PLoS one, 2017. N. Jean, S. Wang, A. Samar, G. Azzari, D. Lobell, and S. Ermon. Tile2vec: Unsupervised representa- tion learning for spatially distributed data. In AAAI, 2019. P. Kansakar and F. Hossain. A review of applications of satellite earth observation data for global societal benefit and stewardship of planet earth. Space Policy, 2016. H. Kerner, G. Tseng, I. Becker-Reshef, C. Nakalembe, B. Barker, B. Munshell, M. Paliyam, and M. Hosseini. Rapid response crop maps in data sparse regions. In ACM SIGKDD Conference on Data Mining and Knowledge Discovery Workshops, 2020. A. Krafft. ASU researcher combats food insecurity with AI. https://news.asu.edu/20230303-solutions- asu-researcher-combats-food-insecurity-ai. Accessed: 2023-09-21. M. Kshirsagar, C. Robinson, S. Yang, S. Gholami, I. Klyuzhin, S. Mukherjee, M. Nasir, A. Ortiz, F. Oviedo, D. Tanner, et al. Becoming good at ai for good. In AAAI/ACM Conference on AI, Ethics, and Society, 2021.
ZH: [待翻译 / pending]


---

<!-- Page 14 -->
## Page 14

**[C052]**
EN: A. Lacoste, N. Lehmann, P. Rodriguez, E. D. Sherwin, H. Kerner, B. Lütjens, J. A. Irvin, D. Dao, H. Alemohammad, A. Drouin, et al. Geo-bench: Toward foundation models for earth monitoring. arXiv preprint arXiv:2306.03831, 2023. O. Manas, A. Lacoste, X. Giró-i Nieto, D. Vazquez, and P. Rodriguez. Seasonal contrast: Unsuper- vised pre-training from uncurated remote sensing data. In CVPR, 2021. C. Nakalembe and H. Kerner. Considerations for ai-eo for agriculture in sub-saharan africa. Environ- mental Research Letters, 2023. C. Nakalembe, C. Justice, H. Kerner, C. Justice, and I. Becker-Reshef. Sowing seeds of food security in africa. Eos (Washington. DC), 102, 2021. M. Neumann, A. S. Pinto, X. Zhai, and N. Houlsby. In-domain representation learning for remote sensing. arXiv preprint arXiv:1911.06721, 2019. C. Parkinson, A. Ward, and M. King. Earth science reference handbook. National Aeronautics and Space Administration: Washington, DC, USA, 2006. C. Pelletier, G. I. Webb, and F. Petitjean. Temporal convolutional neural network for the classification of satellite image time series. Remote Sensing, 2019. K. Rao, A. P. Williams, J. F. Flefil, and A. G. Konings. Sar-enhanced mapping of live fuel moisture content. Remote Sensing of Environment, 2020. C. J. Reed, R. Gupta, S. Li, S. Brockman, C. Funk, B. Clipp, S. Candido, M. Uyttendaele, and T. Darrell. Scale-mae: A scale-aware masked autoencoder for multiscale geospatial representation learning. arXiv preprint arXiv:2212.14532, 2022. C. Robinson, L. Hou, K. Malkin, R. Soobitsky, J. Czawlytko, B. Dilkina, and N. Jojic. Large scale high-resolution land cover mapping with multi-resolution data. In CVPR, 2019. E. Rolf, J. Proctor, T. Carleton, I. Bolliger, V. Shankar, M. Ishihara, B. Recht, and S. Hsiang. A generalizable and accessible approach to machine learning with global satellite imagery. Nature communications, 2021. J. W. Rouse, R. H. Haas, J. A. Schell, D. W. Deering, et al. Monitoring vegetation systems in the great plains with erts. NASA Spec. Publ, 351(1):309, 1974. M. Rußwurm and M. Körner. Self-attention for raw optical satellite time series classification. ISPRS journal of photogrammetry and remote sensing, 2020. M. Rußwurm, S. Wang, M. Korner, and D. Lobell. Meta-learning for few-shot land cover classification. In CVPR Workshops, pages 200–201, 2020. V. Sainte Fare Garnot, L. Landrieu, S. Giordano, and N. Chehata. Satellite image time series classification with pixel-set encoders and temporal self-attention. CVPR, 2020. E. Strubell, A. Ganesh, and A. McCallum. Energy and policy considerations for deep learning in nlp. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2019. M. Sudmanns, D. Tiede, H. Augustin, and S. Lang. Assessing global sentinel-2 coverage dynamics and data availability for operational earth observation (eo) applications using the eo-compass. International journal of digital earth, 2019. M. Tarasiou, E. Chavez, and S. Zafeiriou. ViTs for SITS: Vision Transformers for Satellite Image Time Series. In CVPR, 2023. G. Tseng, I. Zvonkov, C. L. Nakalembe, and H. Kerner. Cropharvest: A global dataset for crop-type classification. In NeurIPS, Datasets and Benchmarks Track, 2021. URL https://openreview. net/forum?id=JtjzUXPEaCu. G. Tseng, H. Kerner, and D. Rolnick. TIML: Task-informed meta-learning for crop type mapping. In AI for Agriculture and Food Systems at AAAI, 2022. D. Tuia, K. Schindler, B. Demir, G. Camps-Valls, X. X. Zhu, M. Kochupillai, S. Džeroski, J. N. van Rijn, H. H. Hoos, F. Del Frate, et al. Artificial intelligence to advance earth observation: a perspective. arXiv preprint arXiv:2305.08413, 2023. K. Van Tricht. Mapping crops at global scale! what works and what doesn’t? https://blog.vito. be/remotesensing/worldcereal-benchmarking, 2021. Accessed: 2023-07-31.
ZH: [待翻译 / pending]


---

<!-- Page 15 -->
## Page 15

**[C053]**
EN: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. NeurIPS, 2017. P. Voosen. Europe builds ‘digital twin’of earth to hone climate forecasts, 2020. S. Wang, W. Chen, S. M. Xie, G. Azzari, and D. B. Lobell. Weakly supervised deep learning for segmentation of remote sensing imagery. Remote Sensing, 2020a. S. Wang, S. Di Tommaso, J. M. Deines, and D. B. Lobell. Mapping twenty years of corn and soybean across the us midwest using the landsat archive. Scientific Data, 2020b. B. Yifang, P. Gong, and C. Gini. Global land cover mapping using earth observation satellite data: Recent progresses and challenges. ISPRS journal of photogrammetry and remote sensing, 2015. J. You, X. Li, M. Low, D. Lobell, and S. Ermon. Deep gaussian process for crop yield prediction based on remote sensing data. Proceedings of the AAAI Conference on Artificial Intelligence, 2017. Y. Yuan and L. Lin. Self-supervised pretraining of transformers for satellite image time series classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 14:474–487, 2020. Y. Yuan, L. Lin, Q. Liu, R. Hang, and Z.-G. Zhou. Sits-former: A pre-trained spatio-spectral-temporal representation model for sentinel-2 time series classification. International Journal of Applied Earth Observation and Geoinformation, 106:102651, 2022.
ZH: [待翻译 / pending]


---

<!-- Page 16 -->
## Page 16

**[C054]**
EN: A     Appendix
ZH: A 附录

### [S015] Reproducibility
**可复现性**

**[C055]**
EN: All code and data used to train and evaluate Presto will be made available upon publication, and the code is currently available at https://github.com/nasaharvest/presto. In addition, we discuss specific implementation details in Appendices A.1 and A.4. We have strived to make the Presto codebase accessible to other practitioners; to this end, we include a demo Jupyter notebook demonstrating how Presto can be applied to a new downstream task, which is available at https: //github.com/nasaharvest/presto/blob/main/downstream_task_demo.ipynb.
ZH: 用于训练和评估 Presto 的所有代码和数据将在发表后公开，代码当前可在 https://github.com/nasaharvest/presto 获取。

**[C056]**
EN: A.1     Pre-training details
ZH: A.1 预训练细节

**[C057]**
EN: We outline training hyperparameters below: • Training length: We train the model for 20 epochs, with a batch size of 4096 (resulting in 5950 batches per epoch). On a single NVIDIA V100 GPU, this takes 43 41 hours. • Optimizer and learning rate: We train the model with an AdamW optimizer. We use a cosine annealing schedule for our learning rate, with a maximum learning rate of 0.001 at the 2nd epoch. We apply a weight decay of 0.05, and βs of (0.9, 0.95). • Masking: We use a masking ratio of 0.75, randomly selecting (for each instance) a masking strategy from the ones described in Section 3.3. If the masking strategy cannot mask the right number of tokens, we randomly mask additional tokens to achieve the correct masking ratio.
ZH: 我们在下面概述训练超参数：• 训练长度：我们训练模型20个 epoch，使用...

**[C058]**
EN: A.1.1    Pre-training data
ZH: A.1.1 预训练数据

**[F006] Figure / 图**
> **EN**: Figure 6: The distribution of the pre-training dataset described in Section 3.1.
> **ZH**: 图6：第3.1节描述的预训练数据集的分布。

**[C059]**
EN: Remote sensing models can be deployed in a wide range of geographies, with few labelled datapoints available at fine-tuning time (Kerner et al., 2020; Böhm et al., 2022). We therefore aim to collect a globally representative pre-training dataset. We achieve this by following the sampling strategy used by Dynamic World (Brown et al., 2022). We divide the Earth into three regions: the Western Hemisphere and two regions in the Eastern Hemisphere. These regions are further divided into ecoregions, and stratified samples are gathered from each region using land cover classes as sampling strata. Figure 6 shows the resulting geographical distribution. Each sample represents a 510 × 510 pixel tile with a spatial resolution of 10 meter per pixel. To obtain pixel-timeseries we grid-sample 2,500 pixels from each sample, yielding a total of 21,535,000 pixel samples (each with 24 one-month timesteps).
ZH: 遥感模型可部署在广泛的地理区域，标注数据点很少...

**[C060]**
EN: A.1.2    Input data
ZH: A.1.2 输入数据

**[C061]**
EN: We leverage the following data products when pre-training Presto:
ZH: 我们在预训练 Presto 时使用以下数据产品：


---

<!-- Page 17 -->
## Page 17

**[T009] Table / 表**
> **EN**: Table 9: Model sizes and FLOPs required to encode a single EuroSat image (or pixel, for Presto), as measured by the thop library. When plotting results in Table 5, we multiply the FLOPs for Presto by the number of pixels encoded for an image. At its highest resolution, EuroSAT images are 64 × 64, so Presto FLOPs for a full resolution image can be obtained by multiplying the per-pixel FLOPs by 4,096. We include this value in brackets for completeness. Model                                    Backbone              Params (M)          MegaFlops SatMAE (RGB) (Cong et al., 2022)         ViT-Large                  303.10         59,685.69 SatMAE (MS) (Cong et al., 2022)          ViT-Large                  305.96        535,515.25 ScaleMAE (Reed et al., 2022)             ViT-Large                  303.10         59,685.69 ConvMAE (Gao et al., 2022)               ConvMAE-Large               88.78         23,315.58 SeCo (Manas et al., 2021)                ResNet-18                   11.69             149.37 GASSL (Ayush et al., 2021)               ResNet-18                   11.69             149.37 Presto RGB pixel (image)                 Presto                       0.40    0.79 (3,235.84) Presto MS pixel (image)                  Presto                       0.40    2.37 (9,707.52)
> **ZH**: 表9：编码单个 EuroSat 图像（或像素，对于 Presto）所需的模型大小和 FLOPs，由 thop 库测量。

- [C062] **EN**: • Sentinel-1 Synthetic Aperture Radar observations (S1): The VV (emit and receive at vertical polarization) and VH (emit at vertical and receive at horizontal polarization) bands: 2 real-valued dynamic values per monthly timestep. • Sentinel-2 Multispectral images (S2): We removed the 60m resolution bands, yielding bands with 10m and 20m resolution with channels in the visible, near-infrared and short-wave infrared range: 10 real-valued dynamic values per timestep. • ERA5 Climate Reanalysis Meteorological data (ERA5): Monthly total precipitation and temper- ature at 2 metres above the ground: 2 real-valued dynamic values per timestep. • NDVI (Rouse et al., 1974): Computed from the red (B4) and near-infrared (B8) Sentinel-2 bands: 1 real-valued dynamic value per timestep. • Dynamic World Land Cover classes (DW, Brown et al., 2022): Land cover classes produced for every non-cloudy Sentinel-2 image: 1 dynamic categorical value from the set of possible classes V per timestep. We took the mode of classes for all timesteps within a month. • Topography data (TG), from the Shuttle Radar Topography Mission’s Digital Elevation Model: The elevation and slope of each pixel, real-valued and static in time. • Coordinates (Loc): 3D static in time Cartesian coordinates computed from the latitude and longi- tude of the pixel’s geographical location: sLoc = [cos(lat) × cos(lon), cos(lat) × sin(lon), sin(lat)].
  **ZH**: • Sentinel-1 合成孔径雷达观测（S1）：VV（垂直极化发射和接收）和 VH（垂直发射、水平接收）波段。

**[C063]**
EN: A.1.3   Channel Groups
ZH: A.1.3 通道组

**[C064]**
EN: As described in Section 3.2, we transform the pixel timeseries x into a number of tokens, where each token is a linear transformation of a subset of the input channels. We group together channels which (i) come from the same sensor or product, (ii) have equivalent native spatial resolutions and (iii) represent similar parts of the electromagnetic spectrum (for Sentinel-2 channel groups). We group the input data into the following channel groups:
ZH: 如第3.2节所述，我们将像素时间序列 x 转换为多个标记，其中每个标记对应一个通道组。

- [C065] **EN**: • Sentinel-1: The VV and VH bands from the Sentinel-1 sensor • Sentinel-2 RGB: The B2, B3 and B4 bands from the Sentinel-2 sensor • Sentinel-2 Red Edge: The B5, B6 and B7 bands from the Sentinel-2 sensor • Sentinel-2 Near Infra Red (10m): The B8 band from the Sentinel-2 sensor • Sentinel-2 Near Infra Red (20m): The B8A band from the Sentinel-2 sensor • Sentinel-2 Short Wave Infra Red: The B11 and B12 bands from the Sentinel-2 sensor • NDVI: The normalized difference vegetation index, calculated from the Sentinel-2 B4 and B8 bands. • ERA5 Climatology: Precipitation and temperature at 2m from the ERA5 Climate Reanalysis product • Topography: The elevation and slope of a pixel, calculated by the SRTM’s DEM • Location: The cartesian coordinates of a pixel, computed from the pixel’s latitude and longitude
  **ZH**: • Sentinel-1：Sentinel-1 传感器的 VV 和 VH 波段 • Sentinel-2 RGB：B2、B3 和 B4 波段...


---

<!-- Page 18 -->
## Page 18

**[T010] Table / 表**
> **EN**: Table 10: Full results for regression tasks from Table 3, including standard error computed from three runs.
> **ZH**: 表10：表3中回归任务的完整结果，包括从三次运行计算的标准误。

**[C066]**
EN: Fuel Moisture      Algae Blooms      Mean difference Linear Regression                   28.20              0.850                   0% Random Forest               23.84 ± 0.42         1.249 ± 0.02               15.7% MOSAIKS-1DRF                 28.75 ± 0.15        0.972 ± 0.01               8.15% PrestoF T (random init.)     26.07 ± 0.52        0.955 ± 0.05               2.40% PrestoF T                   25.28 ± 0.30       0.815 ± 0.03              −7.24% PrestoRF                     25.98 ± 0.66       0.884 ± 0.01             −1.94%
ZH: [表格数据保留原文]

**[C067]**
EN: A.2   FLOP calculations
ZH: A.2 FLOP 计算

**[C068]**
EN: We use the thop library (https://github.com/Lyken17/pytorch-OpCounter) to calculate the FLOPs required to encode a EuroSAT image (as plotted in Table 5(b)). For the SatMAE, ScaleMAE and ConvMAE models, all images were resized to 224 × 224, so the FLOPs required to encode an image is independent of resolution. For Presto, we computed the FLOPs required to encode a single pixel and multiplied this by the number of pixels in an image at each resolution (e.g. the “64” resolution has 64 × 64 pixels, so we multiply the FLOPs required to encode a single pixel by 64 × 64 = 4096). The FLOPs calculated by the thop library are recorded in Table 9.
ZH: 我们使用 thop 库（https://github.com/Lyken17/pytorch-OpCounter）计算编码 EuroSat 图像所需的 FLOPs。

**[C069]**
EN: A.3   Baselines
ZH: A.3 基线

**[C070]**
EN: In addition to task-specific baselines, we benchmark Presto against: • Random Forests: Random forests are powerful baselines in remote sensing as they they remain competitive with state-of-the-art methods (Pelletier et al., 2019; Kerner et al., 2020). Tree-based methods, especially random forests, are commonly deployed in large-scale machine learning for remote sensing applications (Hansen et al., 2013; Van Tricht, 2021; Di Tommaso et al., 2022). • MOSAIKS-1D: We adapt MOSAIKS (Rolf et al., 2021) for timeseries data. MOSAIKS-1D uses patches from the pre-training dataset and convolves over the temporal dimension instead of the spatial dimension. We benchmark MOSAIKS-1D on all timeseries evaluation tasks. Because this does not work for categorical inputs, we exclude Dynamic World. As with Presto, we use the output features with random forests (MOSAIKS-1DRF ) and with regressions (MOSAIKS-1DR ).
ZH: 除特定任务的基线外，我们将 Presto 与以下模型进行基准比较：• 随机森林：基于随机森林的分类器...

**[C071]**
EN: A.4   Downstream Results
ZH: A.4 下游结果

**[C072]**
EN: We include complete results for the evaluation tasks. These include error bars, as well as additional results reported for the CropHarvest (Table 12 and Figure 3), regression tasks (Table 10), EuroSAT (Tables 11, 13 and 14), TreeSatAI (Table 15) and Sen2-Agri100 (Table 16) datasets. We run all downstream classifiers with 3 seeds (0, 42, 84), with the exception of the kNN classifiers and the linear regression (which are deterministic). In the tables in the main paper (Tables 2, 4, 6 and 3) we report the average of these runs; the standard error is reported in Tables 12,15, 16 and 10. • Presto as a feature extractor: When used as a feature extractor, a random forest, regression of K-nearest-neighbours classifier is trained on Presto’s output embeddings. In this case, we use scikit-learn models with the default hyperparameters. For the CropHarvest tasks, the class labels are extremely imbalanced; we therefore set class_weight equal to balanced for those tasks, for both Presto and MOSAIKS-1D. • Fine-tuning Presto: When fine-tuning Presto, we use the same hyperparameters across all tasks: an AdamW optimizer with a learning rate of 3e-4 and weight decay of 0.05. As discussed in Section 5.2, we obtain per-image predictions using Presto by computing a mean and standard deviation of Presto’s output pixels, and passing a concatenation of these two vectors to a downstream classifier. This is illustrated in Figure 4.
ZH: 我们包含评估任务的完整结果，包括误差线以及额外的基线。


---

<!-- Page 19 -->
## Page 19

**[T011] Table / 表**
> **EN**: Table 11: Accuracy results for pre-trained and from-scratch Presto when fine-tuned on EuroSat, at varying resolutions. We hypothesize that the drop in performance for the full resolution (64) RGB input is due to the model construction; the model outputs for all pixels in the image (4,096 pixels for the full resolution) are aggregated and passed to a linear layer for classification, yielding a noisy gradient signal.
> **ZH**: 表11：预训练和从头训练的 Presto 在 EuroSat 上微调时的准确率，在不同分辨率下。

**[C073]**
EN: Resolution                          2                   4               8               16                 32             64 random init.           0.703 ± 0.005      0.684 ± 0.032      0.694 ± 0.013    0.739 ± 0.004   0.750 ± 0.018     0.745 ± 0.009 RGB pre-trained            0.792 ± 0.010      0.837 ± 0.006      0.847 ± 0.016    0.865 ± 0.006   0.872 ± 0.002     0.849 ± 0.004 random init.           0.837 ± 0.014      0.884 ± 0.010      0.895 ± 0.006     0.907 ± 0.13   0.924 ± 0.005     0.924 ± 0.003 MS pre-trained            0.898 ± 0.005      0.925 ± 0.004      0.939 ± 0.000    0.950 ± 0.002   0.958 ± 0.001     0.953 ± 0.004
ZH: [表格数据保留原文]

**[T012] Table / 表**
> **EN**: Table 12: Additional results for the CropHarvest task. In addition to the F1 scores reported in the main paper, we report AUC ROC scores, with standard error bars computed with three runs. Model                           Kenya                 Brazil               Togo     Mean Random Forest            0.559 ± 0.003        0.000 ± 0.000       0.756 ± 0.002     0.441 MOSAIKS-1DR              0.790 ± 0.027        0.746 ± 0.084       0.679 ± 0.024     0.738 F1             TIML                     0.838 ± 0.000        0.835 ± 0.012       0.732 ± 0.002     0.802 PrestoR                0.816 ± 0.000         0.891 ± 0.000      0.798 ± 0.000       0.835 no DW              0.861 ± 0.000             0.888 ± 0.000      0.760 ± 0.000    0.836 Random Forest            0.578 ± 0.006        0.941 ± 0.004       0.892 ± 0.001     0.803 MOSAIKS-1DR              0.693 ± 0.036        0.890 ± 0.038       0.836 ± 0.005     0.806 AUC ROC        TIML                     0.794 ± 0.003        0.988 ± 0.001       0.890 ± 0.000     0.890 PrestoR                0.834 ± 0.000         0.997 ± 0.000      0.921 ± 0.000       0.917 no DW              0.863 ± 0.000             0.989 ± 0.000      0.912 ± 0.000    0.921
> **ZH**: 表12：CropHarvest 任务的额外结果。除正文报告的平均 F1 分数外，还包括按国家划分的结果。

**[C074]**
EN: A.5   Disentangling the effect of pre-training
ZH: A.5 预训练效果的解耦

**[C075]**
EN: To understand the effect of pre-training Presto, we fine-tune Presto and train it from scratch on EuroSat (Table 5), the regression tasks (Table 3 in the main paper) and TreeSatAI (Table 15). We omit the CropHarvest dataset because it was expressly designed as a few-shot-learning dataset. Its small size makes the construction of validation sets with which to control the finetuning (e.g. with early stopping) challenging. Overall, we find a consistent and significant improvement from the use of pre-trained Presto compared to a randomly initialized version of the model. For the EuroSat task, pre-training consistently delivers an incresse in accuracy score > 0.1 (representing increases in accuracy of up to 25%). This effect is consistent with what we observe on the TreeSatAI dataset for S2 data and on the regression tasks (where pre-training reduces RMSE by to 15% on the algae blooms task). For the TreeSatAI dataset with S1 data, pre-training penalizes the model compared to random initialization - we hypothesize that this is due to the difference in input (a single timestep and single channel group image) relative to the pre-training data. The benefit of pre-training effect is especially pronounced on the S2-Agri100 dataset; we hypothesize this is due to the small training set size.
ZH: 为理解预训练 Presto 的效果，我们在 EuroSat 上对 Presto 进行微调和从头训练。

**[C076]**
EN: A.6   Presto’s failure modes
ZH: A.6 Presto 的失效模式

**[C077]**
EN: Presto processes pixel-timeseries independently, without spatial context from other pixels or locations. This means that when we make image-based predictions (such as for scene classification), Presto’s independent pixel representations must be aggregated into a single prediction. We opt for a simple concatenation of the element-wise mean and standard deviation of the representations, from which a classifier makes a prediction. Information gets lost in such a simple aggregation, which impacts Presto’s performance on such tasks.
ZH: Presto 独立处理像素时间序列，不使用来自其他像素或位置的空间上下文。


---

<!-- Page 20 -->
## Page 20

**[T013] Table / 表**
> **EN**: Table 13: Additional results for the EuroSat task - results for the ScaleMAE, SatMAE and ConvMAE models are from (Reed et al., 2022). We report kNN classifier results for different values of k, and at varying input resolutions. Resolution                     16                           32                         64 k                      5        20       100        5        20        100        5      20        100 SatMAE            0.729       0.727     0.695   0.871      0.876     0.854   0.934    0.931       0.913 ScaleMAE          0.751       0.744     0.699   0.912      0.901     0.869   0.960    0.956       0.935 ConvMAE           0.835       0.826     0.788   0.909      0.898     0.863   0.947    0.940       0.914 Presto (RGB)      0.869       0.828     0.713   0.869      0.829     0.712   0.869    0.829       0.713 Presto (MS)       0.916       0.892     0.844   0.920      0.892     0.846   0.921    0.893       0.846
> **ZH**: 表13：EuroSat 任务的额外结果——ScaleMAE、SatMAE 和 ConvMAE 模型的结果来自（Reed 等，2022）。

**[T014] Table / 表**
> **EN**: Table 14: Additional results for the EuroSat task for Presto when run with reduced resolutions (compared to those used by (Reed et al., 2022) and reported in Table 13). We report kNN classifier results for different values of k, and at varying input resolutions. Resolution                      2                            4                          8 k                      5         20      100           5      20       100        5         20     100 Presto (RGB)      0.843       0.811     0.699   0.860      0.820     0.706    0.869   0.826       0.710 Presto (MS)       0.873       0.852     0.799   0.895      0.874     0.824    0.911   0.886       0.838
> **ZH**: 表14：Presto 在降低分辨率下运行时的 EuroSat 任务额外结果。

**[T015] Table / 表**
> **EN**: Table 15: Additional results for the TreeSatAI (as in (Ahlswede et al., 2023), we report precision and recall in addition to F1 score and mAP). In addition, we report the results of finetuning Presto (PrestoF T ) from the pre-trained weights and from a random initialization. Model                      Data     Aggregation                F1             mAP         Precision           Recall MLP                                                        10.09              29.42          33.29             7.13 LightGBM                                                   11.86              32.79          37.96             8.06 PrestoF T (random init.)            Weighted        40.36 ± 0.77       39.77 ± 0.79   30.69 ± 0.82     64.69 ± 1.09 PrestoF T                                           38.69 ± 0.78       37.41 ± 0.58   30.09 ± 0.74     61.20 ± 0.85 PrestoRF                                            38.34 ± 0.07       35.45 ± 0.03   29.67 ± 0.07     57.23 ± 0.06 S1 MLP                                                        12.82              33.09          63.01             7.13 LightGBM                                                   14.07              35.11          55.49             8.06 PrestoF T (random init.)            Micro           42.04 ± 0.73       43.00 ± 0.80   31.20 ± 1.00     64.69 ± 1.09 PrestoF T                                           41.65 ± 0.46       40.75 ± 0.69   31.58 ± 0.47     61.20 ± 0.85 PrestoRF                                            40.79 ± 0.04       38.64 ± 0.02   31.69 ± 0.03     57.23 ± 0.06 MLP                                                        51.97              64.19          74.59            42.23 LightGBM                                                   48.17              61.99          74.27            40.04 PrestoF T (random init.)            Weighted        52.74 ± 0.50       57.24 ± 0.64   45.87 ± 1.17     64.29 ± 1.51 PrestoF T                                           53.63 ± 0.42       59.16 ± 1.24   47.15 ± 1.40     65.11 ± 3.21 PrestoRF                                            55.29 ± 0.08       61.53 ± 0.09   56.93 ± 0.07     58.56 ± 0.09 S2 MLP                                                        54.49              65.83          77.18            42.23 LightGBM                                                   52.52              61.66          76.27            40.04 PrestoF T (random init.)            Micro           52.56 ± 0.41       58.08 ± 0.66   44.56 ± 1.03     64.29 ± 1.51 PrestoF T                                           53.31 ± 0.18       59.77 ± 1.13   45.51 ± 1.46     65.11 ± 3.21 PrestoRF                                            58.29 ± 0.06       63.31 ± 0.06   58.04 ± 0.05     58.56 ± 0.09
> **ZH**: 表15：TreeSatAI 的额外结果（如（Ahlswede 等，2023）所述，我们报告精确度和召回率）。


---

<!-- Page 21 -->
## Page 21

**[T016] Table / 表**
> **EN**: Table 16: Full results on the S2-Agri100 dataset, including standard errors obtained from 3 runs. To obtain standard errors for the SITS-Former, we run the official code (https://github.com/ linlei1214/SITS-Former) with 3 seeds. Best results are highlighted.
> **ZH**: 表16：S2-Agri100 数据集的完整结果，包括从3次运行获得的标准误。

**[C078]**
EN: Params (M)     Pre-trained?              OA                κ               F1 SITS                                     65.13 ± 3.01     0.55 ± 0.03     42.12 ± 0.52 2.5 Former                         ✓          67.03 ± 2.24     0.56 ± 0.02    42.83 ± 0.30 45.98 ± 2.74    0.35 ± 0.02     27.45 ± 0.64 Presto         0.4 ✓          68.89 ± 1.05     0.58 ± 0.01     40.41 ± 0.25
ZH: [表格数据保留原文]

**[F007] Figure / 图**
> **EN**: Figure 7: Accuracy of kNN@5 classifier with Presto RGB representations on the EuroSat dataset vs. the input resolution, for different categories. Some categories have been left out for clarity.
> **ZH**: 图7：使用 Presto RGB 表示的 kNN@5 分类器在 EuroSat 数据集上的准确率 vs. 预训练数据集大小的关系。

**[C079]**
EN: (a) Forest            (b) Annual Crop            (c) Highway                (d) River Figure 8: the RGB bands of example images from EuroSat classes.
ZH: (a) 森林  (b) 一年生作物  (c) 高速公路  (d) 河流  图8：...

**[C080]**
EN: For example, Presto’s performance on the EuroSat dataset reaches a plateau when increasing the input resolution. As Figure 7 shows, this is mainly caused by a failure to accurately predict specific classes (for example, the Highway and River classes). Figure 8 shows example images for these classes, as well as for the Forest and AnnualCrop classes, on which Presto achieves higher accuracies. While in the Forest and AnnualCrop images, most pixels of the image actually represent the labelled class, in the Highway and River images only a relatively small part of the image actually contains the label (a highway or river). We hypothesize that since many pixels in the Highway and River images do not actually represent that class, the crude token-aggregation method we use to represent images is insufficiently discriminative to accurately classify these images. Other pre-trained remote sensing models use much more powerful mechanisms for aggregating spatial information. For example, ViT models convolve over patches and then apply an attention mechanism between spatial patches. If image-based predictions are needed and these predictions are highly dependent on the occurrence of objects in subregions of the image, models which natively process this important spatial information may be better suited. We plan on exploring techniques to mitigate this difficulty with Presto in future work.
ZH: 例如，当增加输入大小时，Presto 在 EuroSat 数据集上的性能达到平台期。


---

## Figures and Tables

### Extracted Assets

- `fig-000.png`
- `fig-001.png`
- `fig-002.png`
- `fig-003.png`
- `fig-004.png`
- `fig-005.png`
- `fig-006.png`
- `fig-007.png`
- `fig-008.png`
- `fig-009.png`
- `fig-010.png`
- `fig-011.png`
- `fig-012.png`
- `fig-013.png`
- `fig-014.png`
- `fig-015.png`
- `fig-016.png`
- `fig-017.png`
- `fig-018.png`
- `fig-019.png`
- `fig-020.png`

### Figure-Asset Mapping

| Figure | Asset File | Description |
|--------|-----------|-------------|
| Figure 1 | assets/fig-001.png ~ fig-005.png | Presto learns from structurally-masked remote sensing pixel-timeseries |
| Figure 2 | assets/fig-006.png ~ fig-009.png | Reconstruction of completely masked channels |
| Figure 3 | assets/fig-010.png ~ fig-011.png | Robustness to incomplete inputs |
| Figure 4 | assets/fig-012.png ~ fig-013.png | Per-image predictions via aggregation |
| Figure 5 | assets/fig-014.png ~ fig-017.png | EuroSat kNN@5 accuracy vs FLOPs |
| Figure 6 | assets/fig-018.png | Pre-training data distribution |
| Figure 7 | assets/fig-019.png | kNN@5 accuracy vs pre-training dataset size |
| Figure 8 | assets/fig-020.png | Class-specific examples |
