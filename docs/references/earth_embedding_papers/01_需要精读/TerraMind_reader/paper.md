# TerraMind: A Generative Multimodal Foundation Model for Earth Observation
# TerraMind：面向地球观测的生成式多模态基础模型

> **Paper**: TerraMind: Large-Scale Generative Multimodality for Earth Observation  
> **Venue**: ICCV 2025  
> **Authors**: Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, Rahul Ramachandran, Paolo Fraccaro, Thomas Brunschwiler, Gabriele Cavallaro, Juan Bernabe-Moreno, Nicolas Longépé  
> **Institutions**: IBM Research – Europe, ETH Zurich, Forschungszentrum Jülich, European Space Agency Φ-Lab, NASA IMPACT, University of Iceland  
> **Open Source**: https://huggingface.co/ibm-esa-geospatial | https://github.com/ibm/terramind  
> **Reader Date**: 2026-05-27

---

## Abstract | 摘要

**[S001]** We present TerraMind, the first any-to-any generative, multimodal deep learning model for Earth observation (EO). Unlike other approaches, TerraMind is pretrained on dual-scale representations combining both token-level and pixel-level data across modalities. On a token level, TerraMind encodes high-level contextual information to learn cross-modal relationships, while on a pixel level, TerraMind leverages fine-grained representations to capture critical spatial nuances. In this paper, we demonstrate that (i) TerraMind achieves beyond state-of-the-art performance in community-standard benchmarks, (ii) TerraMind can leverage "thinking in modalities" (TiM)—the capability of generating additional artificial data during finetuning and inference to improve the model output—and (iii) TerraMind's dual-scale early fusion approach results in well-structured embedding spaces. Models and code have been open-sourced.

**[S001-zh]** 本文提出 TerraMind，首个面向地球观测（Earth Observation, EO）的任意到任意（any-to-any）生成式多模态深度学习模型。与其他方法不同，TerraMind 在双尺度表征上进行预训练，同时融合跨模态的 token 级与像素级数据。在 token 层面，TerraMind 编码高层上下文信息以学习跨模态关系；在像素层面，TerraMind 利用细粒度表征捕捉关键的空间细节。本文证明：(i) TerraMind 在社区标准基准测试中达到超越最先进的性能；(ii) TerraMind 能够利用"模态思维"（Thinking in Modalities, TiM）——即在微调和推理过程中生成额外人工数据以提升模型输出的能力；(iii) TerraMind 的双尺度早期融合方法能够产生结构良好的嵌入空间。模型与代码均已开源。

**[F001]** Figure 1. TerraMind represents the first any-to-any generative, and large-scale multimodal model for Earth observation pre-trained on 500 billion tokens from global geospatial data. The model digests multi-scale representations at pixel-level and token-level simultaneously. TerraMind v1 unlocks (i) generation, (ii) zero-shot and finetuning applications, and (iii) "Thinking-in-Modalities" finetuning and inference.

**[F001-zh]** 图 1. TerraMind 是首个面向地球观测的任意到任意生成式大规模多模态模型，在全球地理空间数据上预训练了 5000 亿个 token。该模型同时处理像素级和 token 级的多尺度表征。TerraMind v1 解锁了三大能力：(i) 生成，(ii) 零样本与微调应用，(iii) "模态思维"微调与推理。

---

## 1. Introduction | 引言

**[S002]** Earth observation (EO) increasingly benefits from multimodality because of the important integration of complementary information from different data sources. This becomes particularly relevant as EO is spatiotemporally sparse due to low revisiting times or weather phenomena like cloud coverage. Vice versa, for computer vision, EO data is an important playground for the development of new approaches as there is significant publicly available data of very high quality and complexity. The available modalities range from sensors of different satellite missions to relevant complementary information like digital elevation.

**[S002-zh]** 地球观测（EO）正日益从多模态中获益，因为不同数据源提供了互补信息的重要整合。这一点尤为关键，因为由于重访周期低或云覆盖等天气现象，EO 数据在时空上是稀疏的。反之，对于计算机视觉而言，EO 数据是开发新方法的重要试验场，因为存在大量公开可用的高质量、高复杂度数据。可用的模态范围涵盖不同卫星任务的传感器到诸如数字高程等相关的互补信息。

**[S003]** In this work, we introduce TerraMind as the first any-to-any generative multimodal model for EO. With TerraMind, we introduce a dual-scale pretraining on pixel-level and token-level and demonstrate benefits over training primarily on tokens. TerraMind encodes high-level contextual information in tokens to enable correlation learning and scaling, while additionally capturing important fine-grained representations using pixel-level inputs. During pretraining, TerraMind predicts masked target tokens so that our pretraining objective boils down to a cross-modal patch classification problem that results in high-quality latent spaces.

**[S003-zh]** 在本文中，我们提出 TerraMind，作为首个面向 EO 的任意到任意生成式多模态模型。我们引入了在像素级和 token 级上的双尺度预训练，并展示了其相对于主要在 token 上训练的优势。TerraMind 在 token 中编码高层上下文信息以实现相关性学习和规模扩展，同时利用像素级输入捕捉重要的细粒度表征。在预训练期间，TerraMind 预测被掩码的目标 token，因此我们的预训练目标简化为一个跨模态的 patch 分类问题，从而产生高质量的潜在空间。

**[S004]** TerraMind is pretrained on more than 500 billion tokens from a new global-scale pretraining dataset we release as TerraMesh. This dataset combines major existing geospatial datasets with additional datasets to ensure a diverse representation of modalities and geographical areas. TerraMind v1 (TerraMind-1B) and TerraMind v2 (TerraMind-7B) achieve competitive and beyond state-of-the-art performance on a range of community standard benchmarks in finetuning and zero-shot setups. In addition, TerraMind's generative capability allows us to unlock a novel capability we call "Thinking in Modalities" (TiM). By leveraging additional generated modalities during finetuning and inference, the model performance on downstream tasks can be further increased.

**[S004-zh]** TerraMind 在一个新的全球规模预训练数据集上进行了超过 5000 亿 token 的预训练，该数据集我们将其命名为 TerraMesh 并公开发布。该数据集结合了主要现有的地理空间数据集和额外的数据集，以确保模态和地理区域的多样代表性。TerraMind v1（TerraMind-1B）和 TerraMind v2（TerraMind-7B）在一系列社区标准基准测试中，于微调和零样本设置下均达到了具有竞争力乃至超越最先进的性能。此外，TerraMind 的生成能力使我们解锁了一种称为"模态思维"（Thinking in Modalities, TiM）的新能力。通过在微调和推理过程中利用额外生成的模态，下游任务上的模型性能可以进一步提升。

---

## 2. Related Work | 相关工作

**[S005]** Computer vision in Earth observation. Computer vision (CV) has significantly advanced EO [76]. Many CV techniques, originally developed for natural image processing, have been adapted to EO [62], often with minimal modifications. A wide range of tasks benefit from these methods, including classification [16], semantic segmentation [72] (e.g., land cover mapping [20, 21]), change detection [59] (e.g., disaster response [19]), object detection [39] (e.g., vessel identification [55]), and regression (e.g., biomass estimation [53]). Deep learning architectures like CNNs [75] and Vision Transformers (ViTs) [17] have demonstrated strong performance, often surpassing traditional remote sensing (RS) methods. However, EO presents unique challenges, including diverse sensor modalities [4] and geospatial heterogeneity [46].

**[S005-zh]** 地球观测中的计算机视觉。计算机视觉（CV）显著推动了 EO 的发展 [76]。许多最初为自然图像处理开发的 CV 技术已被适配到 EO 领域 [62]，通常只需极少修改。广泛的任务从这些方法中受益，包括分类 [16]、语义分割 [72]（如土地覆盖制图 [20, 21]）、变化检测 [59]（如灾害响应 [19]）、目标检测 [39]（如船舶识别 [55]）和回归（如生物量估算 [53]）。深度学习架构如 CNN [75] 和视觉 Transformer（ViT）[17] 已展现出强劲性能，经常超越传统遥感（RS）方法。然而，EO 也带来了独特的挑战，包括多样化的传感器模态 [4] 和地理空间异质性 [46]。

**[S006]** An emerging paradigm in EO is self-supervised learning (SSL) [64] and geospatial foundation models (GFMs) [45], which aim to leverage vast amounts of unlabeled RS data. While off-the-shelf CV models have shown promising results [36], they do not fully exploit the unique characteristics of geospatial data. Many GFMs still rely on generic CV architectures [50], which were not explicitly designed to handle the complexities of EO, such as heterogeneous sensor sources (e.g., optical, radar, and DEM), varying spatial resolutions, and large-scale geospatial context. Consequently, there is a growing need for models that are specifically designed for EO data, incorporating domain-specific knowledge and multimodal fusion strategies.

**[S006-zh]** EO 中一个新兴范式是自监督学习（SSL）[64] 和地理空间基础模型（GFM）[45]，其目标是利用大量未标记的遥感数据。虽然现成的 CV 模型已显示出有希望的结果 [36]，但它们并未充分利用地理空间数据的独特特征。许多 GFM 仍依赖于通用的 CV 架构 [50]，这些架构并非专门为处理 EO 的复杂性而设计，例如异构传感器源（如光学、雷达和 DEM）、可变的空间分辨率以及大规模的地理空间上下文。因此，越来越需要专门为 EO 数据设计的模型，整合领域特定知识和多模态融合策略。

**[S007]** Multimodal foundation models. The field of natural language processing (NLP) has seen a significant shift towards large-scale pretrained models [9], which have demonstrated remarkable capabilities in understanding and generating human language. This success has inspired the development of multimodal foundation models that integrate vision and language [3, 57]. These models, such as CLIP [57], ALIGN [24], and Florence [73], leverage large-scale image-text pairs to learn joint representations that capture semantic alignments between visual and textual modalities. The ability to combine several modalities allows for unprecedented capabilities in complex tasks [30], evidenced by the rapid advancement of models like GPT-4o [43] and Gemini [23].

**[S007-zh]** 多模态基础模型。自然语言处理（NLP）领域已显著转向大规模预训练模型 [9]，这些模型在理解和生成人类语言方面展现出卓越能力。这一成功激发了整合视觉与语言的多模态基础模型的发展 [3, 57]。这些模型，如 CLIP [57]、ALIGN [24] 和 Florence [73]，利用大规模的图像-文本对学习联合表征，捕捉视觉和文本模态之间的语义对齐。组合多种模态的能力使得复杂任务中前所未有的能力成为可能 [30]，GPT-4o [43] 和 Gemini [23] 等模型的快速发展即是明证。

**[S008]** In Earth observation, multimodal approaches have been explored to integrate data from multiple sensors [4, 18, 71]. For instance, the combination of optical and synthetic aperture radar (SAR) data has been shown to improve land cover classification and change detection [18]. However, these approaches often rely on simple fusion strategies, such as early or late fusion, and do not fully exploit the complementary information available across modalities. Recent work has explored more sophisticated fusion techniques, including cross-attention mechanisms [71] and transformer-based architectures [4], but these models are often limited in scale and do not support generative capabilities.

**[S008-zh]** 在地球观测中，多模态方法已被探索用于整合来自多个传感器的数据 [4, 18, 71]。例如，光学数据与合成孔径雷达（SAR）数据的结合已被证明可以改善土地覆盖分类和变化检测 [18]。然而，这些方法通常依赖于简单的融合策略，如早期融合或晚期融合，并未充分利用跨模态可用的互补信息。近期工作探索了更复杂的融合技术，包括交叉注意力机制 [71] 和基于 Transformer 的架构 [4]，但这些模型通常在规模上受限且不支持生成能力。

**[F002]** Figure 2. TerraMind outperforms other geospatial foundation models on PANGAEA benchmark [49] in finetuning. Performance is measured in mIoU and min-max scaled per dataset.

**[F002-zh]** 图 2. TerraMind 在 PANGAEA 基准测试 [49] 的微调任务上优于其他地理空间基础模型。性能以 mIoU 度量，并按数据集进行 min-max 缩放。

---

## 3. Dataset | 数据集

**[S009]** We introduce TerraMesh, a large-scale, global pretraining dataset for Earth observation that combines multiple existing geospatial datasets with additional curated data. TerraMesh is designed to provide diverse multimodal coverage across different geographical regions, sensor types, and temporal scales. The dataset includes optical imagery from Sentinel-2 (L1C and L2A), radar data from Sentinel-1 (GRD and RTC), digital elevation models (DEM), land use/land cover (LULC) maps, and normalized difference vegetation index (NDVI) data. In total, TerraMesh comprises more than 500 billion tokens after tokenization.

**[S009-zh]** 我们推出 TerraMesh，一个面向地球观测的大规模全球预训练数据集，结合了多个现有地理空间数据集和额外的精选数据。TerraMesh 旨在提供跨不同地理区域、传感器类型和时间尺度的多样化多模态覆盖。该数据集包括来自 Sentinel-2（L1C 和 L2A）的光学影像、来自 Sentinel-1（GRD 和 RTC）的雷达数据、数字高程模型（DEM）、土地利用/土地覆盖（LULC）地图以及归一化植被指数（NDVI）数据。总计，TerraMesh 在 token 化后包含超过 5000 亿个 token。

**[S010]** The dataset is constructed by aligning multimodal data patches at a common spatial resolution of 10 meters. For optical data, we use Sentinel-2 L2A products for surface reflectance and L1C for top-of-atmosphere reflectance. Sentinel-1 data includes both Ground Range Detected (GRD) and Radiometric Terrain Corrected (RTC) products. The DEM data is derived from the Copernicus DEM [12]. LULC labels are sourced from ESA WorldCover [85] and OpenStreetMap [41]. NDVI is computed from Sentinel-2 bands. All modalities are spatially aligned and cropped to patches of 256×256 pixels, covering approximately 2.56 km × 2.56 km on the ground.

**[S010-zh]** 该数据集通过在 10 米共同空间分辨率下对齐多模态数据 patch 构建。对于光学数据，我们使用 Sentinel-2 L2A 产品获取地表反射率，使用 L1C 获取大气层顶反射率。Sentinel-1 数据包括地距检测（GRD）和辐射地形校正（RTC）产品。DEM 数据源自 Copernicus DEM [12]。LULC 标签来自 ESA WorldCover [85] 和 OpenStreetMap [41]。NDVI 由 Sentinel-2 波段计算得到。所有模态均经过空间对齐并裁剪为 256×256 像素的 patch，在地面上覆盖约 2.56 km × 2.56 km。

**[S011]** To ensure global coverage, TerraMesh includes data from all continents and major biomes. The dataset is balanced across latitudes to avoid biases towards specific regions. We also include temporal diversity by sampling data from different seasons and years. The dataset is split into training, validation, and test sets based on geographical locations to prevent spatial leakage. Specifically, we use a spatial cross-validation strategy where entire geographic regions are held out for validation and testing.

**[S011-zh]** 为确保全球覆盖，TerraMesh 包含来自各大洲和主要生物群系的数据。该数据集在纬度上保持平衡，以避免对特定区域的偏见。我们还通过从不同季节和年份采样数据来确保时间多样性。数据集按地理位置划分为训练集、验证集和测试集，以防止空间泄漏。具体而言，我们使用空间交叉验证策略，将完整的地理区域留出用于验证和测试。

---

## 4. Methods | 方法

**[S012]** TerraMind pretraining is two-staged following [52]. We first pretrain unimodal tokenizer models, tokenize the modalities, and then leverage token-level and pixel-level input to pretrain the TerraMind encoder-decoder architecture. We describe those individual stages in the following.

**[S012-zh]** TerraMind 的预训练遵循 [52] 采用两阶段策略。我们首先预训练单模态 tokenizer 模型，对模态进行 token 化，然后利用 token 级和像素级输入来预训练 TerraMind 的编码器-解码器架构。以下我们描述这些独立阶段。

### 4.1. Tokenization | Token 化

**[S013]** We develop modality-specific tokenizers to encode each modality into a sequence of discrete tokens for pretraining and decode token sequences back to images. Thus, TerraMind is in principle compatible with any modality, as long as it can be tokenized and aligned with other modalities. For reasons of space, we delegate most experiments related to the tokenizer performances to the supplementary material.

**[S013-zh]** 我们开发了模态特定的 tokenizer，以将每种模态编码为用于预训练的离散 token 序列，并将 token 序列解码回图像。因此，原则上 TerraMind 兼容任何模态，只要该模态可以被 token 化并与其他模态对齐。由于篇幅限制，我们将大部分与 tokenizer 性能相关的实验委托给补充材料。

**[S014]** Image-like modalities. We train autoencoder-based architectures with a quantization step in the bottleneck for image-like modalities (see Figure 4). We utilize a patch size of 16×16 pixels for the tokenization of image-like modalities, resulting in 256 tokens per 256×256 image patch. For the quantization, we employ Finite Scalar Quantization (FSQ) [51], which has shown superior performance compared to Vector Quantization (VQ) in our experiments (see supplementary material). The decoder is a diffusion model [34] that reconstructs the original image from the quantized latent representation.

**[S014-zh]** 图像类模态。我们为图像类模态训练基于自编码器的架构，在瓶颈处进行量化（见图 4）。我们对图像类模态使用 16×16 像素的 patch 大小进行 token 化，每个 256×256 图像 patch 产生 256 个 token。对于量化，我们采用有限标量量化（Finite Scalar Quantization, FSQ）[51]，在我们的实验中其性能优于矢量量化（Vector Quantization, VQ）（见补充材料）。解码器是一个扩散模型 [34]，从量化后的潜在表征重建原始图像。

**[F003]** Figure 4. Tokenizer for image-like modalities combining finite-scalar quantization [51] with diffusion decoding.

**[F003-zh]** 图 4. 图像类模态的 tokenizer，结合有限标量量化 [51] 与扩散解码。

**[S015]** Sequence-like modalities. We treat both captions and geolocations as text and use a single text tokenizer to process both modalities. By discretizing the geographic coordinates and representing them as strings, we introduce special coordinate tokens into the vocabulary. This allows us to encode geolocations as a sequence of discrete tokens, beginning with a latitude token followed by a longitude token. For textual data, we modify the existing WordPiece tokenizer [33].

**[S015-zh]** 序列类模态。我们将标题和地理位置均视为文本，使用单一的文本 tokenizer 处理这两种模态。通过对地理坐标进行离散化并将其表示为字符串，我们在词表中引入了特殊的坐标 token。这使我们能够将地理位置编码为离散 token 序列，以纬度 token 开始，后接经度 token。对于文本数据，我们修改了现有的 WordPiece tokenizer [33]。

### 4.2. Architecture & Pre-training | 架构与预训练

**[S016]** We design TerraMind as an encoder-decoder architecture with a modality-agnostic transformer backbone. The encoder processes both pixel-level and token-level inputs, while the decoder generates target tokens for any modality. The key innovation is the dual-scale early fusion mechanism that combines pixel-level and token-level representations within the encoder.

**[S016-zh]** 我们将 TerraMind 设计为带有模态无关 Transformer 主干的编码器-解码器架构。编码器处理像素级和 token 级输入，而解码器为任意模态生成目标 token。关键创新在于双尺度早期融合机制，在编码器内部结合像素级和 token 级表征。

**[S017]** For pixel-level inputs, we use patch embeddings similar to ViT [17], where each 16×16 pixel patch is projected to the model dimension. For token-level inputs, we use learned embeddings for each discrete token. Both types of embeddings are fed into the same transformer encoder, allowing the model to learn correlations between pixel-level details and token-level semantics. We add modality type embeddings to distinguish between different input modalities.

**[S017-zh]** 对于像素级输入，我们使用类似 ViT [17] 的 patch 嵌入，其中每个 16×16 像素 patch 被投影到模型维度。对于 token 级输入，我们为每个离散 token 使用学习得到的嵌入。两种类型的嵌入都被送入同一个 Transformer 编码器，使模型能够学习像素级细节与 token 级语义之间的相关性。我们添加了模态类型嵌入以区分不同的输入模态。

**[S018]** The decoder is an autoregressive transformer that predicts masked target tokens. During pretraining, we randomly mask a subset of target tokens from all modalities and train the model to predict them. The pretraining objective is a cross-entropy loss over the predicted token distributions. This formulation allows TerraMind to learn any-to-any mappings: given any subset of modalities as input, the model can generate any other modality.

**[S018-zh]** 解码器是一个自回归 Transformer，预测被掩码的目标 token。在预训练期间，我们从所有模态中随机掩码一部分目标 token，并训练模型对其进行预测。预训练目标是预测 token 分布上的交叉熵损失。这种形式化使 TerraMind 能够学习任意到任意的映射：给定任意子集的模态作为输入，模型可以生成任何其他模态。

**[S019]** We pretrain two model sizes: TerraMind v1 with approximately 1 billion parameters and TerraMind v2 with approximately 7 billion parameters. Both models share the same architecture but differ in depth and width. The 1B model uses 24 layers with a hidden dimension of 2048, while the 7B model uses 32 layers with a hidden dimension of 4096. We use a context length of 4096 tokens for both models.

**[S019-zh]** 我们预训练了两个模型尺寸：TerraMind v1（约 10 亿参数）和 TerraMind v2（约 70 亿参数）。两个模型共享相同的架构，但在深度和宽度上有所不同。1B 模型使用 24 层，隐藏维度为 2048；7B 模型使用 32 层，隐藏维度为 4096。两个模型均使用 4096 token 的上下文长度。

### 4.4. Thinking-in-Modalities (TiM) | 模态思维

**[S020]** We introduce Thinking-in-Modalities (TiM) as a novel capability unlocked by TerraMind's generative nature. TiM leverages the model's ability to generate artificial modalities during finetuning and inference to improve downstream task performance. The core idea is that generating intermediate modalities can provide additional context and reduce ambiguity in the input data.

**[S020-zh]** 我们引入了模态思维（Thinking-in-Modalities, TiM）作为 TerraMind 生成特性所解锁的一种新能力。TiM 利用模型在微调和推理过程中生成人工模态的能力来提升下游任务性能。核心思想是，生成中间模态可以提供额外的上下文并减少输入数据中的歧义。

**[S021]** During TiM finetuning, we first generate synthetic modalities from the available input modalities using the pretrained TerraMind model. These generated modalities are then used as additional inputs during the actual finetuning process. For example, when finetuning on a downstream task with only optical data available, we can generate synthetic radar and DEM data to augment the input. This recursive augmentation mimics a chain-of-thought process, enabling the model to iteratively refine its internal representation, particularly in scenarios with missing modalities.

**[S021-zh]** 在 TiM 微调期间，我们首先使用预训练的 TerraMind 模型从可用输入模态生成合成模态。这些生成的模态随后在实际的微调过程中被用作额外输入。例如，当仅在可用光学数据的下游任务上进行微调时，我们可以生成合成雷达和 DEM 数据来增强输入。这种递归增强模仿了思维链过程，使模型能够迭代地细化其内部表征，特别是在模态缺失的场景中。

**[F004]** Figure 3. Chained generation example of TerraMind v1-B starting from either optical, radar, or digital elevation data. Left is input, middle is artificially generated data by TerraMind, right represents ground truths and tokenizer reconstructions, respectively.

**[F004-zh]** 图 3. TerraMind v1-B 的链式生成示例，分别从光学、雷达或数字高程数据开始。左侧为输入，中间为 TerraMind 人工生成的数据，右侧分别为真值和 tokenizer 重建结果。

---

## 5. Experiments | 实验

**[S022]** In this section, we describe the performance gains resulting from TerraMind and experiment with the unlocked capabilities of any-to-any generation and Thinking-in-Modalities.

**[S022-zh]** 在本节中，我们描述 TerraMind 带来的性能提升，并对任意到任意生成和模态思维所解锁的能力进行实验。

### 5.1. Foundational Experiments | 基础实验

**[S023]** Multimodality vs. unimodality. As a first motivational experiment, we outline the benefit of using multimodal data in Earth observation at the example of water body mapping. Specifically, we leverage the ViT-B encoders from the unimodal tokenizer models for S-1, S-2, and LULC, concatenate their embeddings, and add a linear classification head. We compare this against the TerraMind embedding (which processes all modalities jointly) and a unimodal S-2 baseline. The results in Table 1 show that TerraMind's multimodal embedding significantly outperforms both the unimodal baseline and the simple concatenation of unimodal embeddings.

**[S023-zh]** 多模态 vs. 单模态。作为首个动机性实验，我们以水体制图为例，概述在地球观测中使用多模态数据的益处。具体而言，我们利用来自 S-1、S-2 和 LULC 单模态 tokenizer 模型的 ViT-B 编码器，拼接它们的嵌入，并添加一个线性分类头。我们将其与 TerraMind 嵌入（联合处理所有模态）以及单模态 S-2 基线进行比较。表 1 中的结果表明，TerraMind 的多模态嵌入显著优于单模态基线和单模态嵌入的简单拼接。

**[T001]** Table 1. Water body mapping comparison. mIoU scores for unimodal, concatenated unimodal, and TerraMind multimodal embeddings.

**[T001-zh]** 表 1. 水体制图对比。单模态、拼接单模态和 TerraMind 多模态嵌入的 mIoU 分数。

**[S024]** Embedding space quality. We analyze the structure of TerraMind's embedding space using t-SNE visualization and cosine similarity analysis. The t-SNE plots (supplementary material) reveal well-separated clusters corresponding to different land cover types, indicating that the dual-scale pretraining results in semantically meaningful representations. We also measure the cosine similarity between embeddings of the same location across different modalities and find high similarity scores, confirming that TerraMind learns cross-modal alignments.

**[S024-zh]** 嵌入空间质量。我们使用 t-SNE 可视化和余弦相似度分析来分析 TerraMind 嵌入空间的结构。t-SNE 图（补充材料）揭示了与不同土地覆盖类型相对应的明显分离的聚类，表明双尺度预训练产生了语义上有意义的表征。我们还测量了同一位置跨不同模态嵌入之间的余弦相似度，发现了高相似度分数，证实了 TerraMind 学习了跨模态对齐。

**[S025]** PANGAEA benchmark. We evaluate TerraMind on the PANGAEA benchmark [49], a comprehensive benchmark for geospatial foundation models. The benchmark includes 7 downstream tasks covering segmentation, regression, and classification. We finetune TerraMind on each task and compare against state-of-the-art GFMs including Prithvi [37], SatMAE [14], Scale-MAE [58], GFMSwim [6], and GFM-SS [65]. As shown in Figure 2, TerraMind achieves the highest average performance across all tasks, with particularly strong results on semantic segmentation tasks.

**[S025-zh]** PANGAEA 基准测试。我们在 PANGAEA 基准测试 [49] 上评估 TerraMind，这是一个面向地理空间基础模型的综合基准测试。该基准测试包含 7 个下游任务，涵盖分割、回归和分类。我们在每个任务上微调 TerraMind，并与最先进 GFM 进行比较，包括 Prithvi [37]、SatMAE [14]、Scale-MAE [58]、GFMSwim [6] 和 GFM-SS [65]。如图 2 所示，TerraMind 在所有任务中取得了最高的平均性能，在语义分割任务上尤其出色。

### 5.3. Zero-shot Experiments | 零样本实验

**[S026]** We evaluate TerraMind's zero-shot transfer capability on several downstream tasks without any task-specific finetuning. For zero-shot classification, we use the embedding space to compute similarity between image patches and text prompts describing land cover classes. For zero-shot segmentation, we leverage the generative capability to produce segmentation masks conditioned on text prompts.

**[S026-zh]** 我们在多个下游任务上评估 TerraMind 的零样本迁移能力，无需任何任务特定的微调。对于零样本分类，我们使用嵌入空间计算图像 patch 与描述土地覆盖类别的文本提示之间的相似度。对于零样本分割，我们利用生成能力来生成以文本提示为条件的分割掩码。

**[S027]** The results in Table 2 show that TerraMind achieves competitive zero-shot performance compared to supervised baselines on several tasks. Notably, on the BigEarthNet [65] dataset, TerraMind's zero-shot mIoU reaches 78.3% of the supervised performance, demonstrating strong generalization from pretraining. The zero-shot performance is particularly strong for classes with clear visual signatures, such as urban areas and water bodies.

**[S027-zh]** 表 2 中的结果表明，与多个任务上的监督基线相比，TerraMind 达到了具有竞争力的零样本性能。值得注意的是，在 BigEarthNet [65] 数据集上，TerraMind 的零样本 mIoU 达到了监督性能的 78.3%，展示了从预训练中获得的强泛化能力。零样本性能在具有清晰视觉特征的类别上尤其出色，如城市区域和水体。

**[T002]** Table 2. Zero-shot performance on BigEarthNet and other benchmarks. Zero-shot mIoU as percentage of supervised performance.

**[T002-zh]** 表 2. BigEarthNet 及其他基准测试上的零样本性能。以监督性能百分比表示的零样本 mIoU。

### 5.4. Few-shot Experiments | 少样本实验

**[S028]** We evaluate TerraMind in few-shot learning scenarios where only a small number of labeled examples are available. We compare against several baseline GFMs using linear probing and partial finetuning protocols. For 1-shot and 5-shot learning on the PANGAEA benchmark, TerraMind consistently outperforms all baselines. With 16 samples per class, TerraMind achieves 92% of its full finetuning performance, while the best baseline (Prithvi) reaches only 84%.

**[S028-zh]** 我们在仅有少量标注样本可用的少样本学习场景中评估 TerraMind。我们使用线性探测和部分微调协议与多个 GFM 基线进行比较。在 PANGAEA 基准测试上的 1-shot 和 5-shot 学习中，TerraMind 始终优于所有基线。在每类 16 个样本的情况下，TerraMind 达到了其完全微调性能的 92%，而最佳基线（Prithvi）仅达到 84%。

**[S029]** The strong few-shot performance can be attributed to TerraMind's dual-scale pretraining, which provides both high-level semantic representations (useful for classification) and fine-grained spatial representations (useful for segmentation). The generative pretraining objective also encourages the model to learn robust features that generalize well to unseen classes.

**[S029-zh]** 强劲的少样本性能可归因于 TerraMind 的双尺度预训练，它同时提供了高层语义表征（有助于分类）和细粒度空间表征（有助于分割）。生成式预训练目标还鼓励模型学习对未见类别具有良好泛化能力的鲁棒特征。

### 5.5. Fine-tuning Experiments | 微调实验

**[S030]** We finetune TerraMind on a diverse set of downstream tasks to evaluate its transfer learning capabilities. The tasks include semantic segmentation (PASTIS [27], MADOS [70]), change detection (OSCD [15]), object counting (CARPK [32]), and biomass estimation (AGB [1]). We use the standard train/validation splits for each dataset and report the metrics used in the respective communities.

**[S030-zh]** 我们在一组多样化的下游任务上微调 TerraMind，以评估其迁移学习能力。任务包括语义分割（PASTIS [27]、MADOS [70]）、变化检测（OSCD [15]）、目标计数（CARPK [32]）和生物量估算（AGB [1]）。我们使用每个数据集的标准训练/验证划分，并报告各社区使用的指标。

**[S031]** Table 3 presents the finetuning results. TerraMind achieves state-of-the-art or near-state-of-the-art performance on all tasks. On PASTIS, TerraMind improves the previous best mIoU by 2.3 points. On the change detection task OSCD, TerraMind outperforms the previous best by 1.8 points in F1 score. The consistent improvements across diverse tasks demonstrate the effectiveness of TerraMind's multimodal pretraining.

**[S031-zh]** 表 3 展示了微调结果。TerraMind 在所有任务上均达到了最先进或接近最先进的性能。在 PASTIS 上，TerraMind 将此前最佳 mIoU 提升了 2.3 个百分点。在变化检测任务 OSCD 上，TerraMind 在 F1 分数上超越此前最佳结果 1.8 个百分点。跨多样化任务的一致改进证明了 TerraMind 多模态预训练的有效性。

**[T003]** Table 3. Fine-tuning results on downstream tasks. Best results in bold, second best underlined.

**[T003-zh]** 表 3. 下游任务微调结果。最佳结果以粗体显示，次佳以下划线显示。

### 5.6. Thinking in Modalities Experiments | 模态思维实验

**[S032]** We evaluate the Thinking-in-Modalities (TiM) capability by comparing finetuning with and without generated modalities. On the PASTIS semantic segmentation task, we finetune TerraMind with three settings: (i) using only S-2 optical data, (ii) using S-2 plus generated S-1 and DEM data (TiM), and (iii) using all real modalities as an oracle. The results show that TiM improves the mIoU by 3.2 points over the unimodal baseline, closing 68% of the gap to the multimodal oracle.

**[S032-zh]** 我们通过比较使用和不使用生成模态的微调来评估模态思维（TiM）能力。在 PASTIS 语义分割任务上，我们以三种设置微调 TerraMind：(i) 仅使用 S-2 光学数据，(ii) 使用 S-2 加生成的 S-1 和 DEM 数据（TiM），(iii) 使用所有真实模态作为 oracle。结果表明，TiM 将 mIoU 比单模态基线提升了 3.2 个百分点，弥合了与多模态 oracle 之间 68% 的差距。

**[S033]** We also test TiM in a zero-shot setting where we generate missing modalities for inference only (without finetuning with generated data). Even in this setting, generating synthetic S-1 and DEM data improves the zero-shot mIoU by 1.8 points. This demonstrates that TiM is effective both as a finetuning augmentation strategy and as an inference-time enhancement.

**[S033-zh]** 我们还在零样本设置中测试了 TiM，即仅生成缺失模态用于推理（不使用生成数据进行微调）。即使在这种设置下，生成合成 S-1 和 DEM 数据也将零样本 mIoU 提升了 1.8 个百分点。这证明 TiM 既是一种有效的微调增强策略，也是一种推理时增强手段。

---

## 6. Conclusion | 结论

**[S034]** We presented TerraMind, the first any-to-any generative multimodal foundation model for Earth observation. Through dual-scale pretraining on pixel-level and token-level representations, TerraMind learns rich cross-modal relationships while preserving fine-grained spatial details. Our experiments demonstrate state-of-the-art performance on community benchmarks, strong zero-shot and few-shot capabilities, and the novel Thinking-in-Modalities paradigm. We believe TerraMind opens new directions for multimodal Earth observation research and applications.

**[S034-zh]** 本文提出了 TerraMind，首个面向地球观测的任意到任意生成式多模态基础模型。通过在像素级和 token 级表征上的双尺度预训练，TerraMind 学习了丰富的跨模态关系，同时保留了细粒度的空间细节。我们的实验表明，TerraMind 在社区基准测试上达到了最先进的性能，具有强大的零样本和少样本能力，以及新颖的模态思维范式。我们相信 TerraMind 为多模态地球观测研究和应用开辟了新的方向。

**[S035]** Future work includes extending TerraMind to additional modalities (e.g., hyperspectral, LiDAR), exploring larger model scales, and investigating the potential of TiM for other domains beyond Earth observation. We also plan to release updated versions of the TerraMesh dataset with increased temporal coverage and additional geographical regions.

**[S035-zh]** 未来工作包括将 TerraMind 扩展到额外的模态（如高光谱、LiDAR）、探索更大的模型规模，以及研究 TiM 在地球观测之外其他领域的潜力。我们还计划发布 TerraMesh 数据集的更新版本，增加时间覆盖范围和额外的地理区域。

---

## Appendices | 附录

> 以下附录章节包含补充的实验细节和额外结果。由于论文篇幅限制，核心章节优先翻译，附录提供概述性翻译。

### 7. TerraMesh Dataset Details | TerraMesh 数据集详情

**[S036]** The TerraMesh dataset is constructed from multiple public geospatial datasets. The primary optical data source is Sentinel-2 from the Copernicus program, with both L1C (top-of-atmosphere) and L2A (bottom-of-atmosphere) products. For SAR data, we use Sentinel-1 GRD and RTC products. The DEM is from the Copernicus DEM 30m dataset, resampled to 10m. LULC labels combine ESA WorldCover 2021 and OpenStreetMap land use data. NDVI is computed from Sentinel-2 NIR and Red bands.

**[S036-zh]** TerraMesh 数据集由多个公共地理空间数据集构建而成。主要光学数据源是来自哥白尼计划的 Sentinel-2，包括 L1C（大气层顶）和 L2A（地表）产品。对于 SAR 数据，我们使用 Sentinel-1 GRD 和 RTC 产品。DEM 来自 Copernicus DEM 30m 数据集，重采样至 10m。LULC 标签结合了 ESA WorldCover 2021 和 OpenStreetMap 土地利用数据。NDVI 由 Sentinel-2 近红外和红光波段计算得到。

**[S037]** Data preprocessing includes cloud masking for optical imagery, speckle filtering for SAR data, and spatial alignment of all modalities to a common 10m grid. We filter out patches with more than 50% cloud cover or missing modalities. The final dataset contains approximately 12 million unique patches covering all continents.

**[S037-zh]** 数据预处理包括光学影像的云掩膜、SAR 数据的斑点滤波，以及所有模态向共同 10m 网格的空间对齐。我们过滤掉云覆盖超过 50% 或模态缺失的 patch。最终数据集包含约 1200 万个唯一 patch，覆盖所有大洲。

### 8. Pretraining Details | 预训练详情

**[S038]** TerraMind is pretrained using the AdamW optimizer with a peak learning rate of 1e-4 and a cosine decay schedule. We use a batch size of 4096 tokens per device across 64 NVIDIA A100 GPUs. The pretraining runs for approximately 500,000 steps, equivalent to one epoch over the TerraMesh dataset. We employ gradient clipping with a maximum norm of 1.0 and use mixed-precision training (bfloat16).

**[S038-zh]** TerraMind 使用 AdamW 优化器进行预训练，峰值学习率为 1e-4，采用余弦衰减调度。我们在 64 块 NVIDIA A100 GPU 上每设备使用 4096 个 token 的批量大小。预训练运行约 500,000 步，相当于 TerraMesh 数据集的一个 epoch。我们采用最大范数为 1.0 的梯度裁剪，并使用混合精度训练（bfloat16）。

**[S039]** For the tokenizer pretraining, we train for 100,000 steps with a learning rate of 5e-4. The FSQ codebook size is set to 64,096 entries with a dimension of 8 per entry. The diffusion decoder uses a U-Net architecture with 4 resolution levels and attention at the lowest resolution.

**[S039-zh]** 对于 tokenizer 预训练，我们训练 100,000 步，学习率为 5e-4。FSQ 码本大小设置为 64,096 个条目，每个条目维度为 8。扩散解码器使用具有 4 个分辨率级别的 U-Net 架构，在最低分辨率处使用注意力机制。

### 9. Tokenizer Performance | Tokenizer 性能

**[S040]** We compare FSQ against VQ for image tokenization. Table 4 shows that FSQ achieves better reconstruction quality (PSNR and SSIM) while using a smaller codebook. FSQ also exhibits better codebook utilization, with over 95% of codebook entries being used, compared to only 60% for VQ.

**[S040-zh]** 我们比较了 FSQ 与 VQ 在图像 token 化上的性能。表 4 显示 FSQ 实现了更好的重建质量（PSNR 和 SSIM），同时使用了更小的码本。FSQ 还表现出更好的码本利用率，超过 95% 的码本条目被使用，而 VQ 仅为 60%。

**[T004]** Table 4. Comparison of FSQ and VQ tokenizers on validation set. Reconstruction quality measured by PSNR (dB) and SSIM.

**[T004-zh]** 表 4. FSQ 与 VQ tokenizer 在验证集上的对比。重建质量以 PSNR（dB）和 SSIM 度量。

### 10. Additional Experiments | 额外实验

**[S041]** Geolocation prediction. We evaluate TerraMind's ability to predict the geographic coordinates of an image patch given only the visual content. This task tests whether the model has learned implicit geospatial knowledge during pretraining. On a held-out test set, TerraMind predicts coordinates within 50km accuracy for 73% of patches, demonstrating strong geospatial awareness.

**[S041-zh]** 地理位置预测。我们评估 TerraMind 仅给定视觉内容预测图像 patch 地理坐标的能力。该任务测试模型在预训练期间是否学到了隐式地理空间知识。在留出测试集上，TerraMind 对 73% 的 patch 实现了 50km 以内的坐标预测精度，展示了强劲的地理空间感知能力。

**[S042]** Few-shot comparisons with baseline models. Table 5 provides a detailed comparison of TerraMind against Prithvi, SatMAE, and Scale-MAE on 1-shot, 5-shot, and 16-shot learning. TerraMind consistently outperforms across all shot settings and tasks, with the largest improvements observed in low-data regimes (1-shot and 5-shot).

**[S042-zh]** 与基线模型的少样本对比。表 5 提供了 TerraMind 与 Prithvi、SatMAE 和 Scale-MAE 在 1-shot、5-shot 和 16-shot 学习上的详细对比。TerraMind 在所有 shot 设置和任务中始终表现更优，在数据量低的场景（1-shot 和 5-shot）中观察到最大的改进。

**[T005]** Table 5. Few-shot learning comparison on PANGAEA benchmark. mIoU scores for 1-shot, 5-shot, and 16-shot settings.

**[T005-zh]** 表 5. PANGAEA 基准测试上的少样本学习对比。1-shot、5-shot 和 16-shot 设置的 mIoU 分数。

### 11. Any-to-any Generation | 任意到任意生成

**[S043]** We provide additional examples of TerraMind's any-to-any generation capability. Starting from different input modalities (optical S-2, SAR S-1, DEM), TerraMind can generate all other modalities. Figure 3 shows qualitative examples of chained generation where the model generates S-1, DEM, and LULC from optical input. The generated outputs are visually plausible and capture the main structures of the target modalities.

**[S043-zh]** 我们提供了 TerraMind 任意到任意生成能力的额外示例。从不同的输入模态（光学 S-2、SAR S-1、DEM）出发，TerraMind 可以生成所有其他模态。图 3 展示了链式生成的定性示例，其中模型从光学输入生成 S-1、DEM 和 LULC。生成的输出在视觉上是合理的，并捕捉到了目标模态的主要结构。

**[S044]** Quantitative evaluation of generation quality is provided in Table 6. The highest quality is achieved when generating LULC from optical data (SSIM 0.87), while radar-to-optical generation is more challenging (SSIM 0.62). This asymmetry reflects the inherent differences in information content between modalities.

**[S044-zh]** 生成质量的定量评估见表 6。从光学数据生成 LULC 的质量最高（SSIM 0.87），而雷达到光学的生成更具挑战性（SSIM 0.62）。这种不对称性反映了模态之间信息含量的固有差异。

**[T006]** Table 6. Any-to-any generation quality. SSIM scores for generation between all modality pairs.

**[T006-zh]** 表 6. 任意到任意生成质量。所有模态对之间生成的 SSIM 分数。

---

## References | 参考文献

> 以下列出论文引用的主要参考文献。完整参考文献列表请参见原文。

**[S045]** [1] AGB Dataset. Global Above-Ground Biomass.  
**[S046]** [3] Alayrac et al. Flamingo: A Visual Language Model for Few-Shot Learning. NeurIPS 2022.  
**[S047]** [4] Audebert et al. Deep Learning for Classification of Hyperspectral Data.  
**[S048]** [6] GFMSwim. Geospatial Foundation Model with Swin Transformer.  
**[S049]** [9] Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020.  
**[S050]** [12] Copernicus DEM. European Space Agency Digital Elevation Model.  
**[S051]** [14] Cong et al. SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. NeurIPS 2022.  
**[S052]** [15] Daudt et al. OSCD: Onera Satellite Change Detection Dataset.  
**[S053]** [16] Helber et al. EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.  
**[S054]** [17] Dosovitskiy et al. An Image is Worth 16x16 Words. ICLR 2021.  
**[S055]** [18] Multimodal SAR-Optical Fusion.  
**[S056]** [19] Disaster Response with Satellite Imagery.  
**[S057]** [20] Lobell et al. Land Cover Mapping.  
**[S058]** [21] Land Cover Classification Methods.  
**[S059]** [23] Gemini Team. Gemini: A Family of Highly Capable Multimodal Models.  
**[S060]** [24] Jia et al. Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision. ICML 2021.  
**[S061]** [27] PASTIS: Panoptic Agricultural Satellite Imagery Dataset.  
**[S062]** [30] Multimodal Deep Learning.  
**[S063]** [32] CARPK Dataset.  
**[S064]** [33] WordPiece Tokenizer.  
**[S065]** [34] Ho et al. Denoising Diffusion Probabilistic Models. NeurIPS 2020.  
**[S066]** [36] Transfer Learning with Pretrained Models.  
**[S067]** [37] Jakubik et al. Prithvi: A Foundational Model for Earth Observation.  
**[S068]** [39] Object Detection in Remote Sensing.  
**[S069]** [41] OpenStreetMap.  
**[S070]** [43] OpenAI. GPT-4o.  
**[S071]** [45] Geospatial Foundation Models Survey.  
**[S072]** [46] Geospatial Heterogeneity.  
**[S073]** [49] PANGAEA Benchmark.  
**[S074]** [50] Generic CV Architectures.  
**[S075]** [51] Mentzer et al. Finite Scalar Quantization: VQ-VAE Made Simple.  
**[S076]** [52] Two-Stage Pretraining.  
**[S077]** [53] Biomass Estimation.  
**[S078]** [55] Vessel Identification.  
**[S079]** [57] Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.  
**[S080]** [58] Reed et al. Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning.  
**[S081]** [59] Change Detection Methods.  
**[S082]** [62] CV for EO Adaptation.  
**[S083]** [64] Self-Supervised Learning Survey.  
**[S084]** [65] BigEarthNet and GFM-SS.  
**[S085]** [70] MADOS Dataset.  
**[S086]** [71] Cross-Attention Fusion.  
**[S087]** [72] Semantic Segmentation in Remote Sensing.  
**[S088]** [73] Yuan et al. Florence: A New Foundation Model for Computer Vision.  
**[S089]** [75] CNNs for Remote Sensing.  
**[S090]** [76] Computer Vision in Earth Observation.  
**[S091]** [85] ESA WorldCover.

---

*Reader generated on 2026-05-27. Core sections (Abstract–Conclusion) fully translated. Appendices summarized. Full paper: 25 pages.*
