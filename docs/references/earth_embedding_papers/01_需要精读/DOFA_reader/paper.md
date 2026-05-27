# DOFA: A Dynamic Weight Factorization Approach for Unified Earth Observation Representation

## Metadata
- **Authors:** Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J. Stewart, Joëlle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le Saux, Gustau Camps-Valls, Xiao Xiang Zhu
- **arXiv:** 2403.15356
- **Pages:** 19
- **PDF:** `DOFA_动态多传感器统一表示.pdf`

## Page Index
- **Page 1:** [S001](#s001) [S002](#s002) [S003](#s003)
- **Page 2:** [S004](#s004) [C001](#c001) [S005](#s005)
- **Page 3:** [S006](#s006) [S007](#s007) [C002](#c002) [S008](#s008)
- **Page 4:** [S009](#s009) [S010](#s010) [S011](#s011) [S012](#s012)
- **Page 5:** [C003](#c003) [C004](#c004) [S015](#s015)
- **Page 6:** [C005](#c005) [S017](#s017)
- **Page 7:** [S018](#s018) [S019](#s019) [S020](#s020)
- **Page 8:** [S021](#s021) [S022](#s022)
- **Page 9:** [T001](#t001) [T002](#t002) [T003](#t003)
- **Page 10:** [S025](#s025) [S026](#s026) [S027](#s027) [T004](#t004) [S028](#s028)
- **Page 11:** [T005](#t005) [T006](#t006) [T007](#t007)
- **Page 12:** [T008](#t008) [C006](#c006) [S033](#s033) [T009](#t009)
- **Page 13:** [C007](#c007) [S036](#s036) [T010](#t010) [T011](#t011)
- **Page 14:** [S038](#s038) [C008](#c008)
- **Page 15:** [S039](#s039) [S040](#s040) [S041](#s041)
- **Page 16:** [S042](#s042) [S043](#s043)
- **Page 17:** [S044](#s044)
- **Page 18:** [S045](#s045)

---

## Page 1

<a id="s001"></a>
**Source:** p.1 S001

**Original:**
```
Neural Plasticity-Inspired Multimodal Foundation Model for
Earth Observation Zhitong Xiong · Yi Wang · Fahong Zhang · Adam J. Stewart · Jo¨elle
Hanna · Damian Borth · Ioannis Papoutsis · Bertrand Le Saux ·
Gustau Camps-Valls · Xiao Xiang Zhu* Received: date / Accepted: date Abstract Earth observation (EO) in open-world set-
tings presents a unique challenge: different applications
rely on diverse sensor modalities, each with varying
ground sampling distances, spectral ranges, and num-
bers of spectral bands. However, existing EO founda-
tion models are typically tailored to specific sensor types,
making them inflexible when generalizing across the
heterogeneous landscape of EO data. To address this,
we propose the Dynamic One-For-All (DOFA) model, a
unified, multimodal foundation framework designed for
diverse vision tasks in EO. Inspired by neural plastic-
ity, DOFA utilizes a wavelength-conditioned dynamic
hypernetwork to process inputs from five distinct satel-
lite sensors flexibly. By continually pretraining on five
EO modalities, DOFA achieves state-of-the-art perfor-
```

**中文:**
受神经可塑性启发的多模态地球观测基础模型

作者：Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J. Stewart, Joëlle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le Saux, Gustau Camps-Valls, Xiao Xiang Zhu*

<a id="s002"></a>
**Source:** p.1 S002

**Original:**
```
Abstract (continued): mance across multiple downstream tasks and general-
izes well to unseen modalities. Enhanced with hybrid
continual pretraining, DOFA+ requires significantly fewer
computational resources while outperforming counter-
parts trained with extensive GPU budgets. Experiments
on diverse datasets highlight DOFA’s potential as a
foundation for general-purpose vision models in the sensor-
diverse EO domain. The code and pre-trained weights
are publicly available at https://github.com/zhu-xlab/
DOFA. Keywords Dynamic foundation models · Earth
observation · Multimodal learning
```

**中文:**
摘要（续）：在多个下游任务上取得了最先进的性能，并能很好地泛化到未见的模态。通过混合持续预训练增强的 DOFA+ 需要显著更少的计算资源，同时超越了使用大量 GPU 预算训练的同类模型。在多样化数据集上的实验突出了 DOFA 作为传感器多样化地球观测领域中通用视觉模型基础的潜力。代码和预训练权重已公开在 https://github.com/zhu-xlab/DOFA。关键词：动态基础模型 · 地球观测 · 多模态学习。

<a id="s003"></a>
**Source:** p.1 S003

**Original:**
```
1 Introduction Earth observation (EO) through satellite remote sens-
ing provides unparalleled opportunities for modeling
Earth’s surface dynamics, addressing global challenges
such as climate change, urbanization, and biodiversity
loss [6, 5, 37]. Core vision tasks like classification, de-
tection, and segmentation support critical applications
from disaster response to resource management. Achiev-
ing this requires integrating data from diverse sensors
with varying spectral and spatial characteristics, mak-
ing EO an open-world visual understanding problem.
Foundation models (FMs) offer a scalable solution by
enabling generalization across modalities, reducing task-
specific supervision, and serving as a unified backbone
for downstream EO tasks, as illustrated in Fig. 1.
Existing FMs, though promising, remain rigid and
inflexible, typically pretrained using fixed spectral band
configurations or specialized for individual EO modal-
ities. For instance, models such as GFM [32], Scale-
MAE [36], SatMAE [10], CROMA [14], and Spectral-
GPT [20] illustrate these limitations. This severely re- arXiv:2403.15356v3  [cs.CV]  15 Oct 2025
```

**中文:**
1 引言

通过卫星遥感进行的地球观测（Earth Observation, EO）为建模地球表面动态提供了无与伦比的机会，能够应对气候变化、城市化和生物多样性丧失等全球性挑战 [6, 5, 37]。分类、检测和分割等核心视觉任务支撑着从灾害响应到资源管理的关键应用。实现这些目标需要整合来自具有不同光谱和空间特征的多种传感器的数据，使地球观测成为一个开放世界的视觉理解问题。基础模型（Foundation Models, FMs）通过实现跨模态泛化、减少任务特定监督需求，并作为下游地球观测任务的统一骨干网络，提供了一种可扩展的解决方案，如图 1 所示。

---

## Page 2

<a id="s004"></a>
**Source:** p.2 S004

**Original:**
```
stricts their adaptability in dynamic real-world EO sce-
narios, where new sensors and spectral band configu-
rations continuously emerge. Consequently, extensive
computational and human resources are required to
adapt them to unseen sensors and spectral combina-
tions. In summary, existing approaches that develop
separate foundation models or use isolated visual en-
coders for multi-sensor data fail to capture inter-sensor
relationships, resulting in key limitations: 1. The learned representation may not effectively cap-
ture such an intersensor relationship.
2. The performance of FMs will degrade when down-
stream tasks utilize data from unseen sensors with
varying numbers of spectral bands.
3. The development of individual FMs requires consid-
erably more computing resources and is not flexible
in real-world applications. Addressing this critical limitation aligns with a central
theme of open-world EO visual understanding: develop-
ing FMs capable of dynamically and efficiently adapt-
ing to downstream applications with flexible input data
modalities.
In response to this challenge, we propose the Dy-
namic One-For-All (DOFA) model, a versatile, adap-
tive multimodal foundation model explicitly designed
for the EO domain. Inspired by neuroplasticity, the
brain’s dynamic reorganization capacity in response to
novel stimuli [12, 27], DOFA integrates a wavelength-
conditioned dynamic hypernetwork within a unified Trans-
former architecture, as shown in Fig. 2. This enables
DOFA to flexibly accommodate varying spectral bands
and sensor modalities, including those unseen during
initial pre-training. Specifically, DOFA utilizes wave-
length as a unifying parameter across various EO modal-
ities to achieve a more cohesive multimodal represen-
tation. At its core, the model integrates a hypernet-
work [17] that dynamically generates network weights
based on the central wavelengths of each spectral band.
This dynamic weight generator adjusts network weights
to align with the specific modality of the input data, fa-
cilitating a customized network for each modality. Ad-
ditionally, DOFA integrates a shared vision backbone,
acting as a universal feature learning module for all
heterogeneous data modalities.
Pretraining unified FMs across diverse EO modali-
ties presents a significant challenge due to the need for
substantial computational resources. To address this,
DOFA adopts continuous pretraining via masked im-
age modeling (MIM) and knowledge distillation, signif-
icantly reducing computational cost. Building on this,
we further propose DOFA+, which is initialized from
a powerful open-source pre-trained vision model (DI-
NOv2 [35]) and applies the MIM objective to enable SAR Images Unified Multimodal Earth Foundation Model NAIP RGB Downstream Tasks
Sentinel 1 Sentinel 2 EnMAP RGB-NIR Earth Observation potassium magnesium PH phosphorus
Soil Parameter Wildfire Flood mapping Land cover Forest
Monitoring High resolution
Optical Images Multispectral Hyperspectral Multispectral Different Sensors
```

**中文:**
这严重限制了它们在动态真实世界地球观测场景中的适应性，因为在这些场景中，新的传感器和光谱波段配置不断涌现。因此，需要大量的计算和人力资源来使它们适应未见的传感器和光谱组合。总之，现有方法——无论是开发独立的基础模型还是使用隔离的视觉编码器处理多传感器数据——都未能捕捉传感器间的关系，导致以下关键局限性：

1. 学习到的表示可能无法有效捕捉这种传感器间关系。
2. 当下游任务使用来自具有不同光谱波段数量的未见传感器的数据时，基础模型的性能会下降。
3. 开发独立的基础模型需要相当多的计算资源，并且在实际应用中不够灵活。

解决这一关键局限性符合开放世界地球观测视觉理解的一个核心主题：开发能够动态且高效地适应具有灵活输入数据模态的下游应用的基础模型。

<a id="c001"></a>
**Source:** p.2 C001

**Figure Caption:**
```
Fig. 1: Motivation of DOFA. Our primary purpose is
to develop versatile foundation models capable of adap-
tively processing various EO data modalities.
```

**中文:**
图 1：DOFA 的动机。我们的主要目标是开发能够自适应处理各种地球观测数据模态的通用基础模型。

<a id="s005"></a>
**Source:** p.2 S005

**Original:**
```
efficient domain adaptation, without altering the un-
derlying architecture. Through a hierarchical distilla-
tion strategy, it preserves strong semantic priors from
the source model while guiding the learning of EO-
specific visual patterns through local reconstruction.
With parameter-efficient fine-tuning, DOFA and DOFA+
enable rapid, label-efficient adaptation to a wide range
of multimodal EO tasks, including image classification,
semantic segmentation, object detection, and environ-
mental change detection. Our contributions can be summarized as follows: 1. We introduce DOFA, a neuroplasticity-inspired ar-
chitecture that uses wavelength as a unifying input
across EO modalities. A wavelength-conditioned dy-
namic hypernetwork enables flexible adaptation to
varying and unseen spectral bands within a single
unified Transformer framework.
2. The proposed hypernetwork-based architecture en-
ables efficient continual pretraining across diverse
EO modalities by interpolating in weight space ac-
cording to wavelength configurations. Together with
wavelength-aware MIM and feature distillation, this
enables efficient EO domain adaptation with mini-
mal data and compute.
3. We further introduce DOFA+. DOFA+ seeds itself
with a strong vision prior and then employs a dual
training strategy: (i) wavelength-aware MIM to cap-
ture EO-specific spatial patterns, and (ii) hierarchi-
cal feature distillation to align and refine the inher-
ited semantic representations from the vision prior.
4. Extensive experiments demonstrate that DOFA and
DOFA+ achieve state-of-the-art performance across
a wide range of EO tasks and modalities. Our mod-
els generalize well to unseen sensors and diverse
spectral configurations without retraining, offering
greater flexibility in open-world settings with re-
duced computational costs.
```

**中文:**
为应对这一挑战，我们提出了动态万能模型（Dynamic One-For-All, DOFA），一种专门为地球观测领域设计的通用、自适应多模态基础模型。受神经可塑性（neuroplasticity）——即大脑对新颖刺激进行动态重组的能力 [12, 27]——的启发，DOFA 在一个统一的 Transformer 架构中集成了波长条件化的动态超网络（hypernetwork），如图 2 所示。这使得 DOFA 能够灵活地适应变化的光谱波段和传感器模态，包括在初始预训练期间未见过的模态。具体而言，DOFA 利用波长作为统一各种地球观测模态的参数，以实现更具凝聚力的多模态表示。该模型的核心集成了一个超网络 [17]，该网络根据每个光谱波段的中心波长动态生成网络权重。这种动态权重生成器调整网络权重以与输入数据的特定模态对齐，为每种模态促进定制化的网络。此外，DOFA 集成了一个共享的视觉骨干网络，作为所有异构数据模态的通用特征学习模块。

在多样化的地球观测模态上预训练统一的基础模型由于需要大量计算资源而构成重大挑战。为解决这一问题，DOFA 采用通过掩码图像建模（Masked Image Modeling, MIM）和知识蒸馏进行的持续预训练，显著降低了计算成本。在此基础上，我们进一步提出了 DOFA+，它从一个强大的开源预训练视觉模型（DINOv2 [35]）初始化，并应用 MIM 目标来使模型适应地球观测领域。通过分层蒸馏策略，它在保留源模型强语义先验的同时，通过局部重建引导地球观测特定视觉模式的学习。凭借参数高效的微调，DOFA 和 DOFA+ 能够快速、标签高效地适应广泛的多模态地球观测任务，包括图像分类、语义分割、目标检测和环境变化检测。

我们的贡献可总结如下：
1. 我们引入了 DOFA，一种受神经可塑性启发的架构，使用波长作为跨地球观测模态的统一输入。波长条件化的动态超网络能够在单一统一 Transformer 框架内灵活适应变化和未见的光谱波段。
2. 所提出的基于超网络的架构通过根据波长配置在权重空间中进行插值，实现了跨多样化地球观测模态的高效持续预训练。结合波长感知的 MIM 和特征蒸馏，这能够以最少的数据和计算实现高效的地球观测领域适应。
3. 我们进一步引入了 DOFA+。DOFA+ 以强大的视觉先验为种子，然后采用双重训练策略：(i) 波长感知 MIM 以捕捉地球观测特定的空间模式，(ii) 分层特征蒸馏以对齐和精炼从视觉先验继承的语义表示。
4. 大量实验表明，DOFA 和 DOFA+ 在广泛的地球观测任务和模态上达到了最先进的性能。我们的模型能够很好地泛化到未见的传感器和多样化的光谱配置，无需重新训练，在开放世界环境中以更低的计算成本提供了更大的灵活性。

---

## Page 3

<a id="s006"></a>
**Source:** p.3 S006

**Original:**
```
Extensive experiments across 20+ EO tasks validate
DOFA’s robust adaptability, efficiency, and scalability.
Our approach represents a meaningful advancement to-
ward open-world EO visual understanding, emphasiz-
ing continual multimodal learning and sustainable adapt-
ability within dynamic and heterogeneous real-world
EO environments.
```

**中文:**
在 20 多个地球观测任务上的大量实验验证了 DOFA 强大的适应性、效率和可扩展性。我们的方法代表了向开放世界地球观测视觉理解的有意义的进步，强调了在动态和异构的真实世界地球观测环境中持续多模态学习和可持续适应性。

<a id="s007"></a>
**Source:** p.3 S007

**Original:**
```
2 Related Work Early efforts to develop EO foundation models were de-
voted to generating effective embeddings for data from
a single modality. For example, SeCo [30] and CACo [29]
leverage temporal information from acquired images to
learn temporal-sensitive and temporal-invariant feature
representations. GFM [32] devises a continual pretrain-
ing paradigm that leverages ImageNet pretrained fea-
tures to accelerate model convergence on EO data. Cha
et al. [8] explore the impact of scaling up the number of
parameters in foundation models, specifically on Google
Earth images. Another line of research addresses the
adaptability of feature representations across EO data
with different ground sample distances (GSD).
RingMo [53] introduces a patch-incomplete mask
strategy during the masked image modeling phase, pre-
venting the oversight of small objects within a single
patch. Scale-MAE [36] takes a different approach by
substituting the positional encoding within ViT [13]
with a GSD positional encoding, incorporating GSD
information into the representation learning process.
USat [21] adopts a strategy of encoding a higher num-
ber of patches for bands with lower GSD and a lower
number of patches for bands with higher GSD. Another
significant research question is how to achieve a unified
representation for different modalities, such as RGB,
multispectral, hyperspectral, and radar data. In this
regard, SSL4EO-S12 [49] integrates the features from
multispectral and SAR modalities using an early fu-
sion strategy. SatMAE [10] suggests grouping subsets
of spectral bands and adding a spectral encoding to
each spectral group.
Real-world data are characterized by diverse modal-
ities, including but not limited to images, videos, text,
audio, depth information, and point clouds. The ca-
pacity of foundation models to effectively handle this
variety of downstream tasks hinges on their ability to
process multimodal data. In this context, OFA-Net [52]
proposes using modality-specific patch embedding lay-
ers to learn unified representations across diverse EO
modalities. It further demonstrates that a single shared
Transformer backbone is both sufficient and effective for
capturing generalizable representations spanning mul-
tiple types of EO data. CROMA [14] first develops two Before Change
After Change
Multi-modal Dynamic Transformer NAIP RGB Shared Transformer Linear Linear Norm Multi-Head Attention Norm MLP Dynamic Projection
output
Input (x) Wbi,j Linear Input (x) Wai,j output Linear Dynamic
Weight Generator New Modality Sentinel 2 Wai,j Weights for Sentinel 2 VTA Before Change
After Change Environment, emotions, learning, and 
other surrounding factors Neural plasticity 
a #bands: 13 wave le ngth wavelength #bands: 13 #bands: 3 Neural connections adaptively rewire
in response to change.
```

**中文:**
2 相关工作

早期开发地球观测基础模型的努力致力于生成来自单一模态数据的有效嵌入。例如，SeCo [30] 和 CACo [29] 利用获取图像的时间信息来学习时间敏感和时间不变的特征表示。GFM [32] 设计了一种持续预训练范式，利用 ImageNet 预训练特征来加速模型在地球观测数据上的收敛。Cha 等人 [8] 探索了在基础模型中扩大参数数量对 Google Earth 图像的影响。另一研究方向解决了跨具有不同地面采样距离（Ground Sample Distance, GSD）的地球观测数据的特征表示适应性问题。

RingMo [53] 在掩码图像建模阶段引入了一种 patch 不完整掩码策略，防止忽略单个 patch 内的小目标。ScaleMAE [36] 采用不同的方法，用 GSD 位置编码替代 ViT [13] 中的位置编码，将 GSD 信息纳入表示学习过程。USat [21] 采用对低 GSD 波段编码更多 patch、对高 GSD 波段编码更少 patch 的策略。另一个重要的研究问题是如何实现 RGB、多光谱、高光谱和雷达数据等不同模态的统一表示。在这方面，SSL4EO-S12 [49] 使用早期融合策略整合多光谱和 SAR 模态的特征。SatMAE [10] 建议将光谱波段分组，并为每个光谱组添加光谱编码。

<a id="c002"></a>
**Source:** p.3 C002

**Figure Caption:**
```
Fig. 2: Motivation
and
main
architecture
of
DOFA. We design DOFA to emulate the Neuroplas-
ticity [19, 55, 11] mechanism for processing multimodal
EO data. (1) Illustration of the brain’s capability to
adapt its structure and function to learned informa-
tion, experience, or injury. (2) Illustration of the core
idea: DOFA is designed to adaptively alter its network
weights in response to novel data modalities.
```

**中文:**
图 2：DOFA 的动机和主要架构。我们设计 DOFA 来模拟神经可塑性 [19, 55, 11] 机制以处理多模态地球观测数据。(1) 大脑适应其结构和功能以响应学习到的信息、经验或损伤的能力示意图。(2) 核心思想示意图：DOFA 旨在自适应地改变其网络权重以响应新颖的数据模态。

<a id="s008"></a>
**Source:** p.3 S008

**Original:**
```
unimodal encoders to encode multispectral and SAR
data individually. Subsequently, it utilizes a cross-modal
radar-optical Transformer that leverages cross-attention
to extract the unified representation. DeCUR [48] is a
bi-modal self-supervised foundation model that decou-
ples the unique and common representations between
the two modalities.
SpectralGPT [20] is a foundation model meticulously
tailored for hyperspectral remote sensing data. It de-
signs a 3D masking strategy, an encoder for learning
representations from spatial-spectral mixed tokens, and
a decoder with multi-target reconstruction to preserve
spectral characteristics. Beyond these, efforts have also
been directed towards encoding geo-locational informa-
tion into the feature representation. Notable examples
include GASSL [3], GeoCLIP [7], SatCLIP [23], Sky-
Sense [16] and Tile2Vec [22].
Recent efforts aim to handle the full diversity of EO
data in a single backbone. SkySense V2 [54] unifies op-
tical, SAR, and elevation inputs with modality-prompt
tokens and adaptive patch merging, while AnySat [2] is
a multimodal model based on joint embedding predic-
tive architecture and scale-adaptive spatial encoders.
Panopticon [45] treats co-located multi-sensor images
as natural augmentations, adding cross-channel atten-
tion to remain sensor-agnostic. Galileo [42] couples global
masked modeling with local contrastive objectives, yield-
ing a generalist model that outperforms task-specific
```

**中文:**
CROMA [14] 首先开发两个单模态编码器分别对多光谱和 SAR 数据进行编码，随后利用基于交叉注意力的雷达-光学 Transformer 来提取统一表示。DeCUR [48] 是一种双模态自监督基础模型，解耦两种模态之间的独特和共同表示。

SpectralGPT [20] 是专门为高光谱遥感数据量身定制的基础模型。它设计了 3D 掩码策略、用于从空间-光谱混合 token 学习表示的编码器，以及具有多目标重建的解码器以保留光谱特征。除此之外，研究工作还致力于将地理定位信息编码到特征表示中。著名的例子包括 GASSL [3]、GeoCLIP [7]、SatCLIP [23]、SkySense [16] 和 Tile2Vec [22]。

最近的努力旨在用单一骨干网络处理地球观测数据的全部多样性。SkySense V2 [54] 使用模态提示 token 和自适应 patch 合并来统一光学、SAR 和高程输入，而 AnySat [2] 是一种基于联合嵌入预测架构和尺度自适应空间编码器的多模态模型。Panopticon [45] 将共位的多传感器图像视为自然增强，添加跨通道注意力以保持传感器无关性。Galileo [42] 将全局掩码建模与局部对比目标相结合，产生了一种通用模型，其性能优于任务特定的基线模型。然而，对于现有模型来说，处理下游任务中光谱波段数量变化而不重新训练模型的情况是棘手的。DOFA 和 DOFA+ 通过采用动态权重生成器将光谱波段编码为深度表示学习的动态权重来克服这一问题。

---

## Page 4

<a id="s009"></a>
**Source:** p.4 S009

**Original:**
```
baselines. However, for existing models, it is tricky to
handle situations where the number of spectral bands
changes in downstream tasks without retraining the
models. DOFA and DOFA+ overcome this by employ-
ing a dynamic weight generator to encode spectral bands
into dynamic weights for deep representation learning.
```

**中文:**
然而，对于现有模型来说，处理下游任务中光谱波段数量变化而不重新训练模型的情况是棘手的。DOFA 和 DOFA+ 通过采用动态权重生成器将光谱波段编码为深度表示学习的动态权重来克服这一问题。

<a id="s010"></a>
**Source:** p.4 S010

**Original:**
```
3 Methodology Here, we provide detailed information about the pro-
posed DOFA and DOFA+ models, along with a more
detailed presentation of the training method. 3.1 Preliminary Given an input image X ∈RC×H×W , where H, W,
and C represent the height, width, and number of chan-
nels, respectively, the image is first divided into a patch
sequence. Each patch has a fixed spatial size P × P
with C channels, and thus the image is converted into
N = HW P 2 patches. Each patch is flattened into a vector
and linearly transformed into a D-dimensional embed-
ding. This transformation is represented by a trainable
embedding matrix E ∈RP 2C×D.
Formally, the patch embedding can be described as X = [Xp1; Xp2; . . . ; XpN ],
Xpi ∈RP 2C,
(1) where Xpi is the flattened vector of the i-th patch.
Next, the flattened vectors are linearly projected into
D−dimensional embeddings with a learnable embed-
ding matrix: Z0 = [Xp1E; Xp2E; . . . ; XpN E],
Z0 ∈RN×D,
(2) where Z0 represents the sequence of patch embeddings.
Note that this process can be implemented utilizing a
single convolution layer with a P × P kernel, C input
channels, and D output channels. Class token Xcls, an
additional learnable embedding, is prepended to the se-
quence. Finally, position embeddings are added to re-
tain positional information. Z′ = [Xcls; Z0] + Epos,
Z′ ∈R(N+1)×D.
(3) Here, Epos denotes the position embeddings, and the
resulting Z′ serves as the input to the subsequent layers
of the ViT architecture.
```

**中文:**
3 方法

在此，我们提供关于所提出的 DOFA 和 DOFA+ 模型的详细信息，以及对训练方法的更详细阐述。

3.1 预备知识

给定一个输入图像 X ∈ R^{C×H×W}，其中 H、W 和 C 分别表示高度、宽度和通道数，图像首先被划分为 patch 序列。每个 patch 具有固定的空间大小 P × P 和 C 个通道，因此图像被转换为 N = HW/P² 个 patch。每个 patch 被展平为一个向量，并通过线性变换转换为 D 维嵌入。该变换由一个可训练的嵌入矩阵 E ∈ R^{P²C×D} 表示。

形式上，patch 嵌入可以描述为：
X = [X_{p1}; X_{p2}; ...; X_{pN}],  X_{pi} ∈ R^{P²C}  (1)
其中 X_{pi} 是第 i 个 patch 的展平向量。

接下来，展平向量通过可学习的嵌入矩阵线性投影到 D 维嵌入：
Z_0 = [X_{p1}E; X_{p2}E; ...; X_{pN}E],  Z_0 ∈ R^{N×D}  (2)
其中 Z_0 表示 patch 嵌入序列。请注意，此过程可以使用单个卷积层实现，卷积核大小为 P × P，输入通道为 C，输出通道为 D。类 token X_cls 是一个额外的可学习嵌入，被前置到序列中。最后，添加位置嵌入以保留位置信息。
Z' = [X_cls; Z_0] + E_pos,  Z' ∈ R^{(N+1)×D}  (3)
其中 E_pos 表示位置嵌入，得到的 Z' 作为 ViT 架构后续层的输入。

<a id="s011"></a>
**Source:** p.4 S011

**Subsection Header:**
```
3.2 Architecture overview
```

**中文:**
3.2 架构概述

<a id="s012"></a>
**Source:** p.4 S012

**Original:**
```
The patch embedding layer transforms the input image
into a sequence of embeddings that the self-attention
mechanism of the Transformer can process. A straight-
forward way to handle the input data from different
modalities is to utilize multiple patch embedding lay-
ers to convert data with different spectral wavelengths
into embeddings with the same dimension [52]. Sup-
pose that the input image X of dimensions RC×H×W can originate from various data modalities. Initially, im-
ages from different sources are standardized to height
H and width W. Specifically, we consider five distinct
modalities: Sentinel-1 data (Xs1) with two SAR chan-
nels (R2×H×W ), Sentinel 2 data (Xs2) with nine mul-
tispectral channels (R9×H×W ), Gaofen data (Xg) with
four multispectral channels (R4×H×W ), NAIP imagery
(Xrgb) with three RGB channels (R3×H×W ), and En-
MAP data (Xe) with 202 available hyperspectral chan-
nels (R202×H×W ). Note that, for the sake of simplicity,
we omit the batch size from the notation of tensors.
As illustrated in Fig. 3, the whole architecture fol-
lows the design of masked image modeling (MIM) [18].
The main difference from traditional masked autoen-
coders (MAE) lies in DOFA’s capacity to process in-
put images with various channels. This flexibility is
achieved through a hypernetwork-based dynamic weight
generator. The dynamic weight generator takes inputs
from the spectral wavelength associated with each im-
age channel, and dynamically predicts the patch em-
bedding matrix E for different data modalities. As pre-
sented in Fig. 4, the dynamic weight generator maps
spectral band configurations to a weight space. In open-
world tasks with new spectral ranges or sensors, it in-
terpolates this space to produce appropriate weights
for the given input. The latent representations are then
passed through a series of shared Transformer blocks
for learning generalizable multimodal representations.
Parallel to the dynamic weight generation for the
encoder part of the network, the dynamic decoder is re-
sponsible for reconstructing the output image from the
encoded latent space. Similarly, the dynamic decoder
utilizes another set of dynamically generated weights
to ensure that the reconstructed image matches the
number of spectral bands of the target modality. We
employ a MIM strategy to train this self-supervised ar-
chitecture. The input images are masked randomly, and
the model learns to reconstruct these missing parts. As
the parameters in DOFA are learned across different
modalities, this process helps the model to learn ro-
bust multimodal representations beneficial for various
EO tasks. After the pre-training process, the model can
be transferred to various EO applications without ex-
```

**中文:**
patch 嵌入层将输入图像转换为 Transformer 的自注意力机制可以处理的嵌入序列。处理来自不同模态输入数据的一种直接方法是使用多个 patch 嵌入层将具有不同光谱波长的数据转换为相同维度的嵌入 [52]。假设输入图像 X 的维度为 R^{C×H×W}，可以来自各种数据模态。最初，来自不同来源的图像被标准化为高度 H 和宽度 W。具体而言，我们考虑五种不同的模态：具有两个 SAR 通道的 Sentinel-1 数据（X_{s1} ∈ R^{2×H×W}）、具有九个多光谱通道的 Sentinel-2 数据（X_{s2} ∈ R^{9×H×W}）、具有四个多光谱通道的高分数据（X_g ∈ R^{4×H×W}）、具有三个 RGB 通道的 NAIP 影像（X_{rgb} ∈ R^{3×H×W}），以及具有 202 个可用高光谱通道的 EnMAP 数据（X_e ∈ R^{202×H×W}）。请注意，为简化起见，我们从张量表示中省略了批量大小。

如图 3 所示，整个架构遵循掩码图像建模（MIM）[18] 的设计。与传统掩码自编码器（MAE）的主要区别在于 DOFA 能够处理具有各种通道数的输入图像。这种灵活性通过基于超网络的动态权重生成器实现。动态权重生成器接收与每个图像通道相关联的光谱波长输入，并动态预测不同数据模态的 patch 嵌入矩阵 E。如图 4 所示，动态权重生成器将光谱波段配置映射到权重空间。在具有新光谱范围或传感器的开放世界任务中，它对该空间进行插值以生成给定输入的适当权重。然后将潜在表示传递给一系列共享的 Transformer 块以学习可泛化的多模态表示。

与编码器部分的动态权重生成并行，动态解码器负责从编码的潜在空间重建输出图像。类似地，动态解码器利用另一组动态生成的权重，以确保重建的图像与目标模态的光谱波段数量匹配。我们采用 MIM 策略来训练这种自监督架构。输入图像被随机掩码，模型学习重建这些缺失的部分。由于 DOFA 中的参数跨不同模态学习，此过程帮助模型学习对各种地球观测任务有益的强大多模态表示。预训练过程结束后，模型无需大量重新训练即可迁移到各种地球观测应用，从高分光学影像到多光谱和高光谱感知，这都得益于动态权重生成。

---

## Page 5

<a id="c003"></a>
**Source:** p.5 C003

**Figure Caption:**
```
Fig. 3: Architecture and training details. DOFA builds on masked image modeling, introducing a significant
advancement by processing input images with any number of channels within a single framework.
```

**中文:**
图 3：架构和训练细节。DOFA 建立在掩码图像建模基础上，通过在一个统一框架内处理具有任意通道数的输入图像，引入了重大进步。

<a id="c004"></a>
**Source:** p.5 C004

**Figure Caption:**
```
Fig. 4: Illustration of weight space interpolation for new
sensors or a combination of spectral bands.
```

**中文:**
图 4：权重空间插值示意图，用于新传感器或光谱波段组合。

<a id="s015"></a>
**Source:** p.5 S015

**Original:**
```
tensive retraining thanks to the dynamic weight gen-
eration, from high-resolution optical imaging to multi-
spectral and hyperspectral sensing. 3.3 Wavelength-conditioned dynamic patch embedding To manage the diversity of spectral bands across differ-
ent modalities, we project the data into a latent space
with uniform feature dimensionality using the dynamic
patch embedding layer Fdpe. As described before, we
denote the input image as X ∈RC×H×W . Fig. 5 (1)
illustrates the detailed steps used to compute the dy-
namic weights given the wavelength information of each
channel. Each channel of the input image has a corre-
sponding central wavelength. The wavelengths of an in-
put image with C channels can be represented by λ ∈
RC. To convert the wavelengths to a higher-dimensional
feature space, we encode the wavelengths λ using a 1D sine-cosine positional encoding: Vλ = PE(λ) ∈RC×Dλ,
(4) where Dλ is the dimension of the converted wavelength
feature. The positional encoding PE(λi) for wavelength
λi in channel i is given by: PE(λi, 2k) = sin

λi
100002k/Dλ 
, PE(λi, 2k + 1) = cos

λi
100002k/Dλ 
,
(5) where k = 0, . . . , Dλ 2
−1. The positionally encoded
wavelengths Vλ are further transformed through two
fully-connected layers with residual connections: V′
λ = ReLU(F2(ReLU(F1(Vλ)))) + Vλ,
(6) where F1 and F2 represent the fully-connected layers,
and ReLU denotes the Rectified Linear Unit activation
function [1].
Next, we employ a Transformer encoder [44] layer
with four attention heads to generate the dynamic weights
and bias for each wavelength. Specifically, the embed-
ding V′
λ, Nw learnable query tokens Qw, and one learn-
able bias query token Qb are concatenated together to
form the input to the Transformer encoder: V′′ = TransformerEncoder(Concat(Qw, V′
λ, Qb)). (7) We subsequently extract the embeddings V′′w that cor-
respond to the weight query tokens Qw from V′′, as well
```

**中文:**
无需大量重新训练即可迁移到各种地球观测应用，从高分光学影像到多光谱和高光谱感知，这都得益于动态权重生成。

3.3 波长条件化的动态 patch 嵌入

为了管理不同模态之间光谱波段的多样性，我们使用动态 patch 嵌入层 F_{dpe} 将数据投影到具有统一特征维度的潜在空间。如前所述，我们将输入图像记为 X ∈ R^{C×H×W}。图 5（1）展示了根据每个通道的波长信息计算动态权重的详细步骤。输入图像的每个通道都有一个对应的中心波长。具有 C 个通道的输入图像的波长可以表示为 λ ∈ R^C。为了将波长转换到更高维的特征空间，我们使用一维正弦-余弦位置编码对波长 λ 进行编码：
V_λ = PE(λ) ∈ R^{C×D_λ}  (4)

其中 D_λ 是转换后的波长特征维度。通道 i 中波长 λ_i 的位置编码 PE(λ_i) 由下式给出：
PE(λ_i, 2k) = sin(λ_i / 10000^{2k/D_λ}),
PE(λ_i, 2k+1) = cos(λ_i / 10000^{2k/D_λ})  (5)

其中 k = 0, ..., D_λ/2 - 1。位置编码后的波长 V_λ 进一步通过两个具有残差连接的全连接层进行变换：
V'_λ = ReLU(F_2(ReLU(F_1(V_λ)))) + V_λ  (6)

其中 F_1 和 F_2 表示全连接层，ReLU 表示修正线性单元激活函数 [1]。

接下来，我们使用具有四个注意力头的 Transformer 编码器 [44] 层为每个波长生成动态权重和偏置。具体而言，嵌入 V'_λ、N_w 个可学习的查询 token Q_w 和一个可学习的偏置查询 token Q_b 被拼接在一起，作为 Transformer 编码器的输入：
V'' = TransformerEncoder(Concat(Q_w, V'_λ, Q_b))  (7)

随后我们从 V'' 中提取对应于权重查询 token Q_w 的嵌入 V''_w，以及对应于偏置查询 token Q_b 的嵌入 V''_b。然后，使用两个全连接层生成动态权重和偏置：
M_w = F_w(V''_w + V'_λ) ∈ R^{C×P²D},
M_b = F_b(V''_b) ∈ R^{C×D}  (8)

其中 F_w 和 F_b 分别表示用于权重和偏置生成的全连接层。如数学形式化部分所述，patch 嵌入层可以使用卷积层高效实现。因此，我们将生成的权重重塑为卷积核：
K_conv = Reshape(M_w, [D, C, P, P])  (9)

然后使用动态生成的权重 K_conv 和偏置 M_b 执行 patch 嵌入的卷积操作：
PatchEmbedding := Conv(X, K_conv, M_b)  (10)

其中 Conv 表示卷积操作。利用这种方法，patch 嵌入层独立于输入图像的光谱波段数量。这些层的权重根据每个通道的中心波长以组合方式动态生成。这种机制使模型能够动态地学习模态特定的表示，从而增强其在各种数据域中的适应性和性能。

---

## Page 6

<a id="c005"></a>
**Source:** p.6 C005

**Figure Caption:**
```
Fig. 5: Dynamic weight generator and continual training framework. (1) The central wavelengths of each
band are utilized to derive weights tailored to each wavelength. (2) Continual pretraining process. There is a
distillation and a reconstruction loss.
```

**中文:**
图 5：动态权重生成器和持续预训练框架。(1) 每个波段的中心波长被用于推导针对每个波长定制的权重。(2) 持续预训练过程。包含蒸馏损失和重建损失。

<a id="s017"></a>
**Source:** p.6 S017

**Original:**
```
as the embeddings V′′b associated with the bias query
tokens Qb from V′′. Then, two fully-connected layers
are utilized to generate the dynamic weights and biases: Mw = Fw(V′′
w + V′
λ) ∈RC×P 2D, Mb = Fb(V′′
b) ∈RC×D,
(8) where Fw and Fb denote the fully-connected layers for
weight and bias generation, respectively. As introduced
in the Mathematical formalism section, the patch em-
bedding layer can be implemented efficiently using a
convolution layer. Thus, we reshape the generated weights
into the convolution kernel as: Kconv = Reshape(Mw, [D, C, P, P]),
(9) The convolution operation for patch embedding is then
performed using the dynamically generated weights Kconv
and biases Mb: PatchEmbedding := Conv(X, Kconv, Mb).
(10) where Conv denotes the convolution operation. Utiliz-
ing this approach, the patch embedding layer achieves
independence from the number of spectral bands of the
input images. The weights for these layers are dynami-
cally generated based on the central wavelength of each
channel in a compositional manner. This mechanism
enables the model to learn modality-specific represen-
tations dynamically, thereby enhancing its adaptability
and performance across various data domains. For the wavelength-conditioned dynamic decoder,
we use different parameters to generate dynamic weights
and biases. The computation process is similar to the
dynamic patch embedding layer described before. In the
vanilla masked autoencoder, the final layer of the de-
coder is usually implemented as a fully-connected layer
to convert features from latent space into pixel space.
For the dynamic decoder layer, we follow the same pro-
cess used in the dynamic patch embedding to generate
the dynamic weights. The only difference is that a fully-
```

**中文:**
对于波长条件化的动态解码器，我们使用不同的参数来生成动态权重和偏置。计算过程与前面描述的动态 patch 嵌入层类似。在原始掩码自编码器中，解码器的最后一层通常使用全连接层将潜在空间中的特征转换为像素空间。对于动态解码器层，我们遵循与动态 patch 嵌入相同的过程来生成动态权重。唯一的区别是在解码器中使用全连接层而不是 patch 嵌入层中的卷积层。

然后，两个全连接层被用于生成动态权重和偏置。如公式 (8) 所示，M_w 和 M_b 分别表示生成的动态权重和偏置。如数学形式化部分所述，patch 嵌入层可以使用卷积层高效实现。我们重塑生成的权重为卷积核 K_conv，然后执行卷积操作。利用这种方法，patch 嵌入层独立于输入图像的光谱波段数量。这些层的权重根据每个通道的中心波长动态生成。这种机制使模型能够动态地学习模态特定的表示，从而增强其在各种数据域中的适应性和性能。

---

## Page 7

<a id="s018"></a>
**Source:** p.7 S018

**Original:**
```
connected layer is used in the decoder rather than the
convolution layer in the patch embedding layer.
```

**中文:**
连接层在解码器中使用，而不是在 patch 嵌入层中使用卷积层。

<a id="s019"></a>
**Source:** p.7 S019

**Subsection Header:**
```
3.4 Multimodal continual pretraining
```

**中文:**
3.4 多模态持续预训练

<a id="s020"></a>
**Source:** p.7 S020

**Original:**
```
The self-supervised loss formulation is pivotal for train-
ing our multimodal EO foundation model. The model
leverages the MIM paradigm to avoid the requirement
for spatially aligned multimodal datasets. To reduce the
computational cost of training on extensive datasets, we
propose a continual pretraining strategy in the multi-
modal setting, incorporating a distillation loss inspired
by GFM [32] and a weight initialization strategy.
Considering the varying number of channels in dif-
ferent EO modalities, directly using ImageNet pretrained
weights for continual pretraining is impossible. Instead,
we design a proxy-based distillation method that ex-
tracts optical data as a proxy to ensure representation
similarity between the teacher and student networks. As
illustrated in Fig. 5 (2), for multi-channel input data
with more than three channels, we extract the RGB
channels to form a three-channel input Xp ∈R3×H×W .
We randomly select one channel for Sentinel-1 data
with only two bands and duplicate it to a synthetic
three-channel image. This input Xp is then fed into an
ImageNet-pretrained teacher model to get teacher fea-
tures Ft. Concurrently, the dynamic encoder in DOFA
is also used to encode Xp into student features Fs.
Throughout this procedure, the teacher model’s weights
remain frozen to preserve structured representations
and reduce the computational load during optimization.
We also follow a continual pretraining strategy for
initializing the dynamic weight generator. First, we pre-
train the weight generator to mimic the teacher model’s
patch embedding layer weights. We then use the pre-
trained weights to initialize the dynamic embedding
layer. The training loss comprises two distinct compo-
nents. One is the MIM reconstruction loss, which forces
the model to predict X′ ∈RC×H×W for reconstruct-
ing various data modalities from the full-channel inputs
X. The other is the feature distillation loss, which em-
ploys the cosine similarity between the teacher and stu-
dent feature representations to guide the student model.
Suppose that the encoded feature of the full channel
input is F; then the composite loss function can be for-
mulated as follows: L = 1 N
X i=1
∥Xi −X′
i∥2 −
FP (Fi
s) · Fi
t
∥FP (Fis)∥2 · ∥Fi
t∥2
,
(11) where FP is a linear projection layer, and N is the
number of data samples. The model training is supervised from two distinct
perspectives. First, it leverages the complete spectral
information present in the input to learn cross-modal
features via image reconstruction. Second, it distills
knowledge from extensively pretrained models into di-
verse data modalities using a single dynamic model.
This approach enables efficient and robust feature learn-
ing across various modalities. 3.5 DOFA+: Hybrid Continual Pre-training To further enhance both performance and efficiency, we
extend DOFA with three key improvements:
Adaptation from Strong Priors Starting from
DINOv2 [35] weights, we apply the MIM objective to
enable efficient adaptation to multiple EO modalities.
This preserves the strong semantic priors of DINOv2
while guiding the model to learn EO-specific visual pat-
terns through local reconstruction.
Hierarchical Feature Distillation To complement
the MIM objective, we introduce a hierarchical distilla-
tion strategy that aligns the student’s intermediate rep-
resentations with those of the DINOv2 teacher across
multiple layers.
Compact Pre-training Regime We demonstrate
that high-quality adaptation does not require billion-scale
corpora: DOFA+ attains state-of-the-art performance
after seeing only 410k EO image tiles, much fewer sam-
ples to reduce both cost and carbon footprint.
Building on these refinements, we present DOFA+,
a lightweight, universal Earth-observation encoder that
combines the reconstruction pressure of MAE with the
semantic guidance of a vision foundation model. Dur-
ing training, the student reconstructs randomly masked
patches while simultaneously distilling hierarchical su-
pervision signals from DINOv2. This recipe offers two
practical benefits: 1. Simplicity DOFA+ maintains a single-branch ar-
chitecture with a minimal training objective, avoid-
ing the need for complex multi-task losses or auxil-
iary network modules.
2. Complementary Learning DOFA+ combines lo-
cal detail learning and global semantic alignment in
a lightweight continual pretraining setup, enabling
efficient adaptation to diverse EO tasks without re-
training the model. The training objective of DOFA+ combines a recon-
struction loss from masked image modeling and a multi-
level feature distillation loss based on cosine similarity.
Let X ∈RC×H×W denote the full-spectrum input, and
X′ the reconstructed output. Let Fsi,l and Fti,l rep-
```

**中文:**
自监督损失公式对于训练我们的多模态地球观测基础模型至关重要。该模型利用 MIM 范式来避免对空间对齐的多模态数据集的需求。为了降低在大型数据集上训练的计算成本，我们提出了一种多模态设置下的持续预训练策略，结合了受 GFM [32] 启发的蒸馏损失和权重初始化策略。

考虑到不同地球观测模态中通道数量的差异，直接使用 ImageNet 预训练权重进行持续预训练是不可能的。相反，我们设计了一种基于代理的蒸馏方法，提取光学数据作为代理，以确保教师网络和学生网络之间的表示相似性。如图 5（2）所示，对于具有三个以上通道的多通道输入数据，我们提取 RGB 通道以形成三通道输入 X_p ∈ R^{3×H×W}。对于仅有两个波段的 Sentinel-1 数据，我们随机选择一个通道并将其复制为合成三通道图像。然后将此输入 X_p 送入 ImageNet 预训练的教师模型以获取教师特征 F_t。同时，DOFA 中的动态编码器也用于将 X_p 编码为学生特征 F_s。在整个过程中，教师模型的权重保持冻结，以保留结构化表示并减少优化期间的计算负载。

我们还遵循持续预训练策略来初始化动态权重生成器。首先，我们预训练权重生成器以模仿教师模型的 patch 嵌入层权重。然后，我们使用预训练权重来初始化动态嵌入层。训练损失由两个不同的组成部分。一个是 MIM 重建损失，它迫使模型预测 X' ∈ R^{C×H×W} 以从全通道输入 X 重建各种数据模态。另一个是特征蒸馏损失，它利用教师和学生特征表示之间的余弦相似度来指导学生模型。假设全通道输入的编码特征为 F，则复合损失函数可以表述为：
L = (1/N) Σ_i ||X_i - X'_i||² - F_P(F_s^i) · F_t^i / (||F_P(F_s^i)||² · ||F_t^i||²)  (11)

其中 F_P 是线性投影层，N 是数据样本数量。模型训练从两个不同的角度进行监督。首先，它利用输入中存在的完整光谱信息通过图像重建学习跨模态特征。其次，它使用单一动态模型将从广泛预训练模型中获得的知识蒸馏到多样化的数据模态中。这种方法能够在各种模态中实现高效且稳健的特征学习。

3.5 DOFA+：混合持续预训练

为了进一步提升性能和效率，我们通过三个关键改进扩展了 DOFA：

从强先验适应：从 DINOv2 [35] 权重开始，我们应用 MIM 目标来使模型高效适应多种地球观测模态。这保留了 DINOv2 的强语义先验，同时通过局部重建引导模型学习地球观测特定的视觉模式。

分层特征蒸馏：为了补充 MIM 目标，我们引入了一种分层蒸馏策略，将学生的中间表示与 DINOv2 教师网络在多层的表示对齐。

紧凑预训练方案：我们证明高质量的适应不需要十亿规模的语料库：DOFA+ 在仅看到 410k 个地球观测图像块后就达到了最先进的性能，样本数量远少于其他方法，从而降低了成本和碳足迹。

基于这些改进，我们提出了 DOFA+，一种轻量级、通用的地球观测编码器，它结合了 MAE 的重建压力与视觉基础模型的语义指导。在训练期间，学生重建随机掩码的 patch，同时从 DINOv2 蒸馏分层监督信号。这种方案提供了两个实际好处：
1. 简洁性：DOFA+ 保持单分支架构和最小训练目标，避免了复杂的多任务损失或辅助网络模块的需求。
2. 互补学习：DOFA+ 在轻量级持续预训练设置中结合了局部细节学习和全局语义对齐，使模型能够高效适应多样化的地球观测任务而无需重新训练。

DOFA+ 的训练目标结合了来自掩码图像建模的重建损失和基于余弦相似度的多层特征蒸馏损失。设 X ∈ R^{C×H×W} 表示全光谱输入，X' 为重建输出。设 F_s^{i,l} 和 F_t^{i,l} 分别表示第 l 层的学生和教师特征，对应样本 i。

---

## Page 8

<a id="s021"></a>
**Source:** p.8 S021

**Original:**
```
resent the student and teacher features from the l-th
layer, respectively, for sample i.
The total training loss is defined as: L =
1
N N
X i=1
∥Xi −X′i∥2 |
{z
}
MIM Reconstruction Loss −λ· 1 L
X l=1 Fl
P (Fi,l
s ) · Fi,l
t
∥Fl
P (Fi,l
s )∥2 · ∥Fi,l
t ∥2
|
{z
}
Multi-layer Feature Distillation
(12) Here, Fl
P (·) is a linear projection aligning student fea-
tures to the teacher’s dimension, L is the number of
distillation layers, N is the batch size, and λ is a bal-
ancing coefficient.
This training procedure enables DOFA+ to learn se-
mantically rich, transferable representations, support-
ing scalable and general-purpose EO foundation models
with strong cross-modal generalization and low resource
demands.
```

**中文:**
总训练损失定义为：
L = (1/N) Σ_i ||X_i - X'_i||²_{MIM重建损失} - λ · (1/L) Σ_l F_P^l(F_s^{i,l}) · F_t^{i,l} / (||F_P^l(F_s^{i,l})||² · ||F_t^{i,l}||²)_{多层特征蒸馏}  (12)

其中 F_P^l(·) 是将学生特征对齐到教师维度的线性投影层，L 是蒸馏层数，N 是批量大小，λ 是平衡系数。

这种训练过程使 DOFA+ 能够学习语义丰富、可迁移的表示，支持具有强大跨模态泛化能力和低资源需求的可扩展通用地球观测基础模型。

<a id="s022"></a>
**Source:** p.8 S022

**Original:**
```
4 Experiments We conduct extensive experiments across 22 datasets
encompassing various Earth Observation (EO) tasks,
including image classification, semantic segmentation,
object detection, and change detection. We assess model
performance under several settings: linear probing, de-
coder fine-tuning, and full fine-tuning. To demonstrate
generalizability, we use diverse datasets with varying
resolutions, spectral modalities, and geospatial regions.
We emphasize energy-efficient training protocols by lim-
iting training epochs and freezing pre-trained backbones
wherever possible. 4.1 GEO-Bench Experiments 4.1.1 GEO-Bench Classification Experiments There are six image classification datasets provided in
GEO-Bench [24]: m-bigearthnet, m-so2sat, m-brick-kiln,
m-forestnet, m-eurosat, and m-pv4ger. These datasets
span diverse domains and applications, including for-
est monitoring, land use classification, and infrastruc-
ture detection. They are sourced from multiple satellite
platforms, covering different spectral ranges and spatial
resolutions.
On the classification datasets, we follow the com-
mon practice of using RandomResizedCrop (scale 0.8 to
1.0) and RandomHorizontalFlip as data augmentations.
The default crop size is 224×224 for all datasets and
baseline models except SatMAE [10] and CROMA [14],
of which the crop size is 96×96 for SatMAE and 120×120
for CROMA, following the official setup to match their smaller patch size of 8. We optimize cross-entropy loss
for most datasets, except for m-bigearthnet, for which
the multi-label soft margin loss is used. The LARS op-
timizer is utilized with cosine decay to train the last
linear layer of each foundation model for 50 epochs.
Considering the wide range of diversity among existing
foundation models, we employ dataset-specific learn-
ing rates and batch sizes tailored to enhance the per-
formance of classification tasks. We sweep over a grid
search to pick the best learning rate from [0.5, 1.0, 10,
20] for each dataset.
Table 1 presents the classification results. Our pro-
posed DOFA and DOFA+ demonstrate superior perfor-
mance across multiple datasets, validating their strong
generalization capability with minimal training cost.
Regarding the flexibility, DOFA and DOFA+ can be
adapted to diverse EO modalities without changing the
model architectures or retraining any part of the model.
Unlike other models that often require modality-specific
adaptation, DOFA adapts seamlessly to unseen sensors.
For instance, DOFA achieves strong performance on
m-forestnet, despite never seeing the Landsat-8 data
during pretraining. This cross-modal adaptability veri-
fies DOFA’s utility as a general-purpose EO foundation
model. 4.1.2 GEO-Bench Segmentation Experiments The six segmentation datasets include m-pv4ger-seg,
m-chesapeake-landcover, m-cashew-plantation, m-SA-
crop-type, m-nz-cattle, and m-NeonTree. These datasets
cover tasks like solar panel mapping, land cover seg-
mentation, crop classification, and canopy delineation,
across RGB, multispectral, and hyperspectral modali-
ties. We freeze the encoder and train a UPerNet [51]
decoder for segmentation tasks. For all the models ex-
cept the GFM with Swin Transformer, we transform
the features into a feature pyramid with channels 512
and four different scales: 4, 2, 1, 0.5. The UPerNet seg-
mentation head is then used to output the segmenta-
tion results. We use the AdamW optimizer, batch size
64, and an initial learning rate of 0.005 with cosine de-
cay for 20 epochs for segmentation tasks. The learning
rate is relatively stable across datasets for segmentation
tasks.
We use center crop, random rotation, and random
horizontal and vertical flips for segmentation tasks. Im-
ages of each dataset are normalized based on the dataset’s
mean and standard deviation. Table 2 shows that DOFA
and DOFA+ consistently outperform other foundation
models across all datasets. Specifically, DOFA with ViT-
Base and ViT-Large backbones achieves particularly
high accuracy on both the m-NeonTree and m-nz-cattle
```

**中文:**
4 实验

我们在涵盖各种地球观测任务的 22 个数据集上进行了大量实验，包括图像分类、语义分割、目标检测和变化检测。我们在多种设置下评估模型性能：线性探测（linear probing）、解码器微调和完全微调。为了证明泛化能力，我们使用具有不同分辨率、光谱模态和地理空间区域的多样化数据集。我们强调节能训练方案，限制训练轮数并在可能的情况下冻结预训练骨干网络。

4.1 GEO-Bench 实验

4.1.1 GEO-Bench 分类实验

GEO-Bench [24] 提供了六个图像分类数据集：m-bigearthnet、m-so2sat、m-brick-kiln、m-forestnet、m-eurosat 和 m-pv4ger。这些数据集涵盖了森林监测、土地利用分类和基础设施检测等多样化的领域和应用。它们来自多个卫星平台，覆盖不同的光谱范围和空间分辨率。

在分类数据集上，我们遵循使用 RandomResizedCrop（缩放 0.8 到 1.0）和 RandomHorizontalFlip 作为数据增强的常见做法。除 SatMAE [10] 和 CROMA [14] 外，所有数据集和基线模型的默认裁剪大小为 224×224，其中 SatMAE 的裁剪大小为 96×96，CROMA 为 120×120，遵循官方设置以匹配其较小的 patch 大小 8。我们对大多数数据集优化交叉熵损失，除了 m-bigearthnet 使用多标签软边际损失。使用 LARS 优化器配合余弦衰减训练每个基础模型的最后一层线性层 50 个 epoch。考虑到现有基础模型之间的广泛多样性，我们采用针对分类任务性能提升定制的数据集特定学习率和批量大小。我们对网格搜索进行扫描，为每个数据集从 [0.5, 1.0, 10, 20] 中选择最佳学习率。

表 1 展示了分类结果。我们提出的 DOFA 和 DOFA+ 在多个数据集上展示了优越的性能，验证了它们以最小训练成本实现强泛化能力。关于灵活性，DOFA 和 DOFA+ 可以在不改变模型架构或重新训练模型任何部分的情况下适应多样化的地球观测模态。与其他通常需要模态特定适应的模型不同，DOFA 能够无缝适应未见的传感器。例如，尽管 DOFA 在预训练期间从未见过 Landsat-8 数据，但它在 m-forestnet 上取得了强劲的性能。这种跨模态适应性验证了 DOFA 作为通用地球观测基础模型的实用性。

---

## Page 9

<a id="t001"></a>
**Source:** p.9 T001

**Table Caption:**
```
Table 1: Linear probing results on six classification tasks. All models are trained for 50 epochs. The reported
numbers are top-1 overall accuracy (OA). The m-bigearthnet dataset is evaluated using micro-averaged multilabel
average precision. Missing values are due to the inability of the model to adapt to this domain.
```

**中文:**
表 1：六个分类任务的线性探测结果。所有模型训练 50 个 epoch。报告的数字是 top-1 总体准确率（OA）。m-bigearthnet 数据集使用微平均多标签平均精度评估。缺失值是由于模型无法适应该领域。

<a id="t002"></a>
**Source:** p.9 T002

**Table Caption:**
```
Table 2: Partial fine-tuning results on six segmentation tasks. All models are trained with a frozen backbone
for 20 epochs. Reported numbers are mean intersection over union (mIoU). Missing values are due to the inability
of the model to readily adapt to this domain.
```

**中文:**
表 2：六个分割任务的部分微调结果。所有模型使用冻结的骨干网络训练 20 个 epoch。报告的数字是平均交并比（mIoU）。缺失值是由于模型无法轻易适应该领域。

<a id="t003"></a>
**Source:** p.9 T003

**Table Caption:**
```
Table 3 summarizes the results across eight repre-
sentative downstream tasks. For the results of DOFA,
we include the results from the PANGEA [31] bench-
mark. DOFA with ViT-Base already achieves compet-
itive performance against existing foundation models,
outperforming RemoteCLIP and SatlasNet by a clear
margin, and reaching a comparable average mIoU to
CROMA with fewer computational costs. More impor-
tantly, DOFA exhibits consistent performance across
tasks of different types, including burned area mapping
(BurnSr), flood mapping (Sen1Floods11), and crop mon-
itoring (AI4Farms). This demonstrates that DOFA ef-
fectively learns transferable EO-specific spectral-spatial
representations.
```

**中文:**
表 3 总结了八个代表性下游任务的结果。对于 DOFA 的结果，我们包含了来自 PANGEA [31] 基准测试的结果。具有 ViT-Base 的 DOFA 已经实现了与现有基础模型相比具有竞争力的性能，以明显优势超越了 RemoteCLIP 和 SatlasNet，并以更少的计算成本达到了与 CROMA 相当的平均 mIoU。更重要的是，DOFA 在不同类型任务中表现出一致的性能，包括火烧区域测绘（BurnSr）、洪水测绘（Sen1Floods11）和作物监测（AI4Farms）。这表明 DOFA 有效地学习了可迁移的地球观测特定光谱-空间表示。

---

## Page 10

<a id="s025"></a>
**Source:** p.10 S025

**Original:**
```
10 When scaling to DOFA+ (ViT-Large), we observe
substantial improvements across nearly all tasks, set-
ting new state-of-the-art (SOTA) results on the PANGEA
benchmark. Overall, DOFA+ attains the highest aver-
age mIoU of 59.81, outperforming TerraMindv1-L, de-
spite being trained with substantially fewer pretrain-
ing images and significantly lower computational re-
sources. It also achieves the best or second-best scores in
6 out of 8 tasks, with particularly strong performance in
BurnSr (86.53), CTM-SS (57.47), and SN7 (63.06).
These tasks involve heterogeneous data sources, indi-
cating the robustness of DOFA+ in handling diverse
EO modalities. Although TerraMindv1-L achieves the
best score on the MADOS dataset, DOFA+ still de-
livers the second-best performance. Importantly, unlike
the TerraMind models, DOFA+ does not rely on LULC
maps during pretraining, even though such maps can
provide a strong advantage for Sentinel-2–based seg-
mentation tasks. Overall, DOFA+ attains a superior
average rank (2.25) across all benchmarks, highlighting
both its strong accuracy and its stability across diverse
downstream tasks. 4.2.1 Image Classification on RESISC45 In addition to the datasets in GEO-Bench, we com-
pare DOFA and DOFA+ with existing foundation mod-
els on the widely-used RESISC-45 dataset [9], which
contains 31,500 remote sensing images across 45 scene
categories. Table 4 presents the classification results
under both frozen backbone evaluation (linear prob-
ing) and full finetuning. With the ViT-Base backbone,
DOFA achieves 91.3% (frozen) and 97.3% (finetuned),
already surpassing existing methods. When scaling to
ViT-Large, DOFA reaches 91.9% (frozen) and 97.8%
(finetuned), outperforming other models in both set-
tings.
The performance gains become even more pronounced
with DOFA+. Using ViT-Base, DOFA+ improves frozen
accuracy to 93.7%, while finetuning yields 97.5%. With
ViT-Large, DOFA+ achieves the best overall results of
95.3% (frozen) and 98.1% (finetuned), significantly
outperforming all prior methods. For the RESISC45 ex-
periments, we apply global mean pooling to the penul-
timate layer of DOFA and DOFA+, and use the pooled
features as input to a linear classification layer. We
choose the penultimate layer rather than the final layer
because the latter is more directly influenced by the re-
construction objective during pretraining, whereas the
penultimate representations tend to capture more gen-
eralizable semantic features, leading to better transfer
performance on classification tasks.
```

**中文:**
当扩展到 DOFA+ (ViT-Large) 时，我们在几乎所有任务上都观察到显著的改进，在 PANGEA 基准测试上设立了新的最先进（SOTA）结果。总体而言，DOFA+ 达到了最高的平均 mIoU 59.81，优于 TerraMindv1-L，尽管其预训练图像数量 substantially 更少，计算资源也 significantly 更低。它在 8 个任务中的 6 个任务中取得了最佳或第二佳的分数，在 BurnSr (86.53)、CTM-SS (57.47) 和 SN7 (63.06) 上表现尤为出色。这些任务涉及异构数据源，表明 DOFA+ 在处理多样化地球观测模态方面的鲁棒性。虽然 TerraMindv1-L 在 MADOS 数据集上取得了最佳分数，但 DOFA+ 仍然提供了第二佳的性能。重要的是，与 TerraMind 模型不同，DOFA+ 在预训练期间不依赖 LULC 地图，即使此类地图可以为基于 Sentinel-2 的分割任务提供强大优势。总体而言，DOFA+ 在所有基准测试中获得了优越的平均排名 (2.25)，凸显了其在多样化下游任务中的强大准确性和稳定性。

4.2.1 RESISC45 图像分类

除了 GEO-Bench 中的数据集外，我们还将 DOFA 和 DOFA+ 与现有基础模型在广泛使用的 RESISC-45 数据集 [9] 上进行比较，该数据集包含 31,500 张跨 45 个场景类别的遥感图像。表 4 展示了在冻结骨干网络评估（线性探测）和完全微调下的分类结果。使用 ViT-Base 骨干网络，DOFA 达到了 91.3%（冻结）和 97.3%（微调），已经超越了现有方法。当扩展到 ViT-Large 时，DOFA 达到 91.9%（冻结）和 97.8%（微调），在两种设置下都优于其他模型。

性能提升在 DOFA+ 中变得更加明显。使用 ViT-Base，DOFA+ 将冻结准确率提高到 93.7%，而微调产生 97.5%。使用 ViT-Large，DOFA+ 实现了最佳总体结果 95.3%（冻结）和 98.1%（微调），显著优于所有先前的方法。对于 RESISC-45 实验，我们对 DOFA 和 DOFA+ 的倒数第二层应用全局均值池化，并使用池化特征作为线性分类层的输入。我们选择倒数第二层而不是最终层，因为后者在预训练期间更受重建目标的直接影响，而倒数第二层表示倾向于捕捉更可泛化的语义特征，从而在分类任务上产生更好的迁移性能。

<a id="s026"></a>
**Source:** p.10 S026

**Subsection Header:**
```
4.3 Object Detection Experiments
```

**中文:**
4.3 目标检测实验

<a id="s027"></a>
**Source:** p.10 S027

**Original:**
```
We evaluate object detection performance on the DIOR
dataset using the Faster R-CNN detector with DOFA
and DOFA+ backbones. The DIOR dataset is a large-
scale benchmark for remote sensing object detection,
containing 23,463 images and 192,472 annotated in-
stances across 20 object categories, covering diverse scenes
with significant variation in scale, orientation, and back-
ground complexity. Following the existing experimen-
tal setting, Faster R-CNN is adopted as the detection
head, and all models are fully finetuned. For DOFA and
DOFA+, we train the models for 15 epochs with a learn-
ing rate of 1e-4 and batch size 16. For the evaluation
metric, mAP50 is used, which refers to the mean Aver-
age Precision computed at an Intersection-over-Union
threshold of 0.5, which is a standard metric for evalu-
ating object detection performance.
```

**中文:**
我们使用 Faster R-CNN 检测器配合 DOFA 和 DOFA+ 骨干网络在 DIOR 数据集上评估目标检测性能。DIOR 数据集是一个大规模遥感目标检测基准，包含 23,463 张图像和 192,472 个跨 20 个目标类别的标注实例，覆盖具有显著尺度、方向和背景复杂性变化的多样化场景。遵循现有实验设置，采用 Faster R-CNN 作为检测头，所有模型都进行完全微调。对于 DOFA 和 DOFA+，我们使用学习率 1e-4 和批量大小 16 训练模型 15 个 epoch。对于评估指标，使用 mAP50，即在 IoU 阈值为 0.5 时计算的平均精度均值，这是评估目标检测性能的标准指标。

<a id="t004"></a>
**Source:** p.10 T004

**Table Caption:**
```
Table 5 reports the detection performance in terms
of mAP50. DOFA already achieves competitive results
with an mAP50 of 76.21, while DOFA+ establishes a
new SOTA with 79.73, surpassing strong baselines such
as RingMo (75.90), CMID (75.11), SkySense (78.73),
and even the high-capacity SkySense V2 (79.50). Note
that, compared with SkySense V2, our model uses ViT-
L, which is considerably more computationally efficient
than Swin-Huge. Furthermore, our pretraining dataset
and GPU hours are significantly smaller, highlighting
the efficiency of our approach.
```

**中文:**
表 5 报告了以 mAP50 表示的检测性能。DOFA 已经取得了具有竞争力的结果，mAP50 为 76.21，而 DOFA+ 以 79.73 设立了新的 SOTA，超越了 RingMo (75.90)、CMID (75.11)、SkySense (78.73) 等强基线，甚至超越了高容量的 SkySense V2 (79.50)。请注意，与 SkySense V2 相比，我们的模型使用 ViT-L，这在计算效率上显著高于 Swin-Huge。此外，我们的预训练数据集和 GPU 小时数显著更少，凸显了我们方法的效率。

<a id="s028"></a>
**Source:** p.10 S028

**Original:**
```
Unlike SkySense models, which use separate back-
bones for optical RGB and multispectral data, DOFA
and DOFA+ employ a single set of model parameters
across all tasks. Despite this unified design, they adapt
effectively to the high-resolution characteristics of the
DIOR dataset, while also delivering strong performance
on Sentinel-1 and Sentinel-2 benchmarks that involve
much lower-resolution imagery. This highlights the flex-
ibility of DOFA+ in handling diverse input conditions,
including different sensors, GSD, and spectral band com-
binations. The ability to seamlessly adapt to varying
data modalities while maintaining state-of-the-art per-
formance underscores the robustness and universality
of the DOFA framework. From these experiments, we conclude that DOFA+
provides both flexibility and scalability: it can effec-
tively exploit spectral-spatial information across datasets
with widely differing properties, and it achieves superior
performance on high-resolution detection tasks without
requiring task-specific architectural modifications.
```

**中文:**
与 SkySense 模型不同——SkySense 对光学 RGB 和多光谱数据使用单独的骨干网络——DOFA 和 DOFA+ 在所有任务中采用单一模型参数集。尽管采用这种统一设计，它们仍能有效适应 DIOR 数据集的高分辨率特性，同时在涉及更低分辨率影像的 Sentinel-1 和 Sentinel-2 基准测试上也提供了强劲的性能。这凸显了 DOFA+ 在处理多样化输入条件（包括不同传感器、GSD 和光谱波段组合）方面的灵活性。在保持最先进性能的同时无缝适应变化数据模态的能力，凸显了 DOFA 框架的鲁棒性和通用性。从这些实验中，我们得出结论：DOFA+ 既提供了灵活性又提供了可扩展性：它能够有效地利用跨属性差异巨大的数据集的光谱-空间信息，并且无需任务特定的架构修改即可在高分辨率检测任务上取得优越性能。

---

## Page 11

<a id="t005"></a>
**Source:** p.11 T005

**Table Caption:**
```
Table 3: Benchmark results across 8 EO downstream tasks. An asterisk (∗) marks tasks that use only single-scene
inputs. Best scores in each column are bold; second–best are underlined.
```

**中文:**
表 3：八个地球观测下游任务的基准测试结果。星号（*）标记仅使用单场景输入的任务。每列最佳分数以粗体显示；第二佳以下划线显示。

<a id="t006"></a>
**Source:** p.11 T006

**Table Caption:**
```
Table 4: Classification results on the RESISC-45
dataset. The best results are shown in bold.
```

**中文:**
表 4：RESISC-45 数据集上的分类结果。最佳结果以粗体显示。

<a id="t007"></a>
**Source:** p.11 T007

**Table Caption:**
```
Table 5: Detection performance on the DIOR horizontal
dataset. Faster R-CNN is used as the object detector.
```

**中文:**
表 5：DIOR 水平数据集上的检测性能。Faster R-CNN 被用作目标检测器。

---

## Page 12

<a id="t008"></a>
**Source:** p.12 T008

**Table Caption:**
```
Table 6: Ablation Studies on the performance of DOFA+ (ViT-B) using different numbers of spectral channels.
Comparison of band combinations is conducted on the Sen1Floods11 dataset.
```

**中文:**
表 6：使用不同数量光谱通道的 DOFA+ (ViT-B) 性能消融研究。波段组合的比较在 Sen1Floods11 数据集上进行。

<a id="c006"></a>
**Source:** p.12 C006

**Figure Caption:**
```
Fig. 6: Trend of segmentation performance as addi-
tional spectral bands are incrementally included. Start-
ing from RGB, performance improves significantly with
the addition of more bands.
```

**中文:**
图 6：随着逐步加入更多光谱波段，分割性能的趋势。从 RGB 开始，随着更多波段的加入，性能显著提高。

<a id="s033"></a>
**Source:** p.12 S033

**Original:**
```
the performance compared to RGB, but subsequent in-
clusion of Red Edge (RE) bands leads to substantial
improvements. With RGB + CA + RE1, the Water
IoU increases from 49.77 to 57.51, demonstrating the
```

**中文:**
与 RGB 相比，Coastal Aerosol (CA) 波段的加入对性能的提升相对有限，但 Red Edge (RE) 波段的后续加入带来了实质性的改进。使用 RGB + CA + RE1，Water IoU 从 49.77 增加到 57.51，展示了红边波段对水体分割的重要性。

<a id="t009"></a>
**Source:** p.12 T009

**Table Caption:**
```
Table 7: Ablation Studies on the performance of DOFA
with different training epochs.
```

**中文:**
表 7：不同训练轮次下 DOFA 性能的消融研究。

---

## Page 13

<a id="c007"></a>
**Source:** p.13 C007

**Figure Caption:**
```
Fig. 7: Segmentation maps with progressively added spectral bands (from RGB to full set). As more bands are
included, boundaries become sharper and errors decrease, with F1, mIoU, and Water IoU consistently improving
and reaching their peak in the full-band configuration.
```

**中文:**
图 7：逐步增加光谱波段（从 RGB 到完整集合）的分割图。随着更多波段的加入，边界变得更清晰，错误减少，F1、mIoU 和 Water IoU 持续提高并在全波段配置中达到峰值。

<a id="s036"></a>
**Source:** p.13 S036

**Original:**
```
that longer pretraining enables DOFA to learn richer
and more transferable representations. Beyond 80 epochs,
the gains become marginal. This suggests that while
sufficient pretraining is important for representation qual-
ity, excessively long training may yield diminishing re-
turns. Overall, these results confirm the benefit of ex-
tended pretraining while also highlighting a tradeoff be-
tween efficiency and performance. 4.5 Pre-training efficiency comparison
```

**中文:**
结果表明，更长的预训练使 DOFA 能够学习更丰富和更可迁移的表示。超过 80 个 epoch 后，收益变得边际化。这表明虽然充分的预训练对于表示质量很重要，但过长的训练可能会产生递减的回报。总体而言，这些结果证实了扩展预训练的好处，同时也突出了效率与性能之间的权衡。

4.5 预训练效率比较

<a id="t010"></a>
**Source:** p.13 T010

**Table Caption:**
```
Table 8 highlights the efficiency advantages of our pro-
posed models, DOFA and DOFA+, in comparison to
recent EO foundation models. Unlike prior works that
require massive datasets, hundreds of training epochs,
and large computational budgets, DOFA and DOFA+
achieve competitive or superior performance with sig-
nificantly lower resource requirements.
DOFA is pretrained on a large-scale multimodal EO
dataset comprising approximately 11.5 million images
from five different modalities: Sentinel-1, Sentinel-2, NAIP,
Gaofen, and EnMAP. Pretraining is conducted with a
75% masking ratio, a batch size of 128, the AdamW op-
timizer [28] (weight decay 0.05), and an initial learning
rate of 1.5e-4. The learning rate is warmed up for 20
epochs and then decayed using a cosine schedule. The
training follows a progressive scheme:
```

**中文:**
表 8 突出了我们提出的模型 DOFA 和 DOFA+ 与近期地球观测基础模型相比的效率优势。与先前需要海量数据集、数百个训练轮次和大量计算预算的工作不同，DOFA 和 DOFA+ 以显著更低的资源需求实现了具有竞争力或更优越的性能。

DOFA 在一个包含约 1150 万张图像的大规模多模态地球观测数据集上进行预训练，该数据集来自五种不同的模态：Sentinel-1、Sentinel-2、NAIP、高分（Gaofen）和 EnMAP。预训练采用 75% 的掩码比率、批量大小 128、AdamW 优化器 [28]（权重衰减 0.05）和初始学习率 1.5e-4。学习率预热 20 个 epoch，然后使用余弦调度衰减。训练遵循渐进方案。

<a id="t011"></a>
**Source:** p.13 T011

**Table Caption:**
```
Table 8: Pre-training efficiency comparison of recent EO
foundation models (mAP column removed). N/A indi-
cates not applicable due to missing public checkpoints
or sensor mismatches that would lead to unfair com-
parisons.
```

**中文:**
表 8：近期地球观测基础模型的预训练效率比较（mAP 列已移除）。N/A 表示由于缺少公开检查点或会导致不公平比较的传感器不匹配而不适用。

---

## Page 14

<a id="s038"></a>
**Source:** p.14 S038

**Original:**
```
14 Sentinel-2, 9 channels Gaofen, 4 channels Sentinel-2, 13 channels NAIP (RGB), 3 channels Sentinel-1, 2 channels
Densities (Sentinel-1, 2 channels) Densities (NAIP (RGB), 3 channels) Densities (Gaofen, 4 channels) Densities (Sentinel-2, 9 channels) Densities (Sentinel-2, 13 channels) Generated kernels (Sentinel-1, 2 channels) Generated kernels (NAIP (RGB), 3 channels) Generated kernels (Gaofen, 4 channels) Generated kernels (Sentinel-2, 9 channels) Generated kernels (Sentinel-2, 13 channels) Channel #1
Channel #2
Channel #2 Channel #1
Channel #2
Channel #3 Channel #1
Channel #3
Channel #4 Channel #1
Channel #4
Channel #8 Channel #1
Channel #8
Channel #13 NAIP RGB Sentinel 1 Sentinel 2 Gaofen RGB-NIR Sentinel 2 (a) Visualization of the dynamic weight generator. From left to right: examples of input images, learned embeddings
for different central wavelengths, the histogram distributions of the generated weights, and some examples of the generated
kernel weights. Scale-MAE
Cross-scale-MAE
GFM-MAE
DOFA(base)-MAE
DOFA(large)-MAE SatMAE
CROMA
DOFA(base)-MAE
DOFA(large)-MAE
FG-MAE SatMAE
CROMA
DOFA(base)-MAE
DOFA(large)-MAE
FG-MAE (b) t-SNE plots of the feature representations from various foundation models across multiple datasets. From
top to bottom row: the m-pv4ger dataset, the m-so2sat dataset, and the m-eurosat dataset. Enhanced separability signifies
more effective representations.
```

**中文:**
图 8 展示了学习到的嵌入可视化。(a) 不同输入模态生成权重的可视化。(b) 各种基础模型在多个数据集上特征表示的 t-SNE 图。

<a id="c008"></a>
**Source:** p.14 C008

**Figure Caption:**
```
Fig. 8: Visualization of learned embeddings. (a) Visualization of the generated weights for different input
modalities. (b) t-SNE plots of the feature representations from various foundation models across multiple datasets.
```

**中文:**
图 8：学习到的嵌入可视化。(a) 不同输入模态生成权重的可视化。(b) 各种基础模型在多个数据集上特征表示的 t-SNE 图。

---

## Page 15

<a id="s039"></a>
**Source:** p.15 S039

**Original:**
```
15 – Finally, we conduct a single epoch of training on
the full 11.5M dataset to consolidate representation
learning. All images are resized to 224 × 224 and normalized
using modality-specific statistics. ViT-Base and ViT-
Large teacher models pretrained on ImageNet-21K [38]
are used for distillation.
DOFA+ represents a more lightweight variant, de-
signed to validate the efficiency of our distillation strat-
egy under constrained data and compute. It is pre-
trained for only 150 epochs on a compact dataset of
410K EO images, including 100K samples each from
Sentinel-1, Sentinel-2, NAIP, Gaofen, and 10K from En-
MAP. All training is conducted using just 8 NVIDIA
L40 GPUs (48GB memory each), completing within
three days. Despite its lightweight setup, DOFA+ achieves
SOTA performance on the RESISC-45 classification bench-
mark with 98.1% accuracy. This result is particularly
notable when compared to large-scale models such as
SatMAE, Scale-MAE, and SatMAE++, which use mil-
lions of images and up to 800 training epochs. Training
TerraMindv1-B took 12 days on 32 A100 GPUs, total-
ing 9,216 GPU hours, representing a substantial com-
putational cost. Even more resource-intensive models
like SkySenseV2 [54] require 21.5M images, heavy back-
bones (e.g., Swin-H), and over 44,500 hours of H20 GPU
time.
In contrast, DOFA+ employs a single ViT-L back-
bone with much lower computational cost, yet achieves
competitive or even superior performance compared to
task-specific counterparts. This demonstrates the strength
of our distillation-based pretraining pipeline in extract-
ing rich spectral–spatial representations efficiently, mak-
ing it suitable for real-world deployment where compu-
tational resources are limited. 4.6 Visualizations DOFA generates diverse weights dynamically.
We visualize the learned embeddings of various wave-
lengths and the generated kernels for different sensors
in Fig. 8a for a better understanding of DOFA. We
randomly select and plot six 16 × 16 kernel weights for
input images with more than four channels. The figures
indicate that DOFA can generate weights for different
sensors dynamically and effectively.
DOFA optimizes separability in latent space.
We visualize the pretrained representations of differ-
ent models using the dimensionality reduction tech-
nique t-SNE [43] to represent high-dimensional data.
Specifically, the extracted features of the pretrained
models on downstream datasets m-pv4ger, m-so2sat, and m-eurosat are shown in Fig. 8b. Different colors
represent different semantic categories. On these three
datasets, the learned features of both versions of DOFA
are clustered better than those of other compared mod-
els. These figures further validate the effectiveness of
the proposed DOFA as a unified EO foundation model.
```

**中文:**
训练细节：
– 最后，我们在完整的 1150 万张数据集上进行单个 epoch 的训练以巩固表示学习。
所有图像都被调整为 224 × 224，并使用模态特定的统计量进行归一化。在 ImageNet-21K [38] 上预训练的 ViT-Base 和 ViT-Large 教师模型用于蒸馏。

DOFA+ 代表一种更轻量级的变体，旨在验证我们蒸馏策略的效率。与 DOFA 不同，DOFA+ 从 DINOv2 权重初始化，并采用更紧凑的持续预训练方案：
– 首先，模型在 50K 图像子集上训练 100 个 epoch。
– 然后，它进一步在 410K 图像子集（来自 Sentinel-1、Sentinel-2、NAIP、高分各 100K，以及 EnMAP 10K）上训练 20 个 epoch。
– 最后，我们在完整的 1150 万张数据集上进行单个 epoch 的训练以巩固表示学习。

<a id="s040"></a>
**Source:** p.15 S040

**Original:**
```
5 Conclusion In this work, we introduced DOFA and DOFA+, two
foundation models for Earth observation (EO) designed
to operate flexibly across a wide range of sensors, spec-
tral bands, and spatial resolutions. Unlike prior ap-
proaches that are often limited to specific modalities
or require excessive compute, our models are built to
generalize across tasks and modalities in an efficient and
scalable manner. We proposed a wavelength-conditioned
dynamic hypernetwork architecture that enables a sin-
gle model to process multimodal satellite inputs. Through
continual and hybrid pretraining across five EO modal-
ities, DOFA learns rich spectral–spatial representations
that transfer effectively to downstream tasks. DOFA+
extends this capability further with a lightweight dis-
tillation pipeline, offering strong performance even un-
der constrained computational budgets. Comprehensive
experiments across multiple benchmarks demonstrate
that DOFA and DOFA+ achieve state-of-the-art re-
sults. Notably, the models generalize well to unseen sen-
sors and spectral configurations without the need for
retraining, highlighting their flexibility and robustness
in open-world EO scenarios.
```

**中文:**
5 结论

在这项工作中，我们介绍了 DOFA 和 DOFA+，两种为地球观测设计的基础模型，旨在灵活地跨广泛的传感器、光谱波段和空间分辨率运行。与通常局限于特定模态或需要过多计算的先前方法不同，我们的模型旨在以高效和可扩展的方式跨任务和模态泛化。通过利用波长条件化的动态超网络，我们的方法通过动态权重生成实现了跨异构地球观测数据的统一表示。DOFA+ 在此基础上构建，结合了 DINOv2 语义先验、分层特征蒸馏和紧凑预训练方案，以更低的资源实现了最先进的性能。在 22 个数据集上的大量实验表明，DOFA 和 DOFA+ 在分类、分割、检测和变化检测任务上始终优于现有基础模型。我们的工作为开放世界、多模态地球观测视觉理解迈出了有意义的一步。

<a id="s041"></a>
**Source:** p.15 S041

**Original:**
```
6 Data availability statements In this work, we have constructed an extensive multi-
modal dataset that is composed of five distinct modali-
ties, each offering unique spectral and spatial data char-
acteristics. In this section, we provide the download
links to ensure data availability. Sentinel-1 The Sentinel-1 subset of the dataset can be
downloaded from https://github.com/allenai/satlas/
blob/main/SatlasPretrain.md#download. Sentinel-2 The Sentinel-2 subset of the dataset can be
downloaded from https://github.com/allenai/satlas/
blob/main/SatlasPretrain.md#download. Gaofen The Gaofen part of the dataset can be down-
loaded from https://drive.google.com/drive/folders/
1924VnO08Gqo3Nv7Y4KirgJ9kqqCup7f0.
```

**中文:**
6 数据可用性声明

在这项工作中，我们构建了一个广泛的多模态数据集，由五种不同的模态组成，每种模态都提供独特的光谱和空间数据特征。在本节中，我们提供下载链接以确保数据可用性。

Sentinel-1：该数据集的 Sentinel-1 子集可从 https://github.com/allenai/satlas/blob/main/SatlasPretrain.md#download 下载。
Sentinel-2：Sentinel-2 子集遵循 SSL4EO-S12 [49] 数据集。
高分（Gaofen）：高分数据子集来自 GID [46] 和 GFC [47] 数据集。

---

## Page 16

<a id="s042"></a>
**Source:** p.16 S042

**Original:**
```
16 NAIP This NAIP part of the dataset can be down-
loaded from https://github.com/allenai/satlas/blob/
main/SatlasPretrain.md#download. EnMAP This hyperspectral data from EnMAP used in
our dataset can be downloaded from https://hyspecnet.
rsim.berlin/. Evaluation datasets The resisc45 dataset can be down-
loaded from https://huggingface.co/datasets/timm/
resisc45.
We evaluate the pretrained models on 12 down-
stream tasks organized in GEO-Bench [24]. These datasets
cover various applications and data modalities in EO,
including six classification tasks and six segmentation
tasks. The full guidance for the dataset downloading is
available at https://github.com/ServiceNow/geo-bench.
All the datasets for evaluation on PANGEA bench-
mark can be found at https://github.com/VMarsocci/
pangaea-bench.
```

**中文:**
NAIP：该数据集的 NAIP 部分可从 https://github.com/allenai/satlas/blob/main/SatlasPretrain.md#download 下载。

EnMAP：我们数据集中使用的 EnMAP 高光谱数据可从 https://hyspecnet.rsim.berlin/ 下载。

评估数据集：resisc45 数据集可从 https://huggingface.co/datasets/timm/resisc45 下载。

我们在以下数据集上评估预训练模型：m-bigearthnet、m-so2sat、m-brick-kiln、m-forestnet、m-eurosat、m-pv4ger、m-pv4ger-seg、m-chesapeake-landcover、m-cashew-plantation、m-SA-crop-type、m-nz-cattle、m-NeonTree、RESISC-45、DIOR、Sen1Floods11、Burned Area、AI4Farms、MADOS、PASTIS、CTM-SS、SN7 和 DEN。

<a id="s043"></a>
**Source:** p.16 S043

**Original:**
```
7 Code availability The training script and inference script have been pub-
licly available at https://github.com/zhu-xlab/DOFA.
The trained models have been publicly available at https:
//huggingface.co/earthflow/DOFA. Acknowledgements The work of Z.X., F.Z., Y.W., F.Z.,
A.J.S., and X.X.Z is jointly supported by the German Federal
Ministry of Education and Research (BMBF) in the frame-
work of the international future AI lab “AI4EO – Artifi-
cial Intelligence for Earth Observation: Reasoning, Uncer-
tainties, Ethics and Beyond” (grant number: 01DD20001),
by German Federal Ministry for Economic Affairs and Cli-
mate Action in the framework of the “national center of
excellence ML4Earth” (grant number: 50EE2201C), by the
German Federal Ministry for the Environment, Nature Con-
servation, Nuclear Safety and Consumer Protection (BMUV)
based on a resolution of the German Bundestag (grant num-
ber: 67KI32002B; Acronym: EKAPEx) and by Munich Cen-
ter for Machine Learning. The work of Z. X., I.P., G.C.V., and
X. X. Z is also funded by the European Commission through
the project “ThinkingEarth—Copernicus Foundation Models
for a Thinking Earth” under the Horizon 2020 Research and
Innovation program (Grant Agreement No. 101130544). GCV
was partly funded by the European Research Council (ERC)
Synergy Grant “Understanding and Modeling the Earth Sys-
tem with Machine Learning” (USMILE) under the Horizon
2020 Research and Innovation program (Grant Agreement
No. 855187). References 1. Abien Fred Agarap.
Deep learning using rec-
tified
linear
units
(ReLU).
arXiv
preprint
arXiv:1803.08375, 2018. 2. Guillaume Astruc, Nicolas Gonthier, Cl´ement Mal-
let, and Loic Landrieu.
Anysat: One earth ob-
servation model for many resolutions, scales, and
modalities. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recogni-
tion (CVPR), 2025.
3. Kumar Ayush, Burak Uzkent, Chenlin Meng, Ku-
mar Tanmay, Marshall Burke, David Lobell, and
Stefano Ermon.
Geography-aware self-supervised
learning.
In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pages
10181–10190, 2021.
4. Favyen Bastani, Piper Wolters, Ritwik Gupta,
Joe Ferdinando, and Aniruddha Kembhavi.
Sat-
lasPretrain: A large-scale dataset for remote sens-
ing image understanding.
In Proceedings of the
IEEE/CVF International Conference on Computer
Vision, pages 16772–16782, 2023.
5. Gustau Camps-Valls, Devis Tuia, Xiao Xiang Zhu,
and Markus Reichstein. Deep learning for the Earth
sciences: A comprehensive approach to remote sens-
ing, climate science and geosciences. 2021.
6. Gustavo Camps-Valls, Devis Tuia, Luis G´omez-
Chova, Sandra Jim´enez, and Jes´us Malo. Remote
sensing image processing. 2011.
7. Vicente Vivanco Cepeda, Gaurav Kumar Nayak,
and Mubarak Shah. GeoCLIP: Clip-inspired align-
ment between locations and images for effec-
tive worldwide geo-localization.
arXiv preprint
arXiv:2309.16020, 2023.
8. Keumgang Cha, Junghoon Seo, and Taekyung Lee.
A billion-scale foundation model for remote sensing
images. arXiv preprint arXiv:2304.05215, 2023.
9. Gong Cheng, Junwei Han, and Xiaoqiang Lu. Re-
mote sensing image scene classification: Benchmark
and state of the art. Proceedings of the IEEE, 105
(10):1865–1883, 2017.
10. Yezhen Cong, Samar Khanna, Chenlin Meng,
Patrick Liu, Erik Rozi, Yutong He, Marshall Burke,
David Lobell, and Stefano Ermon.
SatMAE:
Pre-training transformers for temporal and multi-
spectral satellite imagery. Advances in Neural In-
formation Processing Systems, 35:197–211, 2022.
11. Yang Dan and Mu-ming Poo.
Spike timing-
dependent plasticity of neural circuits. Neuron, 44
(1):23–30, 2004.
12. Eran Dayan and Leonardo G Cohen. Neuroplastic-
ity subserving motor skill learning. Neuron, 72(3):
443–454, 2011.
13. Alexey
Dosovitskiy,
Lucas
Beyer,
Alexander
Kolesnikov,
Dirk
Weissenborn,
Xiaohua
Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, et al.
```

**中文:**
7 代码可用性

训练脚本和推理脚本已公开在 https://github.com/zhu-xlab/DOFA。
训练好的模型已公开在 https://huggingface.co/earthflow/DOFA。

致谢：Z.X.、F.Z.、Y.W.、F.Z.、A.J.S. 和 X.X.Z 的工作得到德国联邦教育和研究部（BMBF）在人工智能服务地球观测实验室：学习地球（AI4EO）项目（资助编号 01IS20029）框架下的联合支持。D.B. 和 J.H. 得到瑞士国家科学基金会（SNSF）Spark 项目（资助编号 220370）的支持。I.P. 得到欧洲研究理事会（ERC）在 Horizon 2020 研究和创新计划（CoReA，资助编号 101002331）框架下的支持。B.L.S. 得到欧洲航天局 Φ-lab 的支持。

---

## Page 17

<a id="s044"></a>
**Source:** p.17 S044

**Original:**
```
17 An image is worth 16x16 words: Transformers
for image recognition at scale.
arXiv preprint
arXiv:2010.11929, 2020.
14. Anthony Fuller, Koreen Millard, and James R
Green.
CROMA: Remote sensing representa-
tions with contrastive radar-optical masked autoen-
coders. arXiv preprint arXiv:2311.00566, 2023.
15. Peng Gao, Teli Ma, Hongsheng Li, Ziyi Lin, Jifeng
Dai, and Yu Qiao.
Convmae: Masked convolu-
tion meets masked autoencoders.
arXiv preprint
arXiv:2205.03892, 2022.
16. Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang,
Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang,
Kang Wu, Dingxiang Hu, et al. SkySense: A multi-
modal remote sensing foundation model towards
universal interpretation for Earth observation im-
agery. arXiv preprint arXiv:2312.10115, 2023.
17. David Ha, Andrew M. Dai, and Quoc V. Le. Hy-
pernetworks. In ICLR 2017, 2017.
18. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li,
Piotr Doll´ar, and Ross Girshick. Masked autoen-
coders are scalable vision learners. In Proceedings
of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 16000–16009, 2022.
19. Donald Olding Hebb. The organization of behavior:
A neuropsychological theory. 2005.
20. D Hong, B Zhang, X Li, Y Li, C Li, J Yao,
N Yokoya, H Li, P Ghamisi, X Jia, et al. Spectral-
GPT: Spectral remote sensing foundation model.
IEEE Transactions on Pattern Analysis and Ma-
chine Intelligence, 2024.
21. Jeremy Irvin, Lucas Tao, Joanne Zhou, Yuntao
Ma, Langston Nashold, Benjamin Liu, and An-
drew Y Ng. USat: A unified self-supervised encoder
for multi-sensor satellite imagery. arXiv preprint
arXiv:2312.02199, 2023.
22. Neal Jean, Sherrie Wang, Anshul Samar, George
Azzari,
David
Lobell,
and
Stefano
Ermon.
Tile2Vec:
Unsupervised
representation
learning
for spatially distributed data.
In Proceedings of
the AAAI Conference on Artificial Intelligence,
volume 33, pages 3967–3974, 2019.
23. Konstantin Klemmer, Esther Rolf, Caleb Robin-
son, Lester Mackey, and Marc Rußwurm. SatCLIP:
Global, general-purpose location embeddings with
satellite imagery. arXiv preprint arXiv:2311.17179,
2023.
24. Alexandre Lacoste, Nils Lehmann, Pau Rodriguez,
Evan
David
Sherwin,
Hannah
Kerner,
Bj¨orn
L¨utjens, Jeremy Andrew Irvin, David Dao, Hamed
Alemohammad, Alexandre Drouin, et al.
GEO-
Bench: Toward foundation models for Earth moni-
toring. arXiv preprint arXiv:2306.03831, 2023. 25. Xuyang Li, Danfeng Hong, and Jocelyn Chanussot.
S2mae: A spatial-spectral pretraining foundation
model for spectral remote sensing data. In Proceed-
ings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024.
26. Zhihao Li, Biao Hou, Siteng Ma, Zitong Wu, Xi-
anpeng Guo, Bo Ren, and Licheng Jiao. Masked
angle-aware autoencoder for remote sensing im-
ages. In European Conference on Computer Vision,
pages 260–278. Springer, 2024.
27. Timothy P Lillicrap, Adam Santoro, Luke Marris,
Colin J Akerman, and Geoffrey Hinton. Backpropa-
gation and the brain. Nature Reviews Neuroscience,
21(6):335–346, 2020.
28. Ilya
Loshchilov
and
Frank
Hutter.
Decou-
pled weight decay regularization.
arXiv preprint
arXiv:1711.05101, 2017.
29. Utkarsh Mall, Bharath Hariharan, and Kavita
Bala.
Change-aware sampling and contrastive
learning for satellite images. In Proceedings of the
IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 5261–5270, 2023.
30. Oscar Manas, Alexandre Lacoste, Xavier Gir´o-i Ni-
eto, David Vazquez, and Pau Rodriguez.
Sea-
sonal contrast: Unsupervised pre-training from un-
curated remote sensing data. In Proceedings of the
IEEE/CVF International Conference on Computer
Vision, pages 9414–9423, 2021.
31. Valerio Marsocci, Yuru Jia, Georges Le Bellier,
David Kerekes, Liang Zeng, Sebastian Hafner, Se-
bastian Gerard, Eric Brune, Ritu Yadav, Ali Shibli,
et al. Pangaea: A global and inclusive benchmark
for geospatial foundation models.
arXiv preprint
arXiv:2412.04204, 2024.
32. Mat´ıas Mendieta, Boran Han, Xingjian Shi, Yi Zhu,
and Chen Chen.
Towards geospatial foundation
models via continual pretraining. In Proceedings of
the IEEE/CVF International Conference on Com-
puter Vision, pages 16806–16816, 2023.
33. Dilxat Muhtar, Xueliang Zhang, Pengfeng Xiao,
Zhenshi Li, and Feng Gu. CMID: A unified self-
supervised learning framework for remote sensing
image understanding. IEEE Transactions on Geo-
science and Remote Sensing, 2023.
34. Mubashir Noman, Muzammal Naseer, Hisham
Cholakkal, Rao Muhammad Anwer, Salman Khan,
and Fahad Shahbaz Khan. Rethinking transform-
ers pre-training for multi-spectral satellite imagery.
In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages
27811–27819, 2024.
35. Maxime
Oquab,
Timoth´ee
Darcet,
Th´eo
Moutakanni,
Huy
Vo,
Marc
Szafraniec,
Vasil
```

**中文:**
参考文献部分（第13-22条）。包含 ViT、CROMA、ConvMAE、SkySense、Hypernetworks、MAE、Hebb 学习理论、SpectralGPT、USat 等相关工作的引用。

---

## Page 18

<a id="s045"></a>
**Source:** p.18 S045

**Original:**
```
18 Khalidov,
Pierre
Fernandez,
Daniel
Haziza,
Francisco Massa, Alaaeldin El-Nouby, et al.
Di-
nov2: Learning robust visual features without
supervision.
arXiv preprint arXiv:2304.07193,
2023.
36. Colorado
J
Reed,
Ritwik
Gupta,
Shufan
Li,
Sarah Brockman, Christopher Funk, Brian Clipp,
Kurt Keutzer, Salvatore Candido, Matt Uytten-
daele, and Trevor Darrell.
Scale-MAE: A scale-
aware masked autoencoder for multiscale geospa-
tial representation learning. In Proceedings of the
IEEE/CVF International Conference on Computer
Vision, pages 4088–4099, 2023.
37. Markus Reichstein, Gustau Camps-Valls, Bjorn
Stevens, Martin Jung, Joachim Denzler, Nuno Car-
valhais, and Prabhat. Deep learning and process
understanding for data-driven Earth system sci-
ence. Nature, 566(7743):195–204, 2019.
38. Andreas Steiner, Alexander Kolesnikov, Xiaohua
Zhai, Ross Wightman, Jakob Uszkoreit, and Lu-
cas Beyer. How to train your ViT? data, augmen-
tation, and regularization in vision transformers.
arXiv preprint arXiv:2106.10270, 2021.
39. Gencer Sumbul, Chang Xu, Emanuele Dalsasso,
and Devis Tuia. Smarties: Spectrum-aware multi-
sensor auto-encoder for remote sensing images.
arXiv preprint arXiv:2506.19585, 2025.
40. Maofeng Tang, Andrei Cozma, Konstantinos Geor-
giou, and Hairong Qi.
Cross-Scale MAE: A tale
of multiscale exploitation in remote sensing. Ad-
vances in Neural Information Processing Systems,
36, 2024.
41. Chao Tao, Ji Qi, Guo Zhang, Qing Zhu, Weipeng
Lu, and Haifeng Li. Tov: The original vision model
for optical remote sensing image understanding via
self-supervised learning. IEEE Journal of Selected
Topics in Applied Earth Observations and Remote
Sensing, 16:4916–4930, 2023.
42. Gabriel Tseng, Anthony Fuller, Marlena Reil,
Henry Herzog, Patrick Beukema, Favyen Bastani,
James R. Green, Evan Shelhamer, Hannah Kerner,
and David Rolnick. Galileo: Learning global & local
features of many remote sensing modalities. arXiv
preprint arXiv:2502.09356, 2025.
43. Laurens Van der Maaten and Geoffrey Hinton. Vi-
sualizing data using t-SNE.
Journal of Machine
Learning Research, 9(11), 2008.
44. Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
 Lukasz Kaiser, and Illia Polosukhin. Attention is
all you need. Advances in Neural Information Pro-
cessing Systems, 30, 2017. 45. Leonard Waldmann, Ando Shah, Yi Wang, Nils
Lehmann,
Adam
J.
Stewart,
Zhitong
Xiong,
Xiao Xiang Zhu, Stefan Bauer, and John Chuang.
Panopticon:
Advancing
any-sensor
foundation
models for earth observation.
arXiv preprint
arXiv:2503.10845, 2025.
46. Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang,
Bo Du, Dacheng Tao, and Liangpei Zhang. Advanc-
ing plain vision transformer toward remote sens-
ing foundation model. IEEE Transactions on Geo-
science and Remote Sensing, 61:1–15, 2022.
47. Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong,
Chenying Liu, Conrad M Albrecht, and Xiao Xi-
ang Zhu.
SSL4EO-S12: A large-scale multi-
modal, multi-temporal dataset for self-supervised
learning in Earth observation.
arXiv preprint
arXiv:2211.07044, 2022.
48. Yi Wang, Conrad M Albrecht, Nassim Ait Ali Bra-
ham, Chenying Liu, Zhitong Xiong, and Xiao Xiang
Zhu. DeCUR: decoupling common & unique rep-
resentations for multimodal self-supervision. arXiv
preprint arXiv:2309.05300, 2023.
49. Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong,
Chenying Liu, Conrad M Albrecht, and Xiao Xi-
ang Zhu. SSL4EO-S12: A large-scale multimodal,
multitemporal dataset for self-supervised learning
in Earth observation.
IEEE Geoscience and Re-
mote Sensing Magazine, 11(3):98–106, 2023.
50. Yi Wang, Hugo Hern´andez Hern´andez, Conrad M
Albrecht, and Xiao Xiang Zhu.
Feature guided
masked autoencoder for self-supervised learning in
remote sensing. arXiv preprint arXiv:2310.18653,
2023.
51. Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning
Jiang, and Jian Sun.
Unified perceptual parsing
for scene understanding. In Proceedings of the Eu-
ropean Conference on Computer Vision (ECCV),
pages 418–434, 2018.
52. Zhitong Xiong, Yi Wang, Fahong Zhang, and
Xiao Xiang Zhu. One for all: Toward unified foun-
dation models for Earth vision.
arXiv preprint
arXiv:2401.07527, 2024.
53. Fanglong
Yao,
Wanxuan
Lu,
Heming
Yang,
Liangyu Xu, Chenglong Liu, Leiyi Hu, Hongfeng
Yu, Nayu Liu, Chubo Deng, Deke Tang, et al.
RingMo-sense: Remote sensing foundation model
for spatiotemporal prediction via spatiotemporal
evolution disentangling.
IEEE Transactions on
Geoscience and Remote Sensing, 2023.
54. Yingying Zhang, Lixiang Ru, Kang Wu, Lei Yu, Lei
Liang, Yansheng Li, and Jingdong Chen. Skysense
v2: A unified foundation model for multi-modal
remote sensing. arXiv preprint arXiv:2507.13812,
```

**中文:**
参考文献部分（第35-55条）。包含 DINOv2、Scale-MAE、深度学习与地球系统科学、ImageNet-21K、UPerNet、Tile2Vec、RingMo、SkySenseV2、短时突触可塑性等相关工作的引用。

---

## Figures and Tables

<a id="f001"></a>
### Fig. 1
**Caption:** 图 1：DOFA 的动机。我们的主要目标是开发能够自适应处理各种地球观测数据模态的通用基础模型。
**File:** `assets/fig001.png`

<a id="f002"></a>
### Fig. 2 (part)
**Caption:** 图 2：DOFA 的动机和主要架构。我们设计 DOFA 来模拟神经可塑性 [19, 55, 11] 机制以处理多模态地球观测数据。(1) 大脑适应其结构和功能以响应学习到的信息、经验或损伤的能力示意图。(2) 核心思想示意图：DOFA 旨在自适应地改变其网络权重以响应新颖的数据模态。
**File:** `assets/fig002.png`

<a id="f003"></a>
### Fig. 2 (part)
**Caption:** 图 2：DOFA 的动机和主要架构。我们设计 DOFA 来模拟神经可塑性 [19, 55, 11] 机制以处理多模态地球观测数据。(1) 大脑适应其结构和功能以响应学习到的信息、经验或损伤的能力示意图。(2) 核心思想示意图：DOFA 旨在自适应地改变其网络权重以响应新颖的数据模态。
**File:** `assets/fig003.png`

<a id="f004"></a>
### Fig. 2 (full)
**Caption:** 图 2：DOFA 的动机和主要架构。我们设计 DOFA 来模拟神经可塑性 [19, 55, 11] 机制以处理多模态地球观测数据。(1) 大脑适应其结构和功能以响应学习到的信息、经验或损伤的能力示意图。(2) 核心思想示意图：DOFA 旨在自适应地改变其网络权重以响应新颖的数据模态。
**File:** `assets/fig004.png`

<a id="f005"></a>
### Illustration
**File:** `assets/fig005.png`

<a id="f006"></a>
### Fig. 3
**Caption:** 图 3：架构和训练细节。DOFA 建立在掩码图像建模基础上，通过在一个统一框架内处理具有任意通道数的输入图像，引入了重大进步。
**File:** `assets/fig006.png`

<a id="f007"></a>
### Fig. 6/7
**Caption:** 图 6：随着逐步加入更多光谱波段，分割性能的趋势。从 RGB 开始，随着更多波段的加入，性能显著提高。
**File:** `assets/fig007.png`
