# 自研遥感基础模型全国扩展改进计划

> **版本**: v1.1（修订版）  
> **日期**: 2026-04-15  
> **基线模型**: 自研哈尔滨高新区版（57.44M参数，STP架构）  
> **目标**: 研究区拓展至全国 + 高分融合（2m+）+ 模型规模扩展  
> **性质**: 战略规划文档（非执行指令）

---

## 一、当前基线分析

### 1.1 现有模型架构

| 组件 | 配置 | 参数量 |
|------|------|--------|
| Sensor Encoder Bank | 3类输入（S2/S1/Landsat），stem_dim=128 | 0.12M |
| STP Blocks (×8) | precision_dim=256, num_heads=8 | 54.59M |
| VMF Bottleneck | embedding_dim=128, skip_l2_training | 0.03M |
| Per-source Decoders (×7) | Continuous + Categorical | 2.67M |
| Classification Heads | Cosine similarity ×3 | ~0.01M |
| **总计** | — | **57.44M** |

- **Patch尺寸**: 128×128@10m（实际覆盖 1.28km × 1.28km）
- **输入源**: S2(10m, 多光谱)、S1(10m, SAR)、Landsat(30m, 多光谱)
- **重建目标**: 输入3类 + DEM + WorldCover + Dynamic World + JRC Water = 7类
- **已有HR支持**: `s2_hr`/`s1_hr` type_id 已预留，`merge_hr_into_lr` 机制已存在
- **核心创新**: 训练时跳过L2 Norm（解决embedding坍缩）、Temporal Magnitude Loss（时序敏感性）、VMF Bottleneck

### 1.2 当前训练状态

| 指标 | 数值 |
|------|------|
| 训练样本 | 424 patches（哈尔滨高新区） |
| 覆盖面积 | ~695 km² |
| GPU配置 | 2×H100 80GB |
| 有效Batch | 32（2卡 × 2本地 × 8梯度累积） |
| Epoch | 300 |
| Checkpoint | ~658MB |
| uniformity | -2.9 ~ -3.1 |
| CD Head 5-fold AUC | 0.896 |

---

## 二、全国扩展数据规模估算

### 2.1 核心假设

- **研究区**: 中国陆地面积约960万km²，有效覆盖（去除大面积沙漠/冰川/高山）按 **60%** 计 ≈ **576万km²**
- **时间分辨率**: S2/S1/GF 月度至少一景，Landsat 双月一景
- **高分融合**: 整合 GF-1/GF-2/GF-6/GF-7 等国产高分卫星（2m/0.8m/0.65m），覆盖重点地区约 **30-50%**
- **Patch策略**: 以10m为基准网格（128×128），高分数据通过多尺度编码器融合，不扩大基础patch尺寸

### 2.2 原始数据量估算（年度）

| 数据源 | 空间分辨率 | 覆盖方式 | 单景大小 | 频率 | 年数据量 |
|--------|-----------|---------|---------|------|---------|
| Sentinel-2 L2A | 10m | ~1,400 tiles | ~1GB | 月度 | **~16.8 TB** |
| Sentinel-1 GRD | 10m | ~1,400 tiles | ~0.8GB | 月度 | **~13.4 TB** |
| Landsat-8/9 | 30m | ~500 scenes | ~1GB | 双月 | **~3.0 TB** |
| GF-1/2/6/7 高分 | 2m/0.8m/0.65m | 重点地区(~30%) | ~0.5GB | 月度 | **~5.7 TB** |
| DEM (Copernicus 30m) | 30m | 全国一次 | — | 静态 | **~500 GB** |
| WorldCover | 10m | 全国一次 | — | 年度 | **~200 GB** |
| Dynamic World | 10m | 全国 | — | 月度 | **~1.0 TB** |
| JRC Water | 30m | 全国 | — | 年度 | **~500 GB** |
| **原始数据合计** | — | — | — | — | **~41 TB/年** |

> 注：高分数据（GF系列）主要通过自然资源卫星遥感云服务平台获取，重点地区覆盖密度高，偏远地区稀疏。

### 2.3 训练样本（Patch）规模

参考国际基准（Prithvi-EO-2.0: 4.2M samples @ 30m 全球；Satlas: 50.4M images @ 10m），全国模型的采样策略：

| 采样密度 | Patch数量 | 覆盖逻辑 |
|---------|----------|---------|
| 低密度（每200km² 1个） | ~29,000 | 过于稀疏，仅适合粗粒度预训练 |
| **中密度（每50km² 1个）** | **~115,000** | **推荐：平衡覆盖与存储** |
| 高密度（每10km² 1个） | ~576,000 | 接近像素级，存储压力极大 |

**推荐采用"分层采样"策略**：
- **高密度区**（城市、农田、水域交界）：每10km² 1个patch，约占30%面积 → ~17万patches
- **中密度区**（一般农田、林地）：每50km² 1个patch，约占50%面积 → ~5.8万patches
- **低密度区**（荒漠、高山、稀疏植被）：每200km² 1个patch，约占20%面积 → ~0.6万patches
- **总计**：约 **23万 patches**

### 2.4 预处理后存储需求

| 项目 | 估算 |
|------|------|
| 10m基础patch缓存（23万 × 50MB） | ~11.5 TB |
| 高分融合patch缓存（23万 × 额外100MB） | ~23.0 TB |
| 多时相序列扩展（平均6时相） | ~3.0 TB |
| 数据集元数据/索引 | ~500 GB |
| **训练缓存合计** | **~38 TB** |
| 原始数据归档（3年滚动） | ~120 TB |
| 模型Checkpoint（20个 × 1.5GB） | ~30 GB |
| 训练日志/TensorBoard/实验追踪 | ~2 TB |
| 工作空间/临时文件 | ~10 TB |
| **活跃存储总计** | **~50 TB** |
| **总存储（含3年归档）** | **~170 TB** |

---

## 三、三档次模型扩展方案（全部详细展开）

基于当前STP架构，保留核心设计（skip L2 norm training、VMF bottleneck、Temporal Magnitude Loss），按算力预算分为三个档次。

### 3.0 三档次总体对比

| 维度 | 档次A：轻量扩展 | 档次B：标准扩展 | 档次C：大型模型 |
|------|---------------|----------------|---------------|
| **目标参数量** | ~150M | ~300M | ~600M |
| **precision_dim** | 384 | 512 | 768 |
| **embedding_dim** | 192 | 256 | 384 |
| **num_blocks** | 12 | 16 | 24 |
| **num_heads** | 12 | 16 | 24 |
| **HR编码器** | 轻量独立stem（共享STP） | 独立HR分支 + 跨尺度融合 | 完整双塔 + APM模块 |
| **多尺度机制** | 通道拼接扩展 | 空间特征金字塔 + 自适应融合 | SkySense V2风格APM |
| **Batch Size** | 64-128 | 128-256 | 256-512 |
| **所需GPU** | 8×H100 | 16-32×H100 | 64×H100+ |
| **训练时长**（100 epochs） | ~6周 | ~4周 | ~3周 |
| **估算GPU-hours** | ~8,000 | ~15,000 | ~35,000 |
| **年算力成本（云）** | ~40-60万元 | ~80-120万元 | ~180-250万元 |

---

### 3.1 档次A：轻量扩展（~150M参数）

#### 3.1.1 设计目标
以最小硬件投入验证全国数据pipeline和模型扩展思路，适合预算有限或作为后续更大模型的预实验。

#### 3.1.2 架构变更

**传感器编码器扩展：**
```
当前: SensorEncoder → stem(3x3,s=2) → projection → [B,T,256,64,64]
扩展后:
  ├─ LR Branch (S2/S1/Landsat @10m/30m):
  │   ├─ stem_dim: 128 → 192
  │   ├─ out_dim: 256 → 384
  │   └─ 输出: [B,T,384,64,64]
  └─ HR Branch (GF-2/GF-7 @2m/0.8m):
      ├─ 轻量stem: 2层3x3 conv (保持与LR相同深度)
      ├─ HR Adapter: 1x1 conv 映射到384维
      └─ 输出: [B,T,384,64,64]（HR数据先下采样到10m网格后编码）
  Fusion: Concat(LR, HR) → 1x1 conv降维到384
```

**STP主干扩展：**
- precision_dim: 256 → **384**
- num_blocks: 8 → **12**
- num_heads: 8 → **12**
- 注意力机制保持标准MultiheadAttention，启用Flash Attention可选
- **不引入跨尺度注意力**，HR信息通过Fusion后的统一特征传播

**Bottleneck与Decoder：**
- embedding_dim: 128 → **192**
- reconstruction_channels: 6 → **8**（适应更多波段）
- Decoder hidden_mult: 1× → **1.5×**
- 不增加HR-specific decoder

**损失函数：**
- 保持V5配置：uniformity_weight=1.5, temporal_magnitude_weight=0.3
- reconstruction_weight=1.0, consistency_weight=0.05
- 增加 `hr_reconstruction_weight=0.5`（引导HR分支学习）

#### 3.1.3 参数量估算

| 组件 | 参数量 |
|------|--------|
| Sensor Encoder Bank | ~0.3M |
| STP Blocks (×12, dim=384) | ~98M |
| Bottleneck (dim=192) | ~0.07M |
| Per-source Decoders (×7, expanded) | ~48M |
| Classification Heads | ~0.02M |
| **总计** | **~146M** |

#### 3.1.4 训练配置

```yaml
# 训练配置（档次A）
gpu: 8×H100 80GB
distributed: DDP (torchrun --nproc_per_node=8)
effective_batch_size: 128  # 8卡 × 16本地
gradient_accumulation: 1   # 无需累积
epochs: 100
lr: 1e-4
lr_schedule: cosine
warmup_epochs: 10
recon_warmup_epochs: 10
optimizer: AdamW
weight_decay: 0.05
grad_clip_norm: 1.0
```

#### 3.1.5 数据需求
- 训练样本：23万patches（与B/C档相同）
- 数据加载：流式加载（不预加载全量到内存）
- 缓存策略：按epoch预取，NVMe SSD缓存热数据

#### 3.1.6 硬件需求

| 组件 | 配置 | 预算参考 |
|------|------|---------|
| 训练节点 | 1节点 × 8×H100 80GB SXM | 购置~80-120万 / 云租~3-5万/月 |
| CPU | 2×AMD EPYC 9654 或 Intel Xeon Platinum 8480+ | — |
| 内存 | 1-2 TB DDR5 | — |
| 系统盘 | 2×3.84TB NVMe SSD (RAID 1) | — |
| 数据盘 | 8×15.36TB NVMe SSD | — |
| 网络 | 200Gbps InfiniBand | — |
| 数据下载/预处理 | 16核/128GB/10Gbps带宽 | ~5万 |
| 对象存储 | MinIO/Ceph ~200TB | ~15-20万 |

#### 3.1.7 预期效果
- uniformity: -3.0 ~ -3.3
- CD AUC: 0.905 ~ 0.915
- 城市精细地物识别：较基线提升有限（高分信息利用不充分）
- 工程价值：验证全国数据pipeline、训练稳定性、损失收敛曲线

---

### 3.2 档次B：标准扩展（~300M参数）【推荐】

#### 3.2.1 设计目标
业界验证的"甜点"规模，对齐Prithvi-EO-2.0 300M和DOFA 300M。独立HR分支 + 跨尺度融合，显著提升城市精细识别能力，兼顾训练成本与性能。

#### 3.2.2 架构变更

**传感器编码器扩展：**
```
扩展后:
  ├─ LR Branch (S2/S1/Landsat @10m/30m):
  │   ├─ stem_dim: 128 → 256
  │   ├─ out_dim: 256 → 512
  │   └─ 输出: [B,T,512,64,64]
  └─ HR Branch (GF-2/GF-7 @2m/0.8m):
      ├─ 独立HR Stem: 4层3x3 conv（比LR更深，提取高分细节）
      ├─ HR-specific BatchNorm（高分数据分布与S2不同）
      ├─ 多尺度输入: 2m原始 + 4x下采样(8m) + 16x下采样(32m)
      ├─ 金字塔融合: FPN风格，合并多尺度HR特征
      ├─ HR Adapter: 1x1 conv + 3x3 conv 映射到512维
      └─ 输出: [B,T,512,64,64]
  Fusion: Cross-Scale Fusion
      ├─ Concat(LR, HR)
      ├─ 1x1 conv 降维到512
      └─ SE-Block 通道注意力（自适应加权LR/HR信息）
```

**STP主干扩展：**
- precision_dim: 256 → **512**
- num_blocks: 8 → **16**
- num_heads: 8 → **16**
- **跨尺度注意力（Cross-Scale Attention）**: 在Block 8和Block 16处插入
  - Query来自LR特征，Key/Value来自HR特征
  - 让高分细节引导低分特征的精细化
  - 仅在空间维度做cross-attention（时序维度独立）
- **Flash Attention 2/3**: 必选项，512维×16头需要显存优化
- **Gradient Checkpointing**: 保持启用

**Bottleneck与Decoder：**
- embedding_dim: 128 → **256**
- reconstruction_channels: 6 → **10**（增加GF高分波段重建）
- Decoder hidden_mult: 1× → **2×**
- **增加HR-specific Decoder**: 可选模块，专门重建2m分辨率输出
  - 结构：PixelShuffle上采样 ×2 → 3×3 conv → PixelShuffle ×2 → 输出
  - 输出尺寸：256×256@2m（对应原图128×128@10m的区域）

**时间编码扩展：**
- time_code_dim: 64 → **128**
- window_code_dim: 64 → **128**
- 增加**绝对位置编码**（经纬度sin/cos编码，参考Prithvi）

**损失函数：**
- 保持V5全部损失
- 增加 `hr_reconstruction_weight=1.0`
- 增加 `cross_scale_consistency_weight=0.1`（LR/HR特征一致性）

#### 3.2.3 参数量估算

| 组件 | 参数量 |
|------|--------|
| Sensor Encoder Bank (LR+HR) | ~1.2M |
| STP Blocks (×16, dim=512, heads=16) | ~210M |
| Cross-Scale Attention (×2) | ~8M |
| Bottleneck (dim=256) | ~0.13M |
| Per-source Decoders (×7, 2× hidden) | ~65M |
| HR-specific Decoder | ~12M |
| Classification Heads | ~0.05M |
| **总计** | **~296M** |

#### 3.2.4 训练配置

```yaml
# 训练配置（档次B）
gpu: 16-32×H100 80GB
distributed: DDP / FSDP (推荐FSDP以节省显存)
effective_batch_size: 256  # 16卡 × 16本地 或 32卡 × 8本地
gradient_accumulation: 1-2
epochs: 100
lr: 5e-5  # 更大模型需要更小lr
lr_schedule: cosine_with_warmup
warmup_epochs: 10
recon_warmup_epochs: 10
optimizer: AdamW
weight_decay: 0.05
grad_clip_norm: 1.0
# 学习率缩放: lr = base_lr × sqrt(global_batch_size / 256)
```

#### 3.2.5 数据需求
- 训练样本：23万patches（与A档相同）
- 数据加载：推荐预加载到高速NVMe（38TB缓存）
- 数据并行：FSDP的ZeRO-3 stage，模型分片到各卡

#### 3.2.6 硬件需求

| 组件 | 配置 | 预算参考 |
|------|------|---------|
| 训练节点 | 2-4节点 × 8×H100 80GB SXM | 购置~200-400万 / 云租~8-15万/月 |
| CPU | 每节点 2×AMD EPYC 9654 | — |
| 内存 | 每节点 2 TB DDR5 | — |
| 系统盘 | 每节点 2×3.84TB NVMe SSD | — |
| 数据盘 | 每节点 8×15.36TB NVMe SSD | — |
| 网络 | 400Gbps InfiniBand NDR（节点间） | — |
| 并行文件系统 | Lustre/WEKA ~200TB | ~30-50万 |
| 数据下载/预处理集群 | 4节点 × 32核/256GB | ~20万 |
| 对象存储归档 | MinIO/Ceph ~300TB | ~30-40万 |

#### 3.2.7 预期效果
- uniformity: -3.2 ~ -3.5
- CD AUC: 0.92 ~ 0.93
- 城市精细地物识别：较基线显著提升（建筑物边界、道路网络）
- 全国LULC分类：预期OA > 85%

---

### 3.3 档次C：大型模型（~600M参数）

#### 3.3.1 设计目标
追求极致性能，对标Prithvi-EO-2.0 600M和SkySense 2B中的Large配置。适合有充足预算、以发表顶级论文或打造行业标杆为目标的场景。

#### 3.3.2 架构变更

**传感器编码器扩展：**
```
扩展后:
  ├─ LR Branch (S2/S1/Landsat @10m/30m):
  │   ├─ stem_dim: 128 → 384
  │   ├─ out_dim: 256 → 768
  │   └─ 输出: [B,T,768,64,64]
  └─ HR Branch (GF-2/GF-7 @2m/0.8m):
      ├─ 双塔结构:
      │   ├─ Tower-A: 处理2m原始分辨率（4层conv + 2层dilated conv）
      │   └─ Tower-B: 处理4x/16x下采样（标准stem）
      ├─ 塔间融合: Attention-based fusion
      ├─ HR Adapter: 2层1x1 conv + 3x3 conv 映射到768维
      └─ 输出: [B,T,768,64,64]
  Fusion: Adaptive Patch Merging (APM) — 参考SkySense V2
      ├─ 对HR tokens: 每stage选择性2×2 patch merge
      ├─ 对LR tokens: 保持分辨率（通过线性投影）
      └─ 输出: 统一token序列，不同模态空间分辨率自适应
```

**STP主干扩展：**
- precision_dim: 256 → **768**
- num_blocks: 8 → **24**
- num_heads: 8 → **24**
- **跨尺度注意力**: 在Block 6, 12, 18, 24处插入
- **局部-全局混合注意力**: 前8层局部窗口注意力（减少计算），后16层全局注意力
- **Flash Attention 3**: 必选项
- **稀疏注意力**: 对超长序列（>64帧）启用局部+膨胀注意力

**Bottleneck与Decoder：**
- embedding_dim: 128 → **384**
- reconstruction_channels: 6 → **12**（覆盖更多高分波段）
- Decoder hidden_mult: 1× → **3×**
- **HR-specific Decoder**: 必选项，支持2m和0.8m两种输出分辨率
- **多尺度Decoder**: 同时输出10m/2m/0.8m三种分辨率重建

**时间编码扩展：**
- time_code_dim: 64 → **256**
- window_code_dim: 64 → **256**
- relative_time_code_dim: 16 → **64**
- 绝对位置编码（经纬度）+ 海拔编码（DEM高程）

**损失函数：**
- 保持V5全部损失
- `hr_reconstruction_weight=1.5`
- `cross_scale_consistency_weight=0.2`
- 增加 `geo_context_prototype_weight=0.1`（地理上下文原型学习，参考SkySense）

#### 3.3.3 参数量估算

| 组件 | 参数量 |
|------|--------|
| Sensor Encoder Bank (LR+HR双塔) | ~4M |
| STP Blocks (×24, dim=768, heads=24) | ~480M |
| Cross-Scale Attention (×4) | ~32M |
| Bottleneck (dim=384) | ~0.3M |
| Per-source Decoders (×7, 3× hidden) | ~55M |
| HR-specific Decoders (×2) | ~28M |
| Geo-Context Prototype (~1000区域 × 100原型 × 768维) | ~77M |
| Classification Heads | ~0.1M |
| **总计** | **~676M** |

#### 3.3.4 训练配置

```yaml
# 训练配置（档次C）
gpu: 64×H100 80GB 或更多
distributed: FSDP (ZeRO-3) / DeepSpeed
effective_batch_size: 512  # 64卡 × 8本地
gradient_accumulation: 1
epochs: 100
lr: 2e-5  # 大模型需要更保守的lr
lr_schedule: cosine_with_warmup
warmup_epochs: 15
recon_warmup_epochs: 15
optimizer: AdamW
weight_decay: 0.05
grad_clip_norm: 1.0
# 混合精度: bf16 (大模型训练稳定性)
# 激活检查点: 必开
# 编译: torch.compile (inductor模式)
```

#### 3.3.5 数据需求
- 训练样本：建议增加到35-50万patches（更大模型需要更多数据避免过拟合）
- 数据加载：必须预加载到高速并行文件系统
- 数据并行：FSDP ZeRO-3 + 激活检查点 + 梯度压缩

#### 3.3.6 硬件需求

| 组件 | 配置 | 预算参考 |
|------|------|---------|
| 训练集群 | 8+节点 × 8×H100 80GB SXM | 购置~500-800万 / 云租~20-30万/月 |
| CPU | 每节点 2×AMD EPYC 9754 | — |
| 内存 | 每节点 2-4 TB DDR5 | — |
| 网络 | 全互联InfiniBand NDR 400Gbps | — |
| 并行文件系统 | Lustre/WEKA/GPFS ~500TB | ~80-120万 |
| 数据下载/预处理集群 | 8节点 × 64核/512GB | ~50万 |
| 对象存储归档 | Ceph/MinIO ~500TB | ~50-70万 |

#### 3.3.7 预期效果
- uniformity: -3.5 ~ -3.8
- CD AUC: 0.93 ~ 0.95
- 城市精细地物识别：接近商用水平（建筑物IoU > 0.75）
- 全国LULC分类：预期OA > 88%
- 零样本/少样本迁移能力：显著提升

---

## 四、训练数据需求汇总（三档次通用）

### 4.1 数据管道建设

| 阶段 | 任务 | 输出 | 周期 |
|------|------|------|------|
| 数据获取 | STAC API接入（Sentinel Hub, 高分数据中心, Copernicus Data Space） | 原始影像下载 | 持续 |
| 预处理 | 大气校正、云掩膜、正射校正、GF全色-多光谱融合 | L2级产品 | 1-2天/批次 |
| 切分 | 按固定地理网格（10m分辨率）切分128×128 patch | Patch TIFF | 实时 |
| 对齐 | 多源数据时空对齐（同地理位置、近时间窗口） | 对齐patch组 | 实时 |
| 缓存 | 转换为训练格式（Zarr/LMDB/TFRecord） | 训练缓存 | 一次性 |
| 质检 | 云量过滤、NDVI异常检测、重复检测、地类均衡性 | 质量报告 | 每周 |

### 4.2 数据增强策略

| 增强类型 | 方法 | 适用场景 |
|---------|------|---------|
| 空间增强 | 随机翻转、90°旋转、颜色抖动、MixUp | 全部样本 |
| 时序增强 | 随机帧丢弃、时间窗口裁剪、时间戳扰动 | 全部样本 |
| 高分增强 | 随机下采样（2m→5m→10m）、锐化、噪声注入 | HR样本 |
| 源增强 | 随机源丢弃（模拟S2缺失/S1缺失） | 全部样本 |
| 地理增强 | 随机地理位置偏移（模拟配准误差） | 全部样本 |

### 4.3 训练样本配比（推荐）

| 样本类型 | 比例 | 说明 |
|---------|------|------|
| 10m纯LR样本 | 40% | S2+S1+Landsat，无高分 |
| 10m+2m融合样本 | 35% | 高分覆盖区域，主要训练对象 |
| 纯高分样本 | 15% | 仅GF数据，强化HR编码器 |
| 时序变化样本 | 10% | 双窗口（mixed_scale），重点优化CD |

---

## 五、机器与算力需求汇总

### 5.1 三档次硬件配置对比

| 组件 | 档次A | 档次B | 档次C |
|------|-------|-------|-------|
| **训练GPU** | 8×H100 80GB | 16-32×H100 80GB | 64×H100+ 80GB |
| **训练节点** | 1节点 | 2-4节点 | 8+节点 |
| **单节点CPU** | 2×EPYC 9654 | 2×EPYC 9654 | 2×EPYC 9754 |
| **单节点内存** | 1-2 TB | 2 TB | 2-4 TB |
| **节点间网络** | 单机 | 400Gbps IB NDR | 全互联IB NDR |
| **高速存储** | 8×15.36TB NVMe | 每节点8×15.36TB NVMe + 共享并行FS | 每节点8×15.36TB NVMe + 大规模并行FS |
| **并行文件系统** | 不需要 | Lustre/WEKA 200TB | Lustre/WEKA/GPFS 500TB+ |
| **对象存储归档** | MinIO 200TB | MinIO/Ceph 300TB | Ceph/MinIO 500TB |
| **数据预处理集群** | 16核/128GB | 4节点×32核/256GB | 8节点×64核/512GB |
| **年算力成本（云）** | ~40-60万元 | ~80-120万元 | ~180-250万元 |
| **一次性购置成本** | ~100-150万 | ~250-450万 | ~600-1000万 |

### 5.2 存储需求明细（三档次通用）

| 项目 | 容量 | 介质建议 |
|------|------|---------|
| 原始数据（1年） | ~41 TB | 对象存储（冷数据） |
| 原始数据归档（3年） | ~120 TB | 对象存储 |
| 预处理训练缓存 | ~38 TB | NVMe SSD（热数据） |
| 模型Checkpoint | ~30 GB | NVMe SSD |
| 训练日志/TensorBoard | ~2 TB | SATA SSD |
| 工作空间/临时 | ~10 TB | NVMe SSD |
| **活跃存储合计** | **~50 TB** | — |
| **总存储（3年）** | **~170 TB** | — |

### 5.3 软件栈建议

| 层级 | 组件 | 推荐选择 |
|------|------|---------|
| 深度学习框架 | PyTorch | 2.6+ |
| 分布式训练 | DDP / FSDP / DeepSpeed | FSDP for B/C档 |
| 数据加载 | WebDataset / TorchData | WebDataset（高吞吐） |
| 实验追踪 | WandB / MLflow / TensorBoard | WandB（推荐） |
| 数据版本 | DVC / LakeFS | DVC |
| 模型 registry | HuggingFace Hub / MLflow | HuggingFace Hub |
| 容器化 | Docker + Kubernetes / Slurm | Slurm（HPC）或K8s（云） |

---

## 六、1年期进度安排

### Q1（第1-3月）：数据基础设施与架构设计

| 周次 | 任务 | 交付物 | 负责角色 |
|------|------|--------|---------|
| W1-2 | 全国数据目录构建，STAC API接入（Sentinel Hub、高分云服务平台、Copernicus Data Space） | 数据目录系统 | 数据工程师 |
| W3-4 | 数据下载pipeline开发，断点续传，增量更新，数据权限申请 | 自动化下载系统 | 数据工程师 |
| W5-6 | 预处理pipeline：大气校正（Sen2Cor）、云掩膜（SCL/FMask）、GF全色融合、正射校正 | 预处理流水线 | 算法工程师 |
| W7-8 | Patch采样策略实现（分层采样：城市高密度/农村中密度/荒漠低密度） | 采样系统 + 首批测试数据 | 算法工程师 |
| W9-10 | 数据质量验证：云量统计、NDVI分布、地类覆盖均衡性检查、高分数据覆盖度评估 | 数据质量报告 | 数据工程师 |
| W11-12 | 小规模测试集构建（黑龙江省全省，~3万patches），数据缓存系统（Zarr/LMDB） | 黑龙江测试集 + 缓存系统 | 全组 |

**Q1里程碑**: 黑龙江全省测试集就绪，数据pipeline稳定运行，数据质量达标，高分数据权限到位。

### Q2（第4-6月）：模型架构升级与省级验证

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W13-14 | 多尺度SensorEncoder设计（三档次分别实现），HR Branch编码器开发 | 编码器模块 |
| W15-16 | 扩展STP Blocks实现（A: dim=384,blocks=12 / B: dim=512,blocks=16 / C: dim=768,blocks=24），Flash Attention集成 | STP模块 |
| W17-18 | 新Bottleneck与Decoder实现（A: dim=192 / B: dim=256+HR decoder / C: dim=384+多尺度decoder），跨尺度注意力实现 | Bottleneck + Decoder |
| W19-20 | 省级验证训练（黑龙江，3万patches，按选定档次配置，50 epochs） | 省级模型 + 评估报告 |
| W21-22 | 架构调优：根据省级验证结果调整HR融合策略、损失权重、学习率 | 调优后的架构 |
| W23-24 | 全国数据构建完成，全量数据预处理（23万patches），数据缓存优化 | 全国数据集 v1.0 |

**Q2里程碑**: 省级验证模型收敛，全国数据集v1.0完成，架构冻结，选定档次配置锁定。

### Q3（第7-9月）：全国预训练

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W25-26 | 全国预训练启动（warmup + recon warmup），监控uniformity/temporal loss/CD AUC，建立训练看板 | 训练监控系统 |
| W27-30 | 持续预训练（第1-50 epochs），中间评估（每10 epoch验证），动态调整损失权重 | 中期checkpoints |
| W31-34 | 持续预训练（第50-100 epochs），学习率衰减，EMA teacher更新，最佳模型选择 | 最佳checkpoint |
| W35-36 | 完整评估：KNN分类、变化检测AUC、重建质量PSNR/SSIM、地类分布一致性、高分重建质量 | 评估报告 |

**Q3里程碑**: 全国预训练完成，最佳checkpoint选定，uniformity <-3.0（B/C档<-3.2），CD AUC > 0.90（B档>0.92）。

### Q4（第10-12月）：下游任务、推理部署与文档

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W37-38 | 下游任务head开发：变化检测头（月度/年度CD）、LULC分类头（像素级/patch级）、建筑物/道路提取头 | 下游head模块 |
| W39-40 | 全国推理pipeline开发：Tile-based推理、地理坐标保持、拼接融合、分辨率自适应输出 | 推理系统 |
| W41-42 | 下游任务微调（freeze encoder / LoRA / full finetune对比实验），选定最优策略 | 微调模型 |
| W43-44 | 全国 Demo 部署（选定3-5个重点城市：北京、上海、深圳、成都、哈尔滨），性能基准测试 | Demo系统 |
| W45-46 | 模型文档、技术报告、数据说明书、模型卡（Model Card） | 文档集 |
| W47-48 | 模型封装（ONNX/TensorRT优化）、API服务化、容器化部署 | 生产部署包 |

**Q4里程碑**: 下游任务模型可用，全国推理pipeline就绪，Demo上线，模型文档完整。

---

## 七、风险评估与应对

| 风险 | 影响 | 概率 | 应对策略 |
|------|------|------|---------|
| 高分数据获取困难 | 高分融合受阻，被迫退回纯10m | 中 | 提前3个月申请高分数据使用权限；与自然资源卫星应用中心建立合作；准备降级方案（纯S2 10m仍可训练） |
| 存储不足/IO瓶颈 | 训练中断或GPU利用率低 | 中 | 采用WebDataset流式加载+NVMe缓存分层；优先保障热数据在本地NVMe；对象存储用于归档 |
| 训练不稳定/不收敛 | 模型无法使用 | 中 | 从基线best checkpoint逐步扩展（先扩宽再加深）；每档先跑10 epoch验证稳定性；保留完整rollback能力 |
| uniformity-temporal张力恶化 | embedding坍缩或退化 | 低 | 延续基线调参经验（uniformity_weight动态调整）；引入早停机制监控uniformity下限 |
| 多卡通信瓶颈 | 训练效率低，成本超支 | 中 | 使用FSDP替代纯DDP；启用梯度压缩（FP8/INT8）；节点内NVLink + 节点间IB |
| 全国数据偏斜 | 某些地类/区域过拟合 | 中 | 分层采样确保地类均衡；引入逆频率采样；地理上下文原型学习 |
| 高分-低分配准误差 | 多尺度融合效果差 | 中 | 预处理阶段严格正射校正；训练时引入随机偏移增强；配准质量阈值过滤 |

---

## 八、方案选择建议

**综合技术成熟度、资源可控性和预期收益，推荐采用【档次B：标准扩展（300M参数）】作为主力方案**：

1. **模型规模适中**: 300M参数是业界验证过的"甜点"（Prithvi-EO-2.0 300M、DOFA 300M均取得SOTA），兼顾表征能力与训练成本
2. **算力可及**: 16-32×H100可通过云租赁（阿里云灵骏、AutoDL、Lambda Labs、CoreWeave）在可控预算内获得
3. **架构渐进**: 基于现有STP架构扩展，保留基线所有反坍缩和时序改进，技术风险低
4. **高分融合实用**: 独立HR分支 + 跨尺度融合 + SE注意力，显著提升城市精细地物识别能力
5. **1年内可交付**: Q1数据→Q2验证→Q3训练→Q4部署，节奏清晰，风险可控

**若预算极度受限**，可先以档次A验证数据pipeline和基本架构，6个月后评估升级至B档。

**若以发表顶会/打造行业标杆为目标**，且预算充足，可直接启动档次C，但需预留额外3个月调参时间。

---

## 九、附录：国际基准模型对比

| 模型 | 参数 | 训练数据 | 算力 | 分辨率 | 开源 | 全国适用性 |
|------|------|---------|------|--------|------|-----------|
| Prithvi-EO-2.0 | 300M/600M | 4.2M@30m | 21k-58k A100-hours | 30m | ✅ | 中（分辨率低，中国区域未针对性优化） |
| SkySense | 2.06B | 21.5M multimodal | 24.6k A100-hours | 0.1-30m | ❌ | 高（但不开源，无法复用） |
| Satlas | 100M Swin | 50.4M@10m | 未公开 | 10m | ✅ | 高（全球覆盖，但无高分融合） |
| DOFA | 86M-330M | 11.5M multimodal | 576 L40-hours | 1-30m | ✅ | 高（多传感器，但时序能力弱） |
| ScaleMAE | 300M ViT | 360k@0.1-30m | 未公开 | 0.1-30m | ✅ | 中（尺度自适应强，但数据量小） |
| GFM | 未公开 | 大规模 | 未公开 | 10m | ❌ | 未知 |
| **自研全国版（目标B档）** | **300M** | **230K@10m+2m** | **~15k H100-hours** | **2m-30m** | **可控** | **高（中国定制，高分融合，时序敏感）** |

---

## 十、文档输出说明

本文档为战略规划性质，不包含具体代码实现。各档次详细实现方案需在架构设计阶段进一步细化，包括：
- 每档的详细PyTorch模块定义
- FSDP/DeepSpeed分布式配置
- 数据加载器（WebDataset）实现
- 训练循环与损失组合
- 推理优化（ONNX/TensorRT导出）

> **输出位置**: 本文档将输出至项目路径 `/workspace/xuannv/national_scale_expansion_plan.md`，供团队评审和后续阶段参考。
> **性质**: 纯规划文档，不涉及模型训练执行或代码修改。
