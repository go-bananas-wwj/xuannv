# 自研遥感基础模型全国扩展计划 — 档次C：大型模型

> **版本**: v1.0  
> **日期**: 2026-04-15  
> **目标规模**: ~600M+参数  
> **算力需求**: 64×H100+ 80GB  
> **年算力成本**: ~180-250万元（云租赁）

---

## 一、设计定位

追求极致性能，对标Prithvi-EO-2.0 600M和SkySense Large配置。适合：
- 充足预算，以发表顶会/打造行业标杆为目标
- 需要最强零样本/少样本迁移能力
- 对城市精细识别（建筑物IoU>0.75）有硬性要求

---

## 二、模型架构

### 2.1 核心参数

| 参数 | 基线值 | 扩展值 | 变化 |
|------|--------|--------|------|
| precision_dim | 256 | **768** | 3× |
| embedding_dim | 128 | **384** | 3× |
| num_blocks | 8 | **24** | 3× |
| num_heads | 8 | **24** | 3× |
| stem_dim | 128 | **384** | 3× |
| reconstruction_channels | 6 | **12** | +6 |
| decoder_hidden_mult | 1× | **3×** | — |
| time_code_dim | 64 | **256** | 4× |
| window_code_dim | 64 | **256** | 4× |
| relative_time_code_dim | 16 | **64** | 4× |

### 2.2 传感器编码器

```
├─ LR Branch (S2/S1/Landsat @10m/30m)
│   ├─ stem_dim: 384
│   ├─ out_dim: 768
│   └─ 输出: [B,T,768,64,64]
└─ HR Branch (GF-2/GF-7 @2m/0.8m)
    ├─ 双塔结构:
    │   ├─ Tower-A: 2m原始（4层conv + 2层dilated conv）
    │   └─ Tower-B: 4x/16x下采样（标准stem）
    ├─ 塔间融合: Attention-based fusion
    ├─ HR Adapter: 2层1x1 conv + 3x3 conv → 768维
    └─ 输出: [B,T,768,64,64]
Fusion: Adaptive Patch Merging (APM) — 参考SkySense V2
    ├─ HR tokens: 每stage选择性2×2 patch merge
    ├─ LR tokens: 保持分辨率（线性投影）
    └─ 输出: 统一token序列，分辨率自适应
```

### 2.3 STP主干

- precision_dim=768, num_blocks=24, num_heads=24
- **跨尺度注意力**: Block 6, 12, 18, 24处插入
- **局部-全局混合注意力**:
  - 前8层: 局部窗口注意力（减少计算）
  - 后16层: 全局注意力
- **Flash Attention 3**: 必选项
- **稀疏注意力**: 超长序列（>64帧）启用局部+膨胀注意力

### 2.4 Bottleneck与Decoder

- embedding_dim=384
- Per-source Decoder hidden_mult=3×
- **HR-specific Decoder**: 必选项，支持2m和0.8m两种输出
- **多尺度Decoder**: 同时输出10m/2m/0.8m三种分辨率

### 2.5 时间编码扩展

- time_code_dim=256, window_code_dim=256, relative_time_code_dim=64
- 绝对位置编码（经纬度）+ **海拔编码（DEM高程）**

### 2.6 地理上下文原型（Geo-Context Prototype）

- 全国划分~1,000个地理区域
- 每区域100个原型向量，768维
- 参考SkySense，通过Sinkhorn-Knopt算法更新
- 参数: ~1,000 × 100 × 768 = **~77M**

### 2.7 参数量估算

| 组件 | 参数量 |
|------|--------|
| Sensor Encoder Bank (LR+HR双塔) | ~4M |
| STP Blocks (×24, dim=768, heads=24) | ~480M |
| Cross-Scale Attention (×4) | ~32M |
| Bottleneck (dim=384) | ~0.3M |
| Per-source Decoders (×7, 3× hidden) | ~55M |
| HR-specific Decoders (×2) | ~28M |
| Geo-Context Prototype | ~77M |
| Classification Heads | ~0.1M |
| **总计** | **~676M** |

---

## 三、训练配置

```yaml
gpu: 64×H100 80GB 或更多
distributed: FSDP (ZeRO-3) / DeepSpeed
effective_batch_size: 512  # 64卡 × 8本地
gradient_accumulation: 1
epochs: 100
lr: 2e-5  # 大模型需更保守lr
lr_schedule: cosine_with_warmup
warmup_epochs: 15
recon_warmup_epochs: 15
optimizer: AdamW
weight_decay: 0.05
grad_clip_norm: 1.0
mixed_precision: bf16  # 大模型稳定性
activation_checkpointing: required
torch_compile: inductor

# 损失权重
reconstruction_weight: 1.0
uniformity_weight: 1.5
temporal_magnitude_weight: 0.3
consistency_weight: 0.05
classification_weight: 0.03
variance_weight: 0.25
decorrelation_weight: 0.05
orthogonality_weight: 0.01
hr_reconstruction_weight: 1.5           # 新增
cross_scale_consistency_weight: 0.2     # 新增
geo_context_prototype_weight: 0.1       # 新增
```

---

## 四、数据需求

### 4.1 训练样本

- **总量**: 建议35-50万patches（更大模型需更多数据防过拟合）
- 在原23万基础上，增加：
  - 更多高分覆盖区域采样
  - 时序变化样本比例提升至15%
  - 引入季节性增强样本

### 4.2 年度原始数据量

| 数据源 | 年数据量 |
|--------|---------|
| Sentinel-2 L2A | ~16.8 TB |
| Sentinel-1 GRD | ~13.4 TB |
| Landsat-8/9 | ~3.0 TB |
| GF-1/2/6/7 高分 | ~5.7 TB |
| DEM / WorldCover / Dynamic World / JRC Water | ~2.2 TB |
| **合计** | **~41 TB/年** |

### 4.3 存储需求

| 项目 | 容量 |
|------|------|
| 训练缓存（扩大样本） | ~60 TB |
| 原始数据归档（3年） | ~120 TB |
| 模型Checkpoint | ~50 GB |
| 训练日志 | ~5 TB |
| **活跃存储** | **~70 TB** |
| **总存储** | **~200 TB** |

---

## 五、硬件需求

### 5.1 训练集群

| 组件 | 配置 |
|------|------|
| GPU | 8+节点 × 8×H100 80GB SXM |
| CPU | 每节点 2×AMD EPYC 9754 |
| 内存 | 每节点 2-4 TB DDR5 |
| 系统盘 | 每节点 2×3.84TB NVMe SSD |
| 数据盘 | 每节点 8×15.36TB NVMe SSD |
| 网络 | 全互联InfiniBand NDR 400Gbps |

### 5.2 数据基础设施

| 组件 | 配置 |
|------|------|
| 并行文件系统 | Lustre/WEKA/GPFS ~500TB |
| 数据预处理集群 | 8节点 × 64核/512GB |
| 对象存储归档 | Ceph/MinIO ~500TB |

### 5.3 预算参考

| 方式 | 成本 |
|------|------|
| 云租赁（64×H100+） | ~20-30万元/月，年~180-250万元 |
| 一次性购置 | ~500-800万元 |
| 并行文件系统 | ~80-120万元 |
| 对象存储 | ~50-70万元 |
| **首年总投入** | **~230-340万元（云）/ ~630-990万元（购置）** |

---

## 六、进度安排

| 季度 | 关键任务 | 里程碑 |
|------|---------|--------|
| Q1 | 全国数据pipeline，高分数据权限，黑龙江测试集 | 数据就绪 |
| Q2 | APM模块实现，双塔HR编码器，省级验证（50 epochs） | 验证收敛 |
| Q3 | 全国预训练（100 epochs），Geo-Context Prototype训练 | uniformity<-3.5, CD AUC>0.93 |
| Q4 | 下游任务微调，多尺度推理pipeline，Demo | 部署上线 |
| Q4+ | 额外调参优化（大型模型需更多调参时间） | 性能达标 |

> 注：C档建议预留额外1-2个月调参时间。

---

## 七、预期效果

| 指标 | 基线(V5) | 预期(C档) |
|------|---------|----------|
| 参数量 | 57M | ~676M |
| uniformity | -2.9 ~ -3.1 | -3.5 ~ -3.8 |
| CD AUC | 0.896 | 0.93 ~ 0.95 |
| 城市精细识别 | 一般 | 接近商用（建筑物IoU>0.75） |
| 全国LULC分类OA | — | >88% |
| 零样本迁移 | 弱 | 显著提升 |

---

## 八、关键技术亮点

1. **Adaptive Patch Merging (APM)**: 参考SkySense V2，自适应处理多分辨率token
2. **双塔HR编码器**: Tower-A处理原始2m，Tower-B处理多尺度下采样，attention融合
3. **局部-全局混合注意力**: 前层局部窗口降低计算，后层全局捕获长程依赖
4. **地理上下文原型**: 1,000区域 × 100原型，增强区域感知表征
5. **海拔编码**: DEM高程信息融入时间编码，增强地形感知
6. **多尺度Decoder**: 同时输出10m/2m/0.8m，支持多分辨率下游任务

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| 训练不稳定/发散 | bf16混合精度 + 更保守lr + 更长warmup + 梯度裁剪 |
| 多卡通信瓶颈 | DeepSpeed ZeRO-3 + 梯度压缩 + 全互联IB |
| 数据不足（600M需更多数据） | 扩充至50万patches + 强数据增强 + 正则化 |
| 检查点损坏 | 每5 epoch保存 + 异地备份 + md5校验 |
| 训练成本超支 | 阶段性评估，设置硬性预算上限 |
