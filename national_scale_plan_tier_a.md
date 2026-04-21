# 自研遥感基础模型全国扩展计划 — 档次A：轻量扩展

> **版本**: v1.0  
> **日期**: 2026-04-15  
> **目标规模**: ~150M参数  
> **算力需求**: 8×H100 80GB  
> **年算力成本**: ~40-60万元（云租赁）

---

## 一、设计定位

以最小硬件投入验证全国数据pipeline和模型扩展思路。适合：
- 预算有限，先验证数据工程可行性
- 作为后续更大模型的预实验和baseline
- 对城市精细识别要求不高的场景

---

## 二、模型架构

### 2.1 核心参数

| 参数 | 基线值 | 扩展值 | 变化 |
|------|--------|--------|------|
| precision_dim | 256 | **384** | +50% |
| embedding_dim | 128 | **192** | +50% |
| num_blocks | 8 | **12** | +4 |
| num_heads | 8 | **12** | +4 |
| stem_dim | 128 | **192** | +50% |
| reconstruction_channels | 6 | **8** | +2 |
| decoder_hidden_mult | 1× | **1.5×** | — |

### 2.2 传感器编码器

```
├─ LR Branch (S2/S1/Landsat @10m/30m)
│   ├─ stem_dim: 192
│   ├─ out_dim: 384
│   └─ 输出: [B,T,384,64,64]
└─ HR Branch (GF-2/GF-7 @2m/0.8m)
    ├─ 轻量stem: 2层3x3 conv
    ├─ HR Adapter: 1x1 conv → 384维
    └─ 输出: [B,T,384,64,64]（HR先下采样到10m网格后编码）
Fusion: Concat(LR, HR) → 1x1 conv降维到384
```

### 2.3 STP主干

- precision_dim=384, num_blocks=12, num_heads=12
- **不引入跨尺度注意力**，HR信息通过Fusion后的统一特征传播
- Flash Attention可选（非必须）
- Gradient Checkpointing保持启用

### 2.4 Bottleneck与Decoder

- embedding_dim=192
- Per-source Decoder hidden_mult=1.5×
- **无HR-specific decoder**

### 2.5 参数量估算

| 组件 | 参数量 |
|------|--------|
| Sensor Encoder Bank | ~0.3M |
| STP Blocks (×12, dim=384) | ~98M |
| Bottleneck (dim=192) | ~0.07M |
| Per-source Decoders (×7, expanded) | ~48M |
| Classification Heads | ~0.02M |
| **总计** | **~146M** |

---

## 三、训练配置

```yaml
gpu: 8×H100 80GB
distributed: DDP (torchrun --nproc_per_node=8)
effective_batch_size: 128  # 8卡 × 16本地
gradient_accumulation: 1
epochs: 100
lr: 1e-4
lr_schedule: cosine
warmup_epochs: 10
recon_warmup_epochs: 10
optimizer: AdamW
weight_decay: 0.05
grad_clip_norm: 1.0

# 损失权重
reconstruction_weight: 1.0
uniformity_weight: 1.5
temporal_magnitude_weight: 0.3
consistency_weight: 0.05
classification_weight: 0.03
variance_weight: 0.25
decorrelation_weight: 0.05
orthogonality_weight: 0.01
hr_reconstruction_weight: 0.5  # 新增
```

---

## 四、数据需求

### 4.1 训练样本

- **总量**: 约23万patches（全国分层采样）
- **分层策略**:
  - 高密度区（城市/农田/水域交界）：~17万patches
  - 中密度区（一般农田/林地）：~5.8万patches
  - 低密度区（荒漠/高山）：~0.6万patches

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
| 训练缓存 | ~38 TB |
| 原始数据归档（3年） | ~120 TB |
| 模型Checkpoint | ~30 GB |
| 训练日志 | ~2 TB |
| **活跃存储** | **~50 TB** |
| **总存储** | **~170 TB** |

---

## 五、硬件需求

### 5.1 训练节点

| 组件 | 配置 |
|------|------|
| GPU | 8×NVIDIA H100 80GB SXM |
| CPU | 2×AMD EPYC 9654 或 Intel Xeon Platinum 8480+ |
| 内存 | 1-2 TB DDR5 |
| 系统盘 | 2×3.84TB NVMe SSD (RAID 1) |
| 数据盘 | 8×15.36TB NVMe SSD |
| 网络 | 200Gbps InfiniBand |

### 5.2 数据基础设施

| 组件 | 配置 |
|------|------|
| 数据下载服务器 | 16核/128GB/10Gbps带宽 |
| 对象存储 | MinIO/Ceph ~200TB |

### 5.3 预算参考

| 方式 | 成本 |
|------|------|
| 云租赁（8×H100） | ~3-5万元/月，年~40-60万元 |
| 一次性购置 | ~80-120万元 |
| 对象存储 | ~15-20万元 |
| **首年总投入** | **~55-80万元（云）/ ~100-140万元（购置）** |

---

## 六、进度安排

| 季度 | 关键任务 | 里程碑 |
|------|---------|--------|
| Q1 | 全国数据pipeline建设，黑龙江测试集（3万patches） | 数据就绪 |
| Q2 | 模型架构实现，省级验证训练（50 epochs） | 验证收敛 |
| Q3 | 全国预训练（100 epochs），中间评估 | 最佳模型 |
| Q4 | 下游任务微调，推理pipeline，Demo | 部署上线 |

---

## 七、预期效果

| 指标 | 基线(V5) | 预期(A档) |
|------|---------|----------|
| 参数量 | 57M | ~150M |
| uniformity | -2.9 ~ -3.1 | -3.0 ~ -3.3 |
| CD AUC | 0.896 | 0.905 ~ 0.915 |
| 城市精细识别 | 一般 | 有限提升 |

---

## 八、风险与应对

| 风险 | 应对 |
|------|------|
| 高分数据获取困难 | 提前申请权限，准备纯10m降级方案 |
| 训练不稳定 | 从基线best checkpoint逐步扩展，先跑10 epoch验证 |
| 存储IO瓶颈 | WebDataset流式加载，NVMe缓存热数据 |
