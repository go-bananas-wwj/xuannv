# 自研遥感基础模型全国扩展计划 — 档次B：标准扩展（推荐）

> **版本**: v1.0  
> **日期**: 2026-04-15  
> **目标规模**: ~300M参数  
> **算力需求**: 16-32×H100 80GB  
> **年算力成本**: ~80-120万元（云租赁）

---

## 一、设计定位

业界验证的"甜点"规模，对齐Prithvi-EO-2.0 300M和DOFA 300M。独立HR分支 + 跨尺度融合，显著提升城市精细识别能力。**推荐作为主力方案**。

适合：
- 追求性能与成本平衡
- 需要高分融合（2m+）的城市精细应用
- 1年内完整交付的目标

---

## 二、模型架构

### 2.1 核心参数

| 参数 | 基线值 | 扩展值 | 变化 |
|------|--------|--------|------|
| precision_dim | 256 | **512** | 2× |
| embedding_dim | 128 | **256** | 2× |
| num_blocks | 8 | **16** | 2× |
| num_heads | 8 | **16** | 2× |
| stem_dim | 128 | **256** | 2× |
| reconstruction_channels | 6 | **10** | +4 |
| decoder_hidden_mult | 1× | **2×** | — |
| time_code_dim | 64 | **128** | 2× |
| window_code_dim | 64 | **128** | 2× |

### 2.2 传感器编码器

```
├─ LR Branch (S2/S1/Landsat @10m/30m)
│   ├─ stem_dim: 256
│   ├─ out_dim: 512
│   └─ 输出: [B,T,512,64,64]
└─ HR Branch (GF-2/GF-7 @2m/0.8m)
    ├─ 独立HR Stem: 4层3x3 conv（比LR更深）
    ├─ HR-specific BatchNorm
    ├─ 多尺度输入: 2m原始 + 4x下采样 + 16x下采样
    ├─ 金字塔融合: FPN风格
    ├─ HR Adapter: 1x1 conv + 3x3 conv → 512维
    └─ 输出: [B,T,512,64,64]
Fusion: Cross-Scale Fusion
    ├─ Concat(LR, HR)
    ├─ 1x1 conv降维到512
    └─ SE-Block通道注意力
```

### 2.3 STP主干

- precision_dim=512, num_blocks=16, num_heads=16
- **跨尺度注意力（Cross-Scale Attention）**: Block 8和Block 16处插入
  - Query来自LR，Key/Value来自HR
  - 仅空间维度cross-attention，时序维度独立
- **Flash Attention 2/3**: 必选项
- **Gradient Checkpointing**: 保持启用

### 2.4 Bottleneck与Decoder

- embedding_dim=256
- Per-source Decoder hidden_mult=2×
- **HR-specific Decoder**: 可选模块
  - PixelShuffle上采样×2 → 3×3 conv → PixelShuffle×2
  - 输出256×256@2m

### 2.5 时间编码扩展

- time_code_dim=128, window_code_dim=128
- 增加**绝对位置编码**（经纬度sin/cos编码，参考Prithvi）

### 2.6 参数量估算

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

---

## 三、训练配置

```yaml
gpu: 16-32×H100 80GB
distributed: FSDP (推荐，节省显存)
effective_batch_size: 256  # 16卡×16 或 32卡×8
gradient_accumulation: 1-2
epochs: 100
lr: 5e-5  # 更大模型需更小lr
lr_schedule: cosine_with_warmup
warmup_epochs: 10
recon_warmup_epochs: 10
optimizer: AdamW
weight_decay: 0.05
grad_clip_norm: 1.0
# 学习率缩放: lr = base_lr × sqrt(global_batch_size / 256)

# 损失权重
reconstruction_weight: 1.0
uniformity_weight: 1.5
temporal_magnitude_weight: 0.3
consistency_weight: 0.05
classification_weight: 0.03
variance_weight: 0.25
decorrelation_weight: 0.05
orthogonality_weight: 0.01
hr_reconstruction_weight: 1.0           # 新增
cross_scale_consistency_weight: 0.1     # 新增
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

### 5.1 训练集群

| 组件 | 配置 |
|------|------|
| GPU | 2-4节点 × 8×H100 80GB SXM |
| CPU | 每节点 2×AMD EPYC 9654 |
| 内存 | 每节点 2 TB DDR5 |
| 系统盘 | 每节点 2×3.84TB NVMe SSD |
| 数据盘 | 每节点 8×15.36TB NVMe SSD |
| 网络 | 400Gbps InfiniBand NDR |

### 5.2 数据基础设施

| 组件 | 配置 |
|------|------|
| 并行文件系统 | Lustre/WEKA ~200TB |
| 数据预处理集群 | 4节点 × 32核/256GB |
| 对象存储归档 | MinIO/Ceph ~300TB |

### 5.3 预算参考

| 方式 | 成本 |
|------|------|
| 云租赁（16-32×H100） | ~8-15万元/月，年~80-120万元 |
| 一次性购置 | ~200-400万元 |
| 并行文件系统 | ~30-50万元 |
| 对象存储 | ~30-40万元 |
| **首年总投入** | **~110-170万元（云）/ ~260-490万元（购置）** |

---

## 六、进度安排

| 季度 | 关键任务 | 里程碑 |
|------|---------|--------|
| Q1 | 全国数据pipeline建设，黑龙江测试集（3万patches） | 数据就绪 |
| Q2 | 模型架构实现（含Cross-Scale Attention），省级验证训练 | 验证收敛，架构冻结 |
| Q3 | 全国预训练（100 epochs），中间评估，最佳模型选择 | uniformity<-3.2, CD AUC>0.92 |
| Q4 | 下游任务微调（CD/LULC/建筑物），推理pipeline，Demo | 部署上线 |

---

## 七、预期效果

| 指标 | 基线(V5) | 预期(B档) |
|------|---------|----------|
| 参数量 | 57M | ~300M |
| uniformity | -2.9 ~ -3.1 | -3.2 ~ -3.5 |
| CD AUC | 0.896 | 0.92 ~ 0.93 |
| 城市精细识别 | 一般 | 显著提升（建筑物边界、道路网络） |
| 全国LULC分类OA | — | >85% |

---

## 八、关键技术亮点

1. **跨尺度注意力（Cross-Scale Attention）**: 让高分细节主动引导低分特征精细化
2. **SE-Block融合**: 自适应加权LR/HR信息，避免高分噪声淹没低分信号
3. **绝对位置编码**: 引入经纬度信息，增强模型地理感知能力
4. **HR-specific Decoder**: 可选2m分辨率重建头，支持超分辨率下游任务

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| 多卡通信瓶颈 | FSDP ZeRO-3 + 梯度压缩 + NVLink/IB |
| 训练不稳定 | 从基线逐步扩展，每档先跑10 epoch验证 |
| 高分-低分配准误差 | 严格正射校正 + 随机偏移增强 + 质量阈值过滤 |
| 存储IO瓶颈 | WebDataset + NVMe缓存 + 并行文件系统 |
