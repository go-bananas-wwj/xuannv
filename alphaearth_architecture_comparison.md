# AlphaEarth Foundations 架构对比报告

> **对比对象**: 当前 AEF_qwen 训练代码 vs. 官方开源复现 [alphaearth-foundations](https://github.com/Brayden-Zhang/alphaearth-foundations)  
> **生成时间**: 2025-04-15  
> **目的**: 识别关键架构差异，定位当前训练 plateaus 和时间敏感性不足的根因

---

## 1. 总体评估

### 1.1 外部复现可信度: **低** (1/40 规模，未经验证)

| 维度 | 论文声明 | 外部复现 | 当前 AEF_qwen |
|------|---------|----------|--------------|
| 模型参数量 | ~1B | ~100M (1/10) | **57.4M** (1/17) |
| 训练样本 | ~40M 场景 | ~1M 场景 (估计) | **424 patches** (严重不足) |
| 批次大小 | 256 | **16** | 32 (2 GPU × 2 × 8 accum) |
| 训练时长 | 10 epochs @ 40M = 400M steps | ~1 epoch | ~90 epochs @ 424 patches |
| 数据规模 | 全球覆盖 | 单区域 | 单区域 (哈尔滨) |
| 验证方式 | 下游 SOTA 性能 | **无任何验证** | AUC + 月度差异测试 |

### 1.2 关键发现摘要

1. **外部复现不具权威性**: 规模只有论文的 1/40，batch_size=16 远低于论文 256，**无任何验证/评估代码**，uniformity loss 实现与论文不一致 (`torch.roll` vs random rotation)。
2. **当前代码在核心架构上基本正确**: 传感器编码器 → STP 块 → 瓶颈 → 多目标解码器的设计与复现一致，且更完整（支持多源输入、实际时间编码、重建目标）。
3. **训练 plateau 的根本原因是数据量不足 + 正则化过度**: 424 patches 远不足以驱动 57M 参数模型的自监督训练，VICReg 已饱和 (~0.974 冻结)，KoLeo 过高 (~1.8)，重建损失停滞 (~0.28)。
4. **时间敏感性弱的根因**: `adjacent_month` 窗口差异信号太弱 (cos_dist=0.0101)，单区域数据无显著季节性变化，temporal contrastive loss 的 anti-diagonal InfoNCE 实现可能存在问题。

---

## 2. 逐模块架构对比

### 2.1 传感器编码器 (Sensor Encoder)

#### 外部复现
```python
class SensorEncoder(nn.Module):
    def __init__(self, ...):
        self.stem = nn.Conv2d(in_channels, stem_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(stem_dim)
        self.projection = nn.Conv2d(stem_dim, out_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
    
    def forward(self, x):
        x = self.stem(x)  # [B*T, C, H, W] -> [B*T, stem_dim, H/2, W/2]
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.projection(x)  # [B*T, stem_dim, H/2, W/2] -> [B*T, out_dim, H/2, W/2]
        x = self.bn2(x)
        return x
```

#### 当前 AEF_qwen
```python
class SensorEncoder(nn.Module):
    def __init__(self, in_channels, stem_channels, stem_dim, out_dim):
        self.use_adapter = in_channels != stem_channels
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_channels), nn.GELU(),
            )
        self.stem = nn.Sequential(
            nn.Conv2d(stem_channels, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_dim), nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(stem_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim), nn.GELU(),
        )
```

#### 差异分析
| 项目 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| Adapter | ❌ 无 | ✅ 1x1 conv adapter | 当前更灵活，支持可变输入通道 |
| BatchNorm vs GroupNorm | BatchNorm | **BatchNorm** | 一致 |
| GELU | ✅ | ✅ | 一致 |
| 下采样 | stride=2 conv | **stride=2 conv** | 一致，1/2 分辨率 |
| 多源支持 | 单源 | **多源 (S2+S1+Landsat)** | 当前更完整 |
| 输出分辨率 | H/2 × W/2 | **H/2 × W/2** | 一致 |

**结论**: 当前实现更完整，支持多源输入和 adapter。**无差异问题**。

---

### 2.2 STP Block (Space-Time-Precision)

#### 外部复现
```python
class STPBlock(nn.Module):
    def __init__(self, channels, num_heads):
        self.precision_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.precision_gn = nn.GroupNorm(8, channels)
        self.precision_gelu = nn.GELU()
        
        # Time path: downsample 1/2 -> MHA -> upsample
        self.time_down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.time_mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.time_norm = nn.LayerNorm(channels)
        self.time_up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        
        # Space path: downsample 1/4 -> MHA -> upsample
        self.space_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        )
        self.space_mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.space_norm = nn.LayerNorm(channels)
        self.space_up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.residual_norm = nn.GroupNorm(8, channels)
```

#### 差异分析
| 项目 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| Precision path | 3x3 conv + GN + GELU | **相同** | 一致 |
| Time path | down 1/2 → MHA → up 2× | **相同** | 一致 |
| Space path | down 1/4 → MHA → up 4× | **相同** | 一致 |
| Fusion | concat 3 paths + 1x1 conv | **相同** | 一致 |
| Residual | GN + add | **相同** | 一致 |
| Time codes | ❌ **未使用** (forward 签名无 time_codes) | ✅ 实际计算并传入 | **外部复现缺陷！** |
| Frame mask | ❌ **未使用** | ✅ 有效使用 | **外部复现缺陷！** |

**关键发现**: 外部复现的 `STPBlock.forward()` 签名只有 `x`，**没有 time_codes 和 frame_mask 参数**！这意味着：
1. 时间编码完全未传入 STP 块
2. 帧掩码（处理变长序列）未使用
3. 这解释了为什么复现只支持固定长度输入

**结论**: 当前实现在 STP 块层面**更正确**。

---

### 2.3 时间编码 (Time Encoding)

#### 外部复现
```python
class AlphaEarthTimeEncoder(nn.Module):
    """时间编码器 — 使用正弦位置编码."""
    def __init__(self, time_dim=64, max_period=10000):
        self.time_dim = time_dim
        self.max_period = max_period
    
    def forward(self, timestamps):
        # timestamps: [B, T] in days
        angles = timestamps.unsqueeze(-1) / (self.max_period ** 
            (torch.arange(0, self.time_dim, 2, device=timestamps.device) / self.time_dim))
        encoding = torch.zeros(*timestamps.shape, self.time_dim, device=timestamps.device)
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles)
        return encoding
```

#### 当前 AEF_qwen
```python
class TimeCodeEncoder(nn.Module):
    """时间戳 → 正弦编码."""
    def __init__(self, code_dim: int = 64):
        self.code_dim = code_dim
    def forward(self, timestamps_ms: torch.Tensor) -> torch.Tensor:
        # timestamps_ms: [B, T]
        # 使用毫秒时间戳的正弦编码
        ...
```

#### 差异分析
| 项目 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| 编码方式 | 正弦位置编码 (days) | **正弦位置编码 (ms)** | 等价，仅单位不同 |
| 维度 | 64 | **64** | 一致 |
| 实际使用 | ❌ **time_codes 未传入 STP** | ✅ 传入 STP 块 | **外部复现缺陷** |
| Window encoding | ❌ **无** | ✅ WindowCodeEncoder | 当前更完整 |
| Relative time | ❌ **无** | ✅ RelativeTimeCodeEncoder | 当前更完整 |

**结论**: 外部复现虽然定义了时间编码器，但**实际未在 STP 中使用**。当前实现正确使用时间编码。**当前更优**。

---

### 2.4 瓶颈层 (Bottleneck)

#### 外部复现
```python
class AlphaEarthBottleneck(nn.Module):
    """瓶颈层 — 1x1 conv + L2 norm + VMF 采样."""
    def __init__(self, channels, embedding_dim, vmf_kappa=2000):
        self.conv = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.vmf_kappa = vmf_kappa
    
    def forward(self, x):
        # x: [B*T, C, H, W]
        x = self.conv(x)  # [B*T, D, H, W]
        # 全局平均池化
        x = x.mean(dim=[2, 3])  # [B*T, D]
        # L2 归一化
        x = F.normalize(x, p=2, dim=1)
        # VMF 采样
        if self.training:
            noise = torch.randn_like(x) / math.sqrt(self.vmf_kappa)
            x = F.normalize(x + noise, p=2, dim=1)
        return x
```

#### 当前 AEF_qwen
```python
class VMFBottleneck(nn.Module):
    """VMF Bottleneck — 训练时跳过 L2 归一化，推理时做 L2 归一化."""
    def __init__(self, channels, embedding_dim, kappa=2000.0, skip_l2_training=True):
        self.to_embedding = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.kappa = kappa
        self.skip_l2_training = skip_l2_training
    
    def forward(self, features):
        pre_norm_map = self.to_embedding(features)  # [B, D, H, W]
        
        if self.training and self.skip_l2_training:
            # 训练模式: 跳过 L2, 保留原始幅度
            if self.kappa > 0:
                noise = torch.randn_like(pre_norm_map) * math.sqrt(1.0 / self.kappa)
                embedding_map = pre_norm_map + noise
            else:
                embedding_map = pre_norm_map
        else:
            # 推理模式: 标准 L2 + VMF
            direction = F.normalize(pre_norm_map, p=2, dim=1)
            embedding_map = sample_vmf(direction, self.kappa)
        
        pre_norm_vector = pre_norm_map.mean(dim=(-2, -1))  # [B, D]
        embedding_vector = embedding_map.mean(dim=(-2, -1))  # [B, D]
        
        if not (self.training and self.skip_l2_training):
            embedding_vector = F.normalize(embedding_vector, p=2, dim=1)
        
        return embedding_map, embedding_vector, pre_norm_vector, pre_norm_map
```

#### 差异分析
| 项目 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| 1x1 conv | ✅ | ✅ | 一致 |
| 全局池化 | mean(H, W) | **mean(H, W)** | 一致 |
| L2 norm | 始终 L2 | **训练 skip, 推理 L2** | **当前核心改进** |
| VMF noise | ✅ | ✅ | 一致 |
| 输出 | 仅 embedding | **embedding_map + pre_norm** | 当前更丰富 |
| skip_l2 | ❌ 无 | ✅ 有 | 当前反坍缩关键设计 |

**关键差异**: 当前代码实现了**训练时跳过 L2 归一化** (`skip_l2_training=True`)，这是 AEF_qwen 的核心创新之一，用于解决 L2 Jacobian 梯度屏障导致的坍缩问题。

**结论**: 当前实现在瓶颈层**更先进**，skip-L2 设计是合理的改进。

---

### 2.5 目标解码器 (Target Decoders)

#### 外部复现
```python
class AlphaEarthDecoder(nn.Module):
    """简单上采样解码器."""
    def __init__(self, embedding_dim, out_channels):
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, embedding_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, embedding_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embedding_dim // 2, out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, embedding_map):
        # embedding_map: [B, D, H, W]
        return self.upsample(embedding_map)  # [B, C, 4H, 4W]
```

#### 当前 AEF_qwen
```python
class ContinuousDecoder(nn.Module):
    """连续目标解码器 — 条件上采样."""
    def __init__(self, embedding_dim, window_code_dim, relative_time_dim, metadata_dim, out_channels):
        # 使用 FiLM / AdaIN 风格条件注入
        self.conv_blocks = nn.ModuleList([
            ConditionalConvBlock(...),
            ...
        ])
    
    def forward(self, embedding_map, window_code, relative_time, metadata):
        # 条件信息注入到每一层
        ...

class CategoricalDecoder(nn.Module):
    """分类目标解码器."""
    ...
```

#### 差异分析
| 项目 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| 结构 | 简单反卷积上采样 | **条件解码器 (FiLM/AdaIN)** | 当前更复杂 |
| 条件输入 | ❌ 无 | ✅ window_code, relative_time, metadata | 当前更完整 |
| 上采样倍数 | 4× | **4×** | 一致 |
| 多目标支持 | 单目标 | **7 目标分别解码** | 当前更完整 |
| 分类目标 | ❌ 无 | ✅ CategoricalDecoder | 当前更完整 |

**结论**: 当前解码器**更复杂、更完整**，但这是否导致训练困难需要评估。外部复现的简单解码器可能更容易训练，但功能不完整。

---

### 2.6 损失函数

#### 外部复现: Uniformity Loss
```python
def uniformity_loss(z, t=2):
    """批次均匀性损失 — 使用随机旋转."""
    # 关键差异: 使用 torch.roll 而不是真正的随机旋转矩阵!
    z = F.normalize(z, dim=-1)
    # 对 batch 做循环移位 (torch.roll)
    z_shifted = torch.roll(z, shifts=1, dims=0)
    return torch.sum(z * z_shifted, dim=-1).mean()
```

**严重问题**: `torch.roll` 只是循环移位，不是随机旋转！这导致 uniformity loss 实际上在最小化相邻样本的余弦相似度，而非真正的球面均匀分布。

#### 当前 AEF_qwen: Raw Uniformity Loss
```python
def raw_uniformity_loss(embeddings):
    """欧氏空间 uniformity — 无 L2 归一化."""
    # 标准化到零均值、全局单位方差
    z = embeddings - embeddings.mean(dim=0)
    global_std = z.std() + 1e-4
    z = z / global_std
    
    # 计算 pairwise RBF
    N = z.shape[0]; D = z.shape[1]
    t = 2.0 / D
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    pair_mask = torch.triu(torch.ones(N, N, device=z.device, dtype=torch.bool), diagonal=1)
    sq_pdist_pairs = sq_pdist[pair_mask]
    loss = torch.logsumexp(-t * sq_pdist_pairs, dim=0) - math.log(sq_pdist_pairs.shape[0])
    return loss
```

#### VICReg + KoLeo (当前使用)
```python
def vicreg_loss(z_student, z_teacher, lambda_inv=1.0, mu_var=1.0, nu_cov=0.04):
    inv = F.mse_loss(z_student, z_teacher)
    z_all = torch.cat([z_student, z_teacher], dim=0)
    std = torch.sqrt(z_all.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std))
    z_all = z_all - z_all.mean(dim=0, keepdim=True)
    cov = (z_all.T @ z_all) / (z_all.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / cov.shape[0]
    return lambda_inv * inv + mu_var * var + nu_cov * cov_loss

def koleo_loss(x):
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    eye = torch.eye(dists.shape[0], device=dists.device)
    dists = dists + eye * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
```

#### 差异分析
| 损失 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| Uniformity | `torch.roll` (错误实现) | **raw_uniformity (欧氏空间)** | 当前更正确 |
| VICReg | ❌ 无 | ✅ 使用 | 当前使用 |
| KoLeo | ❌ 无 | ✅ 使用 | 当前使用 |
| DINO | ❌ 无 | ✅ 已移除 (不稳定) | — |
| Reconstruction | L1 | **L1 + CE (分类)** | 当前更完整 |
| Temporal contrastive | ❌ 无 | **InfoNCE anti-diagonal** | 当前特有 |

**结论**: 外部复现的损失函数实现**不完整且有 bug** (`torch.roll` 不是随机旋转)。当前损失函数组合更完整，但 VICReg 已饱和，KoLeo 过高，可能需要调整权重。

---

### 2.7 训练配置

#### 外部复现
```python
# 训练参数
batch_size = 16
num_epochs = 1  # 只训练 1 epoch!
learning_rate = 1e-4
weight_decay = 0.05
warmup_steps = 1000  # 未使用
```

#### 当前 AEF_qwen
```yaml
# 训练参数 (configs/qwen_v4_cd_upgrade.yaml)
batch_size: 2  # per GPU
gradient_accumulation_steps: 8  # effective batch = 32
lr: 0.0001
weight_decay: 0.05
warmup_epochs: 10
epochs: 200
vicreg_weight: 1.0
koleo_weight: 0.05
reconstruction_weight: 0.5
temporal_contrastive_weight: 0.5
```

#### 差异分析
| 项目 | 外部复现 | 当前 AEF_qwen | 评价 |
|------|---------|--------------|------|
| Batch size | 16 | **32** (effective) | 当前更大 |
| Epochs | 1 | **200** | 当前训练更充分 |
| LR scheduler | ❌ 无 | ✅ cosine | 当前更完整 |
| Grad clip | ❌ 无 | **3.0** | 当前更稳定 |
| AMP | ❌ 无 | ✅ fp16 + GradScaler | 当前更高效 |
| EMA Teacher | ❌ 无 | ✅ 0.996 momentum | 当前更完整 |
| DDP | ❌ 无 | ✅ 2 GPU | 当前更完整 |

---

## 3. 关键差异与问题定位

### 3.1 当前代码的优势 (vs 外部复现)

1. **多源输入支持**: 实际支持 S2+S1+Landsat 三源输入
2. **时间编码真正使用**: time_codes 实际传入 STP 块
3. **更完整的解码器**: 7 类目标分别解码，支持条件生成
4. **skip-L2 bottleneck**: 解决梯度屏障问题
5. **更完整的训练流程**: DDP、EMA、AMP、梯度裁剪、学习率调度
6. **raw_uniformity 损失**: 比外部复现的 `torch.roll` 更正确

### 3.2 当前代码的问题

#### 问题 1: 数据量严重不足 (最关键)
| 指标 | 论文 | 外部复现 | 当前 AEF_qwen |
|------|------|---------|--------------|
| 训练样本 | ~40M 场景 | ~1M | **424 patches** |
| 有效 batch size | 256 | 16 | **32** |
| 每 epoch 步数 | ~156K | ~62K | **~53** |

**分析**: 424 patches × ~7 frames = ~3K 场景，有效 batch=32，每 epoch 只有 **53 步**。训练 90 epoch 仅约 4,770 步，相当于论文的 **0.003%** 训练量。

**影响**: 
- 模型无法学到有意义的表征
- 正则化损失 (VICReg, KoLeo) 占主导
- 重建损失停滞在 ~0.28

#### 问题 2: VICReg 饱和
- 当前值: ~0.974 (冻结)
- 含义: invariance + variance + covariance 已达架构上限
- 可能原因: expander_dim=512 可能不足以产生多样表征，或 batch 太小 (32) 导致协方差估计不准

#### 问题 3: KoLeo 过高
- 当前值: ~1.76-1.85
- 理想范围: 0.5-1.5
- 含义: 最近邻距离太小，嵌入空间仍有聚集趋势
- 可能原因: batch 太小 (32)，koleo 依赖全局 batch 计算最近邻

#### 问题 4: Temporal Contrastive 信号太弱
- 月度差异测试: mean cos_dist = 0.0101 (几乎相同)
- 原因分析:
  1. `adjacent_month` 窗口差异本身很小 (相邻月份的地表变化有限)
  2. 单区域 (哈尔滨) 数据缺乏显著季节性变化
  3. Anti-diagonal InfoNCE 在 batch=2 时退化 (只有一个负样本)
  4. `encode_dual_window` 两次前向传播效率低

#### 问题 5: 重建损失停滞
- 当前值: ~0.28
- 可能原因:
  1. 解码器过于复杂，条件信息未有效利用
  2. 数据量太少，重建目标过于多样 (7 类)
  3. recon_weight=0.5 可能太低

### 3.3 外部复现的问题 (参考价值有限)

1. **未验证**: 无评估代码，不知道是否真的能工作
2. **规模太小**: 1/40 论文规模，batch=16 (论文 256)
3. **Uniformity loss bug**: `torch.roll` 不是随机旋转
4. **时间编码未使用**: 定义了但未传入 STP
5. **无多源支持**: 仅演示单源训练
6. **无文本/几何**: text adapter 和 geometry encoder 定义但未训练

---

## 4. 建议改进方向

### 4.1 短期 (立即执行)

1. **增加数据量** (最高优先级)
   - 扩大 patch 覆盖范围 (哈尔滨 → 更多城市/区域)
   - 使用季节性变化更明显的区域 (农业区、森林区)
   - 考虑使用公开数据集 (如 Sentinel-2 时间序列)

2. **调整 temporal contrastive**
   - 使用更大的时间间隔 (非相邻月份，如 3-6 个月)
   - 增大 batch size 以改善 InfoNCE
   - 考虑使用 `non_overlap` 模式替代 `adjacent_month`

3. **调整损失权重**
   - 提高 reconstruction_weight (0.5 → 1.0 或更高)
   - 降低 vicreg_weight (1.0 → 0.5) 或暂时关闭
   - 调整 koleo_weight (0.05 → 0.1)

### 4.2 中期 (本周)

1. **验证外部复现**: 尝试运行外部复现代码，验证其是否能收敛
2. **简化解码器**: 测试外部复现的简单解码器是否能改善重建
3. **增大模型**: 当前 57M 可能太小，考虑增大 precision_dim / embedding_dim
4. **数据增强**: 添加空间增强 (旋转、翻转、颜色抖动)

### 4.3 长期 (本月)

1. **大规模数据收集**: 按论文规模收集数据 (~40M 场景)
2. **分布式训练**: 使用更多 GPU (但当前硬件限制为 2 卡)
3. **预训练 + 微调**: 在大规模数据上预训练，在小数据上微调
4. **对比学习改进**: 考虑使用 DINOv2 或 iBOT 的改进版本

---

## 5. 附录: 详细参数对比表

### 5.1 模型架构参数

| 参数 | 外部复现 | 当前 AEF_qwen | 论文 |
|------|---------|--------------|------|
| stem_dim | 64 | **64** | 64 |
| precision_dim | 256 | **256** | 256 |
| embedding_dim | 64 | **128** | 64 |
| num_heads | 8 | **8** | 8 |
| num_blocks | 4 | **4** | 4 |
| time_code_dim | 64 | **64** | 64 |
| vmf_kappa | 2000 | **2000** | 2000 |
| expander_dim | N/A | **512** | N/A |
| 总参数量 | ~100M | **57.4M** | ~1B |

### 5.2 训练参数

| 参数 | 外部复现 | 当前 AEF_qwen | 论文 |
|------|---------|--------------|------|
| batch_size | 16 | **2 (per GPU)** | 256 |
| effective batch | 16 | **32** | 256 |
| epochs | 1 | **200** | 10 |
| LR | 1e-4 | **1e-4** | 1e-4 |
| weight_decay | 0.05 | **0.05** | 0.05 |
| warmup | 未使用 | **10 epochs** | ? |
| grad_clip | 无 | **3.0** | ? |
| AMP | 无 | **fp16** | ? |

### 5.3 损失权重

| 损失 | 外部复现 | 当前 AEF_qwen |
|------|---------|--------------|
| reconstruction | 1.0 | **0.5** |
| uniformity | 1.0 (torch.roll) | **0.0 (使用 VICReg)** |
| VICReg | N/A | **1.0** |
| KoLeo | N/A | **0.05** |
| DINO | N/A | **0.0 (已移除)** |
| temporal | N/A | **0.5** |
| CT recon | N/A | **0.5** |

---

## 6. 结论

1. **当前代码在架构层面基本正确**，且比外部复现更完整、更忠实于论文设计。
2. **外部复现参考价值有限**: 规模太小 (1/40)，有 bug (`torch.roll`)，未验证。
3. **训练 plateau 的根本原因是数据量不足**: 424 patches 无法驱动 57M 参数的自监督模型。
4. **时间敏感性弱的原因**: 相邻月份差异信号太弱 + batch 太小 + 单区域数据缺乏变化。
5. **优先行动**: 
   - 立即扩大数据量 (最重要)
   - 调整 temporal contrastive 为更大间隔
   - 提高 reconstruction 权重
   - 考虑简化解码器或增大模型容量

---

*报告结束*
