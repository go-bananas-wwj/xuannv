# AlphaEarth Foundations 架构对比与问题分析报告

**生成日期**: 2025-04-15  
**对比对象**: 
- 当前实现: `/workspace/xuannv/src/` (AEF_qwen)
- 外部复现: `https://github.com/Brayden-Zhang/alphaearth-foundations`
- 参考论文: AlphaEarth Foundations (Google DeepMind, 2025)

---

## 1. 执行摘要

经过对当前代码、外部复现代码和论文描述的详细对比分析，发现**当前架构在高层设计上方向正确，但存在多个关键实现细节与论文和外部复现不一致的问题**。这些差异可能导致训练 plateau 和 temporal sensitivity 不足。

### 关键发现
1. **STP Block 架构差异巨大**: 当前实现使用单通道维度的三条路径，而论文/复现使用三个不同通道维度 (d_s=1024, d_t=512, d_p=128) 并在不同分辨率上操作
2. **传感器编码器过复杂**: 当前使用 Conv2d 编码器，而论文使用简单的 Linear 投影 (IndividualSourceEncoder)
3. **Temporal Summarizer 实现过于简化**: 当前使用简单点积注意力，复现使用 multi-head attention pooling
4. **损失函数组合偏离论文**: 论文使用 Uniformity + Consistency + Reconstruction，当前使用 VICReg + KoLeo + Reconstruction
5. **数据量严重不足**: 424 patches vs 论文 40M 场景

---

## 2. 逐模块详细对比

### 2.1 传感器编码器 (Input Encoding)

#### 论文/外部复现
```python
class IndividualSourceEncoder(nn.Module):
    def __init__(self, source_dim: int, latent_dim: int):
        self.encoder = nn.Sequential(
            nn.Linear(source_dim, latent_dim // 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)  # x: (B*T*H*W, C)
```

**特点**:
- 简单的 MLP 投影
- 按像素独立处理 (将 H,W 展平后与 C 一起处理)
- 输入格式: `(B, T, H, W, C)` → 展平为 `(B*T*H*W, C)` → 编码

#### 当前实现
```python
class SensorEncoder(nn.Module):
    def __init__(self, in_channels, stem_channels, stem_dim, out_dim):
        self.adapter = nn.Conv2d(...)  # 可选
        self.stem = nn.Sequential(
            nn.Conv2d(stem_channels, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_dim), nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(stem_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim), nn.GELU(),
        )
```

**特点**:
- 2D 卷积编码器 (stride=2 下采样)
- 按帧独立处理 (B*T, C, H, W)
- 输出: `(B, S, T, out_dim, H/2, W/2)`

#### 差异分析
| 维度 | 论文/复现 | 当前实现 | 影响 |
|------|----------|---------|------|
| 编码方式 | Linear MLP | Conv2d + BN + GELU | 当前引入了空间归纳偏置，可能限制模型学习能力 |
| 下采样 | 无 (保持 H×W) | stride=2 (H/2 × W/2) | 当前丢失空间信息 |
| 数据格式 | (B,T,H,W,C) 通道在最后 | (B,S,T,C,H,W) 通道在第三维 | 不一致但可兼容 |
| 多源融合 | 编码后 concat 到 C 维度 | 每源独立编码后 stack | 逻辑等价 |

**⚠️ 潜在问题**: 当前的 Conv2d 编码器在 stride=2 时进行了空间下采样，这与论文中"保持分辨率"的描述不符。论文的 STP 编码器自己处理下采样。

---

### 2.2 STP 编码器 (核心架构差异)

#### 论文/外部复现
```python
class STPEncoder(nn.Module):
    def __init__(self, input_channels, d_s=1024, d_t=512, d_p=128, num_blocks=15):
        # 三个不同维度的路径
        self.space_dim = d_s      # 1024
        self.time_dim = d_t       # 512
        self.precision_dim = d_p  # 128
        
        # 分别投影到三个不同维度
        self.space_projection = nn.Linear(input_channels, d_s)
        self.time_projection = nn.Linear(input_channels, d_t)
        # precision 保持 input_channels
        
        # 下采样到不同分辨率
        # space: 1/16L, time: 1/8L, precision: 1/2L
        
        # 15 个 STP blocks
        self.blocks = nn.ModuleList([STPBlock(d_s, d_t, d_p) for _ in range(15)])
```

**STPBlock 结构**:
```python
class STPBlock(nn.Module):
    def __init__(self, space_dim, time_dim, precision_dim):
        self.space_op = SpaceOperator(space_dim)      # ViT attention @ 1/16L
        self.time_op = TimeOperator(time_dim)          # Time attention @ 1/8L
        self.precision_op = PrecisionOperator(precision_dim)  # Conv @ 1/2L
        
        # 跨尺度信息交换 (learned Laplacian pyramid)
        self.space_to_time = LearnedSpatialResampling(space_dim, time_dim, 2.0)
        self.space_to_precision = LearnedSpatialResampling(space_dim, precision_dim, 8.0)
        # ... 其他 4 个方向
    
    def forward(self, space_x, time_x, precision_x, timestamps):
        # 各路径独立操作
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps)
        precision_out = self.precision_op(precision_x)
        
        # 跨尺度交换
        space_exchange = space + time_to_space + precision_to_space
        time_exchange = time + space_to_time + precision_to_time
        precision_exchange = precision + space_to_precision + time_to_precision
        
        return space_out, time_out, precision_out
```

#### 当前实现
```python
class STPBlock(nn.Module):
    def __init__(self, channels, num_heads):
        self.precision_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        
        self.time_down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.time_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.time_up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        
        self.space_down = nn.Sequential(...)  # 两次 stride=2
        self.space_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.space_up = nn.Sequential(...)  # 两次 upsample
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GroupNorm(8, channels), nn.GELU(),
        )
```

#### 关键差异对比表

| 特性 | 论文/复现 | 当前实现 | 严重程度 |
|------|----------|---------|---------|
| **通道维度** | 3 个不同维度 (1024/512/128) | 单一维度 (256) | 🔴 **高** |
| **空间分辨率** | space@1/16L, time@1/8L, precision@1/2L | 统一 @ 1/2L | 🔴 **高** |
| **跨尺度交换** | LearnedSpatialResampling (6 个方向) | Fusion 1x1 conv | 🔴 **高** |
| **Time Operator** | 原生支持时间编码注入 | time_codes 作为外部输入 | 🟡 中 |
| **Block 数量** | 15 blocks | 4-8 blocks | 🔴 **高** |
| **参数量** | ~1B (论文) / ~100M (复现) | 57.4M | 🔴 **高** |

#### 问题分析

**🔴 严重问题 1: 单一通道维度**
- 论文明确三个路径有不同容量：space (高维，处理低频全局信息), time (中维), precision (低维，处理高频局部信息)
- 当前实现统一使用 256 维，导致所有路径竞争相同的表示容量
- 这破坏了多尺度信息分离的设计意图

**🔴 严重问题 2: 缺乏跨尺度信息交换**
- 论文使用 `LearnedSpatialResampling` 在不同分辨率间交换信息 (类似 Laplacian pyramid)
- 当前实现仅在相同分辨率上 fusion，没有真正的跨尺度信息流动
- 这导致空间/时间/精度路径成为"孤岛"

**🔴 严重问题 3: Block 数量不足**
- 论文: 15 blocks
- 当前: 4-8 blocks
- 这直接影响模型的表示能力

---

### 2.3 时间编码与 Summarizer

#### 论文/外部复现
```python
class TemporalSummarizer(nn.Module):
    def __init__(self, feature_dim, embed_dim=64, num_heads=8):
        self.summarizer_q = SummaryPeriodEncoder(dim=feature_dim)
        self.time_pool = TimePooling(dim=feature_dim, num_heads=num_heads)
        self.proj_64 = nn.Linear(feature_dim, embed_dim, bias=False)
    
    def forward(self, feats, timestamps, valid_periods, mask):
        q = self.summarizer_q(valid_periods)  # (B, C)
        z = self.time_pool(feats, q, mask=mask)  # (B, H, W, C)
        mu = self.proj_64(z)  # (B, H, W, 64)
        mu = F.normalize(mu, p=2, dim=-1)
        return mu

class TimePooling(nn.Module):
    """Multi-head attention over time at each (h,w)"""
    def forward(self, feats, q, mask):
        # q: (B, C) single query per sample
        # feats: (B, T, H, W, C)
        # 在每个 (h,w) 位置做 multi-head attention over T
        ...
```

**特点**:
- `SummaryPeriodEncoder`: 编码 valid_period [t_s, t_e] 为查询向量
- `TimePooling`: 在每个空间位置独立做 temporal attention
- 输出直接投影到 64D 并 L2 归一化

#### 当前实现
```python
# encode_frames 中的 summarization
pooled = x.mean(dim=(-2, -1))  # [B, T, C]
query = self.summary_query(window_code)[:, None, :]  # [B, 1, C]
attn_scores = torch.sum(query * pooled, dim=-1)  # [B, T]
attn_scores = attn_scores.masked_fill(~mask, -1e9)
attn = torch.softmax(attn_scores, dim=-1)  # [B, T]
summary = torch.einsum("bt,btchw->bchw", attn, x)  # [B, C, H, W]
```

**特点**:
- 使用单查询向量 (来自 window_code)
- 在所有空间位置共享相同的 temporal attention
- 先 spatial pool 再做 temporal attention

#### 差异分析
| 特性 | 论文/复现 | 当前实现 | 影响 |
|------|----------|---------|------|
| 查询构造 | SummaryPeriodEncoder (编码 [ts, te, duration]) | Linear(window_code) | 信息更少 |
| Attention 粒度 | 每个 (h,w) 独立做 temporal attention | 全局 spatial pool 后统一做 | 空间信息丢失 |
| 输出格式 | (B, H, W, 64) 保持空间结构 | (B, C, H, W) 然后过 bottleneck | 等价 |
| L2 归一化 | 直接归一化到 S^63 | skip-L2 训练技巧 | 当前更稳定 |

**⚠️ 问题**: 当前的 summarization 在所有空间位置使用相同的 temporal weights，这忽略了不同空间位置可能有不同的时间动态。

---

### 2.4 Bottleneck (VMF)

#### 论文/外部复现
```python
# 在 TemporalSummarizer 中
self.proj_64 = nn.Linear(feature_dim, embed_dim, bias=False)
mu = self.proj_64(z)  # (B, H, W, 64)
mu = F.normalize(mu, p=2, dim=-1)  # 直接 L2 归一化
```

#### 当前实现
```python
class VMFBottleneck(nn.Module):
    def forward(self, features):
        pre_norm_map = self.to_embedding(features)  # [B, D, H, W]
        if self.training and self.skip_l2_training:
            embedding_map = pre_norm_map + noise  # 跳过 L2
        else:
            direction = F.normalize(pre_norm_map, p=2, dim=1)
            embedding_map = sample_vmf(direction, self.kappa)
        ...
```

#### 评估
- **当前实现更优**: skip-L2 训练是合理的改进，解决了梯度屏障问题
- 外部复现直接 L2 归一化，可能导致训练困难

---

### 2.5 解码器

#### 论文/外部复现 (Implicit Decoder)
```python
class VonMisesFisherDecoder(nn.Module):
    def __init__(self, embedding_dim, source_dims, geometry_dim=16):
        for source, dim in source_dims.items():
            input_dim = embedding_dim + geometry_dim + embedding_dim
            self.source_decoders[source] = nn.Sequential(
                nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
                nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
                nn.Linear(256, dim)
            )
    
    def forward(self, embeddings, geometry_metadata, timestamps, valid_period, source):
        # 在每个像素位置：concat[vmf_sample, geometry, timecode]
        # 过 MLP 得到重建值
        ...
```

**特点**:
- 隐式解码器：每个像素位置独立过 MLP
- 条件输入：embedding + geometry + timecode
- 输出和 embedding 同分辨率 (1/2L)

#### 当前实现 (Explicit Decoder)
```python
class ContinuousDecoder(nn.Module):
    def __init__(self, embedding_dim, window_code_dim, relative_time_dim, metadata_dim, out_channels):
        self.injector = ConditionInjector(...)
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )
```

**特点**:
- 显式解码器：使用 2D 卷积
- 条件注入：通过 gate 机制融合条件信息
- 输出和 embedding 同分辨率

#### 评估
- 两种方式都合理
- 论文使用隐式 MLP 解码器，当前使用显式 Conv2d 解码器
- 隐式解码器更灵活 (可以解码任意分辨率/位置)，显式解码器更高效

---

### 2.6 损失函数

#### 论文公式 (Equation 3)
```
L_total = a * L_reconstruction + b * L_uniformity + c * L_consistency + d * L_text

其中:
- a = 1.0, b = 0.05, c = 0.02, d = 0.001
- L_uniformity: batch uniformity (Equation 4)
- L_consistency: teacher-student consistency (Equation 5)
```

#### 外部复现
```python
class AEFLoss:
    def __init__(self):
        self.reconstruction_weight = 1.0
        self.uniformity_weight = 0.05
        self.consistency_weight = 0.02
        self.text_weight = 0.001
    
    def batch_uniformity_loss(self, embeddings):
        x = F.normalize(x, p=2, dim=-1)
        x_prime = torch.roll(x, shifts=1, dims=0)  # ⚠️ 错误实现！
        dots = (x * x_prime).sum(dim=-1).abs()
        return dots.mean()
    
    def consistency_loss(self, teacher, student):
        mu = F.normalize(teacher, p=2, dim=-1)
        mu_s = F.normalize(student, p=2, dim=-1)
        return ((1.0 - (mu * mu_s).sum(dim=-1)) * 0.5).mean()
```

**⚠️ 外部复现的 bug**: `torch.roll` 不是随机旋转！只是循环移位，不能实现真正的均匀分布。

#### 当前实现
```python
# V4 训练器
loss = (
    recon_weight * recon
    + ct_recon_w * ct_recon
    + vicreg_weight * vicreg      # 1.0
    + koleo_weight * koleo        # 0.05
    + temporal_w * temporal       # 0.5
)
```

#### 差异分析
| 损失 | 论文 | 外部复现 | 当前实现 | 评价 |
|------|------|---------|---------|------|
| Reconstruction | ✅ L1 | ✅ L1 | ✅ L1 | 一致 |
| Uniformity | ✅ Equation 4 | ⚠️ torch.roll (bug) | ❌ 未使用 | 当前缺失 |
| Consistency | ✅ Equation 5 | ✅ 实现 | ❌ 未使用 | 当前缺失 |
| VICReg | ❌ 无 | ❌ 无 | ✅ 使用 | 当前特有 |
| KoLeo | ❌ 无 | ❌ 无 | ✅ 使用 | 当前特有 |
| Temporal | ❌ 无 | ❌ 无 | ✅ 使用 | 当前特有 |
| DINO | ❌ 无 | ❌ 无 | ❌ 已移除 | — |
| Text/CLIP | ✅ 0.001 | ✅ 实现 | ❌ 未使用 | — |

**⚠️ 关键问题**: 当前完全移除了论文的 Uniformity + Consistency 损失组合，改用 VICReg + KoLeo。虽然 VICReg + KoLeo 是合理的替代，但:
1. VICReg 已饱和 (~0.974 冻结)
2. KoLeo 过高 (~1.8，理想 0.5-1.5)
3. 缺少 teacher-student consistency 损失可能导致 embedding 不稳定

---

### 2.7 数据与训练配置

#### 论文
- 训练数据: ~40M 场景 (全球覆盖)
- Batch size: 256
- 训练步数: 100K steps
- 输入: 多源 (S2, S1, Landsat, 气候等)

#### 外部复现
- 训练数据: OlmoEarth Landsat 子集 (~1M 场景，实际 1/40)
- Batch size: 16
- 训练步数: 20K steps
- 输入: 仅 Landsat (单源)
- **未验证**: 无评估代码

#### 当前实现
- 训练数据: 424 patches (哈尔滨单区域)
- Batch size: 2 per GPU × 2 GPU × 8 accum = 32 effective
- 训练: 200 epochs (~53 steps/epoch)
- 输入: S2 + S1 + Landsat (3源)

#### 数据量对比
| 指标 | 论文 | 外部复现 | 当前实现 |
|------|------|---------|---------|
| 场景数 | ~40M | ~1M | **424 patches** |
| 每 epoch 步数 | ~156K | ~62K | **~53** |
| 总训练步数 (90 epoch) | — | — | **~4,770** |
| 相对比例 | 100% | ~0.4% | **~0.003%** |

---

## 3. 已确认的问题列表

### 🔴 严重问题 (High Priority)

#### 问题 1: STP Block 架构与论文严重不符
- **症状**: VICReg 饱和 (~0.974)，KoLeo 过高，重建停滞
- **根因**: 
  - 单一通道维度 (256) vs 论文三维度 (1024/512/128)
  - 缺乏跨尺度信息交换 (LearnedSpatialResampling)
  - Block 数量不足 (4-8 vs 15)
- **影响**: 模型容量严重不足，无法学习有效的多尺度时空表示

#### 问题 2: 数据量不足 (最根本问题)
- **症状**: 所有损失 plateau
- **根因**: 424 patches 仅相当于论文 0.003% 的数据量
- **影响**: 模型无法学到有意义的表征，正则化损失占主导

#### 问题 3: Temporal Summarizer 过于简化
- **症状**: 月度差异测试 cos_dist=0.0101 (几乎无差异)
- **根因**: 使用全局 spatial pool + 统一 temporal attention，而非 per-pixel attention
- **影响**: 不同空间位置的时间动态被平均掉

### 🟡 中等问题 (Medium Priority)

#### 问题 4: 损失函数组合偏离论文
- 当前: VICReg + KoLeo + Reconstruction
- 论文: Uniformity + Consistency + Reconstruction (+ Text)
- VICReg 已饱和，说明其已无法提供有效梯度

#### 问题 5: 传感器编码器下采样过早
- 当前在输入阶段就 stride=2 下采样
- 论文保持分辨率，让 STP 编码器处理下采样
- 可能导致高频信息丢失

#### 问题 6: 外部复现的 Uniformity Loss 有 Bug
- `torch.roll` 不是随机旋转
- 但该 bug 不影响我们的代码 (我们没使用这个损失)

### 🟢 低优先级 (Low Priority)

#### 问题 7: 缺少 Text Alignment
- 论文有 text contrastive loss (weight=0.001)
- 当前未实现
- 对变化检测影响不大

#### 问题 8: Geometry Metadata 未使用
- 论文使用 16D geometry metadata (传感器几何信息)
- 当前metadata只有4维 (window_pos, window_width, ...)
- 影响较小

---

## 4. 建议修复方案

### 短期修复 (立即执行)

#### 修复 1: 调整损失权重
```yaml
training:
  vicreg_weight: 0.0      # VICReg 已饱和，关闭
  koleo_weight: 0.1       # 提高 KoLeo 以对抗坍缩
  reconstruction_weight: 1.0  # 提高重建权重
  temporal_contrastive_weight: 1.0  # 提高时序对比
```

#### 修复 2: 增大时间窗口差异
```yaml
data:
  window_mode: "non_overlap"  # 替代 adjacent_month
```
- `adjacent_month` 差异太小 (相邻月份地表变化不大)
- `non_overlap` 使用长间隔窗口，差异信号更强

#### 修复 3: 增大有效 batch size
```yaml
training:
  gradient_accumulation_steps: 16  # 从 8 增大到 16
  # effective batch = 2 * 2 * 16 = 64
```
- KoLeo 和 temporal InfoNCE 都依赖 batch size

### 中期修复 (本周)

#### 修复 4: 重构 STP Block (对齐论文)
需修改:
1. `src/models/blocks.py`: 实现三通道维度 STPBlock
2. `src/models/model.py`: 添加 pathway projections 和 learned resampling
3. `src/models/time_encoding.py`: 增强 SummaryPeriodEncoder

参考实现 (基于外部复现):
```python
class STPBlockV2(nn.Module):
    def __init__(self, space_dim=512, time_dim=256, precision_dim=128):
        self.space_op = SpaceOperator(space_dim)
        self.time_op = TimeOperator(time_dim)
        self.precision_op = PrecisionOperator(precision_dim)
        
        # 跨尺度交换
        self.space_to_time = LearnedSpatialResampling(space_dim, time_dim, 2.0)
        self.time_to_space = LearnedSpatialResampling(time_dim, space_dim, 0.5)
        # ... 其他 4 个方向
```

#### 修复 5: 简化传感器编码器
```python
# 替代当前的 Conv2d 编码器
class IndividualSourceEncoder(nn.Module):
    def __init__(self, source_dim, latent_dim):
        self.encoder = nn.Sequential(
            nn.Linear(source_dim, latent_dim // 2),
            nn.GELU(), nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
```

#### 修复 6: 添加 Teacher-Student Consistency 损失
```python
def consistency_loss(teacher_emb, student_emb):
    t = F.normalize(teacher_emb, p=2, dim=-1)
    s = F.normalize(student_emb, p=2, dim=-1)
    return ((1.0 - (t * s).sum(dim=-1)) * 0.5).mean()
```

### 长期修复 (本月)

#### 修复 7: 扩大数据集
- 收集更多区域的数据 (农业区、森林区、城市区)
- 目标: 至少 10K-100K patches

#### 修复 8: 增大模型容量
- 增大 precision_dim: 256 → 512
- 增大 block 数量: 4 → 8-12
- 或使用论文的三维度设计

---

## 5. 验证建议

### 5.1 快速验证 (1-2 天)
1. **调整损失权重** + **non_overlap 模式** → 训练 20 epoch 观察 temporal loss 是否下降
2. **增大 accum steps** → 观察 KoLeo 是否降低到 1.5 以下

### 5.2 中期验证 (1 周)
1. **重构 STP Block** → 对比重建损失和 temporal sensitivity
2. **简化传感器编码器** → 观察是否减少参数并提高训练稳定性

### 5.3 长期验证 (1 月)
1. **扩大数据集** → 这是最根本的解决方案
2. **完整对齐论文架构** → 验证是否达到预期的 temporal sensitivity

---

## 6. 结论

### 6.1 当前代码状态评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构正确性 | 🟡 6/10 | 高层方向正确，但 STP 实现与论文差异大 |
| 损失函数 | 🟡 5/10 | VICReg+KoLeo 替代了论文的 Uniformity+Consistency，效果不佳 |
| 数据管道 | 🟢 8/10 | 多源输入、时间编码、数据增强实现完善 |
| 训练稳定性 | 🟡 6/10 | GradScaler、DDP、EMA 都正确，但 plateau 严重 |
| 代码质量 | 🟢 8/10 | 结构清晰，模块化良好 |

### 6.2 核心问题排序
1. **数据量不足** (根本原因) - 无法通过代码修复
2. **STP Block 架构差异** - 需要重构
3. **Temporal Summarizer 简化** - 需要改进
4. **损失函数组合** - 需要调整

### 6.3 推荐行动
1. **立即**: 调整损失权重 + 切换到 non_overlap 模式 + 增大 batch
2. **本周**: 重构 STP Block (对齐论文三维度设计)
3. **本月**: 扩大数据集至 10K+ patches

---

*报告结束*
