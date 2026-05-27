# AEF 复现对比分析报告

> 对比对象: 我们的项目 (xuannv_embdding) vs Brayden-Zhang/alphaearth-foundations (GitHub 最完整训练复现)
> 分析时间: 2026-05-16

---

## 一、数据预处理

### 1.1 归一化方式

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| S2 | `log(x+1)/10` → z-score → `±6σ clip` | **min-max per-band** (band_min, band_max) |
| S1 | `clip[-30,10]` → z-score → `±6σ clip` | min-max |
| 统计量 | 预计算 JSON，全局统计 | **运行时动态计算 per-sample** |

**分析:**
- Brayden-Zhang 的 min-max 简单但**不对数据分布做假设**
- 我们的 `log(x+1)/10` 假设数据是反射率，但大庆/海淀数据是 float32 0-1，与哈尔滨 uint16 不同，已确认存在 bug
- 他的动态 min-max **避免了跨数据源格式不一致问题**

### 1.2 时间戳处理

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 来源 | 从 TIFF 文件名解析 (YYYYMMDD) | 从 CSV metadata 读取，fallback 到 `base_timestamp + t*month_ms` |
| 精度 | 精确到日 | 粗略到月 |

**分析:**
- 我们的时间戳更精确，符合论文设计
- 他的 fallback 策略可能导致时间信息不准确，但这只是因为他用的 OlmoEarth 数据集没有时间戳

### 1.3 目标帧选择

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 策略 | 从当月可用帧中**随机采样 1 帧** | 选择**最接近时间中心的那 1 帧** |
| 代码 | `random.choice(valid_indices)` | `idx = (ts - center).abs().argmin(dim=1)` |

**分析:**
- 我们的随机采样**增加数据多样性**，论文也说 "randomly selected"
- 他的中心帧选择更稳定，但多样性差
- **两者都不是 bug，是设计选择**

### 1.4 输入帧数

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 上限 | max_frames=32，超过取前 32 | 由 OlmoEarth 数据集决定 (约 10 帧/年) |
| 子采样 | 超过 32 取排序后前 32 (确定性) | 无子采样，全部使用 |

**分析:**
- 我们的**确定性截断是 bug**: 永远只能看到最早的 32 帧
- 应改为随机采样 32 帧

---

## 二、Encoder 架构

### 2.1 输入格式

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| Tensor 格式 | `[B, S, T, C, H, W]` (channel-first) | `[B, T, H, W, C]` (channel-last) |
| 多源处理 | SensorEncoderBank: 每源独立 Conv2d → 堆叠 | IndividualSourceEncoder: Linear 投影 → 拼接 channel |
| 空间降采样 | SensorEncoder stem: `stride=2` Conv2d | `adaptive_avg_pool2d` 到 1/2L |

**分析:**
- 他的 channel-last 是 transformer 社区惯例，我们的 channel-first 是 PyTorch 图像惯例
- **两者都可以工作，不是 bug**
- 但他用 `adaptive_avg_pool2d` 而不是 stride=2 conv，**丢失了高频信息**，可能是简化

### 2.2 STP Block 核心差异

这是**最重大的架构差异**。

#### 论文原文 (Figure 2D):
> "simultaneous pathways at different resolutions to maintain efficiency and spatial precision"
> "Each block terminates with learned Laplacian pyramid rescaling, enabling cross-resolution state passing"

#### Brayden-Zhang 的实现:
- 三条路径**独立存在**，各有自己的特征图
- Space: 1/16L, Time: 1/8L, Precision: 1/2L
- **每个 block 内有显式的跨尺度交换**:
  - `space_to_time`, `space_to_precision`
  - `time_to_space`, `time_to_precision`
  - `precision_to_space`, `precision_to_time`
- 使用 `LearnedSpatialResampling` (可学习的 Laplacian pyramid rescaling)

#### 我们的实现:
- 三条路径**在同一个 feature map 上操作**
- 每个 block:
  1. Precision path: 3×3 conv (保持 1/2L)
  2. Time path: 下采样 1/2 → attention → 上采样回来
  3. Space path: 下采样 1/4 → attention → 上采样回来
  4. 三条路径 concat → 1×1 fusion → + residual

**关键问题:**

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| **没有跨尺度状态传递** | 🔴 高 | 论文明确说 "cross-resolution state passing"，我们只是 concat+fusion，没有在不同分辨率之间传递信息 |
| **Space/Time 路径在 block 内上采样回来** | 🟡 中 | 论文中 Space path 应始终保持在 1/16L，我们的实现在每个 block 内下采样再上采样，增加了计算量且可能破坏多尺度信息 |
| **Block 数量不足** | 🟡 中 | 论文 15 blocks，我们只有 8 个 |
| **路径维度不一致** | 🟡 中 | 论文 Space=1024, Time=512, Precision=128，我们三条路径都是同一维度 (channels=256) |

### 2.3 Temporal Summarizer

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| Query 来源 | `summary_query(window_code)` | `SummaryPeriodEncoder(valid_period)` (编码 start/end/duration) |
| Attention | `query · pooled` 点积 → softmax | Single-query MHA (MultiheadAttention) |
| 输出 | `einsum("bt,btchw->bchw")` | TimePooling: MHA over time at each (h,w) |

**分析:**
- 他的 Summarizer 更复杂，用 MHA 对每个空间位置独立做时间聚合
- 我们的更简单，全局 pooling 后加权
- **论文没有明确说明 summarizer 的实现细节，两者都可能是合理的近似**

### 2.4 论文配置 vs 我们的配置

| 参数 | 论文 | Brayden-Zhang | 我们 |
|------|------|---------------|------|
| Precision dim (d_p) | 128 | 128 | 256 |
| Time dim (d_t) | 512 | 512 | 256 |
| Space dim (d_s) | 1024 | 1024 | 256 |
| STP blocks | 15 | 6 (small) | 8 |
| Embedding dim | 64 | 64 | 64 |
| VMF κ | 8e3 | 10 (训练) | 5000 |
| Patch size | 256? | 256 | 128 |

**分析:**
- 我们的 d_p/d_t/d_s 都是 256，**没有区分三条路径的维度**，论文明确区分
- 这可能影响模型 capacity，尤其是 Space path 需要更大维度来捕捉全局上下文

---

## 三、Bottleneck

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 结构 | `VMFBottleneck`: Conv1×1 → L2 norm | `proj_64`: Linear(feature_dim, 64) → L2 norm |
| Skip L2 | V13 支持 `skip_l2_training` | 始终 L2 normalize |
| VMF noise | 训练时 `mean + N(0, 1/κ)` | 同上 |
| 输出 | `embedding_map`, `embedding`, `pre_norm` | `mu_t`, `img_embed_t` |

**分析:**
- 两者核心逻辑相同（Conv/Linear + L2 norm）
- 我们的 `skip_l2_training` 是 V13 的创新，论文没有这个设计
- **不是 bug，是我们的优化尝试**

---

## 四、Decoder（已详细分析过，补充要点）

| 项目 | 我们的代码 | Brayden-Zhang | 论文 |
|------|-----------|---------------|------|
| 类型 | Conv2d (2层 3×3) | **逐像素 MLP** (3层 512→256) | **逐像素 MLP** (2层 width=512) |
| 条件注入 | ConditionInjector (加法) | 直接 concat embedding+geo+time | 同上 |
| 上采样 | **无** | **Bilinear upsample** | Re-gridding |
| 目标分辨率 | 64×64 | 256×256 (原始) | 原始分辨率 |

**结论: 我们的 Decoder 设计偏离论文，是明确的改进点。**

---

## 五、损失函数

### 5.1 Reconstruction Loss

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 类型 | L1 (连续) / CE (分类) | L1 (连续) / CE (分类) |
| 目标尺寸 | 64×64 | 256×256 |
| Mask | `recon_mask` (50% 像素 mask) | 无 mask |
| 权重 | per-source `source_recon_weights` | 固定 1.0 |

**分析:**
- 我们的 `recon_mask` 是 V13-MAE 设计，论文没有这个
- 他的 loss 直接在全分辨率上计算， ours 在降采样后计算

### 5.2 Uniformity Loss

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 实现 | `batch_uniformity_loss_l2` (gather + 矩阵运算) | `torch.roll` 后点积 |
| 公式 | `Σ\|u_i · u'_i\|` | `mean(\|x · x'\|)` (相同) |
| 空间 | 只在 `pre_norm` 向量上计算 | 在 `embedding_map` (所有像素) 上计算 |

**分析:**
- 他在所有像素上计算 uniformity (B×H×W 个向量)，我们在 batch 级别 (B 个向量)
- 论文说 "batch-rotated embeddings"，**他的实现更接近论文**
-  ours 的 batch 级别 uniformity 可能不够强，特别是 H×W=64×64=4096 个位置只共享 B 个约束

### 5.3 Consistency Loss

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 公式 | MSE(`teacher_emb`, `student_emb`) | `(1 - cos) / 2` |
| 空间 | 全局 mean embedding | 全局 mean embedding |

**分析:**
- 他的 `(1-cos)/2` 更标准（与 cosine distance 一致）
- 我们的 MSE 在球面上不太合适（因为 embedding 已 L2 norm）
- **建议改为 `(1-cos)/2`**

### 5.4 其他 Loss

| Loss | 我们的代码 | Brayden-Zhang | 论文 |
|------|-----------|---------------|------|
| Variance/Covariance (VICReg) | ✅ 有 | ❌ 无 | ❌ 无 |
| Decorrelation | ✅ 有 | ❌ 无 | ❌ 无 |
| Coding Rate | ✅ 有 (V14) | ❌ 无 | ❌ 无 |
| Orthogonality | ✅ 有 | ❌ 无 | ❌ 无 |
| Text-contrastive | ❌ 无 | ✅ 有 (可选) | ✅ 有 |
| Classification | ✅ 有 (aux) | ❌ 无 | ❌ 无 |

**分析:**
- 我们加了大量论文没有的 loss (VICReg, decorr, coding rate, orth)
- 这些是反坍缩的尝试，但可能**过度约束了 embedding 空间**
- 论文只用 4 个 loss: recon + uniform + consist + text

---

## 六、Teacher-Student 机制

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 参数共享 | ✅ 共享 (同一个 model) | ✅ 共享 (同一个 model) |
| Teacher 输入 | 完整输入 | 完整输入 |
| Student 扰动 | Source drop + Frame drop + Truncation | Source drop + Frame drop / Half-year drop |
| S2 是否 drop | **从不 drop** (prob=0) | **从不 drop** (prob=0) |

**分析:**
- 两者都保留了 S2 不 drop 的设计
- 我们的扰动策略更复杂（3-stage），他的更简单
- **都不是 bug**

---

## 七、训练流程

| 项目 | 我们的代码 | Brayden-Zhang |
|------|-----------|---------------|
| 优化器 | AdamW + cosine warmup | Adam (无 warmup) |
| 学习率 | 1e-4 ~ 1e-3 (配置不同) | 1e-4 |
| Batch size | 4 (DDP 6 卡 → 有效 24) | 4-32 (单卡) |
| 训练步数 | 50 epochs × 200 steps = 10k | 20k steps |
| 梯度累积 | 部分实验 accum=4 | 无 |
| AMP/BF16 | ❌ 禁用 | ❌ 禁用 |
| Checkpoint | 每 epoch / 每 10 epoch | 每 step (覆盖) |

**分析:**
- 我们的训练量只有他的 1/2 (10k vs 20k steps)
- 我们的 batch size 有效 24，比他大，这是优势
- 但数据 bug 导致这些训练都不可信

---

## 八、关键 Bug / 设计缺陷汇总

### 🔴 高优先级 (必须修)

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | **数据预处理 bug**: 统计量在原始值计算，但 z-score 用于 log 变换后的值 | `transforms.py` | S2 信号几乎丢失，所有训练结果不可信 |
| 2 | **STP Block 缺少跨尺度信息传递**: 论文明确要 Laplacian pyramid rescaling，我们只有 concat+fusion | `blocks.py` | 多尺度信息无法有效融合，Space/Time path 成为独立计算 |
| 3 | **Decoder 无上采样**: 目标 64×64，论文要求原始分辨率 | `decoders.py` + `model.py` | 重建任务过于简单，无法检验 embedding 质量 |
| 4 | **输入帧确定性截断**: 超过 32 帧永远只能看到最早的 | `dataset.py` `_get_item` | 数据多样性严重受限 |

### 🟡 中优先级 (建议修)

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 5 | **STP Block 三条路径同维度**: 论文 Space=1024, Time=512, Precision=128 | `blocks.py` + `model.py` | 模型 capacity 不足，Space path 无法捕捉足够全局上下文 |
| 6 | **Block 数量不足**: 8 vs 论文 15 | `model.py` | 网络深度不够 |
| 7 | **Uniformity 只在 batch 级别**: 应在所有像素上计算 | `losses.py` | 约束强度不够，embedding 空间可能不均匀 |
| 8 | **Consistency loss 用 MSE**: 球面上应使用 cosine distance | `ddp_v14_trainer.py` | 度量不合适 |
| 9 | **Loss 过多**: VICReg + decorr + coding rate + orth，论文只有 4 个 | `losses.py` + trainer | 过度约束，可能互相冲突 |

### 🟢 低优先级 (可选优化)

| # | 问题 | 说明 |
|---|------|------|
| 10 | 目标帧随机采样 vs 中心帧 | 论文说 random，我们已实现，比 Brayden-Zhang 好 |
| 11 | Channel-first vs channel-last | 不影响正确性 |
| 12 | Patch size 128 vs 256 | 资源限制，可以接受 |
| 13 | SensorEncoderBank 用 Conv2d vs Linear | Conv2d 保留空间结构，可能更好 |

---

## 九、修复建议优先级

**第一阶段（等数据 bug 修复后）:**
1. 修数据 bug → 重新训练基线
2. 加上 Decoder 上采样，目标保持 128×128
3. 输入帧超过 32 时随机采样

**第二阶段（如果基线仍然效果差）:**
4. 把 Conv2d Decoder 换成逐像素 MLP
5. Consistency loss 改为 `(1-cos)/2`
6. Uniformity 在所有像素上计算

**第三阶段（大改动，需评估）:**
7. 重构 STP Block，加入跨尺度信息传递（Laplacian pyramid exchange）
8. 区分三条路径的维度（Space=1024, Time=512, Precision=128）
9. 增加到 15 个 blocks
10. 简化 Loss，去掉 VICReg/coding rate 等论文没有的项

---

## 十、参考代码

- Brayden-Zhang/alphaearth-foundations: https://github.com/Brayden-Zhang/alphaearth-foundations
  - `src/alphaearth/architecture/encoder.py` — STPEncoder
  - `src/alphaearth/architecture/STPBlock.py` — STPBlock with pyramid exchange
  - `src/alphaearth/architecture/decoder.py` — VonMisesFisherDecoder (MLP)
  - `src/alphaearth/architecture/aef_module.py` — AlphaEarthFoundations (upsample)
  - `src/alphaearth/loss_function.py` — AEFLoss
  - `src/alphaearth/training.py` — Trainer

- AEF 论文: arXiv:2507.22291
  - Supplemental S2.4: Architectural details
  - Figure 2: Model architecture
  - S16.2.3: Reconstruction objective
