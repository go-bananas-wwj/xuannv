# AEF 官方论文 vs 当前 xuannv 实现：详细架构对比

> 基于 AEF 官方论文 arXiv:2507.22291 (Google DeepMind, 2025.07) 与当前项目 `v12-clean-dynamic` 分支 (HEAD: 9dcd930)

---

## 一、整体架构概览

| 维度 | AEF 官方 | 当前 xuannv 实现 | 差异评级 |
|------|---------|-----------------|---------|
| **模型类型** | 自监督 Embedding Field Model | 自监督 + 蒸馏（AEF + OlmoEarth） | 🔴 重大差异 |
| **Embedding 维度** | 64D | 64D | 🟢 一致 |
| **输入分辨率** | 128×128 (1280m@10m) | 128×128 | 🟢 一致 |
| **输出分辨率** | 128×128 | **64×64** | 🔴 **关键差异** |
| **STP Block 数** | 论文未明确（非官方复现用 6） | 6 | 🟡 可能一致 |
| **VMF kappa** | **κ = 8000** (固定) | **κ = 10** (可学习) | 🔴 **关键差异** |
| **文本对齐** | 有 (CLIP-like, weight=0.001) | 有接口但未启用 | 🟡 功能缺失 |

---

## 二、Encoder 架构逐层对比

### 2.1 输入预处理 `IndividualSourceEncoder`

| 维度 | AEF 官方 | 当前 xuannv |
|------|---------|------------|
| **输入源** | S2, S1, Landsat, PALSAR-2, GEDI, **文本** | tianyi_sar, S1, S2, Landsat, Planet |
| **每源编码** | 独立 MLP | 独立 MLP (2层: in→16→32) |
| **latent_dim** | 未明确 | 32 per source |
| **总输入通道** | 未明确 | 160 (= 32 × 5 sources) |

**差异**：AEF 官方有 **6 个输入源**（含 GEDI/文本），当前只有 5 个光学/SAR 源。文本源在 AEF 中用于地理语义对齐，当前未加载。

### 2.2 STPEncoder 三通路

| 通路 | 维度 | AEF 官方 | 当前 xuannv | 差异 |
|------|------|---------|------------|------|
| **Space** | d_s | 512 | 512 | 🟢 一致 |
| **Time** | d_t | 256 | 256 | 🟢 一致 |
| **Precision** | d_p | 64 | 64 | 🟢 一致 |
| **Space 分辨率** | 1/16L | 8×8 | 8×8 | 🟢 一致 |
| **Time 分辨率** | 1/8L | 16×16 | 16×16 | 🟢 一致 |
| **Precision 分辨率** | 1/2L | **64×64** | **64×64** | 🟢 一致 |

**结论**：Encoder 内部三通路的分辨率和维度**完全一致**。

### 2.3 STPBlock 跨尺度交换

| 交换方向 | AEF 官方 | 当前 xuannv | 差异 |
|---------|---------|------------|------|
| `space_to_time` | ConvTranspose2d (k=4,s=2,p=1) | ConvTranspose2d (k=4,s=2,p=1) | 🟢 一致 |
| `space_to_precision` | Conv2d stride=8 | Conv2d stride=8 | 🟢 一致 |
| `time_to_space` | Conv2d stride=2 | Conv2d stride=2 | 🟢 一致 |
| `time_to_precision` | Conv2d stride=4 | Conv2d stride=4 | 🟢 一致 |
| `precision_to_space` | Conv2d stride=8 | Conv2d stride=8 | 🟢 一致 |
| `precision_to_time` | Conv2d stride=4 | Conv2d stride=4 | 🟢 一致 |

**已知问题**：`ConvTranspose2d(k=4,s=2,p=1)` 会产生 **棋盘格伪影**（checkerboard artifacts）。AEF 官方论文未明确说明如何处理，但官方 embedding 输出为 128×128，而当前实现输出为 64×64——如果 `final_space_resample` 从 8×8 上采样到 128×128（scale=16），`ConvTranspose2d` 需要连续执行 4 次（每次 ×2），棋盘格问题会被放大。

### 2.4 Operator 实现

| Operator | AEF 官方 | 当前 xuannv | 差异 |
|---------|---------|------------|------|
| **SpaceOperator** | ViT MHA @ 1/16L | ViT MHA @ 1/16L (8 heads, d_h=64) | 🟢 一致 |
| **TimeOperator** | Time-axial MHA @ 1/8L + sinusoidal | Time-axial MHA @ 1/8L + sinusoidal | 🟢 一致 |
| **PrecisionOperator** | 3×3 Conv, GroupNorm, GELU | 3×3 Conv, GroupNorm, GELU | 🟢 一致 |

---

## 三、TemporalSummarizer 对比

| 组件 | AEF 官方 | 当前 xuannv | 差异 |
|------|---------|------------|------|
| **SummaryPeriodEncoder** | Sinusoidal time encoding of [t_s, t_e, duration] | 相同实现 | 🟢 一致 |
| **TimePooling** | Single-query multi-head attention over T | 相同实现 (8 heads) | 🟢 一致 |
| **Projection** | `Linear(d_p → 64)` | `Linear(64 → 64, bias=False)` | 🟡 细微差异 |
| **L2 Norm** | **输出前做 L2 norm** (单位球面) | **训练时 skip L2 norm** (pre-norm) | 🔴 **设计分歧** |
| **proj_64 初始化** | 未明确 | `xavier_normal_(gain=2.0)` | 🟡 增强初始化 |

### 关于 L2 Norm 的设计分歧

这是当前项目**最重要的设计变更**之一：

- **AEF 官方**：Summarizer 输出先做 L2 norm，embedding 分布在单位球面 S^63 上。Batch Uniformity Loss 在 L2 空间计算。
- **当前 xuannv**：Summarizer **不做 L2 norm**，输出 pre-norm 向量。所有反坍缩损失（raw_uniformity, VICReg, erank, coding_rate, magnitude）在欧氏空间计算。

**论文理由**：绕过 L2 norm 的 Jacobian 梯度屏障——当 embedding 接近单位球面时，L2 norm 的梯度会趋向于 0，导致 uniformity loss 的梯度消失。

**代价**：
1. Embedding 不再保证在单位球面上，破坏了 VMF 采样的前提假设
2. Consistency Loss 计算前需要手动做 L2 norm
3. AEF 蒸馏时也需要手动做 L2 norm

---

## 四、VonMisesFisherDecoder 对比

| 维度 | AEF 官方 | 当前 xuannv | 差异 |
|------|---------|------------|------|
| **VMF kappa** | **κ = 8000** (固定超参) | **κ = 10** (可学习, init=10) | 🔴 **关键差异** |
| **采样方式** | deterministic (num_samples=1) | deterministic (num_samples=1) | 🟢 一致 |
| **Geometry dim** | 16 | 16 | 🟢 一致 |
| **Timecode** | Sinusoidal encoding of relative position | 相同 | 🟢 一致 |
| **Decoder MLP** | embedding+geo+time → 512 → 256 → C_src | 相同结构 | 🟢 一致 |
| **输出上采样** | 无（直接输出目标分辨率） | **F.interpolate 到 128×128** | 🔴 差异 |

### VMF kappa 差异分析

这是**最核心的超参数差异**：

- **κ = 8000**（官方）：VMF 分布极度尖锐，几乎确定性。`sample_von_mises_fisher` 中 noise_scale = 1/8000 ≈ 0.000125，采样几乎就是 μ 本身。这意味着 decoder 直接以 embedding 方向为条件重建，没有随机性。
- **κ = 10**（当前）：VMF 分布非常平坦。noise_scale = 1/10 = 0.1，采样会在 μ 附近较大范围内波动。

**影响**：
1. κ=10 时 decoder 接收的输入方向不稳定，增加了重建难度
2. 较低的 κ 理论上应该增加鲁棒性，但实际可能导致重建信号模糊
3. 可学习的 κ 可能收敛到错误值（如过小导致过度噪声）

---

## 五、Loss 设计对比

### 5.1 AEF 官方（4 项损失）

| 损失 | 权重 | 计算空间 | 作用 |
|------|------|---------|------|
| **Reconstruction** | α = **1.0** | 像素空间 | 主目标，自适应 L1/CE |
| **Batch Uniformity** | b = **0.05** | L2-normed | 推 embedding 到单位球面均匀分布 |
| **Consistency** | c = **0.02** | L2-normed | Teacher-Student 一致性 |
| **Text Contrastive** | d = **0.001** | L2-normed | CLIP 对齐地理文本 |

**特点**：
- 4 项损失，结构极简
- 权重差距不大（最大 1.0 vs 0.001 = 1000×）
- Batch Uniformity 是**唯一**的反坍缩机制

### 5.2 当前 xuannv（12 项损失）

| 损失 | 权重 | 计算空间 | 来源 |
|------|------|---------|------|
| Reconstruction | 0.5 | 像素 | AEF 官方 |
| Uniformity (L2) | 2.0 | L2-normed | AEF 官方 |
| **Raw Uniformity** | **20.0** | **pre-norm** | 新增 |
| **Variance (VICReg)** | **50.0** | **pre-norm** | 新增 |
| **Covariance (VICReg)** | **5.0** | **pre-norm** | 新增 |
| **ERank** | **5.0** | **pre-norm** | 新增 |
| **Coding Rate** | **2.0** | **pre-norm** | 新增 |
| **Magnitude** | **5.0** | **pre-norm** | 新增 |
| Consistency | 0.02 | L2-normed | AEF 官方 |
| CLIP (text) | 0.001 | L2-normed | AEF 官方（未启用） |
| **AEF Distill** | **0.5** | L2-normed | **新增** |
| Decorrelation | 0.0 | pre-norm | 已关闭 |

**特点**：
- 12 项损失，结构极复杂
- 权重差距巨大（最大 50.0 vs 0.001 = 50000×）
- **pre-norm 空间损失总权重 = 87.0**，远大于 reconstruction 的 0.5
- 引入了 AEF 官方 embedding 蒸馏（0.5）

### 5.3 Loss 权重对比图

```
AEF 官方:     Recon ████ (1.0)    Uniformity ▏ (0.05)   Consistency ▏ (0.02)   Text ▏ (0.001)

当前 xuannv:  Recon █ (0.5)       Uniformity ▍ (2.0)    RawUnif ████████ (20.0)  Var ████████████████████ (50.0)
              Cov ▌ (5.0)         ERank ▌ (5.0)        Coding ▏ (2.0)          Mag ▌ (5.0)
              Consist ▏ (0.02)    Distill ▍ (0.5)
```

**问题诊断**：
1. **VICReg Variance 权重 50.0 过高**：这可能主导梯度，导致模型优先满足方差约束而非重建
2. **Reconstruction 权重 0.5 过低**：主目标被边缘化
3. **多重反坍缩机制冲突**：raw_uniformity + VICReg + erank + coding_rate + magnitude 同时作用，可能存在梯度方向冲突
4. **蒸馏 loss (0.5) 与 reconstruction (0.5) 同权重**：模型可能优先复制 AEF 教师而非学习自有表示

---

## 六、训练配置对比

| 维度 | AEF 官方 | 当前 xuannv | 差异 |
|------|---------|------------|------|
| **数据规模** | **5,145,244** global locations | **424** patches, Harbin | 🔴 12000× 差距 |
| **Batch size** | **256** | **4** (有效 16 with grad accum) | 🔴 64× 差距 |
| **训练步数** | **100,000** | ~5,000–20,000 | 🔴 5–20× 差距 |
| **Patch size** | 128×128 | 128×128 | 🟢 一致 |
| **空间分辨率** | 10m (1280m patch) | ~10m (哈尔滨) | 🟢 一致 |
| **优化器** | AdamW (lr=1e-4, warmup=2000) | AdamW (lr=1e-4, warmup=2000) | 🟢 一致 |
| **DDP** | 多 GPU (推测 8–64 卡) | 4–8 NPU (hccl) | 🟡 后端不同 |
| **find_unused_params** | 未明确 | True | 🟡 保守设置 |

### 数据规模差异的量化

```
AEF 官方:  5,145,244 locations × 128×128 patch ≈ 84.4 billion pixels
当前:      424 patches × 128×128 patch ≈ 6.96 million pixels

差距: 84.4B / 6.96M ≈ 12,126×
```

---

## 七、关键差异总结与问题诊断

### 🔴 一级问题（可能导致训练失败）

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 1 | **输出分辨率 64×64** | 空间信息损失 4× | 改为 128×128（需解决 OOM） |
| 2 | **VMF kappa = 10** | decoder 接收过度噪声输入 | 改为固定 κ=8000 或可学习但 init=8000 |
| 3 | **Reconstruction weight = 0.5** | 主目标被边缘化 | 恢复到 1.0 |
| 4 | **VICReg Variance weight = 50.0** | 可能主导梯度 | 降低到 1.0–5.0 |
| 5 | **12 项 loss 同时优化** | 梯度冲突、训练不稳定 | 精简到 4–5 项 |
| 6 | **数据量 424 patches** | 严重过拟合风险 | 需数据增强或更多数据 |

### 🟡 二级问题（影响性能但非致命）

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 7 | **Skip L2 norm 设计** | 破坏 VMF 假设 | 考虑在 inference 时恢复 L2 norm |
| 8 | **ConvTranspose2d 棋盘格** | 空间伪影 | 替换为 `Upsample+Conv2d` |
| 9 | **Batch size = 4** | uniformity loss 统计不稳定 | 增大到 32+ 或使用 memory bank |
| 10 | **AEF 蒸馏** | 教师本身可能非最优 | 分阶段训练：先蒸馏对齐，再自监督 |

### 🟢 一致的部分（无需改动）

- STP 三通路架构（Space/Time/Precision）
- 通路维度（512/256/64）
- Operator 实现（MHA + Conv）
- TemporalSummarizer 结构（PeriodEncoder + TimePooling）
- Decoder 结构（VMF + MLP）
- 输入分辨率（128×128）
- Embedding 维度（64D）

---

## 八、重构建议

基于以上对比，重构优先级如下：

### Phase 1: 修复核心架构缺陷
1. **提升输出分辨率到 128×128**
   - 修改 `STPEncoder` 的 precision pathway：从 `adaptive_avg_pool2d` 到 `H/2` 改为保持 `H`（或使用 stride=1 卷积）
   - 或者：保持 64×64 内部处理，但最终 `final_*_resample` 上采样到 128×128
2. **修复 VMF kappa**
   - 改为 `self.log_kappa = nn.Parameter(torch.log(torch.tensor(8000.0)))`
   - 或者固定为 8000（不需要学习）
3. **替换 ConvTranspose2d**
   - `LearnedSpatialResampling` 中 scale > 1 时使用 `nn.Upsample(scale_factor=2, mode='bilinear') + nn.Conv2d` 替代 `ConvTranspose2d`

### Phase 2: 简化 Loss 设计
1. **恢复官方 Loss 结构**：Reconstruction (1.0) + Uniformity (0.05) + Consistency (0.02)
2. **可选保留 1–2 项 pre-norm 反坍缩**：如 raw_uniformity (1.0) + variance (1.0)
3. **关闭冗余损失**：erank, coding_rate, magnitude, decorrelation

### Phase 3: 数据与训练
1. **分阶段训练**
   - 阶段 1（0–1000 steps）：高蒸馏权重（2.0–5.0），低 reconstruction（0.1），对齐 AEF 教师
   - 阶段 2（1000+ steps）：恢复正常权重，逐步降低蒸馏至 0.1–0.2
2. **数据增强**
   - 对 424 patches 做随机旋转/翻转/时序子采样
3. **增大 batch size**
   - 使用 gradient accumulation 达到有效 batch 32+

---

*文档生成时间: 2026-06-06*
*参考: AEF 官方论文 arXiv:2507.22291, 当前项目 HEAD: 9dcd930*
