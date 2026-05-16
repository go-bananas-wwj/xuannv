# AEF 对齐检查 — 详细发现

## 日期: 2026-05-16

---

## 1. Mask 可视化结果 ✅

变化检测 mask 数据合理：
- june: 45/424 patches changed, 0.200% pixels
- aug: 21/424 patches changed, 0.056% pixels  
- september: 24/424 patches changed, 0.096% pixels
- october: 27/424 patches changed, 0.100% pixels

可视化图 (`data/change_masks_visualization.png`) 显示：
- 变化区域以绿色显示在红色背景上，形状合理（线性/块状）
- patch_000000 在所有时期全红（无变化），与预期一致
- 变化比例极低但正常（遥感变化通常是局部的）

---

## 2. /grill-me 对齐检查 — 严重不对齐项

### 🔴 CRITICAL: Decoder 条件注入被完全禁用

**位置**: `src/models/decoders.py` V13

**问题**: `ConditionInjector.forward` 直接返回 embedding，不做任何修改：
```python
def forward(self, embedding, window_code=None, relative_time=None, metadata=None):
    return embedding  # V13: 直接返回，不做条件注入
```

**影响**: 
- Decoder 无法接收 window_code / relative_time
- Reconstruction **不 forcing time-conditioned decoding**
- 这是 **temporal blindness 的根因之一**！

**AEF 原文**: Decoder 应该是条件解码器，接收时间条件并生成时间特定的重建。

---

### 🔴 CRITICAL: Bottleneck 训练时 Skip L2

**位置**: `src/models/bottleneck.py` + 配置

**问题**: 配置 `skip_l2_norm_training: true`，且代码中 `skip_l2_training` 参数生效：
```python
if self.training and self.skip_l2_training:
    embedding_map = pre_norm_map  # 不 L2 归一化！
```

**影响**: 
- 训练时 embedding 不在单位球面上
- Consistency loss 在 non-spherical 空间计算
- Batch uniformity 无法正确工作（期望 L2-normalized 输入）

**AEF 原文**: 训练时 L2 Norm + VMF 噪声，始终在球面上。

---

### 🔴 CRITICAL: Static Target 占比过高

**位置**: `configs/v2_*.yaml`

**问题**: 
- 7 targets: S2, S1, Landsat, DEM, WorldCover, DynamicWorld, JRC_Water
- Static 目标 (DEM/WC/DW/JRC) = 4/7 = 57%
- AEF 原文 static 目标 ≈ 22%

**影响**: Model 被 bias 向 time-agnostic representations（static 目标不随时间变化）

**修复**: 降低 static source 的 reconstruction weight 或移除部分 static 目标

---

### 🟡 HIGH: 多余的反坍缩损失

**AEF 原文损失**: Reconstruction(1.0) + BatchUniformity(0.05) + Consistency(0.02)

**我们当前的损失** (v12 trainer):
- Reconstruction
- Consistency  
- Classification (auxiliary)
- VICReg Variance
- VICReg Covariance
- Decorrelation (Barlow Twins)
- Orthogonality
- Raw Uniformity / Batch Uniformity
- Temporal Cosine Pixel
- ...等等

**影响**: 损失过多可能导致梯度冲突，模型无法找到正确优化方向。

---

### 🟡 HIGH: Temporal Loss 与 AEF 设计冲突

**AEF 原文**: **没有显式 temporal contrastive loss**
- 时间敏感性来自: (1) Teacher-Student consistency + heavy dropout, (2) Reconstruction forcing time-conditioned decoding

**我们**: 有 temporal_cosine_pixel, pixel_temporal_info_nce, gap_aware_temporal 等显式 temporal loss

**影响**: 显式 temporal loss 可能与 AEF 的隐式设计哲学冲突。

---

### 🟡 MEDIUM: 数据年份混合

**当前**: 2025 = 87.7%, 2024 = 6.2%, 2023 = 6.0%
**Minibatch 100 patches**: 97 patches 有 ZERO 2024 数据

**影响**: 虽然大部分 patch 只有 2025 数据，但仍有 3% patch 有跨 year 数据，可能引入混淆。

---

### 🟡 MEDIUM: Image Size 128

**当前**: 128x128
**AEF 可能**: 256x256 或更大

**影响**: 较小分辨率可能限制空间细节编码。

---

## 3. 可能对齐良好的项

| 项目 | 状态 | 说明 |
|------|------|------|
| Teacher-Student 架构 | ✅ | EMA Teacher + Student view 构建已存在 |
| Student dropout | ⚠️ | Frame drop 0.5, source drop 0.3 — 接近 AEF |
| STP Blocks | ⚠️ | 8 blocks, 3 paths — 结构类似 AEF |
| AdamW optimizer | ✅ | 使用 AdamW + cosine schedule |
| VMF kappa | ⚠️ | 50.0 — 可能需要调整到 2000 |

---

## 4. 根因假设

**Temporal blindness 的最可能原因**（按优先级）：

1. **Decoder 条件注入被禁用** → Reconstruction 不 forcing time-conditioned decoding
2. **Bottleneck skip L2 训练时** → Embedding 不在球面上，consistency 无法正确传播时间信号
3. **Static target 占比 57%** → Model 被 bias 向 time-agnostic
4. **多余损失干扰** → VICReg/Decorr/Orth 等损失与 AEF 机制冲突
5. **缺少 AEF 特有的 Batch Uniformity** → 我们的 batch_uniformity_loss_l2 与 AEF 的 cyclic shift 不同

---

## 5. 修复优先级

| 优先级 | 修复项 | 预计影响 |
|--------|--------|----------|
| P0 | 恢复 Decoder 条件注入 | 🔥 最大 |
| P0 | Bottleneck skip_l2=false | 🔥 最大 |
| P0 | 降低 static 权重至 0.1 | 🔥 大 |
| P1 | 移除 VICReg/Decorr/Orth/CLS | 中 |
| P1 | 严格 2025-only 过滤 | 中 |
| P2 | Image size 256 | 小 |
