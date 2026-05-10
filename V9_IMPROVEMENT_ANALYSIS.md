# V9 Temporal Backbone 深度诊断与改进计划

> 分析日期: 2026-05-10 | V9 当前状态: Epoch 62/200, best recon=0.292 (E58)
> 基准对比: V8 Best (E223, recon=0.210, bare AUC=0.51)

---

## 一、5个根因的实际验证

### 根因1: "Temporal Loss 粒度太粗 — 只在 global mean embedding 上算时序 loss"

**结论: ❌ 不成立。** V9 已经实现了 pixel-level temporal loss。

**证据:**
```python
# src/training/losses.py: gap_aware_temporal_cosine_loss
f1 = F.normalize(pre_w1, p=2, dim=1)    # [B, D, H, W]
f2 = F.normalize(pre_w2, p=2, dim=1)
cos_map = (f1 * f2).sum(dim=1)          # [B, H, W] ← 逐像素
target_map = target.view(B, 1, 1).expand(B, H, W)
loss = ((cos_map - target_map).pow(2)).mean() / temperature
```

Loss 在空间维度 `[H, W]` 上逐像素计算 MSE，不是 global mean。且 `pre_w1/pre_w2` 是 **pre-norm map**（L2 前的空间 embedding），保留了完整的幅度信息。

### 根因2: "Uniformity Loss 失效 — t=2/D=0.031 过小，梯度消失"

**结论: ⚠️ 部分成立，但不是当前瓶颈。**

**证据:**
- 当前 `raw_uniformity_loss` 使用 `t = 2.0 / D = 0.031`
- 但 V9 E58 的 `uniform = -3.36`，在健康范围 (-4.0 ~ -1.0)
- 对比 V8 Best: `uniform = -2.08`，V9 反而**更好**（更负 = 更分散）
- 这得益于 soft restart 重新初始化了 bottleneck，打破了 V8 的局部最优

**判断:** t=0.031 确实可能让 gradient 在远距离 pair 上过于平缓，但 uniformity 指标本身健康。这不是 AUC 低的直接原因。

### 根因3: "Reconstruction 主导 — 重建 loss 迫使 embedding 编码静态内容"

**结论: ✅ 成立，这是最大瓶颈。**

**证据:**

| Loss | Weight | E58 实际值 | 对 total loss 贡献 |
|------|--------|-----------|-------------------|
| recon | 1.0 | 0.292 | **0.292** |
| uniform | 1.0 | -3.36 | **-3.36** (负值=奖励) |
| temporal | 0.02 | 2.45 | **0.049** |
| consist | 0.05 | 0.106 | **0.005** |
| cls | 0.08 | 1.11 | **0.089** |
| var | 0.25 | 0.306 | **0.077** |
| decorr | 0.05 | 5.60 | **0.280** |

**关键洞察:**
- recon 对参数的拉力 = 0.292（正梯度，让 embedding 编码可重建信息）
- temporal 对参数的拉力 = 0.049（正梯度，让 embedding 随时间变化）
- **recon 的拉力是 temporal 的 6 倍**
- 当 recon 要求"编码静态内容以便重建"时，temporal 的微弱信号被淹没

更深层原因：decoder 需要从 embedding 中重建 DEM/WorldCover（完全静态的目标）。如果 embedding 被 temporal loss 搞得"时间敏感"了，decoder 重建静态目标会更困难。这是一个**结构性矛盾**。

### 根因4: "数据变化样本稀缺 — 424 个 patch 中变化像素 <1%，且时间 gap 太小"

**结论: ⚠️ 部分成立。**

**实际数据分析:**

| 窗口模式 | 平均 center gap | target_cos | 有效采样率 |
|---------|----------------|-----------|-----------|
| random_split (V9 默认) | **167 天** (5.5月) | 0.084 | 100% |
| non_overlap | **252 天** (8.3月) | -0.379 | **51.2%** |
| mixed_scale_long | **240 天** (7.9月) | -0.317 | **49.3%** |

**解读:**
- random_split 的 gap 实际上已经很大（167 天），target_cos=0.084 要求 cosine similarity 接近 0
- 但 non_overlap/mixed_scale_long 的有效采样率只有 ~50%，大量 patch 无法满足 ≥6 个月的 gap
- S2 平均每 patch **22 帧**，100% ≥ 8 帧，帧数不是问题
- **真正的问题不是 gap 太小，而是 gap 的分布不够多样化**：总是 5.5 个月的单一尺度

### 根因5: "缺少 Teacher-Student"

**结论: ❌ 不成立。** V9 已有完整的 Teacher-Student 体系。

**证据:**
```python
# ddp_v9_temporal_trainer.py
self.teacher = copy.deepcopy(self.model.module)   # EMA Teacher
self.teacher_momentum = 0.996

# Student view 构建
def _build_student_view(...):
    # 随机丢源 (25%)、丢帧 (40%)、截断前后段 (15%)
    ...

# Consistency loss
consist = consistency_loss(teacher_out.embedding, student_out.embedding)
```

---

## 二、14个解决方案的可行性评估

### 🔴 Layer 1: 训练目标重构

#### ① Batch Uniformity Objective (AEF 核心机制)

**当前状态:** 已有 `raw_uniformity_loss` + cross-GPU gather，实现了 batch-level uniformity。

**AEF 原论文的 BatchUniformity = Σ|ui · ui'| 与当前实现的区别:**
- AEF: 循环移位配对，每个 embedding 只有一个配对对象
- 当前: 所有 pair 的 RBF uniformity，更密集

**评估:** 边际收益。**当前 uniform=-3.36 已经很好**，改进空间不大。

#### ② Pixel-Level Temporal Contrastive Loss

**当前状态:** ✅ **已经实现**。`gap_aware_temporal_cosine_loss` 在 `[B, H, W]` 上逐像素计算。

**与建议的区别:** 建议中提到 "1个月→0.9，12个月→0.2" 的线性映射，当前实现是 `target = 1 - 2*gap_norm`（即 12 月→-1，6月→0）。当前的映射更激进。

**评估:** 不是缺失项，是已有但 weight 太轻。

#### ③ 显式变化监督（图像差异作为弱监督）

**当前状态:** ❌ **完全缺失**。

**公式:**
```python
emb_diff = 1 - cosine_similarity(emb_t1, emb_t2)    # [B, H, W]
img_diff = abs(image_t1 - image_t2)                  # [B, C, H, W]
loss = MSE(emb_diff, img_diff.mean(dim=1))           # 对齐到 [B, H, W]
```

**核心价值:** 不需要标注！直接利用输入图像的像素级变化来监督 embedding 的变化模式。
- 建筑物新建区域：图像变化大 → embedding 差异也应大
- 未变化区域：图像变化小 → embedding 差异也应小
- 这给了模型**空间位置级别的变化敏感度**

**评估:** ⭐⭐⭐⭐⭐ **最高优先级**。不增加标注成本，直接解决"变化信号在哪里"的问题。

### 🟡 Layer 2: 防坍缩机制

#### ④ 固定 t=2.0 替代自适应 t=2/D

**当前:** `t = 2.0 / 64 = 0.031`

**问题:** 对 64 维标准化特征，E[||xi-xj||²] ≈ 2D = 128，则 `t * d² = 0.031 * 128 ≈ 4`，exp(-4) = 0.018。确实在远距离 pair 上 gradient 很小。

**评估:** ⭐⭐⭐ 可以尝试，但 uniform 已经健康，预计收益有限。

#### ⑤ CLOA 正交锚点

**当前状态:** ❌ 未实现。

**需要:** 用 WorldCover 类别（11类）作为伪标签，构建正交锚点矩阵。

**评估:** ⭐⭐ 增加语义约束，但实现复杂度高，且分类 loss (cls=0.08) 已经在做类似的事。

#### ⑥ VICReg 方差-协方差约束

**当前状态:** ✅ **已经实现**。`decorrelation_loss` (Barlow Twins) + `variance_regularizer` (VICReg)。

**评估:** 无需改动。

### 🟢 Layer 3: 数据增强

#### ⑦ 合成变化数据（Changen2 思路简化版）

**当前状态:** ❌ 未实现。

**方案:** 在训练时，对输入图像做空间扰动（擦除、平移、替换）生成"伪变化"，要求 embedding 在变化区域产生差异。

**评估:** ⭐⭐⭐⭐ 高价值但工作量大。需要：
1. 实现图像扰动模块
2. 生成变化 mask
3. 修改 temporal loss 支持 per-pixel target

#### ⑧ 强制大时间 Gap 采样

**当前状态:** ⚠️ 部分实现。dataset.py 支持 `mixed_scale` 模式，但 V9 配置未启用。

**数据分析:**
- random_split (默认): gap=167天, 100%有效
- mixed_scale_long: gap=240天, ~50%有效
- 建议: 30%短gap + 40%中gap + 30%长gap

**评估:** ⭐⭐⭐⭐⭐ **容易实现，立即见效**。只需修改 config：
```yaml
data:
  window_mode: "mixed_scale"
  mixed_scale_long_prob: 0.5
  mixed_scale_short_prob: 0.5
```

### 🔵 Layer 4: 架构改进

#### ⑨ Teacher-Student Consistency

**当前状态:** ✅ **已经实现**。

**评估:** 无需改动。

#### ⑩ Difference Module（M-CD 思路）

**当前状态:** ❌ 未实现。

**方案:** 在 STP Encoder 后加显式差分模块：
```python
diff = DWConv(pre_features) - DWConv(post_features)
```

**评估:** ⭐⭐⭐⭐ 架构级改进，但需要修改模型结构、重新训练。如果 V9 收敛后 AUC 仍不达标，这是 V10 的核心方向。

### ⚪ Layer 5: 训练策略

#### ⑪ 课程学习

**当前状态:** ⚠️ 已有部分。30 epoch temporal warmup。

**建议扩展:**
- Phase 1 (E0-30): recon 主导，temporal=0
- Phase 2 (E30-80): temporal 逐步升温，weight 0.02→0.05
- Phase 3 (E80+): temporal 主导，weight 0.05→0.10

**评估:** ⭐⭐⭐⭐ 容易实现，可以配合动态 weight 调度。

#### ⑫ 硬负样本挖掘

**当前状态:** ❌ 未实现。

**方案:** 在 temporal loss 中，只保留"困难"像素（cos_sim 接近 target 的像素）。

**评估:** ⭐⭐ 锦上添花，非核心。

---

## 三、V9 训练状态深度诊断

### 3.1 训练曲线解读

| Epoch | Recon | Uniform | Temporal | 状态 |
|-------|-------|---------|----------|------|
| E001 | 0.998 | -3.32 | 0 | Soft restart, decoder 重置 |
| E010 | 0.595 | -3.32 | 0 | 快速下降期 |
| E020 | 0.373 | -3.37 | 0 | 接近 V8 中期水平 |
| E030 | 0.318 | -3.38 | 0 | Warmup 结束 |
| E031 | 0.309 | -3.34 | **35.6** | ⚠️ Temporal 启动冲击 |
| E034 | 0.403 | -3.22 | 9.78 | ⚠️ Recon 反弹 (temporal 干扰) |
| E040 | 0.346 | -3.30 | 4.41 | 恢复期 |
| E050 | 0.314 | -3.27 | 3.42 | 稳定期 |
| E058 | 0.292 | -3.36 | 2.45 | **Best recon** |
| E062 | 0.300 | -3.23 | 4.01 | 轻微反弹 |

### 3.2 关键发现

**发现1: Temporal loss 成功启动，但 recon 下降速度明显放缓**

| 阶段 | Epoch 范围 | Recon 下降 | 速度 |
|------|-----------|-----------|------|
| Warmup | E1-E30 | 0.998→0.318 | 0.023/epoch |
| Post-temporal | E31-E62 | 0.318→0.300 | **0.0006/epoch** |

E31 后 recon 几乎停滞！这是因为 temporal loss 和 recon loss 在竞争：
- recon 要求 embedding **稳定**（以便 decoder 重建）
- temporal 要求 embedding **变化**（随时间不同）
- 当两者同时优化时，模型找到了一个"折中"：让 embedding 有微弱的时间变化，但不足以显著影响重建

**发现2: Consistency loss 反常升高**

- V8 Best: consist = 0.049
- V9 E58: consist = 0.106

Student 视图（随机丢帧 40% + 丢源 25%）产生的 embedding 与 Teacher（完整输入）的差异比 V8 更大。这说明 **V9 的 embedding 对输入帧的缺失更敏感了**——这是时间敏感性的一个间接信号！但差异方向不一定是"正确"的。

**发现3: Decorrelation 大幅改善**

- V8: decorr = 12.91（过高 = 维度间强相关）
- V9: decorr = 5.60（改善）

Soft restart 打破了 V8 的维度相关性，让 embedding 各维度更加独立。

### 3.3 瓶颈定位

```
┌─────────────────────────────────────────────────────────────┐
│                    V9 瓶颈分析                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Temporal Loss (weight=0.02)                                │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────────────┐    期望: embedding 随时间显著变化       │
│  │  Gap=167天      │         before/after cos_sim → 0.08   │
│  │  target_cos=0.08│                                       │
│  └─────────────────┘                                       │
│     │                                                       │
│     ▼ 但实际...                                             │
│  ┌─────────────────┐                                       │
│  │  Recon Loss     │    拉力: 保持 embedding 稳定           │
│  │  (weight=1.0)   │         以便 decoder 重建静态目标      │
│  └─────────────────┘                                       │
│     │                                                       │
│     ▼                                                       │
│  结果: embedding 只有微弱的 temporal 变化                   │
│        (temporal loss 从 35→3，但 recon 停滞在 0.29)        │
│                                                             │
│  根本原因: 缺少"变化应该发生在哪里"的像素级监督              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、改进计划

### 阶段 0: 继续 V9 训练（当前，不中断）

**行动:** 让 V9 继续跑到 E200。

**理由:**
- E58 recon=0.292，还有下降空间（目标 0.21-0.25）
- 每 20 epoch 保存 checkpoint，随时可以取 best
- 训练健康，没有 NaN/崩溃

**退出条件:**
- recon 连续 10 epoch 不下降 → 提前停止
- 或达到 E200

### 阶段 1: V9.5 快速迭代（基于 V9 checkpoint，1-2 天）

**目标:** 在不改变模型架构的前提下，最大化 temporal 信号。

**改动1: 启用 Mixed-Scale 数据采样**
```yaml
# configs/xuannv_v9_temporal.yaml
data:
  window_mode: "mixed_scale"
  mixed_scale_long_prob: 0.5
  mixed_scale_short_prob: 0.5
  mixed_scale_short_max_gap_ms: 3 * 30 * 24 * 3600 * 1000   # 3个月
  mixed_scale_long_min_gap_ms: 6 * 30 * 24 * 3600 * 1000    # 6个月
```

**预期效果:**
- 50% 的 batch 使用长 gap (≥6月, target_cos ≤ 0)
- 50% 的 batch 使用短 gap (≤3月, target_cos ≥ 0.5)
- 让模型学习"尺度感"：小变化 vs 大变化

**改动2: 动态 Temporal Weight 调度**
```python
# ddp_v9_temporal_trainer.py
# 替代固定 weight=0.02，使用渐进调度
if epoch < 30:
    temporal_w = 0.0
elif epoch < 60:
    temporal_w = 0.02 * min(1.0, (epoch - 30) / 15)
elif epoch < 100:
    temporal_w = 0.02 + 0.03 * min(1.0, (epoch - 60) / 20)  # → 0.05
else:
    temporal_w = 0.05 + 0.03 * min(1.0, (epoch - 100) / 50)  # → 0.08
```

**改动3: 引入 Pixel-Level 图像差异监督（轻量版）**

在现有 `gap_aware_temporal_cosine_loss` 旁边增加：
```python
def pixel_change_supervision_loss(pre_w1, pre_w2, image_t1, image_t2, mask_threshold=0.1):
    """
    利用输入图像差异指导 embedding 差异。
    image_t1/image_t2: [B, C, H, W] 两期 S2 图像（取第一个源的平均）
    """
    # 图像差异
    img_diff = torch.abs(image_t1 - image_t2).mean(dim=1)  # [B, H, W]
    img_diff_norm = (img_diff - img_diff.min()) / (img_diff.max() - img_diff.min() + 1e-6)
    
    # Embedding 差异
    f1 = F.normalize(pre_w1, p=2, dim=1)
    f2 = F.normalize(pre_w2, p=2, dim=1)
    emb_diff = 1.0 - (f1 * f2).sum(dim=1)  # [B, H, W], range [0, 2]
    emb_diff_norm = emb_diff / 2.0
    
    # 加权 MSE：变化大的区域权重更高
    weights = 1.0 + img_diff_norm  # 变化区域权重 2x
    loss = (weights * (emb_diff_norm - img_diff_norm).pow(2)).mean()
    return loss
```

**weight = 0.05**，在 E40 后启用（避免早期干扰 decoder 学习）。

### 阶段 2: V10 架构改进（基于 V9.5 best，3-5 天）

**如果 V9.5 AUC 仍 < 0.65，启动 V10。**

**核心改动: Difference Module + 显式变化分支**

在 STP Blocks 后增加一个轻量的差分编码分支：
```python
class TemporalDifferenceModule(nn.Module):
    """显式编码时间变化。"""
    def __init__(self, channels):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
        )
        self.change_gate = nn.Conv2d(channels // 2, 1, 1)  # 变化注意力图
        
    def forward(self, feat_w1, feat_w2):
        # feat: [B, C, H, W]
        diff = torch.cat([feat_w1, feat_w2], dim=1)
        diff_feat = self.diff_conv(diff)  # [B, C/2, H, W]
        change_score = torch.sigmoid(self.change_gate(diff_feat))  # [B, 1, H, W]
        return diff_feat, change_score
```

**训练目标:**
1. 正常重建（保持 V9 设置）
2. Temporal loss（gap-aware cosine）
3. **新增: Change Consistency Loss**
   - 要求 `change_score` 与图像差异正相关
   - 这强制模型在变化区域产生更大的 embedding 差异

### 阶段 3: 验证与 CD Head 重训练

每个阶段收敛后：
1. 运行 `precompute_embeddings_v8.py` 重新提取 embedding
2. 运行 `verify_sparse_change_hypothesis.py` 验证假设
3. 运行 `train_cd_head_v8_v3.py` 训练 CD Head
4. 运行 `validate_v8_clean.py` 计算 AUC

**AUC 目标:**
- V8 基线: 0.51
- V9.5 目标: 0.60-0.65
- V10 目标: 0.70+

---

## 五、立即行动项

### 今天可以做（不中断 V9 训练）

1. **修改 config 启用 mixed_scale 采样**（下次训练自动生效）
2. **编写 pixel_change_supervision_loss** 函数
3. **准备 V9.5 训练脚本**（从 V9 best soft restart）

### 等 V9 达到 E100 或 recon 停滞后

1. 用 V9 best checkpoint 做 AUC 快速验证
2. 如果 AUC < 0.55 → 立即启动 V9.5
3. 如果 AUC ≥ 0.55 → 继续 V9 到 E200

---

## 六、总结

| 根因/方案 | 在 V9 中是否成立 | 优先级 | 实施阶段 |
|-----------|----------------|--------|---------|
| 1. Temporal 粒度太粗 | ❌ 已有 pixel-level | - | - |
| 2. Uniformity 失效 | ⚠️ t=0.031 偏小，但指标健康 | 低 | 可选 |
| 3. Reconstruction 主导 | ✅ 核心瓶颈 | **高** | V9.5 动态 weight |
| 4. 变化样本稀缺 | ⚠️ Gap 够大但分布单一 | **高** | V9.5 mixed_scale |
| 5. 缺少 Teacher-Student | ❌ 已有 | - | - |
| ③ 显式变化监督 | ❌ 未实现 | **最高** | V9.5 |
| ⑧ 强制大 Gap 采样 | ⚠️ 部分实现 | **高** | V9.5 |
| ⑩ Difference Module | ❌ 未实现 | 高 | V10 |
| ⑪ 课程学习 | ⚠️ 已有 warmup | 中 | V9.5 |

**核心结论:** V9 的训练框架已经相当完善（pixel-level temporal、teacher-student、VICReg 都有），但缺少**像素级的变化监督信号**和**多样化的 gap 采样**。这两点是 V9.5 的改进重点，预计可以在不修改模型架构的情况下显著提升 AUC。
