# Round 8 全面分析与诊断报告

## 一、评估结果汇总

### 1.1 变化检测 (Bare AUC)
| 实验 | AUC Mean | AUC Max | dist_gap |
|------|----------|---------|----------|
| exp8 | 0.5146 | 0.6260 | +0.0021 |
| exp3 | 0.5060 | 0.5758 | +0.0012 |
| exp7 | 0.5057 | 0.5511 | +0.0006 |
| exp4 | 0.5008 | 0.5479 | +0.0001 |
| exp2 | 0.4982 | 0.5559 | -0.0004 |
| exp6 | 0.4978 | 0.5430 | -0.0002 |
| exp5 | 0.4962 | 0.5833 | -0.0004 |
| exp1 | 0.4934 | 0.5305 | -0.0007 |

**结论**: 全部接近随机水平 (0.5)，模型无法区分变化区域与未变化区域。

### 1.2 变化检测 (CD Head AUC, 5-fold CV)
| 实验 | Mean AUC | 最佳 Fold |
|------|----------|-----------|
| exp3 | 0.5848 | 0.6526 |
| exp6 | 0.5749 | 0.6487 |
| exp8 | 0.5698 | 0.6251 |
| exp1 | 0.5652 | 0.6250 |
| exp5 | 0.5585 | 0.6136 |
| exp2 | 0.5504 | 0.5804 |
| exp4 | 0.5493 | 0.5204 |
| exp7 | 0.5482 | 0.5861 |

**结论**: CD Head 比 Bare AUC 提升约 +0.05，但仍远低于及格线 (0.6)。

### 1.3 语义分割与地物提取 (Linear Probe, 评估中)
- WorldCover 语义分割
- Dynamic World 语义分割
- JRC Water 二值分割
- OSM Buildings 二值分割

---

## 二、训练指标分析

### 2.1 最终 Epoch 20 指标

| 指标 | 范围 | 评价 |
|------|------|------|
| **RawUnif** | -5.80 ~ -5.85 | ✅ 优秀，无坍缩 |
| **PreUnif** | -2.38 ~ -2.98 | ✅ 良好 |
| **Recon** | 0.128 ~ 0.143 | ✅ 收敛良好 (-71~75%) |
| **Temporal** | 4.50 ~ 5.19 | ⚠️ 从8.0下降45%，但绝对值仍高 |
| **VICReg** | 6.95 ~ 19.66 | ⚠️ cov项极大 (200-500)，过度去相关 |
| **KoLeo** | 4.76 ~ 5.01 | ✅ 稳定 |
| **Consist** | 0.04 ~ 0.16 | ⚠️ 过低，teacher-student不一致 |

### 2.2 关键发现

#### 🔴 发现 1: Temporal Loss 优化方向错误

`temporal_cosine_pixel_loss` 的公式:
```
loss = cos_map.mean() / temperature
```
其中 `temperature=0.1`。

- Epoch 1: loss=8.0 → cos_sim ≈ 0.80
- Epoch 20: loss=4.7 → cos_sim ≈ 0.47

**问题**: loss 优化的是"让所有像素的 cos_sim → 0"（正交），但这是一个**全局目标**，不区分变化区域和未变化区域。

模型学到的策略可能是:
1. 让两个窗口的embedding整体不同
2. 但这种差异均匀分布在整个空间上
3. 变化区域和未变化区域的差异程度几乎相同 (gap < 0.002)

#### 🔴 发现 2: VICReg Cov 项过大

VICReg 的 cov 项从 0.1-0.3 增长到 200-500，说明协方差去相关化极其激进。

**后果**:
- 模型被迫让embedding的不同维度之间完全不相关
- 时间变化信号可能被"打散"到所有维度
- 空间结构被破坏，无法集中表达变化

#### 🔴 发现 3: Consistency Loss 过低

Teacher-Student 一致性损失仅 0.04-0.16，说明:
- EMA teacher 和 student 的 embedding 差异很大
- Teacher 不稳定，不能提供可靠的监督信号
- 或者 student 的 dropout/frame_drop 过于激进

#### 🔴 发现 4: 训练严重不足

| 参数 | Round 8 | 典型需求 |
|------|---------|----------|
| Epochs | 20 | 100-200+ |
| Steps/epoch | 50 | 200-500 |
| 总步数 | 1,000 | 20,000-100,000 |
| Batch size | 2 | 4-8 |

**计算**: 1,000步 × batch_size 2 = 2,000个样本更新。而数据集有 11,915 个月度样本，模型只看到了约 17% 的数据。

#### 🔴 发现 5: 重建目标缩减导致语义信息丢失

V12/V13 配置将重建目标从 7 类缩减为 4 类（S2+S1+Landsat+DEM），移除了:
- WorldCover (11类语义)
- Dynamic World (9类语义)
- JRC Water (水体)

**后果**: 模型没有通过重建学习任何语义类别信息。语义分割的下游任务需要从零学习，难度大增。

---

## 三、根本原因总结

### 为什么 AUC ≈ 0.5？

```
1. 训练量不足 (20 epoch, 50 steps/epoch)
   → 模型没有充分学习时间变化模式

2. Temporal Loss 是全局目标
   → 均匀推开所有像素，不区分变化/未变化

3. VICReg Cov 过度去相关
   → 打散了时间变化的维度结构

4. 缺乏语义重建监督
   → 模型不知道什么是建筑、水体、植被

5. Teacher-Student 不一致
   → Consistency Loss 失效，稳定性差

6. 验证窗口与训练窗口间隔差异
   → 训练用 6个月间隔，验证用 12个月间隔
   → 模型对 12个月间隔的泛化能力弱
```

---

## 四、下一步改进方案

### 方案 A: 延长训练 + 增加数据 (高优先级)

```yaml
training:
  epochs: 200              # 从 20 → 200
  max_steps_per_epoch: 200  # 从 50 → 200
  batch_size: 4            # 从 2 → 4
  # 总步数: 40,000 (vs 当前 1,000)
```

### 方案 B: 修复 Temporal Loss (高优先级)

**当前问题**: `temporal_cosine_pixel_loss` 是全局目标，不区分空间位置。

**改进方向**:
1. **引入 Masked Temporal Loss**: 只在变化概率高的区域施加 temporal loss
2. **使用 Local-Global 结合**: 像素级 + 全局 mean 结合
3. **引入 Contrastive Hard Negative Mining**: 让变化区域和未变化区域形成对比

```python
# 伪代码: Masked Temporal Loss
def masked_temporal_loss(emb_w1, emb_w2, change_prob):
    cos_map = cosine_similarity(emb_w1, emb_w2)  # [B, H, W]
    # 在变化概率高的区域施加更强的推开力
    weights = change_prob + 0.1  # 避免0权重
    loss = (cos_map * weights).mean() / temperature
    return loss
```

### 方案 C: 恢复语义重建目标 (高优先级)

```yaml
data:
  num_target_sources: 7  # 恢复 WorldCover + Dynamic World + JRC Water
```

**理由**: 模型需要通过重建学习语义概念，才能在下游任务中表现好。

### 方案 D: 调整 VICReg 权重 (中优先级)

```yaml
training:
  variance_weight: 0.25    # 保持
  covariance_weight: 0.01  # 从 0.04 → 0.01 (降低cov去相关化)
```

**理由**: cov项过大导致embedding结构被破坏。

### 方案 E: 引入空间感知的时间变化 (中优先级)

**当前**: `temporal_cosine_pixel_loss` 对每个像素独立计算 cos_sim。

**改进**: 引入空间上下文，让相邻像素的变化模式一致。

```python
# 在 temporal loss 中加入空间平滑约束
spatial_smooth = (cos_map[:, 1:, :] - cos_map[:, :-1, :]).abs().mean()
temporal_loss = cos_map.mean() / temperature + 0.1 * spatial_smooth
```

### 方案 F: 验证窗口对齐 (低优先级)

**当前**: 训练窗口间隔 6个月，验证窗口间隔 12个月。

**改进**: 
1. 训练时也采样 12个月间隔的窗口对
2. 或验证时只比较 6个月间隔的变化

---

## 五、推荐实验配置

基于以上分析，推荐以下配置进行 Round 9:

```yaml
# Round 9 推荐配置
experiment:
  name: round9_expA
  epochs: 100
  
data:
  batch_size: 4
  num_target_sources: 7  # 恢复语义重建
  
training:
  max_steps_per_epoch: 200
  reconstruction_weight: 0.1
  temporal_cosine_pixel_weight: 1.0   # 从 0.5 → 1.0
  pre_norm_uniform_weight: 1.0        # 保持 exp8 最佳值
  consistency_weight: 0.1             # 从 0.05 → 0.1
  covariance_weight: 0.01             # 从 0.04 → 0.01
  variance_weight: 0.25
  vicreg_min_std: 1.0
  
model:
  skip_l2_norm_training: true
  vmf_kappa: 50
```

---

## 六、结论

Round 8 的实验暴露了几个关键问题:

1. **训练量严重不足**: 20 epoch × 50 steps = 1,000步，远远不够
2. **Temporal Loss 设计缺陷**: 全局目标无法区分变化区域
3. **语义信息缺失**: V12 移除语义重建目标，下游任务困难
4. **VICReg 过度去相关**: cov项过大，破坏embedding结构

**短期目标** (Round 9):
- 延长训练至 100 epoch × 200 steps
- 恢复语义重建目标
- 增强 temporal loss 权重 + 降低 cov 权重

**预期效果**:
- Bare AUC: 0.50 → 0.60+
- CD Head AUC: 0.58 → 0.70+
- WorldCover mIoU: 0.10 → 0.25+

---

*报告生成时间: 2026-05-16*
*下游任务评估仍在进行中，结果将补充到本报告*
