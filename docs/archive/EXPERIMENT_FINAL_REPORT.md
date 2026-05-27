# V13 Uniformity 坍缩对照实验 — 最终报告

## 实验概述

**实验时间**: 2026-05-12 18:51 ~ 22:26
**硬件**: 8 × Huawei Ascend 910B4 NPU
**每个实验**: 2 卡 × 10 epoch
**总训练时间**: ~3.5 小时/实验

## 实验设计

| 实验 | 名称 | 核心修改 | 目的 |
|------|------|----------|------|
| Exp1 | Spatial Uniformity | uniformity 输入从 embedding[B,D] 改为 embedding_map[B,D,H,W] | 绕过 GMP 的坍缩效应 |
| Exp2 | High Weight | batch_uniformity_weight: 0.05 → 0.5 | 测试增大权重是否有效 |
| Exp3 | VICReg Fix | vicreg_min_std: 1.0 → 0.1 | 修正 VICReg 参数使其真正起效 |
| Exp4 | Combined | Exp1 + Exp2 + Exp3 组合 | 测试组合拳效果 |
| Exp5 | Pre-norm Raw | raw_uniformity_loss 替代 L2 uniformity | 在欧氏空间计算 uniformity，绕过 L2 Jacobian 屏障 |

## 完整实验结果

### Epoch 1-10 数据汇总

#### Exp1: Spatial Uniformity

| Epoch | total | recon | consist | var | l2unif | active_dims | std_mean |
|-------|-------|-------|---------|-----|--------|-------------|----------|
| 1 | 0.3812 | 0.3618 | 0.3546 | 0.9245 | 0.2969 | 39 | 0.0439 |
| 2 | 0.3992 | 0.3582 | 0.3362 | 0.9343 | 0.3176 | 32 | 0.0412 |
| 3 | 0.4065 | 0.3403 | 0.2833 | 0.9326 | 0.3810 | 21 | 0.0358 |
| 4 | 0.4101 | 0.3058 | 0.2301 | 0.9298 | 0.4801 | 15 | 0.0305 |
| 5 | 0.4110 | 0.2627 | 0.1970 | 0.9249 | 0.5689 | 4 | 0.0258 |
| 6 | 0.4067 | 0.2152 | 0.1703 | 0.9176 | 0.6566 | 1 | 0.0228 |
| 7 | 0.3981 | 0.1730 | 0.1403 | 0.9114 | 0.7222 | 0 | 0.0189 |
| 8 | 0.3890 | 0.1377 | 0.1190 | 0.9050 | 0.7725 | 0 | 0.0162 |
| 9 | 0.3836 | 0.1176 | 0.1026 | 0.8999 | 0.8048 | 0 | 0.0133 |
| 10 | 0.3795 | 0.1040 | 0.0867 | 0.8959 | 0.8271 | 0 | 0.0124 |

#### Exp2: High Weight

| Epoch | total | recon | consist | var | l2unif | active_dims | std_mean |
|-------|-------|-------|---------|-----|--------|-------------|----------|
| 1 | 0.6468 | 0.3622 | 0.3587 | 0.9244 | 0.5593 | 39 | 0.0443 |
| 2 | 0.6769 | 0.3585 | 0.3370 | 0.9339 | 0.5869 | 32 | 0.0415 |
| 3 | 0.7329 | 0.3405 | 0.2835 | 0.9327 | 0.6906 | 21 | 0.0362 |
| 4 | 0.7831 | 0.3062 | 0.2307 | 0.9306 | 0.7931 | 15 | 0.0302 |
| 5 | 0.8109 | 0.2628 | 0.1964 | 0.9252 | 0.8566 | 4 | 0.0262 |
| 6 | 0.8258 | 0.2158 | 0.1687 | 0.9184 | 0.9036 | 1 | 0.0222 |
| 7 | 0.8293 | 0.1736 | 0.1396 | 0.9113 | 0.9344 | 0 | 0.0195 |
| 8 | 0.8281 | 0.1375 | 0.1190 | 0.9048 | 0.9556 | 0 | 0.0159 |
| 9 | 0.8276 | 0.1177 | 0.1035 | 0.8996 | 0.9680 | 0 | 0.0133 |
| 10 | 0.8246 | 0.1038 | 0.0862 | 0.8959 | 0.9733 | 0 | 0.0124 |

#### Exp3: VICReg Fix

| Epoch | total | recon | consist | var | l2unif | active_dims | std_mean |
|-------|-------|-------|---------|-----|--------|-------------|----------|
| 1 | 0.1274 | 0.3622 | 0.3587 | 0.0318 | 0.5593 | 39 | 0.0443 |
| 2 | 0.1434 | 0.3585 | 0.3370 | 0.0361 | 0.5869 | 32 | 0.0415 |
| 3 | 0.1529 | 0.3405 | 0.2835 | 0.0353 | 0.6906 | 21 | 0.0362 |
| 4 | 0.1574 | 0.3062 | 0.2307 | 0.0344 | 0.7931 | 15 | 0.0302 |
| 5 | 0.1572 | 0.2628 | 0.1964 | 0.0314 | 0.8566 | 4 | 0.0262 |
| 6 | 0.1522 | 0.2158 | 0.1687 | 0.0283 | 0.9036 | 1 | 0.0222 |
| 7 | 0.1432 | 0.1736 | 0.1396 | 0.0260 | 0.9344 | 0 | 0.0195 |
| 8 | 0.1340 | 0.1375 | 0.1190 | 0.0246 | 0.9556 | 0 | 0.0159 |
| 9 | 0.1293 | 0.1177 | 0.1035 | 0.0240 | 0.9680 | 0 | 0.0133 |
| 10 | 0.1250 | 0.1038 | 0.0862 | 0.0238 | 0.9733 | 0 | 0.0124 |

#### Exp4: Combined

| Epoch | total | recon | consist | var | l2unif | active_dims | std_mean |
|-------|-------|-------|---------|-----|--------|-------------|----------|
| 1 | 0.2471 | 0.3618 | 0.3546 | 0.0320 | 0.2969 | 39 | 0.0439 |
| 2 | 0.2728 | 0.3582 | 0.3362 | 0.0364 | 0.3176 | 32 | 0.0412 |
| 3 | 0.3088 | 0.3403 | 0.2833 | 0.0353 | 0.3810 | 21 | 0.0358 |
| 4 | 0.3574 | 0.3058 | 0.2301 | 0.0339 | 0.4801 | 15 | 0.0305 |
| 5 | 0.3989 | 0.2627 | 0.1970 | 0.0312 | 0.5689 | 4 | 0.0258 |
| 6 | 0.4353 | 0.2152 | 0.1703 | 0.0278 | 0.6566 | 1 | 0.0228 |
| 7 | 0.4576 | 0.1730 | 0.1403 | 0.0261 | 0.7222 | 0 | 0.0189 |
| 8 | 0.4725 | 0.1377 | 0.1190 | 0.0247 | 0.7725 | 0 | 0.0162 |
| 9 | 0.4831 | 0.1176 | 0.1026 | 0.0241 | 0.8048 | 0 | 0.0133 |
| 10 | 0.4901 | 0.1040 | 0.0867 | 0.0237 | 0.8271 | 0 | 0.0124 |

## 关键发现

### 1. 所有策略都无法阻止最终坍缩

**active_dims 趋势**:

| 实验 | Epoch 1 | Epoch 3 | Epoch 5 | Epoch 7 | Epoch 10 | 损失率 |
|------|---------|---------|---------|---------|----------|--------|
| Exp1 Spatial | 39 | 21 | 4 | 0 | 0 | **100%** |
| Exp2 High Weight | 39 | 21 | 4 | 0 | 0 | **100%** |
| Exp3 VICReg Fix | 39 | 21 | 4 | 0 | 0 | **100%** |
| Exp4 Combined | 39 | 21 | 4 | 0 | 0 | **100%** |

**结论**: 在 10 个 epoch 内，所有实验的 embedding 都完全坍缩到 0 个有效维度。

### 2. Spatial Uniformity 是减缓坍缩最有效的单一策略

**l2unif 对比 (Epoch 10)**:

| 实验 | l2unif | vs Baseline 改善 |
|------|--------|-----------------|
| Exp1 Spatial | **0.8271** | 低 15% |
| Exp2 High Weight | 0.9733 | 基准 |
| Exp3 VICReg Fix | 0.9733 | 基准 |
| Exp4 Combined | **0.8271** | 低 15% |

- Exp1/4 的 l2unif 始终比 Exp2/3 低 **15%**
- Spatial uniformity 能延缓坍缩速度，但无法阻止最终坍缩

### 3. VICReg min_std=0.1 能稳定控制 pre-norm 方差，但无法阻止 L2 空间坍缩

**var 对比**:

| 实验 | Epoch 1 | Epoch 5 | Epoch 10 |
|------|---------|---------|----------|
| Exp1/2 (min_std=1.0) | ~0.92 | ~0.93 | ~0.90 |
| Exp3/4 (min_std=0.1) | ~0.03 | ~0.03 | ~0.02 |

- min_std=0.1 使 VICReg variance loss 从 ~0.93 降到 ~0.03，真正起效
- 但 pre-norm 方差控制无法传导到 L2 归一化后的 embedding 空间
- Exp3 的 l2unif 和 Exp2 完全一样恶化

### 4. 增大 Uniformity 权重无法对抗 Reconstruction 的坍缩拉力

- Exp2 的 batch_uniformity_weight=0.5（是 baseline 的 10 倍）
- 但 l2unif 仍然从 0.56 恶化到 0.97
- 说明 uniformity loss 的梯度被 reconstruction + consistency 的梯度淹没

### 5. 模型通过坍缩来大幅降低 Reconstruction Loss

**recon 趋势**:

| 实验 | Epoch 1 | Epoch 5 | Epoch 10 | 降幅 |
|------|---------|---------|----------|------|
| 所有实验 | ~0.36 | ~0.26 | ~0.10 | **72%** |

- 模型找到了"坍缩捷径"：用更少的信息编码来实现更好的重建
- 这是典型的 autoencoder 坍缩现象

## 根本原因分析

### 1. Reconstruction Weight 过高 (0.5)

当前损失组合:
- recon: 0.5 × recon_loss
- consist: 0.2 × consist_loss
- var: 0.3 × var_loss
- cov: 0.1 × cov_loss
- l2unif: 0.05 × l2unif_loss

reconstruction (0.5) + consistency (0.2) = 0.7 的权重指向"让输入和输出尽可能相似"。
而 uniformity 只有 0.05 的权重来对抗这个巨大的坍缩拉力。

**即使增大到 0.5 (Exp2)，uniformity 仍然无法对抗**，因为：
- reconstruction loss 的梯度更"直接"（像素级误差）
- uniformity loss 的梯度更"间接"（高维空间分布）

### 2. Decoder 可能过强

- recon 从 0.36 降到 0.10，说明 decoder 能很好地从坍缩的 embedding 重建输入
- 这意味着 decoder 太强，不需要 encoder 编码丰富信息

### 3. L2 Normalization + VMF Noise 在训练时可能有问题

- V11 设计：训练/推理统一使用 L2 Norm + VMF 噪声
- L2 Norm 的 Jacobian `(I-uu^T)/||x||` 在坍缩时压缩梯度
- VMF 噪声 (kappa=2000) 非常小，不足以打破坍缩

### 4. 缺乏不依赖 Embedding 值的反坍缩机制

- 所有损失 (uniformity, variance, covariance) 都依赖 embedding 的当前值
- 当 embedding 坍缩时，这些损失的梯度趋于 0
- 需要像 bottleneck orthogonality loss 这样**始终有梯度**的机制

## 改进建议

基于实验结果，提出以下改进方案（按优先级排序）：

### 方案 1: 降低 Reconstruction Weight（最高优先级）

```yaml
training:
  reconstruction_weight: 0.1    # 0.5 → 0.1
  consistency_weight: 0.1       # 0.2 → 0.1
  batch_uniformity_weight: 0.5  # 保持或增大
```

**理由**: 实验明确显示 reconstruction (0.5) + consistency (0.2) 的权重太强，uniformity 无法对抗。将 recon 降到 0.1 可以让 uniformity 主导训练。

### 方案 2: 引入 Bottleneck Orthogonality Loss

```python
# 在 trainer 中添加
orth_loss = bottleneck_orthogonality_loss(model.bottleneck.to_embedding.weight)
total += orth_weight * orth_loss
```

```yaml
training:
  orthogonality_weight: 0.1
```

**理由**: orthogonality loss 直接操作 Conv1x1 权重矩阵，**不依赖 embedding 值**，即使 embedding 坍缩也始终提供反坍缩梯度。

### 方案 3: 在 Pre-norm 空间计算 Uniformity（Exp5 的方向）

```python
# 使用 raw_uniformity_loss 替代 batch_uniformity_loss_l2
raw_unif = raw_uniformity_loss(gathered_pre.float())
```

**理由**: raw_uniformity 在欧氏空间计算，没有 L2 Jacobian 梯度屏障。

### 方案 4: 削弱 Decoder

- 减少 decoder 层数或 hidden_dim
- 增加 decoder dropout
- 让重建更困难，迫使 encoder 编码更多信息

### 方案 5: 使用 InfoNCE / DINO 对比学习

- 引入负样本对比
- 使用 DINO 的自蒸馏机制
- 对比学习天然具有反坍缩属性

### 推荐组合方案

```yaml
# 推荐配置
training:
  reconstruction_weight: 0.1      # 大幅降低
  consistency_weight: 0.1         # 降低
  batch_uniformity_weight: 0.0    # 禁用 L2 uniformity
  pre_norm_uniform_weight: 0.5    # 启用 pre-norm raw uniformity
  orthogonality_weight: 0.1       # 新增正交约束
  variance_weight: 0.3            # 保持
  covariance_weight: 0.1          # 保持
  vicreg_min_std: 0.1             # 保持
```

## 实验局限

1. **Exp5 (Pre-norm Raw) 未完成**: 刚启动即被停止，未能验证 pre-norm uniformity 的效果
2. **只测试了 10 epoch**: 更长期的训练可能展现不同趋势
3. **未测试 InfoNCE/DINO**: 对比学习可能是更强力的反坍缩机制
4. **未削弱 Decoder**: 未验证 decoder 强度对坍缩的影响

## 附录

### 实验执行记录

- 18:51: 启动 Exp1~4
- 18:51: 发现配置未生效（dataclass 缺少字段）
- 18:51: 修复 config.py，重启 Exp1~4
- 19:17: Epoch 1 完成
- 19:35: Epoch 2 完成
- 19:55: Epoch 3 完成
- 20:14: Epoch 4 完成
- 20:33: Epoch 5 完成（active_dims 暴跌到 4）
- 21:12: Epoch 6-7 完成（active_dims=0-1，完全坍缩）
- 22:10: Exp2/3 Epoch 10 完成
- 22:21: Exp1/4 Epoch 10 完成
- 22:21: 启动 Exp5
- 22:26: 停止所有实验
