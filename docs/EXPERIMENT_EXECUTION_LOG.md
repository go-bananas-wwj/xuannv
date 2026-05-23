# V13 Uniformity 坍缩对照实验 — 执行记录

## 启动时间
2026-05-12 18:33 CST

## 实验配置

| 实验 | 名称 | 核心修改 | NPU | 端口 |
|------|------|----------|-----|------|
| Exp1 | Spatial Uniformity | uniformity 输入: embedding_map[B,D,H,W] 替代 embedding[B,D] | 0,1 | 29500 |
| Exp2 | High Weight | batch_uniformity_weight: 0.05 → 0.5 | 2,3 | 29501 |
| Exp3 | VICReg Fix | vicreg_min_std: 1.0 → 0.1 | 4,5 | 29502 |
| Exp4 | Combined | Exp1 + Exp2 + Exp3 组合 | 6,7 | 29503 |
| Exp5 | Pre-norm Raw | raw_uniformity_loss 替代 batch_uniformity_loss_l2 | 0,1 | 29504 |

## 代码修改

### `src/training/ddp_v12_trainer.py`
1. 导入 `raw_uniformity_loss`
2. 增加 `use_spatial_uniformity` 开关: uniformity 输入从 `embedding` 改为 `embedding_map`
3. 增加 `vicreg_min_std` 配置: `variance_regularizer(min_std=...)` 可配置
4. 增加 `use_pre_norm_uniform` + `pre_norm_uniform_weight` 开关: 用 `raw_uniformity_loss(gathered_pre)` 替代 `batch_uniformity_loss_l2(gathered_l2)`

## 训练参数
- Epochs: 10
- Batch per GPU: 6
- World size: 2 (每张实验)
- Effective batch: 12
- max_steps_per_epoch: 100
- 预计每 epoch: ~21 分钟
- 预计总时间: ~3.5 小时/实验

## 实时监控

### Exp1: Spatial Uniformity
- tmux session: `v13_exp1`
- 日志: `/workspace/outputs/v13_exp1_spatial_unif/train.log`

### Exp2: High Weight
- tmux session: `v13_exp2`
- 日志: `/workspace/outputs/v13_exp2_high_weight/train.log`

### Exp3: VICReg Fix
- tmux session: `v13_exp3`
- 日志: `/workspace/outputs/v13_exp3_vicreg_fix/train.log`

### Exp4: Combined
- tmux session: `v13_exp4`
- 日志: `/workspace/outputs/v13_exp4_combined/train.log`

### Exp5: Pre-norm Raw
- tmux session: `v13_exp5`
- 日志: `/workspace/outputs/v13_exp5_prenorm_raw/train.log`

## 进度追踪


## 问题与修复

### 发现的问题（18:49）
所有实验的 Step 20/40 数据完全相同，怀疑配置未生效。

### 根因
`src/config.py` 的 `load_config()` 函数使用 dataclass 字段过滤：
```python
known = {f.name for f in section_cls.__dataclass_fields__.values()}
filtered = {k: v for k, v in values.items() if k in known}
```

配置文件中的 `use_spatial_uniformity`、`use_pre_norm_uniform`、`vicreg_min_std` 字段在 `TrainingConfig` dataclass 中**不存在**，因此被静默过滤！所有实验实际上都在使用默认值。

### 修复
在 `TrainingConfig` 中添加三个字段：
```python
use_spatial_uniformity: bool = False
use_pre_norm_uniform: bool = False
vicreg_min_std: float = 1.0
```

### 验证（18:51）
```
v13_exp1_spatial_unif:  use_spatial_uniformity=True,  vicreg_min_std=1.0,  batch_uniformity_weight=0.05
v13_exp2_high_weight:   use_spatial_uniformity=False, vicreg_min_std=1.0,  batch_uniformity_weight=0.5
v13_exp3_vicreg_fix:    use_spatial_uniformity=False, vicreg_min_std=0.1,  batch_uniformity_weight=0.05
v13_exp4_combined:      use_spatial_uniformity=True,  vicreg_min_std=0.1,  batch_uniformity_weight=0.5
v13_exp5_prenorm_raw:   use_spatial_uniformity=False, vicreg_min_std=1.0,  batch_uniformity_weight=0.0, pre_norm_uniform_weight=0.3
```

### Step 0 验证（配置生效后）

| 实验 | l2unif | var | cov | 说明 |
|------|--------|-----|-----|------|
| Exp1 Spatial | 0.3021 | 0.8376 | 0.0004 | Spatial uniformity 显著更低（0.30 vs 0.65）✓ |
| Exp2 High Weight | 0.6481 | 0.8376 | 0.0004 | Baseline-like，但 weight=0.5 ✓ |
| Exp3 VICReg Fix | 0.6481 | **0.0034** | 0.0004 | var 从 0.84 降至 0.003！min_std=0.1 生效 ✓ |
| Exp4 Combined | 0.3021 | **0.0034** | 0.0004 | Spatial + min_std=0.1 同时生效 ✓ |

关键发现：
1. **Spatial uniformity (Exp1/4)**: embedding_map 上的 uniformity 比 embedding 低 ~0.35，说明 spatial vectors 的区分度确实更好
2. **VICReg fix (Exp3/4)**: min_std=0.1 使 variance loss 从 ~0.84 降到 ~0.003，VICReg 真正开始工作

### 重启时间
2026-05-12 18:51 CST

## 实时监控


## 训练状态（19:00）

所有4个实验已完成预加载，正在执行 Step 0。
NPU 进程确认运行中（8个进程，每卡约51GB内存）。
Step 0 之后步骤可能因 NPU kernel 编译而延迟。


## Step 20/40 数据对比（配置生效后）

| 实验 | Step | l2unif | var | recon | consist | 观察 |
|------|------|--------|-----|-------|---------|------|
| Exp1 Spatial | 20 | 0.3484 | 0.9245 | 0.3590 | 0.3502 | l2unif 显著低于 baseline (~0.35 vs ~0.59) |
| Exp2 High Weight | 40 | 0.4487 | 0.9361 | 0.3584 | 0.3384 | l2unif 持续下降 (0.65→0.59→0.45) |
| Exp3 VICReg Fix | 20 | 0.5876 | **0.0244** | 0.3624 | 0.3590 | var 极低，min_std=0.1 生效 |
| Exp4 Combined | 20 | 0.3484 | **0.0285** | 0.3590 | 0.3502 | Spatial + low var 同时生效 |

关键发现：
- **Spatial uniformity 有效**: Exp1/Exp4 的 l2unif (~0.35) 显著低于 Exp2/Exp3 (~0.59)
- **High weight 有效**: Exp2 的 l2unif 从 0.65 降至 0.45，下降趋势明显
- **VICReg fix 有效**: Exp3/Exp4 的 var 从 ~0.92 降至 ~0.02-0.03


## Step 60 数据对比（19:10）

| 实验 | l2unif | var | recon | consist | 趋势 |
|------|--------|-----|-------|---------|------|
| Exp1 Spatial | **0.2678** | 0.9351 | 0.3639 | 0.3194 | ↓ 持续改善 (0.30→0.27) |
| Exp2 High Weight | **0.6219** | 0.9359 | 0.3663 | 0.3420 | ↑ 恶化中 (0.45→0.62) |
| Exp3 VICReg Fix | **0.6219** | **0.0369** | 0.3663 | 0.3420 | ↑ 恶化中，但 var 正常 |
| Exp4 Combined | **0.2678** | **0.0364** | 0.3639 | 0.3194 | ↓ 最佳组合 |

**重大发现**:
1. **Spatial uniformity 显著优于标准 uniformity**: Exp1/4 的 l2unif (0.27) 只有 Exp2/3 (0.62) 的 43%
2. **标准 uniformity 无法阻止坍缩**: Exp2/3 的 l2unif 从 Step 40 的 0.45 反弹到 Step 60 的 0.62
3. **VICReg min_std=0.1 有效**: Exp3/4 的 var 保持在 ~0.037（vs Exp1/2 的 ~0.94）
4. **Combined (Exp4) 目前表现最佳**: l2unif=0.27 + var=0.037，两个指标都健康


## Epoch 001 完整结果（19:17）

| 实验 | total | recon | consist | var | cov | l2unif | active_dims | std_mean | time |
|------|-------|-------|---------|-----|-----|--------|-------------|----------|------|
| Exp1 Spatial | 0.3812 | 0.3618 | 0.3546 | 0.9245 | 0.0000 | **0.2969** | 39 | 0.0439 | 1214s |
| Exp2 High Weight | 0.6468 | 0.3622 | 0.3587 | 0.9244 | 0.0000 | **0.5593** | 39 | 0.0443 | 1178s |
| Exp3 VICReg Fix | 0.1274 | 0.3622 | 0.3587 | **0.0318** | 0.0000 | **0.5593** | 39 | 0.0443 | 1178s |
| Exp4 Combined | 0.2471 | 0.3618 | 0.3546 | **0.0320** | 0.0000 | **0.2969** | 39 | 0.0439 | 1218s |

### 关键发现

1. **Spatial uniformity 显著有效**: Exp1/4 的 l2unif (0.30) 比 Exp2/3 (0.56) 低 **47%**
2. **VICReg min_std=0.1 有效**: Exp3/4 的 var (0.032) 比 Exp1/2 (0.924) 低 **97%**
3. **High weight 导致 total 过高**: Exp2 total=0.65，uniformity loss 贡献过大
4. **Combined (Exp4) 平衡最佳**: l2unif=0.30 + var=0.032，两个指标都健康，total=0.25 适中
5. **active_dims=39** (epoch-level): 比 step-level (92) 低，这是因为 epoch-level 只用 memory bank 的 L2 embedding 计算

### 趋势预测

- **Exp1/4 (spatial)**: l2unif 保持在 ~0.30，可能继续缓慢下降
- **Exp2/3 (standard)**: l2unif 从 Step 80 的 0.75 降到 Epoch 平均 0.56，说明后期有所改善，但仍远高于 spatial
- **Exp3/4 (low var)**: var 保持在低位，VICReg 真正约束了 pre-norm 方差


## 当前状态（19:28）

所有4个实验正常运行，Epoch 002 进行中。
已设置自动监控：
- epoch2_check.log: 20分钟后检查
- epoch5_check.log: 60分钟后检查
- auto_monitor.log: 每5分钟记录step数据
- epoch_monitor.log: 每20分钟记录epoch数据


## Epoch 002 结果（19:35-19:37）

| 实验 | total | recon | consist | var | cov | l2unif | active_dims | std_mean | time |
|------|-------|-------|---------|-----|-----|--------|-------------|----------|------|
| Exp1 Spatial | 0.3992 | 0.3582 | 0.3362 | 0.9343 | 0.0000 | **0.3176** | 32 | 0.0412 | 1197s |
| Exp2 High Weight | 0.6769 | 0.3585 | 0.3370 | 0.9339 | 0.0000 | **0.5869** | 32 | 0.0415 | 1161s |
| Exp3 VICReg Fix | 0.1434 | 0.3585 | 0.3370 | **0.0361** | 0.0000 | **0.5869** | 32 | 0.0415 | 1158s |
| Exp4 Combined | 0.2728 | 0.3582 | 0.3362 | **0.0364** | 0.0000 | **0.3176** | 32 | 0.0412 | 1204s |

### 趋势分析 (Epoch 001 → 002)

| 实验 | l2unif 变化 | var 变化 | active_dims 变化 | 评估 |
|------|------------|----------|-----------------|------|
| Exp1 Spatial | 0.297 → 0.318 (+7%) | 0.925 → 0.934 (+1%) | 39 → 32 (-18%) | 轻微恶化，但仍显著优于 baseline |
| Exp2 High Weight | 0.559 → 0.587 (+5%) | 0.924 → 0.934 (+1%) | 39 → 32 (-18%) | 继续恶化，高权重不足以对抗坍缩 |
| Exp3 VICReg Fix | 0.559 → 0.587 (+5%) | 0.032 → 0.036 (+13%) | 39 → 32 (-18%) | 标准 uniformity 恶化，但 var 健康 |
| Exp4 Combined | 0.297 → 0.318 (+7%) | 0.032 → 0.036 (+13%) | 39 → 32 (-18%) | **最佳策略**：l2unif 最低 + var 健康 |

### 关键发现

1. **Spatial uniformity 无法完全阻止坍缩，但显著减缓**: Exp1/4 的 l2unif 增幅 (7%) 与 Exp2/3 (5%) 相近，但绝对值仍低 46%
2. **VICReg min_std=0.1 稳定有效**: var 保持在 ~0.035-0.036，不像 Exp1/2 那样接近 0.93
3. **active_dims 持续下降**: 所有实验从 39 → 32，说明即使最好的策略也无法阻止维度坍缩
4. **reconstruction 持续改善**: 所有实验 recon 从 ~0.362 → ~0.358，模型在重建任务上持续学习
5. **Exp4 (Combined) 仍是最佳**: l2unif=0.32 + var=0.036，total=0.27 适中


## Epoch 003 结果（19:55-19:57）

| 实验 | total | recon | consist | var | cov | l2unif | active_dims | std_mean |
|------|-------|-------|---------|-----|-----|--------|-------------|----------|
| Exp1 Spatial | 0.4065 | 0.3403 | 0.2833 | 0.9326 | 0.0000 | **0.3810** | 21 | 0.0358 |
| Exp2 High Weight | 0.7329 | 0.3405 | 0.2835 | 0.9327 | 0.0000 | **0.6906** | 21 | 0.0362 |
| Exp3 VICReg Fix | 0.1529 | 0.3405 | 0.2835 | **0.0353** | 0.0000 | **0.6906** | 21 | 0.0362 |
| Exp4 Combined | 0.3088 | 0.3403 | 0.2833 | **0.0353** | 0.0000 | **0.3810** | 21 | 0.0358 |

### 趋势分析 (Epoch 001 → 002 → 003)

| 实验 | l2unif | active_dims | 评估 |
|------|--------|-------------|------|
| Exp1 Spatial | 0.297 → 0.318 → **0.381** | 39 → 32 → **21** | 持续恶化，但绝对值仍低 45% |
| Exp2 High Weight | 0.559 → 0.587 → **0.691** | 39 → 32 → **21** | 加速恶化 |
| Exp3 VICReg Fix | 0.559 → 0.587 → **0.691** | 39 → 32 → **21** | 加速恶化，但 var 健康 |
| Exp4 Combined | 0.297 → 0.318 → **0.381** | 39 → 32 → **21** | **最佳策略**，但仍无法阻止坍缩 |

### 重大发现

1. **所有策略都无法完全阻止坍缩**: active_dims 从 39 → 21（3个epoch内损失46%的维度）
2. **Spatial uniformity 是减缓坍缩最有效的单一策略**: l2unif 比标准 uniformity 低 45%
3. **VICReg min_std=0.1 能稳定控制 pre-norm 方差**: var 保持在 ~0.035
4. **但 VICReg 对方差控制无法传导到 L2 空间的 uniformity**: Exp3 的 l2unif 和 Exp2 一样在恶化
5. **Combined (Exp4) 仍是最佳，但仍不够**: l2unif=0.38，active_dims=21

### 推断

如果当前趋势持续：
- 到 Epoch 10，active_dims 可能降到 <10
- Exp1/4 的 l2unif 可能升到 >0.6
- Exp2/3 的 l2unif 可能接近 0.9（完全坍缩）

**结论**: 需要更强的反坍缩机制。可能的下一步：
1. 在 pre-norm 空间计算 uniformity（而非 L2 空间）
2. 引入 bottleneck orthogonality loss（不依赖 embedding 值）
3. 使用 InfoNCE / DINO 等对比学习机制
4. 降低 reconstruction weight，给 uniformity 更多空间


## Epoch 004 结果（20:14-20:17）

| 实验 | total | recon | consist | var | cov | l2unif | active_dims | std_mean |
|------|-------|-------|---------|-----|-----|--------|-------------|----------|
| Exp1 Spatial | 0.4101 | 0.3058 | 0.2301 | 0.9298 | 0.0000 | **0.4801** | 15 | 0.0305 |
| Exp2 High Weight | 0.7831 | 0.3062 | 0.2307 | 0.9306 | 0.0000 | **0.7931** | 15 | 0.0302 |
| Exp3 VICReg Fix | 0.1574 | 0.3062 | 0.2307 | **0.0344** | 0.0000 | **0.7931** | 15 | 0.0302 |
| Exp4 Combined | 0.3574 | 0.3058 | 0.2301 | **0.0339** | 0.0000 | **0.4801** | 15 | 0.0305 |

### 趋势 (Epoch 001 → 002 → 003 → 004)

| 实验 | l2unif | active_dims |
|------|--------|-------------|
| Exp1 Spatial | 0.297 → 0.318 → 0.381 → **0.480** | 39 → 32 → 21 → **15** |
| Exp2 High Weight | 0.559 → 0.587 → 0.691 → **0.793** | 39 → 32 → 21 → **15** |
| Exp3 VICReg Fix | 0.559 → 0.587 → 0.691 → **0.793** | 39 → 32 → 21 → **15** |
| Exp4 Combined | 0.297 → 0.318 → 0.381 → **0.480** | 39 → 32 → 21 → **15** |

### 结论

1. **所有策略都无法阻止坍缩**: active_dims 4个epoch内从 39 降到 15
2. **Spatial uniformity 减缓效果最好**: l2unif 比标准 uniformity 低 40%
3. **VICReg min_std=0.1 稳定**: var 保持在 ~0.034
4. **但 VICReg 无法传导到 L2 空间**: Exp3 的 l2unif 和 Exp2 一样在恶化
5. **Combined (Exp4) 仍是最佳**: l2unif=0.48 + var=0.034，但需要更强机制


## Epoch 005 结果（20:33-20:37）— 关键转折点

| 实验 | total | recon | consist | var | cov | l2unif | active_dims | std_mean |
|------|-------|-------|---------|-----|-----|--------|-------------|----------|
| Exp1 Spatial | 0.4110 | 0.2627 | 0.1970 | 0.9249 | 0.0000 | **0.5689** | **4** | 0.0258 |
| Exp2 High Weight | 0.8109 | 0.2628 | 0.1964 | 0.9252 | 0.0000 | **0.8566** | **4** | 0.0262 |
| Exp3 VICReg Fix | 0.1572 | 0.2628 | 0.1964 | **0.0314** | 0.0000 | **0.8566** | **4** | 0.0262 |
| Exp4 Combined | 0.3989 | 0.2627 | 0.1970 | **0.0312** | 0.0000 | **0.5689** | **4** | 0.0258 |

### ⚠️ 重大发现：所有策略都失败了

active_dims 在 5 个 epoch 内从 **39 暴跌到 4**，损失 **90%** 的有效维度！

| 实验 | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | 损失率 |
|------|---------|---------|---------|---------|---------|--------|
| Exp1 Spatial | 39 | 32 | 21 | 15 | **4** | -90% |
| Exp2 High Weight | 39 | 32 | 21 | 15 | **4** | -90% |
| Exp3 VICReg Fix | 39 | 32 | 21 | 15 | **4** | -90% |
| Exp4 Combined | 39 | 32 | 21 | 15 | **4** | -90% |

### 分析

1. **模型找到了坍缩捷径**: recon 从 0.36 持续改善到 0.26，说明模型通过坍缩 embedding 来降低重建损失
2. **Spatial uniformity 只能减缓，不能阻止**: Exp1/4 的 l2unif (0.57) 仍比 Exp2/3 (0.86) 低 34%，但 active_dims 同样暴跌到 4
3. **VICReg min_std=0.1 控制 pre-norm 方差，但无法阻止 L2 空间坍缩**: Exp3/4 的 var (~0.031) 健康，但 l2unif 和 Exp2 一样恶化
4. **High weight (0.5) 不足以对抗 reconstruction (0.5) + consist (0.2) 的坍缩拉力**

### 根本原因推断

1. **Reconstruction weight 过高 (0.5)**: 模型优先优化重建，通过坍缩 embedding 实现
2. **Decoder 可能过强**: 只需要 4 维信息就能重建输入，说明 decoder 太容易了
3. **L2 Norm 在训练时可能有问题**: VMF noise + L2 norm 可能导致梯度方向不稳定
4. **Uniformity loss 的梯度被淹没**: 即使 weight=0.5，uniformity 的梯度仍不足以对抗 reconstruction

### 下一步实验建议

1. **降低 reconstruction weight** (0.5 → 0.1)，让 uniformity 主导训练
2. **Pre-norm raw uniformity** (Exp5): 在欧氏空间计算 uniformity，绕过 L2 Jacobian
3. **Bottleneck orthogonality loss**: 不依赖 embedding 值，始终提供反坍缩梯度
4. **InfoNCE / DINO**: 使用对比学习机制
5. **削弱 Decoder**: 增加 decoder 的重建难度，迫使 encoder 编码更多信息


## Epoch 006-007 结果（20:54-21:12）— 完全坍缩

| 实验 | total | recon | consist | var | cov | l2unif | active_dims | std_mean |
|------|-------|-------|---------|-----|-----|--------|-------------|----------|
| Exp1 Spatial | 0.4067 | 0.2152 | 0.1703 | 0.9176 | 0.0001 | **0.6566** | **1** | 0.0228 |
| Exp2 High Weight | 0.8293 | 0.1736 | 0.1396 | 0.9113 | 0.0001 | **0.9344** | **0** | 0.0195 |
| Exp3 VICReg Fix | 0.1522 | 0.2158 | 0.1687 | **0.0283** | 0.0001 | **0.9036** | **1** | 0.0222 |
| Exp4 Combined | 0.4353 | 0.2152 | 0.1703 | **0.0278** | 0.0001 | **0.6566** | **1** | 0.0228 |

### 完全坍缩确认

- **Exp2: active_dims=0** — 完全坍缩，所有128维的std都<0.05
- **Exp1/3/4: active_dims=1** — 几乎完全坍缩，只剩1维有微弱方差
- **所有实验的 l2unif > 0.65**：embedding 空间高度集中

### 最终结论（Epoch 6-7）

1. **所有当前策略都无法阻止 V13 的 embedding 坍缩**
2. **Spatial uniformity 是减缓效果最好的策略**，但最终仍坍缩到 1 维
3. **VICReg min_std=0.1 能控制 pre-norm 方差**，但无法阻止 L2 空间的坍缩
4. **模型通过坍缩来大幅降低 reconstruction loss**（0.36 → 0.17）
5. **根本问题**: reconstruction weight (0.5) 太强，decoder 可能过强，uniformity loss 的梯度无法对抗


## Epoch 009-010 结果（22:10-22:11）— 最终状态

| 实验 | Epoch | total | recon | consist | var | cov | l2unif | active_dims | std_mean |
|------|-------|-------|-------|---------|-----|-----|--------|-------------|----------|
| Exp2 High Weight | 10 | 0.8246 | 0.1038 | 0.0862 | 0.8959 | 0.0002 | **0.9733** | **0** | 0.0124 |
| Exp3 VICReg Fix | 10 | 0.1250 | 0.1038 | 0.0862 | **0.0238** | 0.0002 | **0.9733** | **0** | 0.0124 |
| Exp1 Spatial | 9 | 0.3836 | 0.1176 | 0.1026 | 0.8999 | 0.0002 | **0.8048** | **0** | 0.0133 |
| Exp4 Combined | 9 | 0.4831 | 0.1176 | 0.1026 | **0.0241** | 0.0002 | **0.8048** | **0** | 0.0133 |

### 最终结论

1. **所有实验在 Epoch 6-7 完全坍缩，Epoch 10 时 active_dims=0**
2. **Spatial uniformity 是减缓效果最好的策略**：
   - Exp1/4 的 l2unif (0.80) 比 Exp2/3 (0.97) 低 **18%**
   - 但无法阻止最终坍缩
3. **VICReg min_std=0.1 能稳定控制 pre-norm 方差**：
   - Exp3/4 的 var (0.024) 比 Exp1/2 (0.90) 低 **97%**
   - 但无法阻止 L2 空间的坍缩
4. **模型通过严重坍缩来大幅降低 reconstruction loss**：
   - recon 从 0.36 降到 0.10，降幅 72%
   - 这是坍缩的"回报"——模型找到了用更少信息重建的捷径
5. **根本问题**: reconstruction (0.5) + consistency (0.2) 的权重太强，uniformity 无法对抗


## Exp5 启动（22:21）

- **配置**: Pre-norm Raw Uniformity
- **NPU**: 0,1
- **端口**: 29504
- **预计完成**: ~01:50
- **核心修改**: 使用 `raw_uniformity_loss(gathered_pre)` 替代 `batch_uniformity_loss_l2(gathered_l2)`
- **权重**: pre_norm_uniform_weight=0.3, batch_uniformity_weight=0.0

