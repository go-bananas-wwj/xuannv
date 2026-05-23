# V13 Round 4 — 更加发散的并行预训练实验计划

> 基于 Round 1~3 及 Extended 实验结论，设计 4 组并行实验进一步探索 embedding 发散极限。
> 
> **当前最佳方案**: Spatial VICReg (var=0.3, cov=0.1) on `[B×H×W, D]` + recon=0 → 107/128 active_dims
> 
> **目标**: 突破 110/128，找到全量数据 (424p) 的快速发散方案，验证渐进式重建策略。

---

## 当前 NPU 状态

| NPU | 当前任务 | 状态 |
|-----|---------|------|
| 0 | ExpM Extended (recon=0, 30ep) | 运行中 (E21/30) |
| 1 | ExpL Extended (recon=0.5, 30ep) | 运行中 (E20/30) |
| **2** | **空闲** | **可用 ✅** |
| **3** | **空闲** | **可用 ✅** |
| **4** | **空闲** | **可用 ✅** |
| **5** | **空闲** | **可用 ✅** |
| 6 | 空闲 | 可用 |
| 7 | 空闲 | 可用 |

**说明**: NPU 0-1 的旧实验将在约 1.5h 后自然结束。新实验使用 NPU 2-5（或 2-3 + 4-5）。

---

## 实验总览

| 实验 | NPU | 核心假设 | 关键改动 | Epochs |
|------|-----|---------|---------|--------|
| **ExpP** | 2 | 更强 VICReg 约束可突破 107 上限 | var=1.0, cov=0.5 | 20 |
| **ExpQ** | 3 | Covariance 可能不必要，仅靠 Variance 即可 | var=1.0, cov=0 | 20 |
| **ExpR** | 4 | 渐进式 recon 比固定 staged 更平滑有效 | recon 0→0.2 linear | 20 |
| **ExpS** | 5 | 全量数据在更长训练下可达 100+ | 424p, 20ep | 20 |

---

## 实验一：ExpP — 强化 VICReg 约束 (High VICReg)

### 假设
当前成功配置 `var=0.3, cov=0.1` 的约束强度可能不足以让模型突破 107/128 的平台期。将权重提高至 `var=1.0, cov=0.5`（约 3-5 倍），可能：
1. 在同样 epoch 内达到更高的 active_dims（115+）
2. 加速全量数据发散（为 ExpS 提供参考）

### 配置

```yaml
experiment:
  name: v13_round4_expP_high_vicreg
  output_dir: /workspace/outputs/v13_round4_expP_high_vicreg

data:
  max_patches: 50  # 快速迭代

training:
  reconstruction_weight: 0.0
  consistency_weight: 0.0
  variance_weight: 1.0      # ← 从 0.3 提升到 1.0
  covariance_weight: 0.5    # ← 从 0.1 提升到 0.5
  use_spatial_vicreg: true
  batch_uniformity_weight: 0.0
  
  lr: 0.000001
  warmup_epochs: 10
  max_steps_per_epoch: 50
  epochs: 20
```

### 成功标准
- Epoch 10: active_dims ≥ 100（不低于 ExpM 基线）
- Epoch 20: active_dims ≥ 110（突破平台期）
- std_mean ≥ 0.075（比 ExpM 的 0.070 更高）

### 风险
- 权重过高可能导致 loss 数值过大，训练不稳定
- 可能遇到 NaN（但 grad_clip=1.0 应可缓解）

---

## 实验二：ExpQ — Variance-only 分解测试

### 假设
VICReg 的两个组件中，`variance_regularizer`（强制每维标准差 ≥ min_std）是防止坍缩的核心机制，`covariance_loss`（去相关）是辅助。测试 **仅保留 variance、完全去掉 covariance** 是否仍能有效发散。

如果成功，说明可以进一步简化损失函数，降低训练复杂度。

### 配置

```yaml
experiment:
  name: v13_round4_expQ_variance_only
  output_dir: /workspace/outputs/v13_round4_expQ_variance_only

data:
  max_patches: 50

training:
  reconstruction_weight: 0.0
  consistency_weight: 0.0
  variance_weight: 1.0      # 保留 variance
  covariance_weight: 0.0    # ← 完全去掉 covariance
  use_spatial_vicreg: true
  batch_uniformity_weight: 0.0
  
  lr: 0.000001
  warmup_epochs: 10
  max_steps_per_epoch: 50
  epochs: 20
```

### 成功标准
- Epoch 10: active_dims ≥ 90（接近 ExpM 基线 93）
- Epoch 20: active_dims ≥ 100
- 与 ExpP 对比，判断 covariance 的贡献度

### 预期结果
- **乐观**: variance 单独即可达到 100+，covariance 非必要
- **悲观**: 无 covariance 时维度间高度相关，虽然 std 高但信息冗余，active_dims 虚高

---

## 实验三：ExpR — 渐进式重建 (Progressive Reconstruction)

### 假设
Staged 训练（先 recon=0 发散，再固定 recon=0.1）已成功，但存在两个问题：
1. 需要手动分阶段，不够优雅
2. 固定 recon=0.1 可能不是最优值

**渐进式策略**: 让 `reconstruction_weight` 从 0 开始，每 epoch 线性增加，最终达到 0.2。这样：
- 早期（Epoch 0-5）: recon≈0，模型自由发散
- 中期（Epoch 5-15）: recon 逐渐增加到 0.15，模型开始学重建但不至于坍缩
- 后期（Epoch 15-20）: recon=0.2，测试更高重建压力下的稳定性

### 代码修改

需要在 `src/training/ddp_v12_trainer.py` 的 `train_epoch()` 中增加动态 recon weight 计算：

```python
# 在 epoch 开始时计算当前 recon weight
recon_ramp_epochs = getattr(t, 'recon_ramp_epochs', 0)
if recon_ramp_epochs > 0 and epoch < recon_ramp_epochs:
    current_recon_weight = t.reconstruction_weight * (epoch / recon_ramp_epochs)
else:
    current_recon_weight = t.reconstruction_weight
```

修改量：约 5 行代码，纯增加逻辑，不影响现有训练。

### 配置

```yaml
experiment:
  name: v13_round4_expR_progressive_recon
  output_dir: /workspace/outputs/v13_round4_expR_progressive_recon

data:
  max_patches: 50

training:
  reconstruction_weight: 0.2      # 最终目标 recon 权重
  recon_ramp_epochs: 20           # ← 新增参数：20 epoch 内从 0 线性增加到 0.2
  consistency_weight: 0.0
  variance_weight: 0.3
  covariance_weight: 0.1
  use_spatial_vicreg: true
  batch_uniformity_weight: 0.0
  
  lr: 0.000001
  warmup_epochs: 10
  max_steps_per_epoch: 50
  epochs: 20
```

### 成功标准
- Epoch 10: active_dims ≥ 95（recon 已达 0.1，不应低于 Staged 基线）
- Epoch 20: active_dims ≥ 100（recon=0.2 下仍保持分散）
- recon loss ≤ 0.30（模型学会了有意义的重建）

### 与 Staged 的对比优势
- 无需手动加载 checkpoint 切换阶段
- 可从零开始一次训练完成
- recon 值平滑过渡，无突变

---

## 实验四：ExpS — 全量数据长时训练 (Full Data Extended)

### 假设
Full Data 实验（424 patches, recon=0, 10ep）已达到 90/128 且持续上升。如果延长训练到 20 epoch，可能达到 100+。这是验证全量数据可行性的关键实验。

### 配置

```yaml
experiment:
  name: v13_round4_expS_full_data
  output_dir: /workspace/outputs/v13_round4_expS_full_data

data:
  # 不设置 max_patches = 使用全部 424 patches
  # 或使用 max_patches: 200（如果 424 太慢）
  max_patches: 0  # 0 = all patches

training:
  reconstruction_weight: 0.0
  consistency_weight: 0.0
  variance_weight: 0.3
  covariance_weight: 0.1
  use_spatial_vicreg: true
  batch_uniformity_weight: 0.0
  
  lr: 0.000001
  warmup_epochs: 30  # 全量数据需要更长 warmup
  max_steps_per_epoch: 50
  epochs: 20
```

### 成功标准
- Epoch 10: active_dims ≥ 90（不低于之前 Full Data 基线）
- Epoch 20: active_dims ≥ 100（全量数据突破三位数）
- std_mean ≥ 0.070

### 风险
- 全量数据预加载慢（~4-5 分钟 vs 50 patch 的 ~1 分钟）
- 每个 epoch 训练时间更长（~570s vs ~550s）
- 如果发散速度不理想，可能浪费较长时间

---

## 实验对比矩阵

| 维度 | ExpP | ExpQ | ExpR | ExpS |
|------|------|------|------|------|
| 数据量 | 50p | 50p | 50p | 424p |
| recon | 0 | 0 | 0→0.2 | 0 |
| var | 1.0 | 1.0 | 0.3 | 0.3 |
| cov | 0.5 | **0** | 0.1 | 0.1 |
| batch_unif | 0 | 0 | 0 | 0 |
| 需改代码 | ❌ | ❌ | ✅ (5行) | ❌ |
| 预计时间 | ~18m | ~18m | ~18m | ~19m/epoch |
| 核心价值 | 突破上限 | 简化损失 | 优雅重建 | 全量验证 |

---

## 启动命令

```bash
# ExpP — NPU 2
ASCEND_RT_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=30010 \
  scripts/train/train_ddp_v12.py --config configs/v13_round4_expP_high_vicreg.yaml \
  --epochs 20 --save-every 5

# ExpQ — NPU 3
ASCEND_RT_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=30011 \
  scripts/train/train_ddp_v12.py --config configs/v13_round4_expQ_variance_only.yaml \
  --epochs 20 --save-every 5

# ExpR — NPU 4（需先修改 trainer 代码）
ASCEND_RT_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=30012 \
  scripts/train/train_ddp_v12.py --config configs/v13_round4_expR_progressive_recon.yaml \
  --epochs 20 --save-every 5

# ExpS — NPU 5
ASCEND_RT_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=30013 \
  scripts/train/train_ddp_v12.py --config configs/v13_round4_expS_full_data.yaml \
  --epochs 20 --save-every 5
```

---

## 监控方案

使用现有 monitor daemon 模式，每 5 分钟更新：

```bash
# 监控日志目录
/workspace/outputs/v13_fast_logs/
# 监控汇总
/workspace/outputs/v13_fast_logs/round4_summary.txt
```

---

## 预期结果与决策树

```
Round 4 结束后:

IF ExpP (var=1.0, cov=0.5) ≥ 115:
  → 使用更高 VICReg 权重启动正式训练
  
IF ExpQ (cov=0) ≥ 100:
  → 可简化损失函数，去掉 covariance
  
IF ExpR (progressive) ≥ 100 AND recon ≤ 0.30:
  → 采用渐进式 recon 作为正式方案（最优雅）
  
IF ExpS (424p) ≥ 100:
  → 全量数据方案可行，启动正式全量训练
```

---

*计划生成时间: 2026-05-13*
*待审批: 等待用户确认后执行*
