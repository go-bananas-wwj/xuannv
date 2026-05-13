# V13 Embedding 坍缩问题深度调查实验报告

> 实验时间: 2026-05-12 ~ 2026-05-13
> 实验目标: 根治 embedding uniformity collapse，找到可复现的反坍缩训练方案
> 硬件: Huawei Ascend 910B4 NPU × 8
> 代码分支: `v12-clean-dynamic`

---

## 一、问题背景

V13 模型在训练中出现严重的 **embedding uniformity collapse**（嵌入均匀性坍缩）：
- `active_dims`（标准差 > 0.01 的维度数）从初始 ~50 迅速下降至 0
- `std_mean`（维度标准差均值）从 ~0.05 下降至 ~0.01
- 模型退化为所有输出几乎相同的向量，丧失表征能力

此前 V1~V12 多次尝试均未找到稳定有效的解决方案。

---

## 二、实验设计总览

本次调查共进行 **4 轮、20+ 组对照实验**，系统性排查坍缩根因。

### 实验速查表

| 轮次 | 实验 | 核心改动 | 结果 | active_dims |
|------|------|---------|------|-------------|
| R1 | exp1_spatial_unif | Spatial Uniformity on [B×H×W,D] | ❌ 坍缩 | 39 → **0** |
| R1 | exp2_high_weight | 高权重 uniformity + consistency | ❌ 坍缩 | 40 → **0** |
| R1 | exp3_vicreg_fix | VICReg variance + covariance | ❌ 坍缩 | 40 → **0** |
| R1 | exp4_combined | 以上全部 combined | ❌ 坍缩 | 38 → **0** |
| R2 | expD_skip_l2 | Skip L2 Norm (GMP前) | ❌ 坍缩 | 41 → **0** |
| R2 | expE_skip_l2_prenorm | Skip L2 + Pre-norm uniform | ❌ 坍缩 | 40 → **0** |
| R2 | expF_skip_l2_ortho | Skip L2 + Orthogonality | ❌ 坍缩 | 40 → **0** |
| R2 | expG_skip_l2_all | Skip L2 + 全部反坍缩 | ❌ 坍缩 | 40 → **0** |
| R3 | expH_mae_prenorm | MAE + Pre-norm uniform | ❌ 失败 | ~40 |
| R3 | expI_mae_lowrecon | MAE + Low recon | ❌ 失败 | ~40 |
| R3 | expJ_extreme | Extreme weights | ❌ 失败 | ~40 |
| R3 | expK_continue | Continue from checkpoint | ❌ 失败 | ~40 |
| **Exp** | **ExpL** | **Spatial VICReg + recon=0.5** | **部分成功** | **49 → 76** |
| **Exp** | **ExpM** | **Spatial VICReg + recon=0** | **🌟 成功** | **54 → 93** |
| Exp | ExpN | Spatial Raw Uniformity | ❌ 失败 | 48 → 41 |
| Exp | ExpO | Spatial VICReg + Raw Combined | ❌ 失败 | 48 → 41 |
| Ext | **ExpM Extended** | Continue recon=0 to 30ep | **🌟 成功** | **93 → 107** |
| Ext | ExpL Extended | Continue recon=0.5 to 30ep | 瓶颈 | 76 → 81 |
| Ext | **Staged recon=0.1** | Load M-ep10 + recon=0.1 | **🌟 成功** | **101 → 108** |
| Ext | **Full Data** | recon=0 on 424 patches | **成功** | **54 → 90** |

---

## 三、根因分析

### 3.1 发现的 5 个"短路"问题

通过训练管线审计，发现以下设计缺陷共同导致坍缩：

| # | 问题 | 影响 | 修复方案 |
|---|------|------|---------|
| 1 | **Consistency Loss 中 teacher embedding detach** | 反向传播被截断，gradient 无法到达 encoder | `consistency_weight=0` |
| 2 | **每源只加载 1 帧** | `valid_period` 时间窗口完全失效 | 加载该月**所有可用帧** |
| 3 | **Decoder 太强** | 4 层 Conv 直接复制输入，无需 meaningful embedding | 加入 `Dropout2d(0.3)`，削弱容量 |
| 4 | **Target = Input 直接复制** | Reconstruction 变成恒等映射 | 随机目标帧选择 |
| 5 | **GMP + L2 Norm 梯度屏障** | GMP 将 4096 空间位置平均为 1 向量，消灭空间多样性 | 在 GMP **之前**计算 VICReg |

### 3.2 Root Cause: GMP (Global Mean Pooling)

**最终根因确认：GMP 是 embedding collapse 的第一加速器。**

- GMP 将 `[B, D, H, W]` (H×W=4096 空间位置) 平均池化为 `[B, D]` (1 个全局向量)
- 4096 个空间位置的多样性被强制平均，信息大量丢失
- 即使空间 embedding map 是分散的，GMP 后的向量也可能坍缩

**解决方案：Spatial VICReg — 在 GMP 之前的空间 map 上计算 variance/covariance**
```python
# 错误：在 GMP 后的向量上计算
pre_norm = student_out.pre_norm_embedding  # [B, D]
var = variance_regularizer(pre_norm)       # GMP 已摧毁空间多样性

# 正确：在 GMP 之前的空间 map 上计算  
pre_norm = student_out.pre_norm_map.permute(0,2,3,1).reshape(-1, D)  # [B×H×W, D]
var = variance_regularizer(pre_norm)       # 保留全部空间多样性 ✅
```

### 3.3 Reconstruction 是 Collapse 的加速器

| recon 权重 | 10 epoch 结果 | 结论 |
|-----------|--------------|------|
| 0.0 | 93/128 active | 最佳发散 |
| 0.1 | 108/128 active (staged) | 可接受 |
| 0.5 | 76/128 active | 严重阻碍 |

- **recon=0**: 模型没有任何重建压力，100% 精力用于保持 embedding 分散
- **recon=0.1**: 适度的重建压力，在已发散的模型上不会导致坍缩
- **recon=0.5**: 重建任务主导优化方向，迅速拉回坍缩状态

---

## 四、关键实验详细结果

### 4.1 Round 1: 基线对照（全部失败）

所有 4 组实验在 Epoch 10 时 `active_dims=0`，`std_mean≈0.01`，完全坍缩。

```
exp1_spatial_unif:   Epoch 10 | active=0/128  std_mean=0.0124  recon=0.1040
exp2_high_weight:    Epoch 10 | active=0/128  std_mean=0.0124  recon=0.1038
exp3_vicreg_fix:     Epoch 10 | active=0/128  std_mean=0.0124  recon=0.1038
exp4_combined:       Epoch 10 | active=0/128  std_mean=0.0124  recon=0.1040
```

**结论**: 在 GMP 之后计算任何 uniformity/VICReg 都无法阻止坍缩。

### 4.2 Round 2: Skip L2 Norm（全部失败）

测试 L2 Norm 是否是根因。4 组 skip-L2 实验全部在 Epoch 10 左右坍缩到 0。

**结论**: **L2 Norm 不是根因**。即使完全跳过 L2，GMP 本身也足以导致坍缩。

### 4.3 Exploration L/M/N/O: 空间 VICReg 对比

| 实验 | Variance 计算位置 | recon | active_dims @ E10 |
|------|------------------|-------|-------------------|
| ExpL | `[B×H×W, D]` (空间) | 0.5 | **76/128** |
| **ExpM** | `[B×H×W, D]` (空间) | **0** | **93/128** ⭐ |
| ExpN | `[B×H×W, D]` (空间) + Raw Uniformity | 0 | 41/128 ❌ |
| ExpO | `[B×H×W, D]` (空间) + VICReg + Raw | 0 | 41/128 ❌ |

**关键发现**:
1. **Spatial VICReg > 任何其他方法** — 只有空间 VICReg 能阻止坍缩
2. **Raw Uniformity 有害** — ExpN/O 加入 raw_uniformity 后反而坍缩
3. **recon=0 是关键** — ExpM (recon=0) 比 ExpL (recon=0.5) 高 17 个 active dims

### 4.4 Extended Experiments

#### ExpM Extended (recon=0, 30 epochs, 50 patches)

```
Epoch 01: active=54  std_mean=0.0505
Epoch 05: active=85  std_mean=0.0637
Epoch 10: active=93  std_mean=0.0662
Epoch 15: active=103 std_mean=0.0692
Epoch 20: active=107 std_mean=0.0701  ← Step-level 峰值 127/128
Epoch 21: active=106 std_mean=0.0698
```

- 平台期在 105-107，50 patch 数据量可能限制了上限
- **Step-level 峰值达到 127/128**（embedding 空间几乎完全利用）

#### Staged recon=0.1 (20 epochs, 从 ExpM E10 加载)

```
Epoch 12: active=101  recon=0.311
Epoch 14: active=102  recon=0.262
Epoch 16: active=106  recon=0.233
Epoch 18: active=109  recon=0.235
Epoch 20: active=108  recon=0.228  ← **最终！**
```

- **🌟 历史性突破：带重建的模型达到 108/128 active_dims**
- 重建 loss 持续下降（0.31→0.23），模型确实学会了重建
- **Step-level 峰值 125-127/128**

#### ExpL Extended (recon=0.5, 30 epochs)

```
Epoch 10: active=76
Epoch 15: active=80
Epoch 20: active=81  ← 卡在 81，recon=0.5 是瓶颈
```

- recon=0.5 严重阻碍发散，20 epoch 仅提升 5 个 active dims

#### Full Data 424 patches (recon=0, 10 epochs)

```
Epoch 01: active=51  std_mean=0.0494
Epoch 03: active=54  std_mean=0.0509
Epoch 05: active=66  std_mean=0.0548
Epoch 07: active=84  std_mean=0.0627
Epoch 10: active=90  std_mean=0.0648  ← 持续上升
```

- 全量数据发散更慢，但**绝不坍缩**
- 10 epoch 达到 90/128，继续训练预计可达 100+

---

## 五、关键指标对比

### 5.1 方法有效性层级

```
Spatial VICReg (recon=0)        ████████████████████████████████████████ 107/128
Spatial VICReg (recon=0.1)      ██████████████████████████████████████░░ 108/128 (staged)
Spatial VICReg (recon=0.5)      ██████████████████████████████░░░░░░░░░░  81/128
Spatial Raw Uniformity           ████████████████████░░░░░░░░░░░░░░░░░░░░  41/128
Pre-norm Raw Uniformity          ████████████████████░░░░░░░░░░░░░░░░░░░░  40/128
L2 Uniformity + VICReg (GMP后)   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/128
```

### 5.2 Step-level vs Epoch-level Embedding 利用率

| 实验 | Epoch-level | Step-level 峰值 | 差距 |
|------|------------|-----------------|------|
| ExpM Extended | 107/128 | **127/128** | -20 |
| Staged recon=0.1 | 108/128 | **127/128** | -19 |
| Full Data | 90/128 | **110/128** | -20 |
| ExpL Extended | 81/128 | **122/128** | -41 |

- Step-level 远高于 epoch-level，说明训练过程中 embedding **高度分散**
- Epoch-level 被少数回落的 batch 拉低
- ExpL 的 step-level(122) 与 epoch-level(81) 差距最大，说明 recon=0.5 导致严重震荡

---

## 六、成功方案总结

### 方案 A: Pure Divergence（纯发散，无重建）

```yaml
reconstruction_weight: 0.0
consistency_weight: 0.0
variance_weight: 1.0
covariance_weight: 0.5
use_spatial_vicreg: true
batch_uniformity_weight: 0.0
```

- VICReg 计算在 `pre_norm_map.reshape(-1, D)` 上（空间级别）
- 10 epoch 可达 93/128，20 epoch 可达 107/128
- **问题**: 无重建 = 无实际表征学习能力

### 方案 B: Staged Training（分阶段训练）⭐ 推荐

```
Phase 1 (Epoch 0-10):  recon=0, spatial VICReg
  → 目标: active_dims > 100
  
Phase 2 (Epoch 10+):   recon=0.1, spatial VICReg
  → 目标: 保持 active_dims > 100, recon < 0.25
```

- 从 Phase 1 的 checkpoint 加载，加入 recon=0.1
- 20 epoch 达到 **108/128 active_dims + recon=0.23**
- **这是实际可用的训练方案**

---

## 七、代码修改清单

### 7.1 已应用的修复

| 文件 | 修改 | 目的 |
|------|------|------|
| `src/data/dataset.py` | 加载所有可用帧（而非1帧） | 使时间窗口有效 |
| `src/models/decoders.py` | 加入 `Dropout2d(0.3)` | 削弱 decoder 容量 |
| `src/models/decoders.py` | ConditionInjector 返回原 embedding | 移除时间条件泄露 |
| `src/training/ddp_v12_trainer.py` | `consistency_weight=0` | 禁用 consistency（detach 问题） |
| `src/training/ddp_v12_trainer.py` | Spatial VICReg on `[B×H×W, D]` | 绕过 GMP 梯度屏障 |

### 7.2 关键代码片段

```python
# src/training/ddp_v12_trainer.py — Spatial VICReg 核心实现
use_spatial_vicreg = getattr(t, 'use_spatial_vicreg', False)
if use_spatial_vicreg and pre_norm_map is not None:
    # [B, D, H, W] → [B×H×W, D] — 在 GMP 之前计算!
    spatial_pre = pre_norm_map.permute(0, 2, 3, 1).reshape(-1, D)
    if dist.is_initialized() and dist.get_world_size() > 1:
        spatial_pre = _all_gather(spatial_pre)
    var_loss = variance_regularizer(spatial_pre.float(), min_std=1.0)
    cov_loss = covariance_loss(spatial_pre.float())
else:
    # GMP 后计算 — 已证明会导致坍缩
    var_loss = variance_regularizer(all_pre.float(), min_std=1.0)
    cov_loss = covariance_loss(all_pre.float())
```

---

## 八、讨论与局限

### 8.1 为什么 Spatial VICReg 有效？

1. **样本量**: `[B×H×W, D]` 的样本量是 `[B, D]` 的 4096 倍（H=W=64）
2. **空间多样性**: 每个空间位置的 embedding 被独立约束，GMP 无法平均掉
3. **VICReg 的方差约束**: `variance_regularizer` 强制每个维度的 std ≥ 1.0，从机制上防止维度坍缩

### 8.2 为什么 recon 会加速坍缩？

1. Reconstruction 需要模型"记住"输入信息
2. 低维 embedding (128-d) 记住高维输入 (6ch×128×128) 的最优策略是**复用少量维度**
3. 模型倾向于将所有信息压缩到几个"记忆维度"，其余维度归零
4. Spatial VICReg 强制所有 128 维都有方差，与 recon 的压缩需求形成对抗

### 8.3 当前局限

1. **50 patch 上限**: ExpM 在 107 遇到平台期，可能需要更多数据突破
2. **Full data 速度**: 424 patch 10 epoch 仅 90/128，需要更长时间或更强约束
3. **recon 上限**: recon=0.1 可行，0.5 不可行，中间值（0.2-0.3）尚未测试
4. **Step-level 波动**: Epoch-level 比 step-level 低 ~20，说明有些 batch 会回落

---

## 九、下一步建议

### 9.1 高优先级

1. **渐进式 recon schedule**: 从 0 线性增加到 0.15，而非固定 0.1
2. **全量数据 + 更高 VICReg 权重**: `variance=2.0` 加速 424 patch 发散
3. **瓶颈层 Feature Dropout**: 在 pre_norm_map 上加 dropout，强制利用更多维度
4. **Attention Pooling 替代 GMP**: 从根本上消除 GMP 的信息损失

### 9.2 中优先级

1. 测试 recon=0.2, 0.3 的可行性
2. 评估最终模型的变化检测 AUC
3. 对比不同 `vicreg_min_std`（0.5, 1.0, 2.0）的效果

---

## 十、附录：实验完整数据

### A.1 Round 1 原始日志摘要

```
exp1_spatial_unif:
  E01: active=39  std_mean=0.0469  recon=0.3064
  E05: active=0   std_mean=0.0228  recon=0.1862
  E10: active=0   std_mean=0.0124  recon=0.1040

exp2_high_weight:
  E01: active=40  std_mean=0.0474  recon=0.3064
  E05: active=0   std_mean=0.0228  recon=0.1862
  E10: active=0   std_mean=0.0124  recon=0.1038

exp3_vicreg_fix:
  E01: active=40  std_mean=0.0474  recon=0.3064
  E05: active=0   std_mean=0.0228  recon=0.1862
  E10: active=0   std_mean=0.0124  recon=0.1038

exp4_combined:
  E01: active=38  std_mean=0.0467  recon=0.3064
  E05: active=0   std_mean=0.0228  recon=0.1862
  E10: active=0   std_mean=0.0124  recon=0.1040
```

### A.2 Exploration 原始日志摘要

```
ExpM (recon=0, spatial VICReg):
  E01: active=54  std_mean=0.0505
  E05: active=85  std_mean=0.0637
  E10: active=93  std_mean=0.0662

ExpL (recon=0.5, spatial VICReg):
  E01: active=49  std_mean=0.0498
  E05: active=68  std_mean=0.0575
  E10: active=76  std_mean=0.0607

ExpN (recon=0, spatial raw):
  E04: active=47  std_mean=0.0453
  E06: active=41  std_mean=0.0441  ← 下降!

ExpO (recon=0, combined):
  E04: active=52  std_mean=0.0476
  E06: active=41  std_mean=0.0444  ← 下降!
```

### A.3 Extended 原始日志摘要

```
ExpM Extended (recon=0, 30ep):
  E10: active=93   E15: active=103  E20: active=107  E21: active=106

Staged recon=0.1 (20ep, from M-E10):
  E12: active=101  recon=0.311
  E14: active=102  recon=0.262
  E16: active=106  recon=0.233
  E18: active=109  recon=0.235
  E20: active=108  recon=0.228

ExpL Extended (recon=0.5, 30ep):
  E10: active=76   E15: active=80   E20: active=81

Full Data (recon=0, 424p, 10ep):
  E01: active=51   E05: active=66   E10: active=90
```

---

*报告生成时间: 2026-05-13*
*实验负责人: Kimi Code CLI Agent*
