# V13 Round 5 — 重建核心 + 维度对齐AEF论文的并行实验计划

> 基于文献调研和用户反馈，重新定位问题：**重建是AEF核心任务，不可去掉；60+维活跃不等于坍缩（AEF论文用64维）；应测试64维embedding**
>
> **文献关键发现**:
> - AEF论文: Embedding **D=64**, κ=8e3, 使用 reconstruction + batch_uniformity + consistency + text_contrastive
> - VICReg论文: λ(invariance)=25, μ(variance)=25, ν(covariance)=1, 两者都是必需的
> - AEF在**512 TPU v4**上训练**100k steps**，使用Adam优化器

---

## 问题重新定位

### 之前的错误假设

| 错误假设 | 正确理解 |
|---------|---------|
| "active_dims=60 = 坍缩" | AEF论文用**64维**，60+活跃**已超过论文维度** |
| "recon=0才能防止坍缩" | **Reconstruction是AEF核心任务**，不能去掉 |
| "128维必须全部活跃" | 60-70维活跃在128维下是**正常利用率** (~50%) |
| "越高VICReg权重越好" | VICReg论文比例 var:cov = **25:1**，不是越大越好 |

### 新的核心问题

**不是**: "如何让128维全部活跃？"

**而是**: "如何在**带reconstruction**的情况下，让模型学习到**有效的64/128维embedding**？"

---

## 实验设计

### 4个并行实验（NPU 0-3，各20 epoch）

| 实验 | NPU | embedding_dim | Recon | Uniformity方法 | Consistency | vmf_kappa | 目的 |
|------|-----|--------------|-------|---------------|------------|-----------|------|
| **ExpT** | 0 | **64** | 1.0 | Spatial VICReg | 0.2 | 8000 | 64维 + Spatial VICReg |
| **ExpU** | 1 | **64** | 1.0 | batch_uniformity (GMP后) | 0.2 | 8000 | 64维 + 原始AEF配置 |
| **ExpV** | 2 | **128** | 1.0 | Spatial VICReg | 0.2 | 8000 | 128维 + Spatial VICReg |
| **ExpW** | 3 | **128** | 1.0 | batch_uniformity (GMP后) | 0.2 | 8000 | 128维 + 原始AEF配置 |

### 对比矩阵

```
                64维                128维
            ┌─────────────────┬─────────────────┐
Spatial     │  ExpT (NPU 0)   │  ExpV (NPU 2)   │
VICReg      │  核心实验        │  对比维度影响    │
            ├─────────────────┼─────────────────┤
batch_unif  │  ExpU (NPU 1)   │  ExpW (NPU 3)   │
(GMP后)     │  原始AEF配置     │  128维AEF配置   │
            └─────────────────┴─────────────────┘
```

---

## 关键超参

所有实验统一使用以下配置：

```yaml
# 损失权重（对齐AEF+VICReg）
reconstruction_weight: 1.0       # AEF核心
consistency_weight: 0.2          # Teacher-Student对齐
variance_weight: 0.3             # Spatial VICReg
batch_uniformity_weight: 0.1     # batch_uniformity (L2球面)
covariance_weight: 0.1           # Spatial VICReg

# 模型参数（对齐论文）
vmf_kappa: 8000.0                # AEF论文: κ=8e3

# 训练
lr: 0.000001
warmup_epochs: 10
max_steps_per_epoch: 50
epochs: 20
```

**注意**: Spatial VICReg 实验使用 `variance + covariance`（不启用batch_uniformity）；batch_uniformity 实验使用 `batch_uniformity`（不启用variance/covariance）。

---

## 成功标准

**不同于之前**：不再追求 active_dims=100+

| 指标 | 成功标准 | 说明 |
|------|---------|------|
| active_dims (64维) | ≥ 50 | 78%利用率 = 良好 |
| active_dims (128维) | ≥ 60 | 47%利用率 = 良好 |
| recon loss | < 0.3 | 模型学会了重建 |
| std_mean | > 0.04 | 维度有适度分散 |
| 变化检测AUC | > 0.7 | 下游任务有效 |

---

## 代码修改

### 1. embedding_dim=64 的模型支持

YAML中直接设置 `embedding_dim: 64`，模型会自动适配所有层。

### 2. vmf_kappa=8000

YAML中设置 `vmf_kappa: 8000.0`。

### 3. 无需修改 trainer 代码

当前代码已支持：
- `use_spatial_vicreg: true/false` 切换
- `embedding_dim` 可变
- `vmf_kappa` 可变
- `consistency_weight` 可调

---

## 启动命令

```bash
# ExpT — NPU 0: 64维 + Spatial VICReg
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=30020 \
  scripts/train/train_ddp_v12.py --config configs/v13_round5_expT_64d_spatial.yaml \
  --epochs 20 --save-every 5

# ExpU — NPU 1: 64维 + batch_uniformity
ASCEND_RT_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=30021 \
  scripts/train/train_ddp_v12.py --config configs/v13_round5_expU_64d_batchunif.yaml \
  --epochs 20 --save-every 5

# ExpV — NPU 2: 128维 + Spatial VICReg
ASCEND_RT_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=30022 \
  scripts/train/train_ddp_v12.py --config configs/v13_round5_expV_128d_spatial.yaml \
  --epochs 20 --save-every 5

# ExpW — NPU 3: 128维 + batch_uniformity
ASCEND_RT_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=30023 \
  scripts/train/train_ddp_v12.py --config configs/v13_round5_expW_128d_batchunif.yaml \
  --epochs 20 --save-every 5
```

---

## 预期结果

| 实验 | 预期 active_dims | 预期 recon | 预期结论 |
|------|-----------------|-----------|---------|
| ExpT (64d+Spatial) | 50-60 | < 0.3 | **可能最优** |
| ExpU (64d+batch) | 40-50 | < 0.3 | 测试batch_uniformity效果 |
| ExpV (128d+Spatial) | 60-70 | < 0.3 | 128维对比 |
| ExpW (128d+batch) | 50-60 | < 0.3 | 128维+batch_uniformity |

**如果 ExpT 成功** (active≥50, recon<0.3):
→ 这是正式训练的最佳起点：64维 + recon + Spatial VICReg

---

## 时间线

- **Epoch 1**: 约10分钟后可见初步结果
- **Epoch 5**: 约50分钟后判断趋势
- **Epoch 10**: 约1.5小时后中期评估
- **Epoch 20**: 约3小时后最终结果

---

*计划生成: 2026-05-13*
*状态: 等待用户审批*
