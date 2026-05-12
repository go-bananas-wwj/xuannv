# V12 per-dimension Uniformity 修复计划

## 背景

用户明确要求：
> "不要算平均距离，而是算每个维度的距离。"

当前 `batch_uniformity_loss_l2` 计算的是**所有样本对之间的平均 dot product**（all-pairs 平均距离），用户认为这种方式：
1. 对于小数据集（424 patches）会强制所有样本互相远离，不现实
2. "平均"会稀释真正重要的坍缩信号
3. 应该用 **per-dimension** 的方式评估和惩罚坍缩

## 分析

### 为什么 all-pairs 平均距离不适合我们的数据？

| 问题 | 说明 |
|------|------|
| 小数据稀释 | 424 patches 中大量同类地表（农田），同类 patch 的 embedding 本就应该相似，all-pairs 强行惩罚这种相似性 |
| 梯度稀释 | all-pairs 的梯度被 N² 平均，对于大 batch 有效，但小 batch 时单个 pair 的梯度太弱 |
| 评估失真 | uniform=0.35 可能反映的是"同类 patch 相似"而非"全部坍缩" |

### per-dimension 方式的优势

1. **与样本数量无关**：只关心 128 个维度每个是否有信息量
2. **不惩罚同类相似**：同类 patch 相似是正常的，只要它们在每个维度上都有变化
3. **指标更直观**："128 个维度中 100 个活跃"比"平均 dot product=0.35"更有意义

## 方案

### 损失函数设计

用 **两个互补的 per-dimension 损失** 替换 `batch_uniformity_loss_l2`：

#### 1. Covariance Loss（维度去相关）

```python
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """VICReg covariance loss — 惩罚维度之间的相关性.
    
    如果 embedding 坍缩到少量方向，协方差矩阵的非对角元素会很大。
    与 L2 归一化完全兼容。
    """
    N = embeddings.shape[0]
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (emb_centered.T @ emb_centered) / (N - 1)  # [D, D]
    # 只惩罚非对角元素
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()
```

**特性**：
- 如果所有样本坍缩到同一个方向 → cov ≈ 全 1 矩阵 → loss ≈ 1.0
- 如果维度完全去相关 → cov 对角矩阵 → loss ≈ 0
- 与 L2 空间兼容（不需要 gamma 参数）
- 只关心 128 个维度之间的关系，不关心样本数量

#### 2. Active Dimension Loss（维度活跃度）

```python
def active_dim_loss(embeddings: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """惩罚死亡维度 — 每个维度的 std 必须大于阈值.
    
    L2-normalized embedding 的理论均匀 std = 1/sqrt(D) ≈ 0.088，
    但哈尔滨数据不可能达到均匀。threshold=0.02 是"有信息量"的最低标准。
    """
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-6)  # [D]
    # 死亡维度：std < threshold
    dead_penalty = F.relu(threshold - std).mean()  # 死亡维度的惩罚
    return dead_penalty
```

**特性**：
- 如果 128 个维度 std 都 < 0.02 → loss ≈ 0.02
- 如果所有维度 std > 0.02 → loss = 0
- 当前状态：128/128 维度 std ≈ 0.011 → loss ≈ 0.009
- weight=0.3 → contribution ≈ 0.003（偏弱，但作为辅助损失足够）

**为什么 threshold=0.02？**
- 不是理论均匀值 0.088（哈尔滨数据不可能达到）
- 是"有信息量"的最低标准：std=0.02 意味着该维度至少有 ±0.06 的变化范围
- 经验值：V10 时期 std<0.05 被视为坍缩，threshold 设为一半（0.02）作为"有信息量"标准

### 权重配置

```yaml
training:
  reconstruction_weight: 0.5
  consistency_weight: 0.2
  covariance_weight: 0.3      # 主反坍缩损失（替代 batch_uniformity）
  active_dim_weight: 0.3      # 辅助反坍缩损失
```

**梯度力量对比**（基于 epoch 1 实测值）：
- recon: 1.16 × 0.5 = 0.58
- consist: 0.37 × 0.2 = 0.074
- covariance: ~0.5 × 0.3 = 0.15（假设 cov off-diag ≈ 0.5）
- active_dim: 0.009 × 0.3 = 0.003

**covariance 的 0.15 与 recon 的 0.58 比例为 1:3.9**，足以竞争。

### 日志指标设计

每个 step 输出：
```
[Step N] recon=X.XXX consist=X.XXX cov=X.XXX active=X.XXX 
         std=[min/mean/max] active_dims=N/128 cov_offdiag=X.XXX
```

- `cov`: covariance loss 值（目标：下降）
- `active`: active_dim loss 值（目标：下降）
- `std=[min/mean/max]`: per-dimension std 统计
- `active_dims`: std > 0.02 的维度数（目标：>100/128）
- `cov_offdiag`: 协方差矩阵非对角元素的均值（目标：接近 0）

### 监控指标与目标

| 指标 | 当前值 | 健康目标 | 含义 |
|------|--------|---------|------|
| cov loss | ~0.5 | <0.1 | 维度去相关程度 |
| active loss | ~0.009 | <0.001 | 死亡维度数量 |
| active_dims | ~0/128 | >100/128 | 有信息量的维度数 |
| cov_offdiag | ~0.5 | <0.05 | 维度间相关性 |
| recon | ~1.16 | <0.5 | 重建质量 |

## 改动文件清单

1. **`src/training/losses.py`** — 新增 `covariance_loss()` 和 `active_dim_loss()`
2. **`src/training/ddp_v12_trainer.py`** — 
   - 移除 `batch_uniformity_loss_l2` 导入和使用
   - 添加 `covariance_loss` 和 `active_dim_loss` 导入
   - 在 `student_out.embedding` 上计算两个损失
   - 更新日志输出（per-dim 统计）
3. **`configs/xuannv_v12_clean.yaml`** — 
   - 移除 `batch_uniformity_weight`
   - 添加 `covariance_weight: 0.3` 和 `active_dim_weight: 0.3`

## 验证计划

1. 跑 10 个 epoch
2. 每 2 个 epoch 检查一次日志，确认：
   - cov loss 是否下降
   - active_dims 是否增加
   - recon 是否继续下降
3. 10 个 epoch 后决定是否继续或调整

## 风险

1. **Covariance loss 计算量**：O(D²) = 128×128 = 16K，可以忽略
2. **L2 空间限制**：covariance 在 L2 空间有效，但可能不如欧氏空间敏感
3. **数据多样性不足**：即使维度去相关，如果样本本身多样性低，active_dims 可能仍然不高
