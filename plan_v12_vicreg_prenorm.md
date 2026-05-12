# V12 VICReg + Pre-norm + Memory Bank 修复计划

## 用户明确需求

1. **核心损失**：使用 `variance_regularizer`（VICReg variance loss）
2. **关键问题**：讨论并解决"是否在 L2 norm 上计算"
3. **Memory Bank**：恢复并用于扩大有效 batch

## 核心结论：必须在 Pre-norm 空间计算 VICReg

### 为什么 L2 空间不行？

| 空间 | 理论均匀 std | gamma 设定 | 当前 std | loss | 问题 |
|------|-------------|-----------|---------|------|------|
| **L2 (128D)** | 1/√128 ≈ **0.088** | γ=0.05（当前） | 0.011 | 0.05-0.011=0.039 | γ<理论值，完全均匀也不惩罚 |
| **L2 (128D)** | 0.088 | γ=0.088 | 0.011 | 0.077 | 即使完全均匀 loss=0，无超额推动力 |
| **Pre-norm** | 无上限（可达 >1.0） | γ=1.0 | ~0.5-2.0 | 0.0-1.0 | VICReg 设计意图，梯度充足 |

**VICReg 论文 Table 8 明确说**："None None: 68.6"（无归一化最佳），"l2 None: 65.1"（L2 归一化降 3.5 点）。

### Pre-norm 空间的优势

1. **gamma=1.0 有效**：std 可以达到 >1.0，loss 有明确的目标和梯度
2. **VICReg 设计意图**：论文和官方代码都在无归一化空间计算
3. **梯度充足**：当前 pre-norm std 估计 ~0.5-2.0，loss 力量与 recon 相当

### Pre-norm 的已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| V10 推理坍缩 | pre-norm uniformity 不传递到 L2 | **L2 空间增加轻量辅助损失**（低 weight） |
| L2 后丢失幅度信息 | L2 归一化抹除了 pre-norm 的幅度差异 | 不需要保留幅度，只需要方向多样性 |
| 需要 true pre-norm | V11 bottleneck 返回的是 L2-normalized | **修改 bottleneck 额外返回 pre-norm** |

## 方案设计

### 架构：Pre-norm 主损失 + L2 辅助监督 + Memory Bank

```
Training:
  features → [Conv1x1] → pre_norm_map → [L2 Norm] → embedding_map
                              ↓                          ↓
                    VICReg variance(γ=1.0)        batch_uniformity(轻量)
                    + covariance                    (weight=0.05)
                    + memory bank(K=512)
                    (weight=0.3)

Inference:
  features → [Conv1x1] → [L2 Norm] → embedding_map
```

### 损失组合

| 损失 | 空间 | 权重 | 作用 |
|------|------|------|------|
| **Reconstruction** | 解码器输出 | 0.5 | 主任务 |
| **Consistency** | L2 embedding | 0.2 | Teacher-student 对齐 |
| **VICReg Variance** | Pre-norm | 0.3 | **主反坍缩** |
| **VICReg Covariance** | Pre-norm | 0.1 | 维度去相关 |
| **Batch Uniformity** | L2 embedding | 0.05 | **L2 空间轻量监督** |

### Memory Bank 设计

- **对象**：`pre_norm_embedding`（不是 L2 embedding）
- **大小**：K=512
- **用途**：与当前 batch 的 pre_norm embedding 拼接，扩大 VICReg 的统计样本量
- **有效 batch**：96 (current) + 512 (bank) = 608

### 为什么 L2 辅助需要但 weight 很低？

- **作用**：确保 L2 空间也有一定多样性，不完全依赖 pre-norm → L2 的传递
- **为什么低 weight**：
  - 哈尔滨数据在 L2 空间不可能完全均匀（理论 uniform≈0.07，实际 0.3-0.4 可接受）
  - 低 weight 足以防止 L2 完全坍缩（uniform > 0.9）
  - 不强行破坏同类 patch 的相似性

## 文件改动清单

### 1. `src/models/bottleneck.py`

**修改**：`forward()` 返回真正的 pre-norm（而不是 L2-normalized）

```python
# 当前（V11）: pre_norm 和 embedding 相同
def forward(self, features):
    pre_norm_map = self.to_embedding(features)
    embedding_map = self._apply_norm(pre_norm_map)
    embedding_vector = embedding_map.mean(dim=(-2, -1))
    embedding_vector = F.normalize(embedding_vector, p=2, dim=1)
    return embedding_map, embedding_vector, embedding_vector, embedding_map  # pre=post

# 修改后: 返回真正的 pre-norm
def forward(self, features):
    pre_norm_map = self.to_embedding(features)
    embedding_map = self._apply_norm(pre_norm_map)
    embedding_vector = embedding_map.mean(dim=(-2, -1))
    embedding_vector = F.normalize(embedding_vector, p=2, dim=1)
    
    pre_norm_vector = pre_norm_map.mean(dim=(-2, -1))
    
    return embedding_map, embedding_vector, pre_norm_vector, pre_norm_map
```

**影响**：
- `model.py` 中 `pre_norm_embedding` 变为真正的 pre-norm
- `bottleneck_cls_head(pre_norm)` 现在接收的是 pre-norm（可能影响 cls head，但 V12 不使用 cls head）
- 需要确认 `forward_dual_window` 也返回 true pre-norm

### 2. `src/models/model.py`

**验证**：确认 `pre_norm_embedding` 和 `pre_norm_map` 的使用不会出问题
- `bottleneck_cls_head(pre_norm)` — V12 不使用（dummy loss）
- `encode_dual_window` 中的 `pre_w1/pre_w2` — 需要确认是 true pre-norm

### 3. `src/training/losses.py`

**修改**：
1. `variance_regularizer`：改为 `unbiased=False`（VICReg 官方实现）
2. 新增 `covariance_loss` 函数

```python
def variance_regularizer(embeddings, min_std=1.0):
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(min_std - std).mean()

def covariance_loss(embeddings):
    N = embeddings.shape[0]
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (emb_centered.T @ emb_centered) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()
```

### 4. `src/training/ddp_v12_trainer.py`

**修改**：
1. 恢复 `variance_regularizer` + `covariance_loss` 导入
2. 恢复 `EmbeddingMemoryBank` 导入
3. `train_epoch`：
   - 在 `student_out.pre_norm_embedding` 上计算 variance + covariance（+ memory bank）
   - 在 `student_out.embedding` 上计算轻量 batch_uniformity（weight=0.05）
   - 日志显示：variance loss + covariance loss + active_dims + L2 uniform
4. 添加 per-dimension std 诊断

### 5. `configs/xuannv_v12_clean.yaml`

**修改**：
```yaml
training:
  reconstruction_weight: 0.5
  consistency_weight: 0.2
  variance_weight: 0.3
  covariance_weight: 0.1
  batch_uniformity_weight: 0.05
  memory_bank_size: 512
```

## 日志指标设计

```
[Step N] recon=X.XXX consist=X.XXX var=X.XXX cov=X.XXX l2unif=X.XXX
         bank=608/512 std=[0.XXX/0.XXX/0.XXX] active_dims=N/128
```

| 指标 | 含义 | 目标 |
|------|------|------|
| var | VICReg variance loss | <0.5 |
| cov | VICReg covariance loss | <0.1 |
| l2unif | L2 batch_uniformity | <0.5（可接受 0.3-0.4） |
| active_dims | pre-norm std > 0.05 维度数 | >100/128 |
| std | per-dim std min/mean/max | mean > 0.5 |

## 验证计划

1. **前 3 个 epoch**：
   - 确认 active_dims 从 ~0 上升到 >50
   - 确认 var 从 ~1.0 下降到 <0.5
   - 确认 L2 uniform 不飙升到 >0.9

2. **10 个 epoch 后**：
   - active_dims > 100/128
   - L2 uniform 稳定在 0.3-0.5
   - recon < 0.5
   - 跑 AUC 验证

## 风险与应对

| 风险 | 应对 |
|------|------|
| bottleneck 改 pre-norm 影响其他模块 | 验证 `forward_dual_window` 和 `bottleneck_cls_head` 不会 crash |
| pre-norm → L2 传递失效 | L2 轻量 batch_uniformity 作为保险 |
| Memory Bank 初始为空 | 前 5-6 个 step 只算当前 batch，bank 填满后生效 |
| Covariance 计算慢 | O(128²) = 16K 操作，可忽略 |
