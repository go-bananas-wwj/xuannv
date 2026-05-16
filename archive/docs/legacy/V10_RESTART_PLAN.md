# V10 问题总结与重启训练计划

## 一、问题总结

### 1.1 Embedding 严重坍缩（核心问题）

| 指标 | V10 E100 实测 | 健康值 | 状态 |
|------|--------------|--------|------|
| Patch间 Mean Cos Sim | **0.905** | ~0.0 | ❌ 严重坍缩 |
| Stable Rank | **1.10 / 64** | ~64 | ❌ 仅利用1.7%维度 |
| 活跃维度 (std>0.1) | **0 / 64** | ~64 | ❌ 全灭 |
| Temporal Distance < 0.05 | **93.2%** | ~50% | ❌ 时间盲 |
| RankMe | **0.505** | ~1.0 | ❌ 低质量 |

**根因**：`skip_l2_training` 导致 pre-norm 空间的 uniformity 无法传递到 L2-norm 后的推理空间。

> AEF 论文原文：**batch uniformity 在 L2-normalized 空间计算**，不是 pre-norm 空间。

### 1.2 训练相关 Bug

| Bug | 影响 | 修复方案 |
|-----|------|---------|
| Uniformity loss 计算在 pre-norm 空间 | 推理时坍缩 | 改在 L2-norm 空间计算 |
| AEF batch uniformity 缺失 | 无 batch 级防坍缩机制 | 新增 batch_uniformity_loss |
| Decorrelation loss 异常高 (6.3) | 梯度冲突 | V11 已移除 |
| VMF Kappa 渐进 (50→500) | 不一致 | 参考 AEF 固定 8000 |
| Temporal loss 权重过低 (0.08) | 时间信号弱 | 保持不变或微调 |

### 1.3 数据问题

| 问题 | 现状 | 方案 |
|------|------|------|
| 数据量不足 | 424 patches | 下载大庆/齐齐哈尔/海淀 (1200新增) |
| S1/Landsat 帧数少 | 每 patch 仅 2 帧 | 增大 filterBounds buffer |
| 下载进程不稳定 | 已重启 3 次 | 改用 systemd/supervisor 监控 |

### 1.4 评估问题

| 问题 | 原因 | 方案 |
|------|------|------|
| Bare AUC 样本不足 | 标注与 patch 匹配率低 | 修复匹配逻辑 |
| NPU _masked_softmax fallback CPU | 算子不支持 | 暂时接受，或改 CPU 评估 |

---

## 二、重启方案对比

### 方案 A：快速修复（推荐）

**目标**：最小改动，立即恢复训练，验证修复效果。

**改动**：
1. 修改 `ddp_v10_temporal_trainer.py`：
   - `raw_uniformity_loss` 前对 `gathered_pre_norm` 做 L2 normalize
   - 新增 `batch_uniformity_loss`（AEF 核心机制）
2. 修改 `bottleneck.py`：
   - `_apply_norm` 中 kappa 固定为 8000（参考 AEF）
3. 恢复训练：从 E100 checkpoint 继续

**优点**：改动小，见效快，不影响当前下载进度
**缺点**：不解决数据量不足的根本问题
**预计时间**：再跑 50-100 epoch（8-16小时）看效果

### 方案 B：V11 完整版

**目标**：等数据下载完成后，全面启动 V11 架构。

**改动**：
1. 数据扩展（1200新增 patches）
2. 源特定重建权重
3. 移除 decorrelation/orthogonality
4. 强化 consistency (0.05→0.2)
5. 静态目标时间编码弱化

**优点**：从根本上解决问题
**缺点**：需要等数据下载完成（12-16小时）

### 方案 C：折中（推荐）

**目标**：先快速修复继续训练，同时等数据，数据就绪后切换 V11。

**步骤**：
1. **立即**：方案 A 快速修复，恢复 V10 训练
2. **并行**：数据下载继续（tmux后台）
3. **E120 评估**：验证快速修复是否有效
4. **数据就绪后**：soft restart 到 V11

---

## 三、详细执行计划（方案 C）

### Phase 1：代码修复（1小时）

#### 1.1 修改 `src/training/losses.py` — 新增 batch_uniformity_loss

```python
def batch_uniformity_loss(embedding_map: torch.Tensor) -> torch.Tensor:
    """AEF 核心机制: batch-level uniformity on L2-normalized embeddings.
    
    Args:
        embedding_map: [B, D, H, W] L2-normalized embedding map
    Returns:
        loss: scalar (minimize to spread embeddings on sphere)
    """
    B, D, H, W = embedding_map.shape
    vectors = embedding_map.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
    shifted = torch.roll(vectors, shifts=1, dims=0)
    uniformity = torch.mean(torch.abs(torch.sum(vectors * shifted, dim=1)))
    return uniformity
```

#### 1.2 修改 `src/training/ddp_v10_temporal_trainer.py`

**改动点 1**：uniformity 在 L2-norm 空间计算
```python
# 原代码：
uniform = raw_uniformity_loss(gathered_pre_norm.float())

# 新代码：
gathered_l2_norm = F.normalize(gathered_pre_norm, p=2, dim=1)
uniform = raw_uniformity_loss(gathered_l2_norm.float())
```

**改动点 2**：新增 batch_uniformity_loss
```python
batch_uniform_w = getattr(t, 'batch_uniformity_weight', 0.05)
batch_uniform = torch.tensor(0.0, device=self.device)
if batch_uniform_w > 0:
    # 使用 embedding_map (已经 L2 normalized)
    batch_uniform = batch_uniformity_loss(student_out.embedding_map.float())
```

**改动点 3**：加入 total loss
```python
total = (
    recon_weight * recon
    + consist_w * consist
    + ...
    + batch_uniform_w * batch_uniform  # 新增
)
```

#### 1.3 修改 `src/models/bottleneck.py`

```python
# kappa 固定为 8000（AEF配置），移除渐进调度
self.kappa = 8000.0  # 固定
```

#### 1.4 修改 `configs/xuannv_v10_temporal.yaml`

```yaml
# 新增
batch_uniformity_weight: 0.05

# 移除（已置0）
decorrelation_weight: 0.0
orthogonality_weight: 0.0
```

### Phase 2：恢复训练（立即）

```bash
# 从 E100 恢复
torchrun --nproc_per_node=8 \
    scripts/train/train_ddp_v10.py \
    --config configs/xuannv_v10_temporal.yaml \
    --resume /workspace/outputs/xuannv_backbone_v10_temporal/epoch_100.pt \
    --epochs 200
```

### Phase 3：监控与评估

| 检查点 | 时间 | 评估内容 |
|--------|------|---------|
| E120 | ~3小时后 | 提取 embedding，计算 patch-level 分析 |
| E150 | ~8小时后 | 完整评估（Bare AUC + 质量指标） |

### Phase 4：数据就绪后切换 V11

当大庆/齐齐哈尔/海淀各 400 patches 下载完成后：
1. 停止 V10 训练
2. 计算新数据统计
3. 修改 dataset 支持多城市
4. Soft restart 到 V11（保留 encoder，重置 bottleneck/decoder）

---

## 四、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| Batch uniformity 无效 | 中 | 回退到 V11 完整方案 |
| NPU 算子继续 fallback CPU | 高 | 接受性能损失，或多卡并行提取 |
| 数据下载再次中断 | 中 | 用 supervisor/systemd 自动重启 |
| E120 评估仍坍缩 | 中 | 立即切换 V11，不等数据 |

---

## 五、预期效果

如果 batch uniformity 修复有效：
- Patch 间 mean cos sim: 0.905 → **0.5-0.7**
- Stable Rank: 1.10 → **10-30**
- Temporal distance > 0.2: 1.3% → **20-40%**
- RankMe: 0.505 → **0.7-0.9**
