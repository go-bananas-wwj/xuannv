# 启用 Temporal Loss 详细计划

## 一、为什么 AUC = 0.4968？根本原因

### 核心诊断

当前训练损失中 **temporal_magnitude_weight = 0.0**，模型从未被要求学习"不同时间 = 不同 embedding"。

验证时的 Score 分布：
```
mean=0.0037, max=0.4573
```
→ 几乎所有 patch 的 before/after cosine similarity ≈ 1.0，模型是**时间盲**的。

### 损失函数分解（当前 E68）

| 损失 | 权重 | 作用 | 状态 |
|------|------|------|------|
| Reconstruction | 1.0 | 重建像素 | ✅ 正常 |
| VICReg | 2.0 | 均匀分布 | ✅ 正常 |
| KoLeo | 0.5 | 推开最近邻 | ✅ 正常 |
| Consistency | 0.03 | Teacher-Student一致 | ✅ 正常 |
| **Temporal Contrastive** | **0.0** | **学习时间差异** | ❌ **禁用** |

> **结论**：没有 Temporal Loss，模型学到的 embedding 对时间完全不敏感，Bare AUC 必然接近随机。

---

## 二、Temporal Loss 是怎么计算的？

项目中有 5 种 Temporal Loss 实现（`src/training/losses.py`），当前使用的是 `temporal_contrastive_loss`：

### 1. temporal_contrastive_loss（当前代码默认）

```python
def temporal_contrastive_loss(emb_w1, emb_w2, temperature=0.05, target_margin=0.2):
    # 1. 对两个窗口的 embedding map 做 spatial global mean
    flat_w1 = emb_w1.reshape(B, D, -1).mean(dim=-1)  # [B, D]
    flat_w2 = emb_w2.reshape(B, D, -1).mean(dim=-1)
    
    # 2. L2 归一化
    flat_w1 = F.normalize(flat_w1, p=2, dim=-1)
    flat_w2 = F.normalize(flat_w2, p=2, dim=-1)
    
    # 3. 计算 cosine similarity
    cos_sim = (flat_w1 * flat_w2).sum(-1)  # [B], 范围 [-1, 1]
    
    # 4. Hinge Loss: 只惩罚 cos_sim > target_margin 的部分
    loss = F.relu(cos_sim - target_margin).mean() / temperature
    return loss
```

**直观理解**：
- 如果两个时间窗口的 embedding 太相似（cos_sim > 0.2），就产生惩罚
- 目标：让不同时间的 embedding 夹角 ≥ 78°（arccos(0.2) ≈ 78°）
- temperature=0.05 让梯度更尖锐

**问题**：Hinge Loss 在 cos_sim ≤ 0.2 时梯度为 0，模型达到 margin 后就停止优化。

### 2. temporal_cosine_pixel_loss（推荐替代）

```python
def temporal_cosine_pixel_loss(emb_w1, emb_w2, temperature=0.05):
    # 逐像素归一化
    f1 = F.normalize(emb_w1, p=2, dim=1)  # [B, D, H, W]
    f2 = F.normalize(emb_w2, p=2, dim=1)
    
    # 逐像素 cosine similarity
    cos_map = (f1 * f2).sum(dim=1)  # [B, H, W]
    
    # 直接最小化所有像素的 cos_sim（非 hinge，持续有梯度）
    loss = cos_map.mean() / temperature
    return loss
```

**优势**：
- **无 hinge**：所有像素持续有梯度，不会因达到 margin 而停止
- **像素级**：强制每个空间位置都产生时间差异
- **更细粒度**：空间信息保留，有利于下游变化检测

### 3. temporal_info_nce_loss（Anti-Diagonal InfoNCE）

```python
def temporal_info_nce_loss(emb_w1, emb_w2, temperature=0.1):
    # 把 batch 内不同样本的 (w1, w2) 当作负样本
    logits = (flat_w1 @ flat_w2.T) / temperature  # [B, B]
    # 对角线 = 正样本（同一地点不同时间应该是"不同"的）
    # 因此用 -logits，让对角线概率最小
    return F.cross_entropy(-logits, labels)
```

**优势**：利用 batch 内多样性，负样本更硬。
**劣势**：batch_size 较小时效果差。

### 4. gap_aware_temporal_cosine_loss（V6.5 使用）

根据时间 gap 大小动态设定 target：
- gap=0 → target=1（相似）
- gap=max → target=-1（相反）

**优势**：符合直觉——间隔越大，差异应该越大。
**劣势**：需要计算时间 gap，数据依赖更强。

### 5. temporal_magnitude_loss

约束 embedding distance ≤ time_gap_norm + margin，只设上限。

---

## 三、推荐方案

### 方案：启用 temporal_cosine_pixel_loss + 适当权重

理由：
1. **无 hinge**：持续有梯度，模型不会停在 margin
2. **像素级**：比 global mean 更细粒度，有利于空间变化检测
3. **简单稳定**：不依赖 batch size，不易崩溃

### 配置变更

```yaml
# configs/xuannv_v7.yaml

training:
  # 启用时序损失
  temporal_magnitude_weight: 0.0   → 0.3
  temporal_magnitude_temperature: 0.1  # pixel loss 用稍大 temperature
  
  # 其他调整（防止新损失冲击训练稳定性）
  consistency_weight: 0.03  → 0.01   # Consistency 可能压制时间差异，降低权重
  
  # 双窗口概率（已配置，无需修改）
  # temporal_window_prob: 0.5
  # temporal_window_min_frames: 4
  # temporal_window_max_frames: 24
```

### Trainer 代码变更

当前 trainer 使用 `temporal_contrastive_loss`：
```python
temporal = temporal_contrastive_loss(emb_w1, emb_w2, temperature=...)
```

改为 `temporal_cosine_pixel_loss`：
```python
from src.training.losses import temporal_cosine_pixel_loss
# ...
temporal = temporal_cosine_pixel_loss(emb_w1, emb_w2, temperature=getattr(t, 'temporal_magnitude_temperature', 0.1))
```

---

## 四、风险评估与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **Temporal Loss 冲击重建** | 中 | Recon 上升 | weight=0.3 起步，不激进 |
| **VICReg + Temporal 冲突** | 低 | 均匀性下降 | 监控 PreUnif，若 > -1.0 报警 |
| **NaN/Inf** | 低 | 训练崩溃 | temperature=0.1 较温和，不易数值爆炸 |
| **Consistency 压制时间差异** | 中 | Temporal 无效 | 降低 consistency_weight 至 0.01 |
| **双窗口数据不足** | 低 | 部分 batch 无 temporal | temporal_window_prob=0.5 已保障 |

---

## 五、执行步骤

### Step 1: 停止当前训练
```bash
tmux kill-session -t v7_train
```

### Step 2: 修改配置
- `configs/xuannv_v7.yaml`: `temporal_magnitude_weight: 0.3`
- `configs/xuannv_v7.yaml`: `consistency_weight: 0.01`

### Step 3: 修改 Trainer
- `src/training/ddp_v7_trainer.py`: 将 `temporal_contrastive_loss` 改为 `temporal_cosine_pixel_loss`
- 添加 `temporal_magnitude_temperature` 读取

### Step 4: 清空输出目录
```bash
rm -f /workspace/outputs/xuannv_backbone_v7_phase1_v2/*.pt
rm -f /workspace/outputs/xuannv_backbone_v7_phase1_v2/*.log
```

### Step 5: 从头训练
```bash
torchrun --nproc_per_node=8 scripts/train/train_ddp_v7.py \
    --config configs/xuannv_v7.yaml \
    --wandb-run-name v7_temporal_enabled
```

### Step 6: 监控指标（关键）

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| `temporal` | 0.1 ~ 2.0 | > 5.0 可能 weight 过高 |
| `recon` | < 0.4 | > 0.6 说明 temporal 冲击重建 |
| `pre_unif` | -4.0 ~ -1.0 | > -0.5 坍缩 |
| `consist` | 0.01 ~ 0.5 | > 1.0 不正常 |

### Step 7: E50 验证
训练到 E50 后，跑 Bare AUC 验证，对比是否从 0.50 → > 0.65。

---

## 六、预期结果

| 指标 | 无 Temporal (E50) | 有 Temporal (E50 预期) |
|------|------------------|----------------------|
| Bare AUC | ~0.50 | **> 0.60** |
| Recon | 0.30 | 0.35 (微升，可接受) |
| VICReg | 0.35 | 0.40 (微升) |
| Temporal Loss | 0.0 | 0.5 ~ 1.5 |

---

## 七、备选方案

如果 `temporal_cosine_pixel_loss` 效果不佳，可以切换：

| 备选 | 场景 | 修改 |
|------|------|------|
| `temporal_contrastive_loss` | 希望稳定收敛 | 用 hinge margin=0.2 |
| `gap_aware_temporal_cosine_loss` | 需要细粒度时间控制 | 传入 time_gap_ms |
| `temporal_info_nce_loss` | batch_size 大时效果好 | 需增大 batch |

---

*计划时间: 2026-05-08*
*当前状态: E68/400, AUC=0.4968, Temporal Loss 禁用*
