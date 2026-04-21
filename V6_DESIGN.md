# AEF_qwen V6 Enhanced Temporal — 设计文档

## 背景

Phase 3 实验结论：
- Pre-norm 幅度信息对分类任务几乎无帮助（除 OSM Buildings 外）
- CNN 空间上下文对多类别任务至关重要（Dynamic World +12.3pp, WorldCover +4.4pp）
- **根本问题在 backbone embedding 质量**：uniformity 仅 -2.7~-3.1（应达 -4），时间区分性不足
- `temporal_magnitude_loss` ≈ 0（hinge loss  rarely violated，无梯度）

V6 目标：**增强 embedding 区分性 + 改善 uniformity + 引入始终有梯度的 pixel-level 时序损失**

---

## 核心变更 (vs V5)

### 1. 增强 Uniformity

| 配置 | V5 | V6 |
|------|-----|-----|
| `uniformity_weight` | 1.5 | **2.5** |
| `gradient_accumulation_steps` | 8 | **12** |
| Effective batch | 32 | **48** |

更大的有效 batch 提升 uniformity 估计质量，更强的权重推动 embedding 更分散。

### 2. 新增空间 Uniformity

从 `pre_norm_map` [B, D, H, W] 中随机采样 **256 个空间位置**，计算额外的 uniformity loss：

```python
spatial_emb = _gather_spatial_embeddings(pre_norm_map, 256)  # [256, D]
spatial_uniform = raw_uniformity_loss(spatial_emb)
```

- 样本数从 B×world_size=4 提升到 256×world_size=512
- 强制空间 embedding 也保持分散

### 3. 新增 Pixel-level Temporal Cosine Loss

**问题**：`temporal_magnitude_loss` 是 hinge loss，当 `dist ≤ time_gap_norm + margin` 时梯度为 0。V5 中该 loss 几乎始终为 0。

**解决**：引入 `temporal_cosine_pixel_loss` —— 每个像素独立 normalize 后直接最小化 cos_sim，**始终有梯度**。

```python
f1 = F.normalize(pre_w1, p=2, dim=1)  # 逐像素 normalize
f2 = F.normalize(pre_w2, p=2, dim=1)
cos_map = (f1 * f2).sum(dim=1)        # [B, H, W]
loss = cos_map.mean() / temperature    # 直接最小化，非 hinge
```

- `weight = 0.5`, `temperature = 0.05`
- 强制不同时间窗口在**每个像素位置**产生方向差异

### 4. 新增 Pixel-level InfoNCE

`pixel_temporal_info_nce_loss` —— Anti-diagonal InfoNCE 的像素级版本：

```python
# 对每个空间位置独立计算 Anti-Diagonal InfoNCE
# 同一 batch 的 w1 和 w2 互为负样本
for each spatial location i:
    logits = (f1[:, i, :] @ f2[:, i, :].T) / temperature
    loss += F.cross_entropy(-logits, labels)
```

- `weight = 0.2`, `temperature = 0.1`
- 采样 16 个空间位置（避免计算量过大）
- 强制模型不能靠"全局时间偏移"作弊

### 5. 降低 Temporal Magnitude Loss 权重

- `weight = 0.3 → 0.1`
- 保留为安全约束（防止 embedding 距离超过时间间隔允许范围）
- 但不再指望它提供主要的时间区分性梯度

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `configs/qwen_v6_enhanced_temporal.yaml` | V6 配置文件 |
| `src/config.py` | 新增 V6 字段到 TrainingConfig dataclass |
| `src/training/ddp_v6_enhanced_temporal_trainer.py` | V6 DDP 训练器 |
| `scripts/train/train_ddp_v6.py` | V6 训练入口 |
| `start_v6_train.sh` | 一键启动脚本 |
| `scripts/test_v6_launch.py` | 快速验证脚本（forward + loss） |

---

## 启动命令

```bash
cd /workspace/xuannv
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
  scripts/train/train_ddp_v6.py \
  --config configs/qwen_v6_enhanced_temporal.yaml \
  --soft-restart /workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt \
  --save-every 20
```

或一键启动：

```bash
bash start_v6_train.sh
```

---

## 预期效果

| 指标 | V5 | V6 目标 |
|------|-----|---------|
| raw_uniformity | -2.7 ~ -3.1 | **-3.5 ~ -4.0** |
| tc_pixel_loss | 0 (未启用) | **> 0 始终有梯度** |
| temporal_magnitude | ~0 (无梯度) | **~0 但仅作约束** |
| 下游 BAcc (CNN Head) | 0.52-0.86 | **+3-8pp 提升** |

---

## 验证计划

1. 训练 50 epochs 后检查 uniformity 是否改善
2. 提取 embedding 验证变化检测 AUC
3. 跑下游 PixelConvHead 对比 V5
