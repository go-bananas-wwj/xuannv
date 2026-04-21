# V5→V6→V6.5 完整实验总结 (2025-04-21)

## 实验概览

| 实验 | GPU 时间 | 核心改动 | 最佳结果 | 结论 |
|------|---------|---------|---------|------|
| V5 (基线) | ~3天 | Mixed Scale + Temporal Magnitude | Bare AUC 0.486, **CD Head AUC 0.9555** | 生产基线 |
| Phase 3 | ~2小时 | Pre-norm + PixelConvHead 下游 | WorldCover BAcc 0.5648 | CNN 空间上下文关键 |
| V6 | ~4小时 | 增强 uniformity + pixel tc loss | Uniformity -4.24→-2.72 (崩溃), AUC 0.489 | Temporal loss 过强 |
| V6.5 | ~6小时 | Gap-aware temporal + 增强 uniformity | Uniformity -4.17→-3.28, AUC 0.479 | 稍好但仍无效 |
| MultiMonth Class | ~1小时 | 5月 mean/std/max 融合 | **WorldCover BAcc 0.6266 (+6.2pp)** | ✅ 分类有效 |
| MultiMonth CD | ~1小时 | 多月份融合 CD Head | AUC 0.9299 (vs 0.9555) | ❌ 检测无效 |

---

## 关键发现

### 1. Backbone Bare AUC 是硬天花板 (~0.49)

无论 uniformity 如何改善，backbone bare AUC 始终 ~0.49:
- V5: uniformity=-3.1, AUC=0.486
- V6 epoch 8: uniformity=-3.64, AUC=0.489
- V6.5 epoch 14: uniformity=-3.66, AUC=0.479

**结论**: 简单 cosine 距离不足以检测变化。变化信号太微弱，需要 CNN 提取局部模式。

### 2. CD Head 是变化检测的唯一有效路径

| 方法 | AUC |
|------|-----|
| Backbone bare (任何版本) | ~0.49 |
| V5 CD Head V3 | **0.9555** |
| MultiMonth CD Head | 0.9299 |

CD Head 的 CNN 能从 embedding 的局部微观模式中提取变化信号，而 cosine 距离不能。

### 3. 多月份融合提升分类但不提升检测

| 任务 | 单月份 | 多月份融合 | Δ |
|------|--------|-----------|---|
| WorldCover BAcc | 0.5648 | **0.6266** | **+6.2pp** ✅ |
| JRC Water F1 | ~0.79 | 0.8226 | +3pp ✅ |
| OSM Buildings F1 | ~0.16 | 0.2325 | +7pp ✅ |
| CD Head AUC | **0.9555** | 0.9299 | -2.6pp ❌ |

**原因**: 多月份融合提供"时序稳定性"特征，帮助分类；但平滑了变化检测需要的"突变"信号。

### 4. Temporal Loss 与 Uniformity 互斥

| 模型 | Epoch 3 Uniformity | Epoch 10+ Uniformity | 变化速度 |
|------|-------------------|---------------------|---------|
| V6 (naive cosine) | -4.24 | -2.72 (epoch 11) | **崩溃** |
| V6.5 (gap-aware) | -4.17 | -3.56 (epoch 10) | **缓慢下降** |

Gap-aware 设计成功减缓崩溃 (3x)，但无法阻止下降趋势。

---

## 最佳配置总结

### 变化检测 (核心任务)
```
Backbone: V5 epoch 161 (skip-L2 training)
Embedding: L2-normalized monthly [128, 64, 64]
CD Head: ChangeDetectionHeadV3 (hidden_dim=64, 2.4M params)
Result: AUC 0.9555
```

### 下游分类
```
Backbone: V5 epoch 161
Embedding: Pre-norm monthly [128, 64, 64]
Head: PixelConvHead (in_dim=384 for multi-month, 3x3+1x1)
Result: WorldCover BAcc 0.6266, JRC Water F1 0.82
```

---

## 生产建议

### 立即部署 (不变)
- **变化检测**: V5 + CD Head V3 (AUC 0.9555)
- **分类**: V5 + MultiMonth Fusion PixelConvHead

### 未来优化方向

#### 高优先级
1. **改进 Change Detection Head**: 尝试更深的架构、多尺度特征、空间注意力
2. **改进 OSM Buildings Head**: 当前 F1=0.23 仍很低，尝试 Focal Loss + 更深网络

#### 中优先级
3. **从头训练 V7**: 若上述都失败，尝试无 soft-restart 的从头训练，使用 gap-aware temporal loss + 极高 uniformity weight

#### 低优先级
4. **继续优化 backbone uniformity**: 已验证对 bare AUC 无帮助，ROI 低

---

## 资产清单

| 资产 | 路径 | 说明 |
|------|------|------|
| V5 Best Checkpoint | `epoch_best_epoch161.pt` | 生产 backbone |
| V5 Monthly Embeddings | `monthly_embeddings_2025/` | L2-normalized, 2120 文件 |
| V5 Pre-norm Embeddings | `monthly_embeddings_2025_prenorm/` | 原始幅度, 2120 文件 |
| V5 CD Head | `monthly_cd_head/monthly_cd_head_v3.pt` | AUC 0.9555 |
| MultiMonth Class Heads | `downstream_convhead_multimonth/` | WorldCover/JRC/OSM |
| V6 Checkpoint | `epoch_best_epoch8.pt` | 已弃用 |
| V6.5 Checkpoint | `epoch_best_epoch14.pt` | 已弃用 |

---

## 经验教训

1. **Uniformity 不是万能药**: 从 -3.1 提升到 -4.2 没有改善下游任务
2. **评估指标必须对齐目标**: 我们优化了 uniformity，但 bare AUC 不提升 → 优化错了目标
3. **Soft Restart 有偏见**: 保留 encoder 权重使新 loss 难以改变结构
4. **多月份融合是双刃剑**: 分类有效，检测有害
5. **CNN 空间上下文是关键**: Phase 3 已验证，所有后续实验都支持这一结论

## 补充实验 (2025-04-21 续)

### CD Head V3 + hidden_dim=128 + OHEM
- 结果: Val AUC 0.9159 (vs V3 hd=64 的 0.9555)
- 结论: **更大的模型反而更差**，小数据集上过拟合

### 最终推荐配置 (不变)
```
Backbone: V5 epoch 161
CD Head: ChangeDetectionHeadV3 (hidden_dim=64, dropout=0.3)
AUC: 0.9555
```

