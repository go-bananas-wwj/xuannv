# V6/V6.5 实验总结与下一步建议

## 实验时间线

| 阶段 | 内容 | 结果 |
|------|------|------|
| V5 (基线) | Mixed Scale + Temporal Magnitude Loss | AUC 0.486 (bare), 0.9555 (CD Head) |
| Phase 3 | Pre-norm + PixelConvHead 下游验证 | CNN 空间上下文关键, pre-norm 幅度无用 |
| V6 | 增强 uniformity + pixel temporal cosine loss | Uniformity -4.24→-2.72 (崩溃), AUC 0.489 |
| V6.5 | Gap-aware temporal cosine loss | Uniformity -4.17→-3.28 (缓慢下降), AUC 0.479 |

---

## 核心发现

### 1. Backbone Bare AUC 是硬瓶颈

无论 uniformity 如何改善 (V5: -3.1, V6: -4.24, V6.5: -4.17), backbone bare AUC 始终 ~0.49 (随机水平)。

| 模型 | Uniformity | Bare AUC | CD Head AUC |
|------|-----------|----------|-------------|
| V5 | -3.1 | 0.486 | **0.9555** |
| V6 epoch 8 | -3.64 | 0.489 | 未测试 |
| V6.5 epoch 14 | -3.66 | 0.479 | 未测试 |

**结论**: 方向性 uniformity 不是 backbone bare AUC 的瓶颈。问题可能在：
- 嵌入空间结构不适合简单 cosine 距离
- 变化信号太微弱，需要 CNN 提取局部模式
- 需要多尺度/上下文信息

### 2. 强 Temporal Loss 与 Uniformity 互斥

| 模型 | Epoch 3 Uniformity | Epoch 10 Uniformity | 变化 |
|------|-------------------|---------------------|------|
| V6 | -4.24 | -2.92 | **-1.32 (崩溃)** |
| V6.5 | -4.17 | -3.56 | **-0.61 (缓慢下降)** |

Gap-aware 设计成功减缓了 uniformity 崩溃速度 (V6 崩溃 3 倍快于 V6.5)，但无法阻止下降趋势。

### 3. Pre-norm 幅度信息几乎无用

Phase 3 结论:
- Pre-norm vs L2-normalized Linear Probe: ΔBAcc < 0.002 (无差异)
- 仅 OSM Buildings (极度不平衡二分类) 有 F1 提升 (+37pp)
- CNN 空间上下文才是关键: Dynamic World +12.3pp, WorldCover +4.4pp

---

## V6/V6.5 训练日志对比

### V6 (已停止于 Epoch 11)

```
Epoch 003: total=-4.28  uniform=-4.239  tc_pixel=10.16  recon=0.622
Epoch 008: total=-8.03  uniform=-3.639  tc_pixel=-0.59   recon=0.266  ← 最佳 total
Epoch 011: total=-6.27  uniform=-2.719  tc_pixel=-2.72   recon=0.240
```

### V6.5 (已停止于 Epoch 17)

```
Epoch 003: total=-8.88  uniform=-4.172  gap_aware=13.12  recon=0.603
Epoch 008: total=-9.97  uniform=-3.646  gap_aware=3.90   recon=0.236
Epoch 014: total=-10.27 uniform=-3.661  gap_aware=2.24   recon=0.195  ← 最佳 total
Epoch 017: total=-8.94  uniform=-3.282  gap_aware=2.61   recon=0.194
```

---

## 失败根因分析

### 根因 1: Soft Restart 保留 Encoder 偏见

V6/V6.5 都是从 V5 checkpoint 软重启 (保留 encoder, 重置 bottleneck/decoder/head)。
- V5 encoder 已经过 161 epochs 训练，embedding 结构根深蒂固
- 新的 temporal loss 试图改变这个结构，但 encoder 权重被保留
- 结果: temporal loss 与 encoder 结构冲突 → uniformity 下降

### 根因 2: Temporal Loss 设计困境

- **Hinge loss (V5 temporal_magnitude)**: 几乎无梯度 (loss≈0)
- **Naive cosine (V6 tc_pixel)**: 始终推到 -1，无视 gap 大小 → 破坏渐变检测
- **Gap-aware MSE (V6.5)**: 更好但仍与 uniformity 冲突

**根本矛盾**: uniformity 要求所有 embedding 分散在球面上；temporal loss 要求某些 embedding 对相似/不同。两者在优化目标上天然冲突。

### 根因 3: 评估指标不匹配

我们一直在优化 uniformity，但:
- Uniformity 改善 ≠ 下游任务改善
- Bare AUC 不提升说明 embedding 的"方向差异"不是变化检测的关键信号
- CD Head 的 CNN 能从局部模式中提取信号，但简单 cosine 距离不能

---

## 下一步建议

### 建议 A: 放弃 Backbone 优化，专注 Downstream Heads (推荐)

基于 Phase 3 发现，**CNN 空间上下文是关键**。应投入资源:
1. **改进 Change Detection Head**: 更大的 receptive field, 多尺度特征, attention
2. **改进 Classification Heads**: 更深的 PixelConvHead, 时序融合 (mean+std+max)
3. **多月份 Embedding 融合**: 5 个月的统计量拼接 (Phase 3 设计但未执行)

**优点**: 直接针对已验证的瓶颈，风险低
**缺点**: 不解决 backbone 根本质量问题

### 建议 B: 从头训练 V7 (激进)

放弃 V5 encoder，从头训练新模型:
- 使用 V6.5 的 gap-aware temporal loss (已验证比 V6 稳定)
- 降低 temporal weight 到 0.1 以下
- 大幅提高 uniformity weight (4.0+) 和 variance weight (1.0+)
- 训练 300 epochs 从头开始

**优点**: 无 encoder 偏见，可能学到更好的结构
**缺点**: 需要 3-5 天 GPU 训练，结果不确定

### 建议 C: 混合策略 (折中)

1. 继续让 V6.5 训练到 epoch 50-100 (它还在缓慢进步)
2. 同时并行开发改进的 CD Head
3. 在 V6.5 达到最佳时，提取 embedding 训练新 CD Head
4. 对比 V5 CD Head AUC 0.9555

**优点**: 两边推进，不放弃已有投入
**缺点**: GPU 资源消耗大

---

## 当前可用资产

| 资产 | 位置 | 状态 |
|------|------|------|
| V5 Best Checkpoint | `/workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt` | ✅ 生产就绪 |
| V5 Pre-norm Embeddings | `/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2025_prenorm/` | ✅ 2120 文件 |
| V5 CD Head | `/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_cd_head/` | ✅ AUC 0.9555 |
| V6 Checkpoint | `/workspace/outputs/aef_qwen_v6_enhanced_temporal/epoch_best_epoch8.pt` | ⚠️ Uniformity 崩溃 |
| V6.5 Checkpoint | `/workspace/outputs/aef_qwen_v6_5_gap_aware/epoch_best_epoch14.pt` | ⚠️ Uniformity 缓慢下降 |
| PixelConvHead 脚本 | `scripts/train_v5_downstream_convhead.py` | ✅ 4 任务可用 |

---

## 决策建议

**短期 (今天)**: 用 V5 checkpoint 运行改进的 CD Head 实验和下游分类实验
**中期 (本周)**: 若 V6.5 后续 checkpoint 有改善，提取 embedding 训练新 CD Head
**长期 (下周)**: 若上述都失败，考虑从头训练 V7
