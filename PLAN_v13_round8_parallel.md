# V13 Round8 并行实验改进计划

> 基于 Round7 全面分析结果制定
> 当前状态：8×NPU 全部空闲，可并行运行 3-4 组实验

---

## 一、Round7 核心问题回顾

| 问题 | 证据 | 严重程度 |
|------|------|---------|
| **训练无跨月对比** | dual window 是同月内前后分割，cosine=0.99998 | 🔴 致命 |
| **Uniformity 不足** | -1.44（V4 为 -3.04） | 🔴 严重 |
| **VMF Kappa 过低** | 固定 15，L2 embedding 被噪声污染 | 🟠 重要 |
| **缺失 Consistency** | 无教师-学生一致性监督 | 🟠 重要 |
| **缺失 Classification** | 无 WorldCover 语义监督 | 🟡 中等 |
| **CD Head 数据不足** | Few-shot K=20 欠拟合 | 🟡 中等 |

---

## 二、改进假设与实验设计

基于问题分析，提出 **4 组并行实验**，每组使用 2 卡 DDP：

### 实验 A：跨月窗口 + 像素级时序损失（核心）

**假设**：如果训练时 dual window 是跨月（6 个月间隔）+ 像素级 cosine 时序损失，模型会学到方向上的时间变化。

**配置变更**：
```yaml
# data
window_mode: "non_overlap"
non_overlap_min_gap_ms: 15552000000  # 6个月

# training
temporal_contrastive_weight: 0.0  # 仍关闭全局temporal
temporal_cosine_pixel_weight: 0.5  # 新增：像素级时序损失

# model
vmf_kappa: 50.0  # 从15提升到50
```

**预期**：bare AUC 0.52 → 0.60+

---

### 实验 B：恢复 V4 反坍缩三件套 + Consistency

**假设**：如果恢复 V4 的 uniformity/variance/decorr + consistency，embedding 空间质量会提升到 -2.5 以下。

**配置变更**：
```yaml
# training
uniformity_weight: 1.0
variance_weight: 0.25
decorrelation_weight: 0.05
consistency_weight: 0.05
classification_weight: 0.03

# 保持同月窗口（对照组）
window_mode: "random_split"  # 不变
```

**预期**：uniformity -1.44 → -2.5，bare AUC 可能不变（空间好但无时序信号）

---

### 实验 C：A + B 组合（全面改进）

**假设**：跨月窗口 + 像素级时序 + 反坍缩三件套 + consistency 的组合效果最佳。

**配置变更**：实验 A + 实验 B 的所有改动合并。

**预期**：bare AUC 0.52 → 0.65+

---

### 实验 D：渐进 VMF Kappa 对照

**假设**：VMF Kappa 从 15 渐进到 100，比固定 50 更稳定。

**配置变更**：
```yaml
# training
kappa_start: 15.0
kappa_end: 100.0
kappa_warmup_epochs: 50

# 其他同实验 C
```

**预期**：训练稳定性更好，embedding 噪声更低

---

## 三、资源分配

| 实验 | 卡数 | NPU | 预计时间 | 输出目录 |
|------|------|-----|---------|---------|
| A | 2 | 0,1 | ~6h | `round8_expA_cross_month_pixel` |
| B | 2 | 2,3 | ~6h | `round8_expB_v4_style` |
| C | 2 | 4,5 | ~6h | `round8_expC_full` |
| D | 2 | 6,7 | ~6h | `round8_expD_progressive_kappa` |

---

## 四、验证指标与终止条件

每 50 epoch 自动验证：

| 指标 | 正常范围 | 异常处理 |
|------|---------|---------|
| `raw_unif` | -4.0 ~ -1.0 | > -0.5 持续 5 epoch → 报告坍缩 |
| `recon` | < 0.3 | warmup 后 > 0.5 → 检查数据 |
| `bare AUC` | > 0.55 (及格) | < 0.5 → 训练无效 |
| `active_dims` | > 100 | < 50 → 坍缩 |

**提前终止条件**：
- raw_unif > -0.5 持续 10 epoch
- recon > 0.5 在 epoch 30 后
- bare AUC < 0.5 在 epoch 100 后

---

## 五、执行步骤

### 阶段 1：准备（30 分钟）
1. 创建 4 个 config YAML
2. 复制 exp1 权重作为 warm-start（可选）
3. 清理旧缓存

### 阶段 2：训练（~6 小时）
1. 4 组实验并行启动（tmux）
2. 每 50 epoch 自动保存 + 验证

### 阶段 3：评估（1 小时）
1. 提取最佳 checkpoint 的月度 embedding
2. 运行 CD Head 训练（全量数据）
3. 对比 4 组 AUC

---

## 六、预期结果

| 实验 | 预期 Bare AUC | 预期 +CD Head | 概率 |
|------|--------------|--------------|------|
| A | 0.58-0.62 | 0.75-0.82 | 60% |
| B | 0.50-0.55 | 0.70-0.78 | 40% |
| C | 0.62-0.68 | 0.80-0.88 | 70% |
| D | 0.60-0.65 | 0.78-0.85 | 65% |

**最佳预期**：实验 C 或 D 达到 bare AUC > 0.65，+CD Head > 0.85

---

## 七、风险与备案

| 风险 | 概率 | 备案 |
|------|------|------|
| 全部实验坍缩 | 20% | 降低 temporal_cosine_pixel_weight 到 0.1 |
| NPU OOM | 15% | 减少 batch_size 或 max_frames |
| 训练崩溃 | 10% | 从 epoch 50 软重启 |
| 效果不明显 | 30% | 增加 training epochs 到 400 |
