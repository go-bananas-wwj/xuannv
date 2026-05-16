# 发现与决策 — Mini Batch Test 失败分析

## 需求
找到能在100-patch子集上快速验证的训练配置，目标：
- RawUnif < -2.0（反坍缩）
- cov < 1.0（VICReg协方差可控）
- Recon < 0.3（重建质量）

## 研究发现

### 发现1：Mini Batch 与 Round 9(exp3) 的关键差异

| 维度 | Round 9 exp3 | Mini Batch exp3 |
|------|--------------|-----------------|
| Patches | 424 | 100 |
| 月度样本 | 11,915 | 2,772 |
| Steps/epoch | 200 | 50 |
| Batch size | 4 | 4 |
| 每epoch采样率 | ~7% | ~7% |
| skip_l2 | false | false |
| cov_weight | 0.001 | 0.001 |
| Epoch 1 cov | 0.043 | 0.214 |
| Epoch 1 RawUnif | -1.56 | -1.55 |

**关键洞察**: 采样率相同(7%)，但Mini Batch的绝对样本数只有1/4。这意味着：
- 每个batch的统计估计更不稳定
- VICReg cov基于batch协方差矩阵，小数据集→高方差估计
- 但exp3在Mini Batch上cov仍是最低的(0.39@Ep6)，说明skip_l2=false有帮助

### 发现2：Epoch 4 是转折点

所有实验的RawUnif在Epoch 4达到最佳，之后恶化：
```
exp3: -1.55 → -1.97 → -1.39
exp5: -1.42 → -1.82 → -1.40
exp6: -1.46 → -1.82 → -1.27
```

**原因假设**: warmup=3 epochs，Epoch 4时学习率达到峰值(1e-4)，可能进入不稳定区域。

### 发现3：高uniform权重加剧cov爆炸

exp4 (uniform=1.0) 的cov最严重(33.2)，说明：
- uniformity loss推动embedding分散
- 但cov项同时要求去相关
- 两者冲突时，低cov权重无法维持约束
- 模型被迫牺牲cov来满足uniformity

### 发现4：空间感知temporal loss有效

exp1 (无空间加权) vs exp2 (有空间加权):
- Temporal Loss: 4.96 vs 1.13
- 验证了weight_map的必要性

## 技术决策
| 决策 | 理由 |
|------|------|
| 提高covariance_weight | 0.001无法抑制cov增长，需0.01+ |
| 延长warmup | 3ep太短，学习率冲击导致不稳定 |
| 保持skip_l2=false | exp3 cov控制最好 |
| 保持空间感知temporal | 已验证有效 |

## 下一步假设（待验证）

1. **假设A**: cov_weight=0.01 + skip_l2=false → cov可控 + RawUnif持续改善
2. **假设B**: 去掉cov项 + 提高variance_weight=1.0 → 简化约束
3. **假设C**: warmup=10 + cov_weight=0.01 + uniform=1.0 → 平滑学习 + 强约束
4. **假设D**: 增加steps/epoch到100 → 更多采样 → 统计更稳定

## 资源
- docs/FAILED_PARAMETERS_LOG.md — 完整失败记录
- /workspace/outputs/mini_batch_monitor.log — 训练监控日志
