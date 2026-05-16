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

---

## 重大发现：VICReg 根本没参与训练！

### 代码验证

`ddp_v7_trainer.py` 中的损失计算：
```python
vicreg = lambda_var * vicreg_var + lambda_cov * vicreg_cov
# lambda_var=1.0 (default), lambda_cov=0.04 (default)

total = (
    recon_weight * recon
    + t.vicreg_weight * vicreg    # vicreg_weight=0.0 !!!
    + ...
)
```

所有配置中 `vicreg_weight: 0.0`，且 `vicreg_lambda_cov` 未设置。

**这意味着**：
1. 配置中的 `covariance_weight: 0.001` 对 `ddp_v7_trainer` **完全无效**
2. VICReg (variance + covariance) **完全不参与反向传播**
3. cov=33.2 只是**统计现象**，反映embedding缺乏去相关约束

### 为什么 Round 9(exp3) cov=0.043？

| 因素 | Round 9 | Mini Batch |
|------|---------|------------|
| skip_l2 | false | false (exp3) / true (others) |
| vicreg_weight | 0.0 | 0.0 |
| 数据集 | 424 patches | 100 patches |

**解释**: skip_l2=false → L2归一化 → 协方差矩阵自然受限（元素有界[-1,1]）
- Round 9大数据集进一步稳定统计
- Mini Batch数据少 + skip_l2=true → cov爆炸

### 反坍缩措施实际生效的只有

```python
total = (
    recon_weight * recon           # 0.1
    + t.consistency_weight * consist  # 0.05
    + t.classification_weight * cls   # 0.03
    + temporal_w * temporal           # 0.5
    + pre_norm_uniform_w * raw_unif   # 0.5
)
```

- `raw_unif`: 推分散，但**不去相关**
- 缺乏 `variance` 约束 → 某些维度可能坍缩
- 缺乏 `covariance` 约束 → 维度高度相关

### 为什么 Epoch 4 后 RawUnif 恶化？

- Epoch 1-4: raw_unif 推embedding分散，RawUnif改善
- Epoch 4+: 学习率达到峰值，embedding开始在球面上"滑动"
- 缺乏去相关约束 → embedding聚集在几个方向 → 维度相关增加
- cov统计值上升 → RawUnif恶化（embedding变集中）

## 修正后的下一步假设

| 假设 | 配置 | 预期效果 |
|------|------|----------|
| A | vicreg_weight=1.0, lambda_cov=1.0, skip_l2=false | VICReg参与+去相关约束 |
| B | vicreg_weight=1.0, lambda_cov=5.0, skip_l2=false | 更强去相关 |
| C | vicreg_weight=1.0, lambda_cov=1.0, uniform=1.0, skip_l2=false | 强反坍缩+去相关 |
