# 失败参数记录 — Mini Batch Test (Epoch 6 提前终止)

## 实验背景
- **数据**: 100 patches (91变化 + 9随机)
- **训练量**: 20 epochs × 50 steps = 1,000 total steps
- **Batch Size**: 4
- **每epoch时间**: ~8分钟
- **终止原因**: covariance失控 + RawUnif恶化

## 共同基础参数（所有8个实验）

| 参数 | 值 | 评价 |
|------|-----|------|
| reconstruction_weight | 0.1 (base) | 正常，Recon从0.51→0.20 |
| consistency_weight | 0.05 | — |
| classification_weight | 0.03 | — |
| orthogonality_weight | 0.01 | — |
| pre_norm_uniform_weight | 0.5 (base) | 可能不足 |
| variance_weight | 0.25 (base) | 正常 |
| **covariance_weight** | **0.001** | ❌ **太低，导致cov爆炸** |
| temporal_cosine_pixel_weight | 0.5 | — |
| warmup_epochs | 3 | — |
| lr | 1e-4 | — |
| grad_clip_norm | 1.0 | — |

## 各实验具体参数与结果

### mb_exp1_baseline
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| temporal_use_spatial_weight | false | Loss=1.95, cov=22.5, RawUnif=-1.23 |
| **问题** | Temporal Loss=4.96（异常高） | 验证了空间加权的必要性 |

### mb_exp2_spatial
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| temporal_use_spatial_weight | true | Loss=0.06, cov=12.2, RawUnif=-1.18 |
| **问题** | cov爆炸，RawUnif恶化 | 基础配置也有问题 |

### mb_exp3_no_skip_l2
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| skip_l2_norm_training | false | Loss=0.50, cov=0.39, RawUnif=-1.39 |
| **评价** | **cov控制最好**，但RawUnif仍恶化 | 相对最优但不够 |

### mb_exp4_high_uniform
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| pre_norm_uniform_weight | 1.0 | Loss=-0.73, cov=33.2, RawUnif=-1.18 |
| **问题** | **cov最严重爆炸**，高uniform反而加剧 | ❌ 不推荐 |

### mb_exp5_high_recon
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| reconstruction_weight | 0.3 | Loss=0.32, cov=4.36, RawUnif=-1.40 |
| **评价** | RawUnif尚可，cov中等 | 相对较好 |

### mb_exp6_high_variance
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| variance_weight | 0.5 | Loss=0.08, cov=6.42, RawUnif=-1.27 |
| **评价** | cov仍增长 | 不够 |

### mb_exp7_no_teacher
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| teacher_momentum | 0 | Loss=0.11, cov=8.73, RawUnif=-1.27 |
| **评价** | 无teacher反而cov更高 | 无关 |

### mb_exp8_combined
| 参数 | 值 | 结果 (Epoch 6) |
|------|-----|----------------|
| recon=0.2, uniform=0.8, var=0.4 | 组合 | Loss=-0.27, cov=11.5, RawUnif=-1.18 |
| **问题** | cov高，组合无优势 | ❌ 不推荐 |

## 关键趋势分析 (Epoch 2 → 4 → 6)

### RawUnif（反坍缩，应越来越负）
```
exp3: -1.55 → -1.97 → -1.39  (先好后坏)
exp5: -1.42 → -1.82 → -1.40  (先好后坏)
exp6: -1.46 → -1.82 → -1.27  (先好后坏)
exp8: -1.41 → -1.81 → -1.18  (先好后坏)
```
**结论**: 所有实验在Epoch 4达到最佳，之后开始恶化！

### cov（应接近0）
```
exp1: 1.05 → 10.19 → 22.51  (持续爆炸)
exp4: 0.79 →  7.78 → 33.20  (最严重)
exp3: 0.21 →  0.53 →  0.39  (唯一控制住的)
```

## 根本原因诊断

1. **covariance_weight=0.001 太低**
   - 无法抑制协方差增长
   - cov爆炸 → 维度高度相关 → embedding坍缩 → RawUnif恶化
   - 恶性循环

2. **pre_norm_uniform_weight=0.5 可能不足**
   - Epoch 4后RawUnif开始回升（恶化）
   - 说明反坍缩推力不够持久

3. **warmup只有3 epochs**
   - 学习率从0快速上升到1e-4
   - 可能在Epoch 4后进入不稳定区域

4. **skip_l2_norm_training=true 的隐患**
   - 训练时不做L2归一化，embedding在欧氏空间
   - 但VICReg cov在欧氏空间计算，可能被幅度影响

## 历史对比

| 轮次 | RawUnif范围 | cov范围 | 结果 |
|------|-------------|---------|------|
| Round 8 | ~0.5 (已坍缩) | 259-477 | ❌ 完全失败 |
| Round 9 (exp3) | -1.56 | ~0.04 | ✅ 但训练57h未完成 |
| Mini Batch (Epoch 4) | -1.47 ~ -1.97 | 0.53 ~ 10.19 | ⚠️ 短暂好转后恶化 |
| Mini Batch (Epoch 6) | -1.18 ~ -1.40 | 0.39 ~ 33.20 | ❌ 恶化 |

## 已验证无效的改进

| 改进方向 | 结果 | 结论 |
|----------|------|------|
| spatial_weight temporal loss | 优于无加权 | ✅ 保留 |
| skip_l2=false | cov控制最好 | ⚠️ 需配合更高cov权重 |
| high_uniform=1.0 | cov爆炸加剧 | ❌ 无效 |
| high_recon=0.3 | 无明显优势 | ❌ 不重要 |
| high_variance=0.5 | cov仍增长 | ❌ 单独不够 |
| no_teacher | 无改善 | ❌ 无关 |
| combined | 无协同效应 | ❌ 不推荐 |

## 下一步假设

1. **提高covariance_weight到0.01或0.05** — 直接解决cov爆炸
2. **提高pre_norm_uniform_weight到1.0+** — 增强反坍缩持久性
3. **延长warmup到10 epochs** — 更平滑的学习率上升
4. **尝试完全去掉cov项** — 只用variance正则
5. **增加训练量** — 1,000 steps可能不足，需要5,000+
