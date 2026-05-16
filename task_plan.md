# 任务计划：Round 10 迭代方案

## 目标
找到能同时做到 **变化检测 AUC > 0.65** + **高质量 Embedding (RawUnif < -2.0)** 的训练配置。

## 核心发现（Mini Batch Test 失败分析）

### 根本原因
**VICReg 根本没参与训练！** `ddp_v7_trainer` 中 `vicreg_weight=0.0`，配置中的 `covariance_weight` 和 `variance_weight` 对该 trainer 完全无效。

实际生效的反坍缩措施只有 `raw_unif`（推分散但不去相关），导致：
- 维度高度相关 → cov 统计值爆炸
- Epoch 4后 embedding 开始坍缩 → RawUnif 恶化

### Round 9(exp3) cov=0.043 的原因
- `skip_l2=false` → L2归一化自然限制协方差
- 大数据集(424p)统计更稳定

## Round 10 迭代方案

### 实验设计（3个配置，8 NPU并行）

| # | 名称 | 关键参数 | 理论依据 |
|---|------|----------|----------|
| 1 | **mb_v2_baseline** | vicreg_weight=1.0, lambda_cov=1.0, lambda_var=1.0, skip_l2=false, temporal_spatial=true | 启用VICReg去相关 + L2归一化双重保障 |
| 2 | **mb_v2_strong_cov** | vicreg_weight=1.0, lambda_cov=5.0, lambda_var=1.0, skip_l2=false, temporal_spatial=true | 5倍去相关力度，更强约束 |
| 3 | **mb_v2_uniform_cov** | vicreg_weight=1.0, lambda_cov=1.0, lambda_var=1.0, pre_norm_uniform=1.0, skip_l2=false, temporal_spatial=true | 强反坍缩 + 中等去相关 |

### 共同基础参数
- 数据: 100 patches (与Mini Batch相同)
- epochs: 20, steps/epoch: 50, batch: 4
- warmup: 5 epochs (比3更平滑)
- reconstruction: 0.1, temporal: 0.5
- save_every: 5

### 监控指标
| 指标 | 及格 | 良好 | 优秀 |
|------|------|------|------|
| RawUnif | < -2.0 | < -3.0 | < -4.0 |
| cov | < 1.0 | < 0.5 | < 0.1 |
| Recon | < 0.3 | < 0.2 | < 0.15 |
| 每epoch趋势 | 持续改善 | — | — |

### 终止条件
- 如果 Epoch 5 前 cov > 5 且持续增长 → 该实验失败
- 如果 RawUnif 连续 2 个epoch恶化 → 该实验失败
- 否则训练到 Epoch 20

## 当前阶段
阶段 3 准备中，等用户审批

## 各阶段

### 阶段 1-2：Mini Batch Test 执行与分析
- [x] 8实验并行训练到Epoch 6
- [x] 发现VICReg未参与训练
- [x] 诊断cov爆炸根因
- **状态：** complete

### 阶段 3：Round 10 配置准备
- [ ] 用户审批方案
- [ ] 创建3个新配置
- [ ] 预生成cache
- **状态：** pending

### 阶段 4：Round 10 训练与监控
- [ ] 启动3×2 NPU 或 3×1 NPU
- [ ] 监控到Epoch 20或提前终止
- [ ] 每5 epoch评估AUC
- **状态：** pending

### 阶段 5：结论
- [ ] 对比3实验结果
- [ ] 确定最优配置
- **状态：** pending

## 已做决策
| 决策 | 理由 |
|------|------|
| 停止Mini Batch Test | VICReg未参与训练，继续无意义 |
| 启用vicreg_weight=1.0 | 让VICReg真正参与去相关 |
| 保持skip_l2=false | Round 9已验证有效限制cov |
| 延长warmup到5 | 3ep太短，学习率冲击导致不稳定 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| cov爆炸(0.2→33) | 1 | 启用VICReg + skip_l2=false |
| RawUnif恶化(-1.97→-1.18) | 1 | VICReg去相关 + 强uniform |
| 配置参数与trainer不匹配 | 1 | 明确vicreg_weight/lambda_cov的作用 |
