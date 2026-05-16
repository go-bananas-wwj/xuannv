# 任务计划：Mini Batch Test 失败分析与 Round 10 迭代方案

## 目标
找到能同时做到 **变化检测 AUC > 0.65** + **高质量 Embedding (RawUnif < -2.0)** 的训练配置，并在 Mini Batch 上快速验证。

## 当前阶段
阶段 2

## 各阶段

### 阶段 1：Mini Batch Test 执行与监控
- [x] 构建100-patch子集
- [x] 预生成共享cache
- [x] 创建8个实验配置
- [x] 并行启动8×1 NPU训练
- [x] 监控到Epoch 6，发现cov爆炸+RawUnif恶化
- [x] 停止训练
- [x] 记录失败参数到 docs/FAILED_PARAMETERS_LOG.md
- **状态：** complete

### 阶段 2：失败原因深度分析
- [ ] 对比Round 9(exp3)与Mini Batch的关键差异
- [ ] 调研VICReg协方差爆炸的根本原因
- [ ] 分析数据量、batch size、skip_l2对cov的影响
- [ ] 形成诊断结论和下一步假设
- **状态：** in_progress

### 阶段 3：Round 10 迭代方案设计
- [ ] 设计3-4个新实验配置（基于分析结论）
- [ ] 修改cov权重、uniform权重、warmup等关键参数
- [ ] 创建配置并验证
- **状态：** pending

### 阶段 4：Round 10 快速验证
- [ ] 预生成cache
- [ ] 并行启动实验
- [ ] 监控到Epoch 10-15
- [ ] 评估AUC和embedding质量
- **状态：** pending

### 阶段 5：结论与交付
- [ ] 对比所有实验结果
- [ ] 确定最优配置
- [ ] 撰写完整报告
- **状态：** pending

## 关键问题
1. 为什么covariance_weight=0.001在Round 9(exp3)上cov=0.043，但在Mini Batch上cov爆炸到33？
2. skip_l2_norm=true vs false 对cov计算的本质影响是什么？
3. 100-patch子集的数据量是否不足以稳定估计VICReg cov？
4. 除了VICReg，还有什么方法可以防止embedding坍缩？

## 已做决策
| 决策 | 理由 |
|------|------|
| 停止Mini Batch Test | cov失控+RawUnif恶化，继续训练浪费算力 |
| 记录失败参数 | 避免重复踩坑，积累知识 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| cov爆炸(0.2→33) | 1 | 待解决：提高cov权重或去掉cov项 |
| RawUnif恶化(-1.97→-1.18) | 1 | 待解决：增强反坍缩力度 |
