# 进度日志

## 会话：2026-05-16 Mini Batch Test

### 阶段 1：执行Mini Batch Test
- **状态：** complete
- **开始时间：** 13:07
- **停止时间：** 13:56
- 执行的操作：
  - 停止Round 9，释放8 NPU
  - 预生成6.7GB共享cache（36秒）
  - 创建8个实验配置
  - 并行启动8×1 NPU训练
  - 监控到Epoch 6，发现cov爆炸+RawUnif恶化
  - 停止所有训练
- 创建/修改的文件：
  - configs/mb_exp*.yaml (8个)
  - configs/mini_batch_test*.yaml (4个)
  - configs/mini_batch_patches.json
  - configs/mini_batch_manifest.json
  - docs/FAILED_PARAMETERS_LOG.md
  - src/config.py (patch_list支持)
  - src/data/dataset.py (patch_list支持)
  - logs_archive/ (整理旧log)
  - outputs/logs/ (整理旧log)

### 阶段 2：失败分析
- **状态：** in_progress
- 执行的操作：
  - 对比Epoch 2/4/6指标趋势
  - 诊断cov爆炸+RawUnif恶化的根因
  - 记录失败参数
  - 创建planning文件
- 关键发现：
  - cov_weight=0.001太低，所有实验cov爆炸
  - Epoch 4是转折点，之后RawUnif恶化
  - skip_l2=false(exp3) cov控制最好
  - 空间感知temporal有效

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| Mini Batch 8实验 | 100p, 20ep, 50steps | RawUnif<-2, cov<1 | RawUnif恶化, cov爆炸 | ❌ 失败 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 13:48 | cov爆炸(0.2→33) | 1 | 提高cov_weight到0.01+ |
| 13:48 | RawUnif恶化(-1.97→-1.18) | 1 | 增强反坍缩 + 稳定训练 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段2：失败分析完成，准备设计Round 10 |
| 我要去哪里？ | 阶段3-4：设计并验证新配置 |
| 目标是什么？ | 找到能同时变化检测+高质量embedding的配置 |
| 我学到了什么？ | cov_weight必须>0.001, warmup需更长, skip_l2=false更好 |
| 我做了什么？ | 执行8实验, 诊断失败, 记录教训 |
