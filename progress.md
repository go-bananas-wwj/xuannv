# 进度日志

## 会话：2026-05-16 Mini Batch Test + Round 10 规划

### 阶段 1-2：Mini Batch Test 执行与深度分析
- **状态：** complete
- **开始时间：** 13:07
- **关键里程碑：**
  - 13:07: 启动8×1 NPU并行训练
  - 13:18: Epoch 1完成，所有实验正常
  - 13:26: Epoch 2完成，RawUnif改善到-1.37~-1.55
  - 13:40: Epoch 3-4完成，RawUnif达到最佳(-1.47~-1.97)
  - 13:48: Epoch 4后RawUnif开始恶化！
  - 13:56: Epoch 6，cov爆炸(22~33)，RawUnif恶化到-1.18~-1.40
  - 13:56: **停止所有训练**
- **关键发现（14:02）：**
  - `ddp_v7_trainer` 中 `vicreg_weight=0.0`
  - 配置中的 `covariance_weight: 0.001` 对该trainer**完全无效**
  - **VICReg根本没参与训练！**
  - cov爆炸只是统计现象，反映embedding缺乏去相关约束
- 创建的文件：
  - docs/FAILED_PARAMETERS_LOG.md
  - task_plan.md / findings.md / progress.md

## 测试结果
| 测试 | 输入 | 预期 | 实际 | 状态 |
|------|------|------|------|------|
| Mini Batch 8实验 | 100p, 20ep, vicreg_weight=0 | RawUnif<-2, cov<1 | RawUnif恶化, cov统计爆炸 | ❌ 失败 |
| 代码审查 | ddp_v7_trainer.py | VICReg参与训练 | vicreg_weight=0, 未参与 | ❌ 配置错误 |

## 错误日志
| 时间戳 | 错误 | 根因 | 解决方案 |
|--------|------|------|----------|
| 13:48 | cov爆炸(0.2→33) | VICReg未参与, 缺乏去相关约束 | 启用vicreg_weight=1.0 |
| 13:48 | RawUnif恶化 | raw_unif不去相关, 维度相关增加 | VICReg去相关 + 强uniform |
| 14:02 | 配置与trainer不匹配 | covariance_weight对ddp_v7无效 | 使用vicreg_weight + lambda_cov |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段2完成，等待用户审批Round 10方案 |
| 我要去哪里？ | 阶段3-4：执行Round 10验证 |
| 目标是什么？ | 找到vicreg参与后能有效反坍缩的配置 |
| 我学到了什么？ | vicreg_weight=0是根本原因，skip_l2=false有帮助 |
| 我做了什么？ | 8实验训练, 深度诊断, 发现配置错误 |
