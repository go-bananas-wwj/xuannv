# AEF 对齐实验 — 任务计划

## 目标
将 xuannv embedding 模型对齐 AEF 原文，解决 embedding temporal blindness（Bare AUC ≈ 0.54）。

## 当前状态
- Branch: `v12-clean-dynamic`
- 8 NPU 全部空闲
- 已确认变化检测 mask 合理（变化比例 0.056%-0.200%）
- v2 8 实验全部失败（AUC ≈ 0.5）

## 已对齐的决策

### 训练方案
- **第一阶段**: 20 epochs × 50 steps，8 实验并行快速验证
- **第二阶段**: 对前 2 个最佳配置，扩展长训验证
- **Checkpoint**: 保存完整权重 + 优化器状态，支持续训
- **待确认**: 是否需要保存 scheduler step 以实现平滑续训 LR schedule

### 评估方法
- **Bare AUC**: 像素级 cosine distance → ROC-AUC
- **多时期评估**: 4月→6月, 6月→8月, 8月→9月, 9月→10月，分别计算后取平均
- **标注来源**: 4个shapefile（june, aug, September, October）
- **评估脚本**: `scripts/eval/validate_aef_bare.py`（已修复为2025时间窗口）

## 阶段

### 阶段 1: 对齐检查与问题诊断 ✅
**状态**: 完成
**输出**: 识别出 10+ 个不对齐点（见 findings.md）

### 阶段 2: 创建 Unified Trainer + 8 实验配置 ✅
**状态**: 完成
**输出**:
- `src/models/decoders.py` — 恢复 ConditionInjector 条件注入
- `src/models/model.py` — 传递时间条件给 decoder
- `src/training/ddp_unified_trainer.py` — AEF 对齐 unified trainer (仅 Recon+BatchUnif+Consist)
- `scripts/train/train_unified.py` — 训练入口
- `configs/aef_*.yaml` — 8 个实验配置
- `scripts/train/launch_aef_8gpu.sh` — 并行启动脚本

### 阶段 3: 并行训练 🔄
**状态**: 进行中
**已启动**: 8 个 tmux session，每卡 1 个实验
**监控**: `tmux list-sessions | grep aef_`
**目标**:
1. 修改 bottleneck: `skip_l2_training=false`（对齐 AEF 训练时 L2+VMF）
2. 修改 decoder: 恢复条件注入（时间条件传递）
3. 修改 target_sources: 降低 static 权重至 AEF 水平（~22%）
4. 修改配置: 64D embedding, 100 patches, 2025-only
5. 移除 AEF 不用的损失: VICReg, Decorr, Orth, CLS
6. 创建 8 个并行实验配置
7. 创建统一训练脚本

### 阶段 3: 并行训练
**状态**: 待开始
**目标**: 8 卡各跑 1 个实验，监控指标

### 阶段 4: 评估与对比
**状态**: 待开始
**目标**: 提取 embedding → 计算 Bare AUC → 对比 8 实验

## 关键决策
- **64D embedding**: 对齐 AEF
- **100 patches**: 快速验证
- **严格 2025-only**: 消除跨 year 混淆
- **Batch size 4**: 保持与 v2 一致
- **Epochs 20**: 快速迭代

## 实验矩阵

| # | 实验名 | 核心变量 | Consistency | Static 权重 | Skip L2 | Decoder 条件 |
|---|--------|----------|-------------|-------------|---------|-------------|
| 1 | aef_baseline | AEF 完全对齐 | 0.02 | 0.1 | false | 恢复 |
| 2 | aef_high_consist | 高一致性 | 0.05 | 0.1 | false | 恢复 |
| 3 | aef_no_static | 无 static | 0.02 | 0.0 | false | 恢复 |
| 4 | aef_skip_l2 | Skip L2 ( ours ) | 0.02 | 0.1 | true | 恢复 |
| 5 | aef_simple_decoder | 简单 decoder | 0.02 | 0.1 | false | 禁用 |
| 6 | aef_128d | 128D 对比 | 0.02 | 0.1 | false | 恢复 |
| 7 | aef_high_kappa | Kappa=2000 | 0.02 | 0.1 | false | 恢复 |
| 8 | aef_batch_unif_only | 仅 BatchUnif+Recon+Consist | 0.02 | 0.1 | false | 恢复 |
