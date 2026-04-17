# AEF_qwen — Agent 上下文

## 项目背景

这是 **AlphaEarth Foundations 改进版**，从零实现于 `/workspace/xuannv/`。
核心使命：**解决嵌入坍缩 + 提升时间敏感性**，使模型能够执行变化检测。

**原版代码位于 `/workspace/AEF/`，任何情况下不要修改。**

## 核心设计决策

1. **输入严格对齐论文**: 只有 `S2`、`S1`、`Landsat` 三类时序图像作为输入。
2. **静态数据仅作目标**: `DEM`、`WorldCover`、`Dynamic World`、`JRC Water` 只参与重建，不输入给 encoder。
3. **训练时跳过 L2 Norm**: `VMFBottleneck(skip_l2_training=true)`，在 pre-norm 空间计算所有反坍缩损失。
4. **推理时标准 L2 + VMF**: 保证 embedding 在球面上。
5. **时间窗口增强**: 训练时随机裁剪 valid_period；V3 引入不重叠双窗口 + 时序对比损失。

## 文件结构

- `src/models/bottleneck.py` — 核心改进 bottleneck
- `src/training/losses.py` — raw_uniformity + 时序对比损失
- `src/data/dataset.py` — 双窗口采样逻辑
- `src/training/trainer.py` — DDP 训练器与损失组合
- `configs/qwen_v*.yaml` — 训练配置
- `validate_v2.py` — 变化检测 AUC 验证
- `demo/app.py` — Gradio 可视化 Demo

## Agent 行为准则

- 处理训练/模型相关任务前，先读取对应版本的 config YAML。
- 修改损失函数或训练逻辑时，确保 `gathered_pre_norm` 被正确使用。
- 调试 AUC 低时，优先检查 temporal contrastive loss 是否生效、双窗口数据是否正确生成。
- 所有文件操作限制在 `/workspace/xuannv/` 内。

## GitHub 仓库同步规则

- **远程仓库**: `git@github.com:go-bananas-wwj/xuannv.git` (私密仓库)
- **主分支**: `main`
- **强制要求**: 每次对代码/配置/文档做任何修改后，**必须**执行 `git add -A && git commit -m "描述" && git push origin main` 同步到远端。
- **提交信息规范**: 用中文或英文简明描述改动内容，包含改动的文件和目的。
- **禁止**: 任何本地修改不同步就结束任务。
