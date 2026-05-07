# xuannv_embdding — Agent 上下文

## 项目背景

这是 **AlphaEarth Foundations 改进版**，从零实现于 `/workspace/xuannv/`。
核心使命：**解决嵌入坍缩 + 提升时间敏感性**，使模型能够执行变化检测。

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

## ★ 当前数据状态与训练准备（2026-05-08）

> **详细文档见根目录 `AGENTS.md` 的「当前数据状态与训练准备」章节。**
> 任何接手训练的 Agent **必须**先阅读该章节。

### 关键摘要

- **数据目录**: `/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered/`
- **S2 数据**: 云筛选后，每月保留最 clear 的 2 帧，平均 ~22 帧/patch
- **配置路径**: 5 个配置文件 (`qwen_v1_scenes.yaml` 等) 的 `manifest_path` 已指向云筛选目录
- **统计数据**: `/workspace/statistics/harbin_scenes/`（7 个 JSON，S2 已重新计算）
- **缓存**: 27.7GB 新缓存已生成，旧缓存已清理
- **已知问题**: Landsat 缺失 17 patch（代码已优雅处理），357 个冬季月份 fallback

### 训练启动（V1 基线）

```bash
cd /workspace/xuannv
conda activate xuannv
npu-smi info  # 检查空闲 NPU
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp.py --config configs/qwen_v1_scenes.yaml \
    --save-every 50 --warmup-epochs 10
```

### 必须监控的指标

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `raw_unif` | -4.0 ~ -1.0 | > -0.5 表示坍缩，训练失败 |
| `recon` | < 0.3 | warmup 后仍 > 0.5 检查数据 |
| `var_reg` | 接近 0 | > 0.5 方差坍缩 |

**如果 raw_unif > -0.5 且持续 5 个 epoch 不下降，立即报告。**

## Agent 行为准则

- **与用户交流时必须使用中文回复**。
- 处理训练/模型相关任务前，先读取对应版本的 config YAML。
- 修改损失函数或训练逻辑时，确保 `gathered_pre_norm` 被正确使用。
- 调试 AUC 低时，优先检查 temporal contrastive loss 是否生效、双窗口数据是否正确生成。
- 所有文件操作限制在 `/workspace/xuannv/` 内。
- **硬件资源约束：运行训练、推理或预计算任务前，请先检查 GPU/NPU 占用情况，选择空闲 GPU/NPU。必要时手动设置 `CUDA_VISIBLE_DEVICES` 或 `ASCEND_RT_VISIBLE_DEVICES`。**
