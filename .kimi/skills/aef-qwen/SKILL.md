---
name: aef-qwen
description: AlphaEarth Foundations 改进版 (xuannv_embdding) 项目专家知识 — 解决嵌入坍缩与时间敏感性问题
---

# xuannv_embdding 项目专家指南

## 项目定位

- **项目路径**: `/workspace/xuannv/`
- **原版路径**: `/workspace/AEF/` (⚠️ 严禁修改)
- **核心目标**: 让 AEF 模型对**时间敏感**，从而支持**变化检测**
- **两大改进**:
  1. **反嵌入坍缩**: 训练时 skip L2 norm，使用 `raw_uniformity_loss` (欧氏空间，自适应 t=2/D)
  2. **时间敏感性**: 训练时随机裁剪 valid_period 窗口 (4~24 帧)，V3 进一步引入双窗口不重叠采样 + 时序对比损失

## 关键实验结果

| 版本 | 配置 | 时间敏感 (cos_sim) | 变化检测 AUC | 状态 |
|------|------|-------------------|--------------|------|
| V1 | `qwen_v1_scenes.yaml` | ~0.97 (不敏感) | ~0.498 (随机) | 基线 |
| V2 | `qwen_v2_temporal.yaml` | 改善中 | 改善中 | 加入时序对比损失 |
| V3 | `qwen_v3_temporal.yaml` | 双窗口采样 | 待验证 | 在 V2 上微调 |
| ExpD | `xuannv_v2_expD_7target_lowrecon.yaml` | 0.578 (cosine) | 0.730 (LR) | 7-target + skip L2 + high uniform |

## 模块速查表

| 文件 | 职责 | 修改频率 |
|------|------|---------|
| `src/models/bottleneck.py` | **VMF Bottleneck** — 训练 skip L2 / 推理 L2 | 低 (核心已稳定) |
| `src/training/losses.py` | **损失函数** — raw_uniformity, decorrelation, variance, orthogonality, temporal_contrastive | 中 |
| `src/data/dataset.py` | **数据集** — 3 输入源 (S2/S1/Landsat), 7 目标源, 时序窗口增强, 双窗口采样 | 中 |
| `src/models/model.py` | **AEFModel** — encode_frames, forward, encode_dual_window | 低 |
| `src/training/trainer.py` | **DDPTrainer** — 损失组合、checkpoint 保存逻辑 | 中 |
| `src/config.py` | YAML 配置加载 | 低 |
| `validate_v2.py` | 在 105 个光学标注上计算 AUC，对比 V1/V2 | 低 |
| `demo/app.py` | Gradio Demo (4 Tabs: 嵌入可视化 / 变化热力图 / 异常检测 / 训练曲线) | 低 |

## 数据设计 (严格对齐论文)

- **输入源 (Input)**: `S2` + `S1` + `Landsat` (仅带时间戳的图像帧)
- **目标源 (Target)**: 输入 3 类 + `DEM` + `WorldCover` + `Dynamic World` + `JRC Water` = 7 类
- **静态数据规则**: DEM/WorldCover/JRC 不作为模型输入，仅作为重建目标
- **文件名兼容**: 支持 `YYYYMMDD` 单景 和 `YYYYQN` 季度格式
- **数据路径**: `/workspace/raw/harbin_scenes`
- **统计路径**: `/workspace/statistics/harbin_scenes`

## 训练启动规范

### V1 基线训练
```bash
cd /workspace/xuannv
export CUDA_VISIBLE_DEVICES=5,6,7
torchrun --nproc_per_node=3 scripts/train_ddp.py \
  --config configs/qwen_v1_scenes.yaml \
  --save-every 50 --warmup-epochs 10
```

### V3 接续训练 (从 V2 checkpoint 继续)
```bash
cd /workspace/xuannv
export CUDA_VISIBLE_DEVICES=5,6,7
torchrun --nproc_per_node=3 scripts/train_ddp.py \
  --config configs/qwen_v3_temporal.yaml \
  --resume /workspace/outputs/xuannv_embdding_v2/epoch_499.pt \
  --save-every 50 --warmup-epochs 5
```

### 关键训练参数
- `batch_size=2` (每 GPU)，`gradient_accumulation_steps=4`
- `max_frames=32`, `image_size=128`
- `embedding_dim=128`, `precision_dim=256`, `num_blocks=8`
- `skip_l2_norm_training=true` (必须开启)
- `vmf_kappa=2000.0`

## 反坍缩四件套 (核心损失组合)

在 `src/training/trainer.py` 中，以下损失在 **pre-norm 空间** 计算：

1. `raw_uniformity_loss(gathered_pre_norm)` — 权重 `uniformity_weight`
2. `decorrelation_loss(gathered_pre_norm)` — 权重 `decorrelation_weight`
3. `variance_regularizer(gathered_pre_norm)` — 权重 `variance_weight`
4. `bottleneck_orthogonality_loss(bnw)` — 权重 `orthogonality_weight`

**良好分散指标**: `raw_unif ≈ -4.0`，`pre_unif ≈ -4.0`

## 时序对比损失

- **V1**: `temporal_contrastive_weight = 0` (关闭)
- **V2/V3**: 开启，使用 `temporal_contrastive_loss(emb_w1, emb_w2)`
- **emb_w1/emb_w2 来源**: `encode_dual_window()` 对同一 batch 用两个不同 `valid_period` 分别编码
- **V3 改进**: 数据集中启用 `non_overlapping_windows=true`，随机抽取两段不重叠的时间窗口

## 验证 Pipeline (推荐)

### 标准验证流程

训练进行中时，**不要停止训练**，直接在空闲 NPU 上并行验证：

```bash
# 1. 提取 Embedding（基础步骤，必须先做）
python scripts/eval/extract_embeddings_v2.py \
    --config configs/xuannv_v2_expD_7target_lowrecon.yaml \
    --checkpoint /workspace/outputs/.../epoch_best_xxx.pt \
    --output-dir /workspace/outputs/.../evaluation/embeddings \
    --device npu:0 \
    --batch-size 4

# 2. Bare CD AUC（零样本基线）
python scripts/eval/evaluate_cd_v2.py \
    --embedding-file /workspace/outputs/.../evaluation/embeddings/patch_embeddings.npz \
    --output-dir /workspace/outputs/.../evaluation/change_detection

# 3. MLP 下游分类
python scripts/eval/evaluate_mlp_v2.py \
    --embedding-file /workspace/outputs/.../evaluation/embeddings/patch_embeddings.npz \
    --output-dir /workspace/outputs/.../evaluation/mlp_downstream \
    --device npu:0 \
    --epochs 50

# 4. Few-Shot CD（冻结 backbone，训练 CD Head）
python scripts/eval/fewshot_change_detection.py \
    --config configs/xuannv_v2_expD_7target_lowrecon.yaml \
    --checkpoint /workspace/outputs/.../epoch_best_xxx.pt \
    --k-shots 1,5,10,20 \
    --n-splits 5 \
    --device npu:0

# 5. Few-Shot 下游分类（基于 embedding，少量像素训练）
python scripts/eval/fewshot_downstream_from_embedding.py \
    --embedding-file /workspace/outputs/.../evaluation/embeddings/patch_embeddings.npz \
    --k-pixels 100,1000,10000,100000 \
    --n-splits 3 \
    --device npu:0
```

### 验证注意事项

| 注意点 | 说明 |
|--------|------|
| **不要停止训练** | 验证和训练并行，用空闲 NPU |
| **batch_size ≤ 4** | NPU 单卡推理时 batch_size 不能太大，否则 Conv2D 内存分配失败 |
| **设备 ID 映射** | 设置 `ASCEND_RT_VISIBLE_DEVICES=7` 后，物理 NPU 7 映射为逻辑 `npu:0` |
| **KNN 不用做** | sklearn KNN 在 142 万像素上极慢（CPU 单线程 20+ 分钟），且结果通常比 MLP 差 |

### 评估脚本说明

| 脚本 | 输入 | 输出 | 耗时 | 是否必须 |
|------|------|------|------|---------|
| `extract_embeddings_v2.py` | config + checkpoint | `patch_embeddings.npz` [N,12,D,H,W] | ~40min (batch=4) | ✅ 必须先做 |
| `evaluate_cd_v2.py` | embedding 文件 | AUC (cosine + LR) | ~5min | ✅ 核心指标 |
| `evaluate_mlp_v2.py` | embedding 文件 | Acc/mIoU (3 tasks) | ~15min | ✅ 下游语义 |
| `fewshot_change_detection.py` | config + checkpoint | Few-Shot CD AUC | ~30min | ⭐ 推荐 |
| `fewshot_downstream_from_embedding.py` | embedding 文件 | Few-Shot Acc | ~20min | ⭐ 推荐 |

## 验证变化检测流程

1. 使用 `validate_v2.py`
2. 默认对比两个 checkpoint:
   - V1: `/workspace/outputs/xuannv_embdding_v1/epoch_399.pt`
   - V2: `/workspace/outputs/xuannv_embdding_v2/epoch_499.pt`
3. 时间窗口（哈尔滨标注为2025年变化）:
   - june.shp: 2025-04 vs 2025-06
   - aug.shp: 2025-06 vs 2025-08
   - September.shp: 2025-08 vs 2025-09
   - October.shp: 2025-09 vs 2025-10
4. 计算 `change_map = (1 - cos_sim) / 2.0`
5. 在 20 个有标注的 patch 上计算 ROC-AUC

## Demo 启动

```bash
cd /workspace/xuannv
export CUDA_VISIBLE_DEVICES=5,6,7
python3 -u demo/app.py --port 7868
```

## 禁忌事项 ❌

1. **永远不要修改 `/workspace/AEF/` 下的任何代码**
2. **不要关闭 `skip_l2_norm_training`** (否则梯度屏障会重新出现)
3. **不要将静态源 (DEM/WorldCover/JRC) 加入输入源** (违反论文设计)
4. **不要期望分类损失有效** (label 全为零，classification_weight 设为 0)

## 调试排查速查

| 现象 | 可能原因 | 排查动作 |
|------|---------|---------|
| AUC ≈ 0.5 | 模型对时间不敏感 | 检查 `cos_sim` 是否接近 1.0；确认 temporal_contrastive_weight > 0 |
| RawUniformity ≈ 0 | 嵌入坍缩 | 检查 skip_l2_norm_training；确认反坍缩四件套权重非零 |
| Reconstruction 极高 | 数据加载失败 / NaN 过多 | 检查 stats_dir 路径；查看 target_mask 是否为全 False |
| DDP 卡住 | 某个进程数据加载异常 | 检查 num_workers；尝试单卡 `CUDA_VISIBLE_DEVICES=5 python3 scripts/train_ddp.py` |
| checkpoint 加载失败 | 键不匹配 | 确认 resume 的 checkpoint 与当前 config 的模型结构一致 |
| `Memory_Allocation_Failure` | batch_size 过大 | 将 batch_size 降到 4 或更小 |
| `Invalid device ID` | ASCEND_RT_VISIBLE_DEVICES 设置后设备映射错误 | 设置 `VISIBLE_DEVICES=7` 后用 `npu:0`，不要再用 `npu:7` |

## 常用 shell 命令

```bash
# 监控训练日志
tail -f /workspace/outputs/xuannv_embdding_v1/training.log

# 快速单卡测试
CUDA_VISIBLE_DEVICES=5 python3 scripts/train_ddp.py --config configs/qwen_v1_scenes.yaml

# 查看已保存 checkpoint
ls -lah /workspace/outputs/xuannv_embdding_v1/*.pt

# 运行变化检测验证
python3 validate_v2.py
```
