# xuannv_embdding — Agent 上下文

## 项目概述

**xuannv_embdding** (包名 `aef-qwen`) 是 **AlphaEarth Foundations (AEF)** 的独立改进版，从零实现于 `/workspace/xuannv/`。

- **核心使命**: 解决嵌入坍缩 (embedding collapse) 与提升时间敏感性 (temporal sensitivity)，使模型能够执行变化检测 (change detection)。
- **与原版关系**: 本项目完全独立实现，仅参考 AEF 接口设计（数据加载协议、模型输入输出格式）。
- **语言与注释**: 项目内代码注释、文档、配置均以中文为主。

### 核心设计决策

1. **输入严格对齐论文**: 只有 `S2`、`S1`、`Landsat` 三类时序图像作为输入。
2. **静态数据仅作目标**: `DEM`、`WorldCover`、`Dynamic World`、`JRC Water` 只参与重建，不输入给 encoder。
3. **训练时跳过 L2 Norm**: `VMFBottleneck(skip_l2_training=true)`，在 pre-norm 空间计算所有反坍缩损失。
4. **推理时标准 L2 + VMF**: 保证 embedding 在球面上。
5. **时间窗口增强**: 训练时随机裁剪 valid_period；V3+ 引入不重叠双窗口 + 时序对比损失。

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python >= 3.9 |
| 深度学习 | PyTorch >= 2.0 |
| 数据 I/O | rasterio, geopandas, numpy |
| 配置 | PyYAML |
| 构建 | setuptools (`pyproject.toml`) |
| Demo | Gradio + matplotlib + folium |
| 推理加速 | torch.autocast (fp16/bf16), gradient checkpointing |
| 分布式 | torchrun + DDP (nccl) |

安装命令:
```bash
cd /workspace/xuannv
pip install -e .
```

**注意**: 项目中没有任何 linting/formatting 配置文件（无 black、flake8、isort、pre-commit 等）。代码风格由开发者手动保持一致。

## 代码组织结构

```
xuannv_embdding/
├── pyproject.toml              # 包配置 (setuptools)
├── configs/
│   └── qwen_v*.yaml            # 训练配置 (支持 _base_ 继承)
├── src/                        # 核心源码包
│   ├── config.py               # YAML -> dataclass 配置系统
│   ├── data/
│   │   ├── dataset.py          # HarbinPatchDataset (3输入/7目标, 内存预加载)
│   │   ├── transforms.py       # 归一化、TIFF读取、时间戳解析
│   │   └── builder.py          # DataLoader 工厂 (DistributedSampler)
│   ├── models/
│   │   ├── model.py            # AEFModel 主模型 (编码→瓶颈→解码)
│   │   ├── bottleneck.py       # VMFBottleneck (训练skip L2, 推理L2+VMF)
│   │   ├── blocks.py           # STPBlock (三路径: Time/Space/Precision)
│   │   ├── sensor_encoders.py  # SensorEncoderBank (多源独立stem)
│   │   ├── decoders.py         # ContinuousDecoder / CategoricalDecoder
│   │   ├── heads.py            # ChangeDetectionHead V1/V2/V3
│   │   ├── downstream_heads.py # PixelMLPHead / PixelConvHead
│   │   └── time_encoding.py    # TimeCode / WindowCode / RelativeTimeCode
│   ├── training/
│   │   ├── losses.py           # raw_uniformity + 时序对比损失全家桶
│   │   ├── trainer.py          # V1 DDP 训练器
│   │   ├── ddp_v4_trainer.py   # V4 DDP 训练器
│   │   ├── ddp_v5_mixed_scale_trainer.py   # V5 混合尺度
│   │   ├── ddp_v6_enhanced_temporal_trainer.py  # V6 增强时序
│   │   ├── ddp_v6_5_gap_aware_trainer.py        # V6.5 gap-aware
│   │   ├── single_gpu_trainer.py
│   │   ├── optimizer.py        # AdamW + cosine warmup scheduler
│   │   └── loops.py            # 重建损失计算辅助
│   ├── inference/
│   │   └── engine.py           # 统一推理引擎 (load_backbone / load_cd_head)
│   └── utils/
│       ├── checkpoint.py       # load_checkpoint / save_checkpoint
│       ├── device.py           # get_device
│       ├── io.py               # IO 辅助
│       └── logging.py          # 日志工具
├── scripts/
│   ├── train/                  # DDP 训练入口 (train_ddp.py / train_ddp_v*.py)
│   ├── eval/                   # AUC 验证、benchmark、embedding 分析
│   ├── inference/              # 月度 embedding 提取
│   ├── visualize/              # 训练诊断、结果可视化
│   └── train_v*_downstream_*.py  # 下游任务训练 (MLP/ConvHead)
├── demo_v2/                    # Gradio 可视化 Demo
│   ├── app.py                  # Demo 主入口
│   ├── config.py               # 模型注册表查询
│   ├── cache_manager.py        # 预计算结果缓存
│   ├── engines/                # 推理引擎封装
│   ├── components/             # 各 Tab UI 组件
│   └── utils/                  # 常量、可视化、地图工具
└── archive/                    # 废弃/实验代码 (只读参考)
```

## 配置系统

所有训练参数通过 `configs/qwen_v*.yaml` 管理，支持继承机制:

```yaml
_base_: qwen_v5_mixed_scale.yaml   # 先加载基础配置，再覆盖
experiment:
  name: aef_qwen_v6_enhanced_temporal
```

加载方式:
```python
from src.config import load_config
cfg = load_config("configs/qwen_v1_scenes.yaml")
```

关键配置段:
- `experiment`: 名称、种子、输出目录 (`/workspace/outputs/{name}`)
- `data`: batch_size, image_size, max_frames, 时序增强参数, 窗口模式
- `model`: embedding_dim, num_blocks, vmf_kappa, skip_l2_norm_training
- `training`: 学习率、损失权重、warmup、检查点保存频率
- `pretrained`: 预训练权重路径 (可选，由训练脚本 `--resume` 或 `--soft-restart` 覆盖)

## 训练系统

### 训练入口

各版本对应独立的训练脚本和训练器类:

| 版本 | 训练脚本 | 训练器类 | 关键特性 |
|------|----------|----------|----------|
| V1 | `scripts/train/train_ddp.py` | `DDPTrainer` | 反坍缩基线 |
| V4 | `scripts/train/train_ddp_v4.py` | `DDPv4Trainer` | 官方对齐 |
| V5 | `scripts/train/train_ddp_v5.py` | `DDPv5MixedScaleTrainer` | 混合尺度双窗口 |
| V6 | `scripts/train/train_ddp_v6.py` | `DDPv6EnhancedTemporalTrainer` | 像素级时序损失 |
| V6.5 | `scripts/train/train_ddp_v6_5.py` | `DDPv6_5GapAwareTrainer` | gap-aware temporal |

启动示例 (V1):
```bash
cd /workspace/xuannv
torchrun --nproc_per_node=2 \
    scripts/train/train_ddp.py --config configs/qwen_v1_scenes.yaml \
    --save-every 50 --warmup-epochs 10
```

启动示例 (V6.5, 从 checkpoint 软重启):
```bash
torchrun --nproc_per_node=2 \
    scripts/train/train_ddp_v6_5.py --config configs/qwen_v6_5_gap_aware.yaml \
    --soft-restart /workspace/outputs/.../epoch_best_xxx.pt \
    --save-every 20
```

### 训练监控指标

训练中应关注以下指标 (由 trainer 打印到 stdout/log):

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `raw_unif` | -4.0 ~ -1.0 | > -0.5 表示坍缩 |
| `pre_unif` | 接近 raw_unif | 差距大说明有问题 |
| `recon` | < 0.3 | > 0.5 重建质量差 |
| `var_reg` | 接近 0 | > 0.5 方差坍缩 |
| `orth` | < 0.3 | > 0.5 权重不正交 |
| `decorr` | < 1.0 | > 2.0 强相关 |

### 损失函数体系

核心损失位于 `src/training/losses.py`:

- **raw_uniformity_loss**: 欧氏空间 uniformity，自适应 t=2/D，梯度永远非零
- **decorrelation_loss**: Barlow Twins 去相关
- **variance_regularizer**: VICReg 方差正则
- **bottleneck_orthogonality_loss**: Conv1x1 权重正交约束
- **temporal_contrastive_loss**: 双窗口 hinge loss (global mean)
- **temporal_cosine_pixel_loss**: 像素级 cosine 时序损失 (V6+)
- **pixel_temporal_info_nce_loss**: 像素级 Anti-Diagonal InfoNCE (V6+)
- **gap_aware_temporal_cosine_loss**: 根据时间 gap 大小动态设定 target (V6.5)
- **reconstruction_loss**: 掩码 L1 (连续) / CE (分类)

### 检查点保存

- 默认保存路径: `/workspace/outputs/{experiment_name}/`
- 定期保存: `epoch_{N}.pt`
- 最佳模型: `epoch_best_epoch{N}.pt` (基于 uniform + recon 平衡)
- 加载时使用 `src.utils.checkpoint.load_checkpoint`，自动处理 `model_state_dict` key。

## 推理与评估

### 统一推理引擎

```python
from src.inference.engine import load_backbone

model, dataset, cfg = load_backbone(
    config_path="configs/qwen_v1_scenes.yaml",
    checkpoint_path="/workspace/outputs/.../epoch_399.pt",
    device="cuda:0",
)
```

### 变化检测验证

主要评估脚本:
- `scripts/eval/validate_v2.py`: V2 变化检测 AUC (105 个光学标注)
- `scripts/eval/validate_v4_official.py`: V4 官方验证
- `scripts/eval/validate_v5_level1_bare.py`: V5 Level-1 裸验证
- `scripts/eval/benchmark_v5_monthly_cd_head.py`: V5 月度 CD Head benchmark

AUC 验证的基本流程:
1. 加载模型 checkpoint
2. 对 before/after 时间窗口分别提取 embedding
3. 计算 cosine distance 或经 CD Head 得到变化概率
4. 与 shapefile 标注对比计算 ROC-AUC

### Embedding 提取

```bash
# 提取所有 patch 的月度 embedding
python scripts/inference/extract_monthly_embeddings_all_patches.py \
    --gpu_idx 0 --total_gpus 2
```

## Demo 系统

Gradio 可视化平台 (`demo_v2/app.py`)，提供以下 Tab:
- Project Introduction
- Data & Embedding Field (数据浏览)
- Spatial Anomaly (空间异常)
- Change Detection (变化检测)
- Three-Type CD (建筑工地/房屋拆除/非农非粮)
- Downstream Tasks (下游任务)
- Model Performance Analysis
- Model Comparison
- AlphaEarth Official

启动:
```bash
cd /workspace/xuannv
python demo_v2/app.py --port 7868
# 或使用脚本:
bash start_gradio.sh
```

Demo 依赖预计算的 embedding maps (`embedding_maps.npy` + `patch_ids.json`)，由 `demo_v2/precompute_cd.py` 生成。

## 开发规范

### 代码风格

- 使用 `from __future__ import annotations` 支持 Python 3.9+ 类型注解
- 类型注解尽量完整，特别是 Tensor shape 在 docstring 中注明
- 模块内注释使用中文
- 使用 `pathlib.Path` 处理路径
- 设备选择统一走 `src.utils.device.get_device`
- checkpoint 统一走 `src.utils.checkpoint.load_checkpoint / save_checkpoint`
- 训练脚本顶部通常设置 `torch.set_num_threads(4)`
- 训练脚本常通过 `sys.path.insert(0, "/workspace/xuannv")` 确保模块导入

### 数据规范

- **输入源** (3类): `s2`, `s1`, `landsat`
- **目标源** (7类): `s2`, `s1`, `landsat`, `dem`, `worldcover`, `dynamic_world`, `jrc_water`
- 时间戳兼容: `YYYYMMDD` (单景) 和 `YYYYQN` (季度)
- 归一化: 光学 `log(x+1)/10 → z-score → ±6σ clip`；SAR `clip[-30,10]dB → z-score`
- 分类源 (WorldCover/DynamicWorld): one-hot 编码

### 模型前向接口

`AEFModel.forward` 参数 (必须按名传参):
- `source_frames`: [B, S, T, C, H, W] 或 [B, T, C, H, W]
- `source_timestamps_ms`, `source_frame_mask`, `source_input_mask`, `source_type_ids`
- `valid_start_ms`, `valid_end_ms`: 定义当前窗口的有效时间范围
- `target_relative_time`, `target_metadata`
- `target_loss_type`, `target_source_idx`: 可选，用于 per-source decoder 路由

输出 `AEFOutput` 字段:
- `embedding_map`: [B, D, H, W] (推理时 L2 normalized)
- `embedding`: [B, D] (全局 mean)
- `pre_norm_embedding`: [B, D] (L2 前，用于反坍缩损失)
- `pre_norm_map`: [B, D, H, W] (L2 前的空间 embedding)
- `reconstructions`, `logits`, `aux_logits`, `bottleneck_logits`

双窗口编码: `AEFModel.encode_dual_window` 返回 `(emb_w1, emb_w2, pre_w1, pre_w2)`。

## 测试策略

项目中没有使用 pytest/unittest 等正式测试框架，验证方式为:

1. **快速冒烟测试**: `scripts/test_v6_launch.py` — 仅跑 5 个 step 确认无 crash、shape 正确
2. **训练器测试**: `scripts/test_v6_trainer.py` — 验证 trainer 的 forward + loss 链路
3. **AUC 验证**: 训练完成后运行 `scripts/eval/validate_v*.py` 计算变化检测 AUC
4. **benchmark**: `scripts/eval/benchmark_*.py` 对月度/季度 embedding 做系统性评估

添加新 trainer 时，建议复制 `test_v6_launch.py` 模式写一个快速验证脚本。

## 部署与运行时

### 训练任务管理

训练通常在 tmux session 中后台运行，例如:
```bash
tmux new -s v6_train
# 在 session 中执行训练命令
```

可用的启动脚本:
- `start_train.sh`: V1 DDP 训练启动器 (GPU 5,6,7)
- `start_v6_train.sh`: V6 训练启动器
- `start_v6_5_train.sh`: V6.5 训练启动器
- `watchdog.sh`: DDP v4 崩溃自动恢复看门狗 (自动 resume 最新 checkpoint，最大重试 10 次)
- `monitor_training.py`: Python 版监控脚本，检测 NaN/Inf 并自动修复重启 (会修改 config 降低 risky weights)

### 输出目录约定

```
/workspace/outputs/{experiment_name}/
├── epoch_{N}.pt                # 定期检查点
├── epoch_best_epoch{N}.pt      # 最佳检查点
├── embeddings/                 # 预计算 embedding
│   ├── embedding_maps.npy
│   └── patch_ids.json
└── train_YYYYMMDD_HHMMSS.log   # 训练日志
```

### 预计算缓存

Demo 使用的预计算结果由 `demo_v2/precompute_cd.py` 生成，依赖 backbone checkpoint。修改模型后需要重新预计算。

### 本机硬件配置

- **GPU**: 8 × NVIDIA GeForce RTX 4090 (24GB 显存)
- **CUDA**: 12.8 (驱动 570.211.01)
- **PyTorch**: 2.11.0+cu126
- **conda 环境**: `aef-qwen` (Python 3.11)
- **激活方式**: `conda activate aef-qwen`

## 安全与注意事项

- **所有文件操作限制在 `/workspace/xuannv/` 内**，不要读写项目外的文件。
- **checkpoint 文件较大** (数百 MB 到数 GB)，不要频繁复制或传输。
- 训练脚本会修改 config YAML (如 `monitor_training.py` 自动降低 loss weights)，注意版本控制。
- Demo 和评估脚本中使用了大量**硬编码绝对路径** (如 `/workspace/raw/harbin_scenes/`、`/workspace/index/harbin/grid/harbin_grid.geojson`)，修改路径时需同步更新所有引用位置。
- `archive/` 目录为废弃代码，**只读参考**，不要修改或依赖其中的逻辑。
- 没有 CI/CD 流程，所有测试和验证需在本地或开发机上手动执行。

## GitHub 仓库同步规则

- **远程仓库**: `git@github.com:go-bananas-wwj/xuannv.git` (私密仓库)
- **主分支**: `main`
- **强制要求**: 每次对代码/配置/文档做任何修改后，**必须**执行:
  ```bash
  git add -A && git commit -m "描述" && git push origin main
  ```
  同步到远端。
- **提交信息规范**: 用中文或英文简明描述改动内容，包含改动的文件和目的。
- **禁止**: 任何本地修改不同步就结束任务。

## Agent 行为准则

- **与用户交流时必须使用中文回复**。
- 处理训练/模型相关任务前，先读取对应版本的 config YAML。
- 修改损失函数或训练逻辑时，确保 `gathered_pre_norm` / `pre_norm_map` 被正确使用。
- 调试 AUC 低时，优先检查:
  1. temporal contrastive loss 是否生效
  2. 双窗口数据是否正确生成
  3. `raw_unif` 是否在正常范围 (-4.0 ~ -1.0)
- 所有文件操作限制在 `/workspace/xuannv/` 内。
- 如需修改 demo 模型注册表，同步更新 `demo_v2/utils/constants.py` 中的 `MODEL_REGISTRY`。
- 启动训练前请先检查 GPU 占用情况（`nvidia-smi`），选择空闲 GPU，必要时手动设置 `CUDA_VISIBLE_devices`。
