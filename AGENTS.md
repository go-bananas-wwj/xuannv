# xuannv_embdding — Agent 上下文

## 项目概述

**xuannv_embdding** (包名 `xuannv`) 是 **AlphaEarth Foundations (AEF)** 的独立改进版，从零实现于 `/workspace/xuannv/`。

- **核心使命**: 解决嵌入坍缩 (embedding collapse) 与提升时间敏感性 (temporal sensitivity)，使模型能够执行变化检测 (change detection)。
- **与原版关系**: 本项目完全独立实现，仅参考 AEF 接口设计（数据加载协议、模型输入输出格式）。
- **语言与注释**: 项目内代码注释、文档、配置均以中文为主。

### 核心设计决策

1. **输入严格对齐论文**: 只有 `S2`、`S1`、`Landsat` 三类时序图像作为输入。
2. **静态数据仅作目标**: `DEM`、`WorldCover`、`Dynamic World`、`JRC Water` 只参与重建，不输入给 encoder。
3. **训练时跳过 L2 Norm**: `VMFBottleneck(skip_l2_training=true)`，在 pre-norm 空间计算所有反坍缩损失。
4. **推理时标准 L2 + VMF**: 保证 embedding 在球面上。
5. **时间窗口增强**: 训练时随机裁剪 valid_period；V3+ 引入不重叠双窗口 + 时序对比损失。

## 硬件环境

- **主计算设备**: 8 × Huawei Ascend 910B4 NPU
- **分布式后端**: `hccl` (替代 nccl)
- **PyTorch NPU**: torch 2.1.0 + torch_npu 2.1.0.post18
- **conda 环境**: `xuannv` (Python 3.11)，激活方式 `conda activate xuannv`
- **注意**: 当前代码已全面适配 NPU (`torch.npu`, `hccl`, `torch_npu.amp.GradScaler`)。`src/utils/device.py` 默认优先返回 NPU 设备。少量遗留启动脚本中仍出现 `CUDA_VISIBLE_DEVICES`，实际训练代码已统一走 NPU 逻辑。

## 技术栈与依赖

| 层级 | 技术 |
|------|------|
| 语言 | Python >= 3.9 |
| 深度学习 | PyTorch >= 2.0 |
| NPU 适配 | torch_npu, CANN |
| 数据 I/O | rasterio, geopandas, numpy |
| 配置 | PyYAML |
| 构建 | setuptools (`pyproject.toml`) |
| 推理加速 | torch.autocast (bf16), gradient checkpointing |
| 分布式 | torchrun + DDP (hccl) |

### 关键配置文件

- **`pyproject.toml`**: 包配置，定义包名 `xuannv`、版本 `0.1.0`、依赖 (`torch>=2.0`, `numpy`, `rasterio`, `geopandas`, `pyyaml`)，使用 `setuptools>=61.0` 构建。
- **`configs/qwen_v*.yaml`**: 训练配置，支持 `_base_` 继承机制。共 14 个配置文件，覆盖 V1~V6.5 各版本。
- **`.gitignore`**: 排除 `__pycache__/`, `*.pt`, `*.pth`, `outputs/`, `*.npy`, `*.tif`, `.vscode/`, `.idea/` 等。

### 安装命令

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
│   └── qwen_v*.yaml            # 14个训练配置 (支持 _base_ 继承)
├── src/                        # 核心源码包 (~6800行 Python)
│   ├── config.py               # YAML -> dataclass 配置系统 (236行)
│   ├── data/
│   │   ├── dataset.py          # HarbinPatchDataset (1000行, 3输入/7目标, 内存预加载)
│   │   ├── transforms.py       # 归一化、TIFF读取、时间戳解析、源定义 (205行)
│   │   └── builder.py          # DataLoader 工厂 (DistributedSampler) (53行)
│   ├── models/
│   │   ├── model.py            # AEFModel 主模型 (355行, 编码→瓶颈→解码)
│   │   ├── bottleneck.py       # VMFBottleneck (90行, 训练skip L2, 推理L2+VMF)
│   │   ├── blocks.py           # STPBlock (128行, 三路径: Time/Space/Precision)
│   │   ├── sensor_encoders.py  # SensorEncoderBank (135行, 多源独立stem)
│   │   ├── decoders.py         # ContinuousDecoder / CategoricalDecoder (124行)
│   │   ├── heads.py            # ChangeDetectionHead V1/V2/V3 (396行)
│   │   ├── downstream_heads.py # PixelMLPHead / PixelConvHead (174行)
│   │   └── time_encoding.py    # TimeCode / WindowCode / RelativeTimeCode (77行)
│   ├── training/
│   │   ├── losses.py           # raw_uniformity + 时序对比损失全家桶 (420行)
│   │   ├── trainer.py          # V1 DDP 训练器 (351行)
│   │   ├── ddp_v4_trainer.py   # V4 DDP 训练器 (402行)
│   │   ├── ddp_v4_official_trainer.py  # V4 官方对齐 (502行)
│   │   ├── ddp_v5_mixed_scale_trainer.py        # V5 混合尺度 (546行)
│   │   ├── ddp_v6_enhanced_temporal_trainer.py  # V6 增强时序 (602行)
│   │   ├── ddp_v6_5_gap_aware_trainer.py        # V6.5 gap-aware (571行)
│   │   ├── single_gpu_trainer.py       # 单卡训练器 (397行)
│   │   ├── optimizer.py        # AdamW + cosine warmup scheduler (54行)
│   │   └── loops.py            # 重建损失计算辅助 (69行)
│   ├── inference/
│   │   └── engine.py           # 统一推理引擎 (165行, load_backbone / load_cd_head)
│   └── utils/
│       ├── checkpoint.py       # load_checkpoint / save_checkpoint (84行)
│       ├── device.py           # get_device (优先NPU) (29行)
│       ├── io.py               # IO 辅助 (43行)
│       └── logging.py          # 日志工具 (21行)
├── scripts/
│   ├── train/                  # DDP 训练入口 (train_ddp.py / train_ddp_v*.py)
│   ├── eval/                   # AUC 验证、benchmark、embedding 分析 (12个脚本)
│   ├── inference/              # 月度 embedding 提取
│   ├── visualize/              # 训练诊断、结果可视化
│   ├── preprocessing/          # 数据预处理 (云筛选、统计计算)
│   ├── test_v6_launch.py       # 快速冒烟测试
│   ├── test_v6_trainer.py      # 训练器链路测试
│   └── train_v*_downstream_*.py  # 下游任务训练 (MLP/ConvHead)
├── archive/                    # 废弃/实验代码 (只读参考)
│   ├── demo_legacy/
│   ├── demo_v2_legacy/
│   ├── root_tools/
│   ├── scripts_deprecated/
│   ├── scripts_experimental/
│   └── scripts_preprocessing/
└── docs/                       # 项目文档 (8个markdown文件)
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
- `data`: batch_size, image_size, max_frames, 时序增强参数, 窗口模式, preload
- `model`: embedding_dim, num_blocks, vmf_kappa, skip_l2_norm_training
- `training`: 学习率、损失权重、warmup、检查点保存频率
- `evaluation`: knn_k, bootstrap_samples
- `pretrained`: 预训练权重路径 (可选，由训练脚本 `--resume` 或 `--soft-restart` 覆盖)

## 模型架构

### AEFModel 前向流程

1. **SensorEncoderBank**: 多源独立 stem (S2/S1/Landsat 各自由 Conv 下采样到统一精度维度)
2. **Time/Window/RelativeTime Encoder**: 时间编码注入
3. **STPBlocks** (×N): Space-Time-Precision 三路径块
   - Time path: 时间轴 MultiheadAttention (降采样 1/8)
   - Space path: 空间 MultiheadAttention (降采样 1/16)
   - Precision path: 3×3 卷积 (保持 1/2)
   - Fusion: 三路径拼接 + 1×1 卷积融合
4. **VMFBottleneck**: Conv1×1 压缩 → 训练时保留原始幅度 / 推理时 L2 Norm + VMF 噪声
5. **Decoders**: 条件解码器 (ContinuousDecoder / CategoricalDecoder)，逐目标源重建

### 输入输出规范

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

### 数据源定义

- **输入源** (3类): `s2`, `s1`, `landsat`
- **目标源** (7类): `s2`, `s1`, `landsat`, `dem`, `worldcover`, `dynamic_world`, `jrc_water`
- 时间戳兼容: `YYYYMMDD` (单景) 和 `YYYYQN` (季度)
- 归一化: 光学 `log(x+1)/10 → z-score → ±6σ clip`；SAR `clip[-30,10]dB → z-score`
- 分类源 (WorldCover/DynamicWorld): one-hot 编码

## 训练系统

### 训练入口

各版本对应独立的训练脚本和训练器类:

| 版本 | 训练脚本 | 训练器类 | 关键特性 |
|------|----------|----------|----------|
| V1 | `scripts/train/train_ddp.py` | `DDPTrainer` | 反坍缩基线 |
| V4 | `scripts/train/train_ddp_v4.py` | `DDPv4Trainer` | 官方对齐 |
| V4官方 | `scripts/train/train_ddp_v4_official.py` | `DDPv4OfficialTrainer` | 官方完整对齐 |
| V5 | `scripts/train/train_ddp_v5.py` | `DDPv5MixedScaleTrainer` | 混合尺度双窗口 |
| V6 | `scripts/train/train_ddp_v6.py` | `DDPv6EnhancedTemporalTrainer` | 像素级时序损失 |
| V6.5 | `scripts/train/train_ddp_v6_5.py` | `DDPv6_5GapAwareTrainer` | gap-aware temporal |

启动示例 (V1 DDP):
```bash
cd /workspace/xuannv
torchrun --nproc_per_node=3 \
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

### 损失函数体系

核心损失位于 `src/training/losses.py`:

- **raw_uniformity_loss**: 欧氏空间 uniformity，自适应 t=2/D，梯度永远非零
- **decorrelation_loss**: Barlow Twins 去相关
- **variance_regularizer**: VICReg 方差正则
- **bottleneck_orthogonality_loss**: Conv1×1 权重正交约束
- **temporal_contrastive_loss**: 双窗口 hinge loss (global mean)
- **temporal_cosine_pixel_loss**: 像素级 cosine 时序损失 (V6+)
- **pixel_temporal_info_nce_loss**: 像素级 Anti-Diagonal InfoNCE (V6+)
- **gap_aware_temporal_cosine_loss**: 根据时间 gap 大小动态设定 target (V6.5)
- **reconstruction_loss**: 掩码 L1 (连续) / CE (分类)

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
    device="npu:0",
)
```

### 变化检测验证

主要评估脚本:
- `scripts/eval/validate_v2.py`: V2 变化检测 AUC (105 个光学标注)
- `scripts/eval/validate_v4_official.py`: V4 官方验证
- `scripts/eval/validate_v5_level1_bare.py`: V5 Level-1 裸验证
- `scripts/eval/validate_v6_level1_bare.py`: V6 Level-1 裸验证
- `scripts/eval/validate_v6_5_level1_bare.py`: V6.5 Level-1 裸验证
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

### 测试策略

项目中没有使用 pytest/unittest 等正式测试框架，验证方式为:

1. **快速冒烟测试**: `scripts/test_v6_launch.py` — 仅跑 5 个 step 确认无 crash、shape 正确
2. **训练器测试**: `scripts/test_v6_trainer.py` — 验证 trainer 的 forward + loss 链路
3. **AUC 验证**: 训练完成后运行 `scripts/eval/validate_v*.py` 计算变化检测 AUC
4. **benchmark**: `scripts/eval/benchmark_*.py` 对月度/季度 embedding 做系统性评估

添加新 trainer 时，建议复制 `test_v6_launch.py` 模式写一个快速验证脚本。

### 数据预处理脚本

- **`scripts/preprocessing/filter_cloudy_frames.py`**: S2 云筛选，按月保留最 clear 的帧
- **`scripts/preprocessing/compute_statistics.py`**: 计算各源数据的 mean/std 统计量

## 部署与运行时

### 训练任务管理

**必须使用 tmux 运行训练，禁止使用 nohup。**

> ⚠️ **教训**: nohup 在会话断开时会发送 SIGHUP 信号，导致 torchrun DDP 进程被终止。2025-05-08 的 V7 Phase1 v2 训练因此中断（Epoch 50）。tmux 是唯一的可靠方案。

#### 启动新训练

```bash
# 创建 tmux session
tmux new-session -d -s v7_train -c /workspace/xuannv

# 在 session 中发送训练命令
tmux send-keys -t v7_train 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3' Enter
tmux send-keys -t v7_train 'conda activate xuannv' Enter
tmux send-keys -t v7_train 'torchrun --nproc_per_node=4 scripts/train/train_ddp_v7.py --config configs/xuannv_v7.yaml --save-every 20' Enter

# detach，训练在后台继续
tmux detach -t v7_train
```

#### 从 checkpoint 恢复训练

```bash
# 创建新 session 恢复
tmux new-session -d -s v7_train -c /workspace/xuannv
tmux send-keys -t v7_train 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3' Enter
tmux send-keys -t v7_train 'conda activate xuannv' Enter
tmux send-keys -t v7_train 'torchrun --nproc_per_node=4 scripts/train/train_ddp_v7.py --config configs/xuannv_v7.yaml --resume /workspace/outputs/xuannv_backbone_v7_phase1_v2/epoch_best_39.pt --save-every 20' Enter
```

#### 常用 tmux 操作

| 操作 | 命令 |
|------|------|
| 查看所有 session | `tmux list-sessions` |
| attach 到 session | `tmux attach -t v7_train` |
| detach（保持后台运行） | 按 `Ctrl+B` 然后按 `D` |
| 查看日志 | `tmux capture-pane -t v7_train -p \| tail -20` |
| 终止 session | `tmux kill-session -t v7_train` |

#### 可用的启动脚本

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



## 安全与注意事项

- **所有文件操作限制在 `/workspace/xuannv/` 内**，不要读写项目外的文件。
- **checkpoint 文件较大** (数百 MB 到数 GB)，不要频繁复制或传输。
- 训练脚本会修改 config YAML (如 `monitor_training.py` 自动降低 loss weights)，注意版本控制。
- Demo 和评估脚本中使用了大量**硬编码绝对路径** (如 `/workspace/raw/phase1_harbin/`、`/workspace/index/harbin/grid/harbin_grid.geojson`)，修改路径时需同步更新所有引用位置。
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
- 启动训练或推理前请先检查 NPU 占用情况（`npu-smi info`），选择空闲 NPU，必要时通过 `ASCEND_RT_VISIBLE_DEVICES` 或脚本参数指定设备。

---

# ★ 当前数据状态与训练准备（2026-05-08 更新）

> **重要**: 以下内容是最近一次数据准备工作的完整记录。任何接手训练的 Agent **必须**先阅读本节，确保理解当前数据状态和训练要求。

## 数据修复历史

### 1. 路径修复（已完成）

- **问题**: 5 个配置文件的 `manifest_path` 指向 `/workspace/raw/phase1_harbin`（季度数据父目录），但代码期望的是日度数据子目录。
- **修复**: 将 `manifest_path` 从 `/workspace/raw/phase1_harbin` 改为 `/workspace/raw/phase1_harbin/harbin_scenes`。
- **影响配置**: `qwen_v1_scenes.yaml`, `qwen_v2_hr_finetune.yaml`, `qwen_v2_hr_from_scratch.yaml`, `qwen_v2_hr_only_small.yaml`, `qwen_v3_continue.yaml`

### 2. 代码路径解析增强（已完成）

- **修改文件**: `src/data/dataset.py`
- **新增方法**: `_resolve_source_dir(source_name, patch_id)` — 自动在 `data_root` 及其子目录中搜索数据源目录，兼容 `harbin_scenes/harbin/s2/patch_*` 的嵌套结构。
- **影响范围**: `_load_input_frames_impl`, `_load_target_frame`, `_preload_all` 目标加载, `_get_worldcover_label`

### 3. Symlink 修复（已完成）

- **问题**: `dem`, `worldcover`, `jrc_water`, `dynamic_world`, `modis_lst`, `modis_ndvi` 的 symlink 断裂。
- **修复**: 重新链接到 `/workspace/raw/phase1_harbin/harbin/` 下的正确目录。

### 4. filter_2025_monthly 修复（已完成）

- **问题**: `qwen_v2_hr_from_scratch.yaml` 和 `qwen_v2_hr_only_small.yaml` 中 `filter_2025_monthly: true`，导致只保留 2025 年 4-10 月数据，过滤掉了所有 2023-2024 训练数据。
- **修复**: 改为 `filter_2025_monthly: false`。

### 5. 统计数据生成（已完成）

- **新增脚本**: `scripts/preprocessing/compute_statistics.py`
- **输出**: `/workspace/statistics/harbin_scenes/{source}_stats.json`（7 个文件）
- **已生成**: s2(6ch), s1(2ch), landsat(6ch), dem(1ch), worldcover(1ch), dynamic_world(1ch), jrc_water(1ch)

### 6. 并行预加载优化（已完成）

- **修改文件**: `src/data/dataset.py`
- **优化**: 非 DDP 环境下使用 `ProcessPoolExecutor(16 workers)` 并行预加载
- **性能**: 首次加载从 ~19 分钟 → 4.2 分钟 (4.6x 加速)，缓存加载 125 秒

### 7. 验证脚本优化（已完成）

- **修改**: 6 个验证脚本创建数据集前设置 `cfg.data.preload = False`
- **效果**: 启动从 ~19 分钟 → 3 秒
- **修改脚本**: `validate_v2.py`, `validate_v4_level1_bare.py`, `validate_v5_level1_bare.py`, `validate_v6_level1_bare.py`, `validate_v6_5_level1_bare.py`, `analyze_v5_embedding_space.py`

### 8. Grid GeoJSON 重建（已完成）

- **问题**: `/workspace/index/harbin/grid/harbin_grid.geojson` 缺失，导致验证脚本和标注解析无法运行。
- **修复**: 从 xuannv_show 仓库的 `patches_meta.json` 验证，并从 TIFF 元数据重建，424 个 feature 完全匹配。
- **输出**: `/workspace/index/harbin/grid/harbin_grid.geojson`

### 9. S2 云筛选预处理（已完成）

- **新增脚本**: `scripts/preprocessing/filter_cloudy_frames.py`
- **策略**: 对每个 patch 的 S2 帧计算 cloud_score（亮度/10000 - NDVI），按月保留最 clear 的 2 帧；全 cloudy 月份 fallback 保留 1 帧。
- **结果**:
  - 原始: 29,707 帧 → 筛选后: 9,321 帧
  - 平均每 patch: ~70 帧 → ~22 帧
  - fallback 月份: 357 / 4839 = 7.4%（主要集中在哈尔滨冬季 1-2 月）
- **新数据目录**: `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/`
- **Symlink**: s1, s1_hr, s2_hr, landsat, dem, worldcover, dynamic_world, jrc_water, modis_lst, modis_ndvi 已链接
- **配置文件更新**: 5 个配置文件的 `manifest_path` 已指向新目录
- **统计数据更新**: S2 stats 已重新计算（mean=1466.11, std=1207.10）
- **缓存更新**: 旧缓存已删除，新缓存 27.7GB 已生成

## 当前数据目录结构

```
/workspace/raw/phase1_harbin/
├── harbin/                          # 季度合成数据 (YYYYQN)，旧数据
│   ├── s2/, s1/, landsat/, dem/, worldcover/, ...
├── harbin_scenes/                   # 原始日度数据 (YYYYMMDD)
│   ├── s2/ (180帧/patch)            ← 原始数据，未筛选
│   ├── s1/ (96帧/patch)
│   ├── s2_hr/ (5帧/patch)
│   ├── s1_hr/ (4帧/patch)
│   ├── landsat/ (76帧/patch)
│   ├── dem -> ../harbin/dem
│   ├── worldcover -> ../harbin/worldcover
│   ├── dynamic_world -> ../harbin/dynamic_world
│   ├── jrc_water -> ../harbin/jrc_water
│   ├── modis_lst -> ../harbin/modis_lst
│   └── modis_ndvi -> ../harbin/modis_ndvi
└── harbin_scenes_cloud_filtered/    # ★ 当前训练使用的数据
    ├── s2/ (~22帧/patch)            ← 云筛选后
    ├── s1 -> ../harbin_scenes/s1
    ├── s1_hr -> ../harbin_scenes/s1_hr
    ├── s2_hr -> ../harbin_scenes/s2_hr
    ├── landsat -> ../harbin_scenes/landsat
    ├── dem -> ../harbin_scenes/dem
    ├── worldcover -> ../harbin_scenes/worldcover
    ├── dynamic_world -> ../harbin_scenes/dynamic_world
    ├── jrc_water -> ../harbin_scenes/jrc_water
    ├── modis_lst -> ../harbin_scenes/modis_lst
    └── modis_ndvi -> ../harbin_scenes/modis_ndvi
```

## 当前各源帧数分布

| 源 | Patch 数 | 最少帧 | 最多帧 | 均值 | 中位数 | 零帧数 |
|----|---------|--------|--------|------|--------|--------|
| S2 (云筛选后) | 424 | 2 | ~40 | 22.0 | ~22 | 0 |
| S1 | 424 | 38 | 100 | 42.7 | 39 | 0 |
| Landsat | 407 | 2 | 76 | 47.9 | 37 | 17 patch 缺失 |
| S2_HR | 424 | 3 | 5 | 5.0 | 5 | 0 |
| S1_HR | 424 | 3 | 4 | 4.0 | 4 | 0 |

## 已知数据问题

1. **Landsat 缺失 17 个 patch**: `patch_000324`~`patch_000327`, `patch_000409`~`patch_000414` 等。代码已优雅处理（`source_input_mask[s_idx]=False`），不会 crash。
2. **S2 冬季 fallback**: 357 个月份（7.4%）全 cloudy，被迫保留最不清的一张。主要集中在 2025-01（272 个）、2024-01（24 个）、2024-02（15 个）。
3. **不同源时间不对齐**: S2/S1/Landsat 的重访周期不同（5天/12天/16天），同一天重叠极少。模型通过时间编码处理，这是设计预期行为。

## 训练启动要求

### 环境检查（启动前必须做）

```bash
# 1. 激活环境
conda activate xuannv

# 2. 检查 NPU 占用（必须选择空闲 NPU）
npu-smi info

# 3. 确认当前分支和提交
# 当前应为 main 分支，commit 72055b7 或更新
```

### Backbone 训练命令（V1 基线）

```bash
cd /workspace/xuannv
# 使用前3个空闲 NPU，3卡 DDP
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp.py --config configs/qwen_v1_scenes.yaml \
    --save-every 50 --warmup-epochs 10
```

### 监控要求

训练中**必须**实时监控以下指标：

| 指标 | 正常范围 | 第1个 epoch 应达到 | 异常处理 |
|------|----------|-------------------|---------|
| `raw_unif` | -4.0 ~ -1.0 | < -0.5 | 如果 > -0.5 持续 5 个 epoch，立即报告 |
| `pre_unif` | 接近 raw_unif | 差距 < 0.5 | 差距大说明 pre_norm 和 norm 空间不一致 |
| `recon` | < 0.3 | < 1.0 (warmup期) | 如果 warmup 后仍 > 0.5，检查数据 |
| `var_reg` | 接近 0 | < 0.5 | > 0.5 表示方差坍缩 |
| `orth` | < 0.3 | < 0.5 | > 0.5 权重不正交 |
| `decorr` | < 1.0 | < 2.0 | > 2.0 强相关 |

**如果 raw_unif > -0.5 且持续不下降，说明 embedding 坍缩，训练失败。**

### 训练中断恢复

```bash
# 从最新 checkpoint resume
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp.py --config configs/qwen_v1_scenes.yaml \
    --resume /workspace/outputs/aef_qwen_v1/epoch_best_xxx.pt
```

### 验证要求

训练每 50 个 epoch 保存一次 checkpoint，应在保存后立即运行验证：

```bash
# V1 验证（需要约 5-10 分钟）
python scripts/eval/validate_v2.py --checkpoint /workspace/outputs/aef_qwen_v1/epoch_50.pt
```

**AUC 目标**: 变化检测 AUC > 0.7 为及格，> 0.8 为良好，> 0.85 为优秀。

## 对训练 Agent 的强制要求

1. **启动前必须检查 NPU 占用**，不要抢占正在运行的训练任务。
2. **不要修改 `filter_2025_monthly`**，当前所有配置均为 `false`。
3. **不要修改 `manifest_path`**，当前所有配置已指向云筛选后的正确目录。
4. **如果需要重新生成统计数据**，使用 `scripts/preprocessing/compute_statistics.py`，不要手写 JSON。
5. **如果需要重新筛选云数据**，使用 `scripts/preprocessing/filter_cloudy_frames.py`，参数：
   ```bash
   python scripts/preprocessing/filter_cloudy_frames.py \
       --max-per-month 2 --cloud-threshold 0.3 --workers 16
   ```
6. **每次修改后必须 git commit + push**。
7. **如果训练出现 NaN/Inf**，先检查是否是 loss weight 过高，不要直接删除 checkpoint。
8. **调试 AUC 低时**，优先检查：
   - temporal contrastive loss 是否生效
   - 双窗口数据是否正确生成
   - `raw_unif` 是否在正常范围

---
*最后更新: 2026-05-08, commit 72055b7*

---

## 实验管理与命名规范（2026-05-13 更新）

### 输出目录命名规范

为避免 `/workspace/outputs/` 目录混乱，所有新实验必须遵循以下命名格式：

```
{version}_{核心参数}_{卡数}card_{日期}_{可选备注}
```

**示例：**
- `v12_recon005_4card_0513` — v12版本, recon=0.05, 4卡, 5月13日
- `v12_recon010_2card_0513_resume` — v12版本, recon=0.10, 2卡, 从checkpoint恢复

**禁止：** 冗长无意义的命名如 `v13_round6_expX_recon005_full_staged_200patches`

### 缓存清理强制要求

- **每个实验的 `dataset_cache_*.pt` 约 26-32GB**
- **训练结束后必须立即删除缓存**，释放磁盘空间
- **自动清理命令：**
  ```bash
  find /workspace/outputs -name "dataset_cache_*.pt" -delete
  ```
- **启动新训练前**，先检查并清理旧缓存：
  ```bash
  du -sh /workspace/outputs/*/dataset_cache_*.pt 2>/dev/null
  ```

### 长时间训练的后台监控

对于需要数小时的训练任务，**必须启动后台监控任务**追踪关键指标：

1. **创建监控脚本** (`scripts/monitor_{实验名}.py`)：
   - 每 5 分钟读取 `train.log`
   - 提取 `active_dims` 和 `recon` 指标
   - 写入实时报告文件

2. **启动方式**（使用 Python 无缓冲模式）：
   ```bash
   cd /workspace/xuannv
   PYTHONUNBUFFERED=1 /root/miniconda3/envs/xuannv/bin/python \
       scripts/monitor_{实验名}.py > /workspace/outputs/{实验名}/monitor.log 2>&1
   ```

3. **监控目标**：
   - 若 `active_dims < 20` 持续 3 个 epoch → **立即报告坍缩**
   - 若 `recon > 0.5` 在 warmup 后持续不降 → **检查数据**
   - 记录完整的 epoch 轨迹到 JSON 历史文件

### 全量数据训练规范

- **必须使用全部 424 个 patch**，禁止用子集（如 200 patches）做实验
- 配置中 `data.num_samples` 必须保持 `424`
- `manifest_path` 必须指向 `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered`

### NPU 使用规范

- **启动前必须执行 `npu-smi info`** 检查占用
- 多卡训练优先使用连续编号的 NPU（如 0,1,2,3）
- 通过 `ASCEND_RT_VISIBLE_DEVICES` 指定设备：
  ```bash
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  ```
- **严禁使用 nohup 启动 torchrun**，必须使用 tmux

### 实验废弃与清理

- **确认失败的实验**（已坍缩或崩溃）应立即删除目录，释放空间
- 保留有价值的 checkpoint（`epoch_best_*.pt`），其余可删
- 删除前确认无需要恢复的 checkpoint

