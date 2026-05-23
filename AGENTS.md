# xuannv_embdding — Agent 上下文

## 项目概述

**xuannv_embdding** (包名 `xuannv`) 是 **AlphaEarth Foundations (AEF)** 的独立改进版，从零实现于 `/workspace/xuannv/`。

- **核心使命**: 解决嵌入坍缩 (embedding collapse) 与提升时间敏感性 (temporal sensitivity)，使模型能够执行变化检测 (change detection)。
- **与原版关系**: 完全独立实现，仅参考 AEF 接口设计。
- **语言**: 项目内代码注释、文档、配置均以中文为主。

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
- **conda 环境**: `xuannv` (Python 3.11) — `conda activate xuannv`
- **注意**: 代码已全面适配 NPU。`src/utils/device.py` 默认优先返回 NPU 设备。

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

- **`pyproject.toml`**: 包配置，包名 `xuannv`、版本 `0.1.0`。安装: `pip install -e .`
- **`configs/*.yaml`**: 训练配置，支持 `_base_` 继承。活跃配置: `config.yaml`（Round 9 基线）、`v12_*.yaml`、`v13_*.yaml`、`mb_*.yaml` 等。
- **`.gitignore`**: 排除 `__pycache__/`, `*.pt`, `*.pth`, `outputs/`, `*.npy`, `*.tif` 等。

**注意**: 项目中无任何 linting/formatting 配置文件，代码风格由开发者手动保持一致。

## 代码组织结构

```
xuannv_embdding/
├── pyproject.toml              # 包配置
├── configs/
│   ├── config.yaml             # Round 9 基线
│   ├── v12_*.yaml              # V12 实验族
│   └── mb_*.yaml               # Mini-Batch 实验族
├── src/                        # 核心源码 (~8000行)
│   ├── config.py               # YAML -> dataclass 配置系统
│   ├── data/
│   │   ├── dataset.py          # HarbinPatchDataset (3输入/7目标, 内存预加载)
│   │   ├── transforms.py       # 归一化、TIFF读取、时间戳解析
│   │   └── builder.py          # DataLoader 工厂
│   ├── models/
│   │   ├── model.py            # AEFModel 主模型
│   │   ├── bottleneck.py       # VMFBottleneck (V13 skip L2 生效)
│   │   ├── blocks.py           # STPBlock (三路径, V13手动MHA)
│   │   ├── sensor_encoders.py  # 多源独立stem
│   │   ├── decoders.py         # ContinuousDecoder / CategoricalDecoder
│   │   ├── heads.py            # ChangeDetectionHead V1/V2/V3
│   │   ├── downstream_heads.py # PixelMLPHead / PixelConvHead
│   │   └── time_encoding.py    # TimeCode / WindowCode
│   ├── training/
│   │   ├── losses.py           # raw_uniformity + 时序对比损失全家桶
│   │   ├── ddp_v13_trainer.py  # V13 DDP 训练器 (最新)
│   │   ├── ddp_v12_trainer.py  # V12 DDP 训练器
│   │   ├── ddp_v7_trainer.py   # V7 DDP 训练器
│   │   ├── optimizer.py        # AdamW + cosine warmup
│   │   ├── loops.py            # 重建损失计算辅助
│   │   ├── memory_bank.py      # Memory Bank
│   │   └── vicreg_loss.py      # VICReg 损失
│   ├── inference/
│   │   └── engine.py           # 统一推理引擎
│   └── utils/
│       ├── checkpoint.py       # load/save_checkpoint
│       ├── device.py           # get_device (优先NPU)
│       ├── io.py               # IO 辅助
│       └── logging.py          # 日志工具
├── scripts/
│   ├── train/                  # DDP 训练入口 (v7/v12/v13)
│   ├── eval/                   # AUC 验证 (~40个脚本)
│   ├── inference/              # 月度 embedding 提取
│   ├── visualize/              # 训练诊断、结果可视化
│   ├── downstream/             # 下游任务训练与评估
│   ├── preprocessing/          # 数据预处理 (云筛选、统计)
│   └── test_v13_smoke.py       # V13 冒烟测试
├── docs/                       # 项目文档
├── archive/                    # 废弃代码 (只读参考)
└── data/                       # 运行时数据输出
```

## 配置系统

所有训练参数通过 `configs/*.yaml` 管理，支持继承机制:

```yaml
_base_: config.yaml
experiment:
  name: my_experiment
```

加载: `from src.config import load_config; cfg = load_config("configs/config.yaml")`

关键配置段: `experiment` (名称/输出目录), `data` (batch_size/max_frames), `model` (embedding_dim/vmf_kappa), `training` (学习率/损失权重/warmup), `evaluation`, `pretrained`。

## 模型架构

### AEFModel 前向流程

1. **SensorEncoderBank**: 多源独立 stem (S2/S1/Landsat)
2. **Time/Window/RelativeTime Encoder**: 时间编码注入
3. **STPBlocks** (×N): Space-Time-Precision 三路径块
   - Time path: 时间轴 MHA (降采样 1/8)
   - Space path: 空间 MHA (降采样 1/16)
   - Precision path: 3×3 卷积 (保持 1/2)
   - Fusion: 三路径拼接 + 1×1 卷积
   - **V13**: 手动实现 `_manual_mha`，避免 NPU fallback 到 CPU
4. **VMFBottleneck**: Conv1×1 压缩 → 训练 skip L2 / 推理 L2 + VMF 噪声
5. **Decoders**: 条件解码器，逐目标源重建

### 输入输出规范

`AEFModel.forward` 参数: `source_frames`, `source_timestamps_ms`, `source_frame_mask`, `source_input_mask`, `source_type_ids`, `valid_start_ms`, `valid_end_ms`, `target_relative_time`, `target_metadata`, `target_loss_type`, `target_source_idx`, `skip_decoder`, `dual_window`。

输出 `AEFOutput`: `embedding_map`, `embedding`, `pre_norm_embedding`, `pre_norm_map`, `reconstructions`, `logits`, `aux_logits`, `bottleneck_logits`, `dual_pre_w2`。

双窗口编码: `encode_dual_window` → `(emb_w1, emb_w2, pre_w1, pre_w2)`; `encode_dual_window_v10` → 额外返回 `change_score, diff_feat`。

### 数据源

- **输入源** (3类): `s2`, `s1`, `landsat`
- **目标源** (7类): `s2`, `s1`, `landsat`, `dem`, `worldcover`, `dynamic_world`, `jrc_water`
- 时间戳兼容: `YYYYMMDD` (单景) 和 `YYYYQN` (季度)
- 归一化: 光学 `log(x+1)/10 → z-score → ±6σ clip`；SAR `clip[-30,10]dB → z-score`
- 分类源: one-hot 编码

## 训练系统

### 训练入口

| 版本 | 训练脚本 | 训练器类 | 关键特性 |
|------|----------|----------|----------|
| V7 | `scripts/train/train_ddp_v7.py` | `DDPv7Trainer` | 官方对齐 + 反坍缩 |
| V12 | `scripts/train/train_ddp_v12.py` | `DDPv12Trainer` | 纯动态重建基线，极简3-loss |
| V13 | `scripts/train/train_ddp_v13.py` | `DDPv13Trainer` | 最新版本，skip L2 真正生效 |

启动示例 (V13):
```bash
cd /workspace/xuannv
torchrun --nproc_per_node=4 \
    scripts/train/train_ddp_v13.py --config configs/config.yaml \
    --save-every 20 --warmup-epochs 10
```

恢复训练: 加 `--resume /workspace/outputs/.../epoch_best_xxx.pt`

### 损失函数体系 (src/training/losses.py)

- **raw_uniformity_loss**: 欧氏空间 uniformity，自适应 t=2/D
- **batch_uniformity_loss_l2**: L2-normalized uniformity，All-Pairs + 空间采样 + 维度 Dropout
- **decorrelation_loss**: Barlow Twins 去相关
- **variance_regularizer / covariance_loss**: VICReg 方差/协方差正则
- **bottleneck_orthogonality_loss**: Conv1×1 权重正交约束
- **temporal_contrastive_loss**: 双窗口 hinge loss
- **l2_temporal_contrastive_loss**: L2 空间时序对比损失
- **temporal_cosine_pixel_loss**: 像素级 cosine 时序损失，支持空间感知加权
- **pixel_temporal_info_nce_loss**: 像素级 Anti-Diagonal InfoNCE
- **gap_aware_temporal_cosine_loss**: 根据时间 gap 动态设定 target
- **temporal_magnitude_loss**: 约束 embedding distance ≤ time_gap_norm
- **change_consistency_loss**: bottleneck change_score 与图像差异对齐
- **reconstruction_loss**: 掩码 L1 (连续) / CE (分类)

### 训练监控指标

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `raw_unif` | -4.0 ~ -1.0 | > -0.5 表示坍缩 |
| `pre_unif` | 接近 raw_unif | 差距大说明有问题 |
| `recon` | < 0.3 | > 0.5 重建质量差 |
| `var_reg` | 接近 0 | > 0.5 方差坍缩 |
| `orth` | < 0.3 | > 0.5 权重不正交 |
| `decorr` | < 1.0 | > 2.0 强相关 |

**如果 raw_unif > -0.5 且持续不下降，说明 embedding 坍缩，训练失败。**

### 检查点

- 保存路径: `/workspace/outputs/{experiment_name}/`
- 定期保存: `epoch_{N}.pt`
- 最佳模型: `epoch_best_epoch{N}.pt`
- 加载: `src.utils.checkpoint.load_checkpoint`

## 推理与评估

### 统一推理引擎

```python
from src.inference.engine import load_backbone
model, dataset, cfg = load_backbone(
    config_path="configs/config.yaml",
    checkpoint_path="/workspace/outputs/.../epoch_399.pt",
    device="npu:0",
)
```

### 变化检测验证脚本

- `scripts/eval/validate_v12_auc.py`: V12 AUC 验证
- `scripts/eval/validate_v12_bare.py`: V12 Bare AUC
- `scripts/eval/fewshot_change_detection.py`: Few-Shot 变化检测
- `scripts/eval/batch_auc_validate.py`: 批量 AUC 验证
- `scripts/eval/comprehensive_downstream_eval.py`: 综合下游评估

AUC 目标: > 0.7 及格，> 0.8 良好，> 0.85 优秀。

### Embedding 提取

```bash
python scripts/inference/extract_monthly_embeddings_all_patches.py \
    --gpu_idx 0 --total_gpus 2
```

## 开发规范

- 使用 `from __future__ import annotations`
- 类型注解完整，Tensor shape 在 docstring 中注明
- 模块内注释使用中文
- 使用 `pathlib.Path`
- 设备选择统一走 `src.utils.device.get_device`
- checkpoint 统一走 `src.utils.checkpoint.load_checkpoint / save_checkpoint`
- 训练脚本顶部通常设置 `torch.set_num_threads(4)`

### 测试策略

1. **冒烟测试**: `scripts/test_v13_smoke.py`
2. **AUC 验证**: `scripts/eval/validate_v*.py`
3. **benchmark**: `scripts/eval/benchmark_*.py`

### 数据预处理

- `scripts/preprocessing/filter_cloudy_frames.py`: S2 云筛选
- `scripts/preprocessing/compute_statistics.py`: 计算 mean/std 统计量

## 部署与运行时

### 训练任务管理

**必须使用 tmux 运行训练，禁止使用 nohup。**

> ⚠️ nohup 在会话断开时发送 SIGHUP 信号，导致 torchrun DDP 进程被终止。

启动新训练:
```bash
tmux new-session -d -s v13_train -c /workspace/xuannv
tmux send-keys -t v13_train 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3' Enter
tmux send-keys -t v13_train 'conda activate xuannv' Enter
tmux send-keys -t v13_train 'torchrun --nproc_per_node=4 scripts/train/train_ddp_v13.py --config configs/config.yaml --save-every 20' Enter
tmux detach -t v13_train
```

常用 tmux 操作:
| 操作 | 命令 |
|------|------|
| 查看 sessions | `tmux list-sessions` |
| attach | `tmux attach -t v13_train` |
| detach | `Ctrl+B` 然后按 `D` |
| 查看日志 | `tmux capture-pane -t v13_train -p \| tail -20` |
| 终止 | `tmux kill-session -t v13_train` |

### 输出目录

```
/workspace/outputs/{experiment_name}/
├── epoch_{N}.pt
├── epoch_best_epoch{N}.pt
├── embeddings/
│   ├── embedding_maps.npy
│   └── patch_ids.json
└── train_YYYYMMDD_HHMMSS.log
```

### 缓存清理

每个实验的 `dataset_cache_*.pt` 约 26-32GB，训练结束后必须删除:
```bash
find /workspace/outputs -name "dataset_cache_*.pt" -delete
```

## 安全与注意事项

- **所有文件操作限制在 `/workspace/xuannv/` 内**
- checkpoint 文件较大 (数百 MB 到数 GB)，不要频繁复制
- Demo 和评估脚本中有大量**硬编码绝对路径**，修改时需同步更新所有引用
- `archive/` 目录为废弃代码，**只读参考**
- 没有 CI/CD，所有测试和验证需手动执行

## GitHub 仓库同步规则

- **远程仓库**: `git@github.com:go-bananas-wwj/xuannv.git` (私密)
- **主分支**: `main`
- **强制要求**: 每次修改后必须执行:
  ```bash
  git add -A && git commit -m "描述" && git push origin main
  ```
- **禁止**: 任何本地修改不同步就结束任务

## Agent 行为准则

- **与用户交流时必须使用中文回复**
- 处理训练/模型相关任务前，先读取对应版本的 config YAML
- 修改损失函数或训练逻辑时，确保 `gathered_pre_norm` / `pre_norm_map` 被正确使用
- 调试 AUC 低时，优先检查: (1) temporal contrastive loss 是否生效 (2) 双窗口数据是否正确生成 (3) `raw_unif` 是否在正常范围 (-4.0 ~ -1.0)
- 启动训练/推理前检查 NPU 占用 (`npu-smi info`)

---

# ★ 当前数据状态与训练准备

> **重要**: 任何接手训练的 Agent **必须**先阅读本节。

## 数据修复历史 (已完成)

| # | 修复项 | 说明 |
|---|--------|------|
| 1 | 路径修复 | `manifest_path` 改为 `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered` |
| 2 | 代码路径解析 | `src/data/dataset.py` 新增 `_resolve_source_dir`，兼容嵌套结构 |
| 3 | Symlink 修复 | dem/worldcover/jrc_water/dynamic_world/modis_lst/modis_ndvi 已重新链接 |
| 4 | 统计数据 | `/workspace/statistics/harbin_scenes/{source}_stats.json` (7个文件) |
| 5 | 并行预加载 | 非 DDP 环境 16 workers，首次加载 ~19分钟 → 4.2分钟 |
| 6 | S2 云筛选 | 原始 29,707 帧 → 筛选后 9,321 帧；每 patch ~22 帧；fallback 7.4% |

## 当前数据目录结构

```
/workspace/raw/phase1_harbin/
├── harbin/                          # 季度合成数据 (旧数据)
├── harbin_scenes/                   # 原始日度数据 (未筛选)
│   ├── s2/ (180帧/patch)
│   ├── s1/ (96帧/patch)
│   ├── landsat/ (76帧/patch)
│   └── ...
└── harbin_scenes_cloud_filtered/    # ★ 当前训练使用的数据
    ├── s2/ (~22帧/patch)            # 云筛选后
    ├── s1 -> ../harbin_scenes/s1
    ├── landsat -> ../harbin_scenes/landsat
    ├── dem -> ../harbin_scenes/dem
    └── ...
```

## 各源帧数分布

| 源 | Patch 数 | 均值 | 零帧数 |
|----|---------|------|--------|
| S2 (云筛选后) | 424 | 22.0 | 0 |
| S1 | 424 | 42.7 | 0 |
| Landsat | 407 | 47.9 | 17 patch 缺失 |
| S2_HR | 424 | 5.0 | 0 |
| S1_HR | 424 | 4.0 | 0 |

## 已知数据问题

1. **Landsat 缺失 17 个 patch**: 代码已优雅处理 (`source_input_mask[s_idx]=False`)，不会 crash。
2. **S2 冬季 fallback**: 357 个月份 (7.4%) 全 cloudy，被迫保留最不清的一张。主要集中在 2025-01 (272个)。
3. **不同源时间不对齐**: S2/S1/Landsat 重访周期不同 (5天/12天/16天)，同一天重叠极少。模型通过时间编码处理，这是设计预期行为。

## 训练启动要求

### 环境检查 (必须)

```bash
conda activate xuannv
npu-smi info   # 选择空闲 NPU
# 确认当前为 main 分支
```

### 训练命令 (V13 最新版)

```bash
cd /workspace/xuannv
torchrun --nproc_per_node=4 \
    scripts/train/train_ddp_v13.py --config configs/config.yaml \
    --save-every 20 --warmup-epochs 10
```

### 训练监控指标

| 指标 | 正常范围 | 第1个 epoch | 异常处理 |
|------|----------|-------------|----------|
| `raw_unif` | -4.0 ~ -1.0 | < -0.5 | > -0.5 持续 5 epoch → 报告坍缩 |
| `pre_unif` | 接近 raw_unif | 差距 < 0.5 | 差距大说明空间不一致 |
| `recon` | < 0.3 | < 1.0 (warmup) | warmup 后 > 0.5 → 检查数据 |
| `var_reg` | 接近 0 | < 0.5 | > 0.5 方差坍缩 |
| `orth` | < 0.3 | < 0.5 | > 0.5 权重不正交 |
| `decorr` | < 1.0 | < 2.0 | > 2.0 强相关 |

### 恢复训练

```bash
torchrun --nproc_per_node=4 \
    scripts/train/train_ddp_v13.py --config configs/config.yaml \
    --resume /workspace/outputs/.../epoch_best_xxx.pt
```

### 验证

```bash
python scripts/eval/validate_v12_auc.py --checkpoint /workspace/outputs/.../epoch_20.pt
```

## 对训练 Agent 的强制要求

1. **启动前检查 NPU 占用**，不要抢占正在运行的训练任务
2. **不要修改 `filter_2025_monthly`**，当前所有配置均为 `false`
3. **不要修改 `manifest_path`**，当前已指向云筛选后的正确目录
4. **重新生成统计数据**: 使用 `scripts/preprocessing/compute_statistics.py`
5. **重新筛选云数据**: `python scripts/preprocessing/filter_cloudy_frames.py --max-per-month 2 --cloud-threshold 0.3 --workers 16`
6. **每次修改后必须 git commit + push**
7. **训练出现 NaN/Inf**: 先检查 loss weight 是否过高，不要直接删除 checkpoint
8. **调试 AUC 低**: 优先检查 temporal contrastive loss / 双窗口数据 / `raw_unif`

---

## 实验管理与命名规范

### 输出目录命名（强制规范）

所有新实验的输出目录必须遵循以下统一命名格式：

```
{分类前缀}_{版本/阶段}_{实验名}_{日期}
```

| 元素 | 说明 | 示例 |
|------|------|------|
| 分类前缀 | `exp`(主实验) / `quick`(短训) / `ablation`(消融) / `base`(基线) / `_`(系统目录) | `exp_v2_E_pure_recon_7card_100ep_0523` |
| 版本/阶段 | `v2`, `v12`, `v13`, `round1~9`, `mb` 等 | `exp_v2_...` |
| 实验名 | 简洁核心特征 | `pure_recon`, `skipL2` |
| 日期 | 目录创建日期 `MMDD` | `_0523` |

**已整理的现有目录示例：**
- `exp_v2_E_pure_recon_7card_100ep_0523` — V2 主实验，纯重建，7卡，5月23日
- `quick_v2_baseline_10ep_0516` — V2 Quick 短训，基线，10 epoch，5月16日
- `ablation_mb_exp1_baseline_0516` — MB 消融实验1，基线，5月16日
- `round5_0517` — Round 5 系列实验，5月17日
- `base_aef_128d_0516` — AEF 基线，128维，5月16日

**禁止：**
- 冗长无意义命名如 `v13_round6_expX_recon005_full_staged_200patches`
- 项目名前缀重复如 `xuannv_v2_expA`
- 缺少日期后缀导致同类型目录难以区分

### 全量数据训练规范

- **必须使用全部 424 个 patch**
- `data.num_samples` 保持 `424`
- `manifest_path` 指向 `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered`

### NPU 使用规范

- 启动前执行 `npu-smi info`
- 多卡优先使用连续编号 NPU (如 0,1,2,3)
- 通过 `ASCEND_RT_VISIBLE_DEVICES` 指定设备
- **严禁 nohup，必须使用 tmux**

### 实验废弃与清理

- 确认失败的实验应立即删除目录，释放空间
- 保留 `epoch_best_*.pt`，其余可删

---

## 已知 Bug 与教训记录

### Bug 1: matplotlib `imshow(origin='lower')` 导致图像上下翻转

**时间**: 2026-05-15 | **影响**: 标注叠加卫星图像上下颠倒

- `rasterio.read()`: `row=0` 对应影像最北端 (y 值最大)
- `imshow(origin='upper', 默认)`: `row=0` 显示在顶部 → **正确**，与地理坐标一致
- `imshow(origin='lower')`: `row=0` 显示在底部 → **错误**，与地理坐标相反

**正确代码**:
```python
ax.imshow(rgb, extent=[left, right, bottom, top])  # 默认 origin='upper'
gdf_patch.boundary.plot(ax=ax, color='lime')
```

### Bug 2: Shapefile CRS 未正确处理导致标注映射错误

**时间**: 2026-05-15 | **影响**: Few-Shot CD AUC 从 ~0.65 → ~0.50 (接近随机)

哈尔滨 shapefile 原始 CRS 为 `None`，原代码直接跳过重投影。

**正确代码**:
```python
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")
if gdf.crs.to_epsg() != 32652:
    gdf = gdf.to_crs(epsg=32652)
```

### Bug 3: 变化检测验证使用错误时间窗口

**时间**: 2026-05-15 | **影响**: Few-Shot CD AUC 接近随机 (0.51)

**根因**:
1. 固定半年窗口错误 (BEFORE=2023-07~12, AFTER=2024-07~12)，但标注对应**2025年相邻月份** (4-6月、6-8月等)
2. S2 数据为 2025 年帧，旧固定窗口内无数据
3. 未按标注月份配对

**正确做法**: 按 shapefile 对应的 2025 年月份区间设置窗口:
- `june.shp` → 2025-04 → 2025-06
- `aug.shp` → 2025-06 → 2025-08
- `September.shp` → 2025-08 → 2025-09
- `October.shp` → 2025-09 → 2025-10

**教训**:
1. 验证前必须检查数据时间范围: `ls *.tif | cut -c1-6 | sort | uniq -c`
2. 标注时间信息必须与数据时间范围匹配
3. 变化检测必须按标注的月份区间配对
4. 哈尔滨变化检测标注全部为 **2025 年**变化，不存在 2023/2024 年标注

## 下游评估自动化 Pipeline

当需要对多个训练实验进行系统性下游评估（embedding 提取 → KNN/MLP/CD → 报告生成）时，使用项目级 skill `aef-evaluation-pipeline`。

### Skill 位置

`.kimi/skills/aef-evaluation-pipeline/SKILL.md`

### 触发条件

- 提到"评估"、"下游"、"embedding 提取"、"变化检测"、"AUC"
- 提到"对比实验"、"调参"、"KNN"、"MLP"、"Pipeline"
- 需要对多个实验做系统性对比

### 核心流程

1. **并行提取 Embedding**: 7 卡并行，断点续传，约 25 分钟/实验
2. **KNN 评估**: 3 任务（WorldCover/JRC Water/Dynamic World），k=5
3. **MLP 评估**: PixelMLPHead(hidden=256)，50 epochs
4. **变化检测 AUC**: 4 时期加权平均（Cosine + LR）
5. **报告生成**: Markdown 汇总 + 排名 + 结论分析

### 关键脚本

- `scripts/eval/launch_all_round4_eval.sh` — 7 卡并行提取
- `scripts/eval/launch_downstream_v2.sh` — 下游评估批量启动
- `scripts/eval/generate_report.py` — 报告生成（待实现）

### 常见陷阱

1. **NPU 设备映射**: `ASCEND_RT_VISIBLE_DEVICES=X` 时 PyTorch 内必须用 `npu:0`
2. **标签映射**: WorldCover 标签是 ESA 编码（10,30,40...），必须映射到 0-based
3. **JRC Water 过滤**: 必须过滤 `label < num_classes`，否则会保留 0-98 全部值

---
*最后更新: 2026-05-17*
