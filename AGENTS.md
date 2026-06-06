<!-- AGENTS.md — xuannv 项目 AI 编码代理上下文 -->

## ⚠️ 强制规则（每次任务开始前必读）

1. **与用户交流必须使用中文回复**。
2. **每次修改后必须执行**: `git add -A && git commit -m "描述" && git push origin v12-clean-dynamic`（当前活跃分支为 `v12-clean-dynamic`，**不要**推送到 `main`）。
3. **禁止使用 `nohup` 运行训练**，必须使用 `tmux`（`nohup` 会在会话断开时 kill `torchrun` DDP 子进程）。
4. **启动训练/推理前检查 NPU 占用**: `npu-smi info`。
5. **所有文件操作限制在 `/workspace/xuannv/` 内**。
6. **`archive/` 目录为废弃代码，只读参考，不得修改**。
7. **训练出现 NaN/Inf**: 先检查 loss weight 是否过高，不要删除 checkpoint；必要时执行 dummy backward 保持 DDP 同步。
8. **不要修改 `manifest_path`**: 哈尔滨配置已指向 `/workspace/xuannv/data_raw/harbin/scenes`，海淀配置已指向 `/workspace/xuannv/data_raw/haidian/scenes`。
9. **不要修改 `filter_2025_monthly`**: 当前所有配置均为 `false`。
10. **全量训练必须使用全部 patch**（哈尔滨 424 个，海淀 320 个），即 `data.num_samples` 按实际区域配置（若使用多区域混合训练，则按 manifest 实际 patch 数配置）。
11. **运行环境**: `conda activate xuannv`，Python 3.11，torch 2.1.0 + torch_npu 2.1.0.post18。

---

## 项目概述

**xuannv**（包名 `xuannv`，版本 `0.1.0`）是 AlphaEarth Foundations (AEF) 的独立改进版，核心目标：

- 生成一次 embedding 能满足多个下游任务（变化检测、土地覆盖分类、水体识别等）。
- 解决 embedding 坍缩，提升时间敏感性。

**核心设计决策**:
- **输入**只有 `S2`、`S1`、`Landsat` 三类时序图像；`DEM`/`WorldCover`/`Dynamic World`/`JRC Water` 仅作重建目标，不进入 encoder。
- **训练时** `VMFBottleneck` skip L2 norm，在 pre-norm 欧氏空间计算反坍缩损失（绕过 L2 Jacobian 梯度屏障）；**推理时**恢复标准 L2 + VMF 噪声，保证 embedding 分布在单位球面上。
- **不重叠双窗口** + 时序对比损失（hinge loss）提升时间区分能力。
- **双教师蒸馏**（可选）：同时蒸馏 AEF（64D）与 OlmoEarth（768D）的知识，通过投影头对齐。

**硬件**: 8 × Huawei Ascend 910B4 NPU，`hccl` 后端。

---

## 技术栈与构建系统

### 依赖管理
- **构建工具**: `setuptools>=61.0`（`pyproject.toml` 定义）。
- **安装命令**: `pip install -e .`（在 `/workspace/xuannv` 下执行）。
- **核心依赖**: `torch>=2.0`, `numpy`, `rasterio`, `geopandas`, `pyyaml`。
- **运行环境**: conda `xuannv`，Python 3.11.15。
- **无 CI/CD 自动化流水线**：`.github/workflows/` 不存在，测试与部署均靠手动执行。

### NPU 适配要点
- 所有 `.cuda()` → `.npu()`。
- 所有 `torch.cuda` → `torch.npu`。
- 所有 `torch.autocast(device_type="cuda")` → `torch.autocast(device_type="npu")`。
- DDP 后端 `backend="nccl"` → `backend="hccl"`。
- 所有训练/推理脚本必须 `import torch_npu`。
- 设备选择统一走 `src.utils.device.get_device`（优先 NPU）。

---

## 代码组织结构

```
xuannv/
├── pyproject.toml              # 包配置（setuptools，极简）
├── configs/
│   ├── config.yaml             # 活跃训练配置（支持 _base_ 继承）
│   ├── config_dual_teacher_v1.yaml
│   ├── config_haidian_v*.yaml  # 海淀区域系列实验配置
│   ├── config_v14_anti_collapse.yaml
│   ├── config_v27_quick_diag.yaml
│   ├── harbin_only_manifest.json
│   ├── multi_region_manifest.json
│   └── v14/
├── src/                        # 核心源码（运行时通过 sys.path 直接 import src）
│   ├── config.py               # YAML → dataclass 配置加载（支持 _base_ 继承）
│   ├── data/
│   │   ├── dataset.py          # HarbinPatchDataset（月度采样，3输入/7目标，内存预加载）
│   │   ├── multi_region_dataset.py  # 多区域混合数据集（Haidian + Harbin）
│   │   ├── transforms.py       # 归一化、TIFF 读取、时间戳解析
│   │   └── builder.py          # DataLoader 工厂（DistributedSampler）
│   ├── models/
│   │   ├── model.py            # AEFModel 主模型 + AEFOutput dataclass
│   │   ├── bottleneck.py       # VMFBottleneck（skip_l2_norm_training 核心）
│   │   ├── blocks.py           # STPEncoder / SpaceOperator / TimeOperator（手动 MHA）
│   │   ├── sensor_encoders.py  # SensorEncoderBank（多源独立 stem）
│   │   ├── decoders.py         # ContinuousDecoder / CategoricalDecoder（逐像素 MLP）
│   │   ├── heads.py            # ChangeDetectionHead / V2 / V3 / MultiClassChangeDetectionHead
│   │   ├── downstream_heads.py # PixelMLPHead / PixelConvHead
│   │   ├── distill_head.py     # OlmoEarth 蒸馏投影头
│   │   └── time_encoding.py    # TimeCode / WindowCode / RelativeTimeCode
│   ├── training/
│   │   ├── trainer.py          # DDPv13Trainer（唯一活跃训练器）
│   │   ├── losses.py           # 全部损失函数（raw_uniformity, VICReg, temporal, distill 等）
│   │   ├── optimizer.py        # AdamW + cosine warmup 调度
│   │   ├── loops.py            # 重建损失辅助
│   │   └── memory_bank.py      # EmbeddingMemoryBank
│   ├── downstream/
│   │   └── heads.py            # SegmentationHead / ClassificationHead / ChangeDetectionHeadSimple
│   ├── inference/
│   │   └── engine.py           # 统一推理引擎（load_backbone, extract_embedding_for_month）
│   └── utils/
│       ├── checkpoint.py       # load/save_checkpoint（支持多 key 回退）
│       ├── device.py           # get_device（优先 NPU）
│       └── __init__.py
├── scripts/
│   ├── train/
│   │   ├── train.py            # 训练入口（torchrun DDP）
│   │   ├── launch_v13.sh       # v13 训练启动脚本示例
│   │   └── train_dual_teacher.sh
│   ├── eval/
│   │   ├── launch_eval.sh      # 一站式评估启动（extract→knn→auc）
│   │   ├── extract_embeddings.py  # embedding 批量提取（支持分片并行）
│   │   ├── knn_eval.py         # KNN 下游评估（WorldCover/JRC Water/Dynamic World）
│   │   ├── auc_eval.py         # 变化检测 AUC 评估
│   │   ├── pipeline.py         # 完整评估流水线
│   │   ├── evaluate_cd_v2.py   # CD Head 评估
│   │   ├── run_periodic_eval.py # 周期完整下游评估（由 train.py 子进程调用）
│   │   └── fewshot_*.py        # Few-Shot 评估脚本
│   ├── distill/
│   │   └── generate_olmoearth_tokens_ddp.py  # OlmoEarth token 生成（8卡 DDP）
│   │   └── download_aef_*.py   # AEF 嵌入下载/生成脚本
│   ├── preprocessing/
│   │   ├── compute_statistics.py   # 计算 mean/std 统计量
│   │   └── filter_cloudy_frames.py # S2 云筛选
│   ├── test_smoke.py           # 冒烟测试（验证月度采样+前向+损失）
│   ├── profile_train_step.py   # 单卡 step profiling
│   └── visualize/              # 可视化脚本
├── preprocessing/              # 数据预处理独立模块（与训练代码解耦）
│   ├── run.py                  # 预处理统一入口（JSON 配置驱动）
│   ├── configs/                # harbin.json / haidian.json / national_china.json
│   ├── pipelines/              # 流水线编排（patchify, statistics, manifest, orchestrator）
│   ├── downloaders/            # GEE / PlanetaryComputer / 本地导入
│   ├── processors/             # S2/S1/Landsat 后处理
│   ├── utils/                  # geo / tiff / logging 工具
│   └── viz/                    # 可视化
├── archive/                    # 废弃代码（只读参考，不得修改）
│   └── src/training/vicreg_loss.py  # 旧版 VICReg 损失（已归档）
├── docs/                       # 项目文档（含 BUG_FIX_LOG.md、业务/专利/参考文献）
├── data_raw/                   # 原始训练数据（哈尔滨/海淀/北京/全国）
├── statistics/                 # 各区域归一化统计量（未纳入 git）
├── outputs/                    # 训练输出、评估结果、缓存（未纳入 git）
└── out/                        # 部分评估脚本的历史输出目录（未纳入 git）
```

**注意**：
- `src/utils/logging.py` **不存在**，日志功能由训练脚本内的 `FileLogger` 类或 `preprocessing/utils/logging.py` 提供。
- `src/training/vicreg_loss.py` **不存在**（已归档至 `archive/src/training/vicreg_loss.py`），当前 VICReg 相关逻辑直接内联在 `losses.py` / `trainer.py` 中。
- 运行时导入方式：训练/推理脚本普遍在顶部执行 `sys.path.insert(0, "/workspace/xuannv")` 后直接 `import src.xxx`，而非通过包名 `import xuannv`。

---

## 配置系统

配置采用 **YAML + Python dataclass**，入口为 `src.config.load_config`。

- **继承机制**: YAML 中可使用 `_base_: xxx.yaml` 进行配置继承，子配置覆盖父配置。
- **五大区块**: `experiment` / `data` / `model` / `training` / `evaluation`。
- **关键字段示例**:
  - `data.num_samples`: patch 总数（哈尔滨 424，海淀 320）。
  - `data.target_sources`: 列出 7 个重建目标及其 `loss_type`（0=连续/L1，1=分类/CE）。
  - `model.skip_l2_norm_training`: `true` 为当前默认（pre-norm 空间训练）。
  - `training.*_weight`: 大量损失权重开关，按需启用。
- **预处理配置**独立于训练配置，采用 JSON 格式，位于 `preprocessing/configs/`，由 `preprocessing/run.py` 消费。

---

## 模型关键接口

### `AEFModel.forward` 主要参数
`source_frames`, `source_timestamps_ms`, `source_frame_mask`, `source_input_mask`, `source_type_ids`, `valid_start_ms`, `valid_end_ms`, `target_relative_time`, `target_metadata`, `skip_decoder`, `dual_window`

### `AEFOutput` 关键字段
- `embedding_map`: `[B, D, H, W]` — 推理时 L2 归一化后的空间 embedding。
- `embedding`: `[B, D]` — 全局平均后的 embedding。
- `pre_norm_embedding`: `[B, D]` — L2 norm 前的原始幅度向量（反坍缩损失在此计算）。
- `pre_norm_map`: `[B, D, H, W]` — 空间 pre-norm embedding。
- `reconstructions`: `[B, T_tgt, C, H, W]` — 各源重建输出。
- `dual_pre_w2`: `[B, D, H, W]` — 第二窗口 pre_norm（用于 temporal loss）。
- `distill_map` / `distill_global`: 投影后的 768D 空间/全局向量（蒸馏用）。
- `patch_id_logits`: 实例判别头输出（打破坍缩用）。

### 双窗口编码
- `encode_dual_window(...)` → `(emb_w1, emb_w2, pre_w1, pre_w2)`
- `encode_dual_window_explicit_diff(...)` → 额外返回 `change_score, diff_feat`

### 推理引擎
```python
from src.inference.engine import load_backbone
model, dataset, cfg = load_backbone(
    config_path="configs/config.yaml",
    checkpoint_path="/workspace/outputs/.../epoch_40.pt",
    device="npu:0",
)
```

---

## 训练系统

### 训练入口
- **活跃脚本**: `scripts/train/train.py`
- **活跃训练器**: `src/training/trainer.py`（类名 `DDPv13Trainer`，保留旧名以兼容 checkpoint）
- **活跃配置**: `configs/config.yaml` 或 `configs/config_dual_teacher_v1.yaml` 等
- **旧版脚本已归档**：README 中提及的 `scripts/train/train_xuannv_v2.py` 实际位于 `archive/scripts/train_legacy/`，不可使用。

### 核心机制
- **Teacher-Student EMA**: Teacher 为 Student 的 EMA 副本（momentum ~0.996），不参与梯度更新。
- **Student 扰动**: 每 step 对 Student 输入做源级 drop、帧级 drop、前后截断（向量化实现，避免 NPU→CPU 同步）。
- **梯度累积**: 通过 `gradient_accumulation_steps` 配置。
- **Memory Bank**: 扩大 uniformity / VICReg 的有效 batch（默认大小 512）。
- **双教师蒸馏（可选）**:
  - AEF Teacher: 64D 直接对齐（cosine distance）。
  - OlmoEarth Teacher: 768D 通过 `distill_head` 投影后对齐。
  - 支持 Curriculum 学习：蒸馏权重从 `curriculum_start_weight` 渐进到 `curriculum_end_weight`。
- **Projector Warmup**: `distill_projector_warmup_epochs` 可在前 N 个 epoch 仅训练投影头，冻结 backbone。

### 损失函数速查 (`src/training/losses.py`)

| 函数 | 用途 | 计算空间 |
|------|------|----------|
| `raw_uniformity_loss` | 欧氏空间 uniformity，自适应 t=2/D | pre-norm |
| `batch_uniformity_loss_l2` | L2 空间 uniformity + 空间采样 + 维度 Dropout | L2-normed |
| `hyperspherical_uniformity_loss` | 球面 uniformity，直接防方向坍缩 | L2-normed |
| `pairwise_cosine_diversity_loss` | 均值两两余弦，坍缩时梯度非零 | L2-normed |
| `decorrelation_loss` | Barlow Twins 去相关 | pre-norm / gathered |
| `variance_regularizer` / `covariance_loss` | VICReg 方差/协方差正则 | pre-norm |
| `bottleneck_orthogonality_loss` | Conv1×1 权重正交约束 | 权重矩阵 |
| `temporal_contrastive_loss` | 双窗口 hinge loss（纯斥力） | pre-norm |
| `gap_aware_temporal_cosine_loss` | 根据时间 gap 动态设定 target margin | pre-norm |
| `reconstruction_loss` | 掩码 L1（连续）/ CE（分类） | 像素空间 |
| `inter_patch_infonce_loss` | Inter-Patch InfoNCE (NT-Xent) | pre-norm |
| `erank_maximization_loss` | 列方差熵最大化（SVD 替代，NPU-native） | pre-norm |
| `coding_rate_loss` | MCR² log-det 对角近似（NPU-native） | pre-norm |
| `latent_mim_loss` | 潜在空间掩码预测（LMIM） | pre-norm map |
| `pixel_change_supervision_loss` | 像素级变化弱监督 | pre-norm map |
| `aef_batch_uniformity_loss` | AEF 循环移位 batch uniformity | L2-normed |

### 训练监控指标

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `raw_unif` | -4.0 ~ -1.0 | **> -0.5 持续 5 epoch → embedding 坍缩，训练失败** |
| `pre_unif` | 接近 `raw_unif` | 差距 > 0.5 → 空间不一致 |
| `recon` | < 0.3 | warmup 后 > 0.5 → 检查数据路径 |
| `var_reg` | ≈ 0 | > 0.5 → 方差坍缩 |
| `orth` | < 0.3 | > 0.5 → 权重不正交 |
| `decorr` | < 1.0 | > 2.0 → 强相关 |
| `erank` | > 32 (D=64) | < 10 → 严重维度坍缩 |
| `active_dims` | 64/64 | < 50 → 维度坍缩 |
| `std_mean` | > 0.60 | < 0.50 → 方差不足 |

调试 AUC 低时，优先检查：① `temporal_contrastive_loss` 是否生效 ② 双窗口数据是否正确生成 ③ `raw_unif` 是否正常。

### tmux 训练管理

```bash
# 启动
tmux new-session -d -s train -c /workspace/xuannv
tmux send-keys -t train 'conda activate xuannv' Enter
tmux send-keys -t train 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3' Enter
tmux send-keys -t train 'torchrun --nproc_per_node=4 scripts/train/train.py --config configs/config.yaml --save-every 20' Enter
tmux detach -t train

# 查看日志
tmux capture-pane -t train -p | tail -20
# 终止
tmux kill-session -t train
```

### Checkpoint 路径规则

```
/workspace/outputs/{experiment_name}/
├── epoch_{N}.pt
├── epoch_best_epoch{N}.pt
└── train_YYYYMMDD_HHMMSS.log
```

---

## 评估工作流

### 一站式评估（推荐）

```bash
bash scripts/eval/launch_eval.sh \
    --checkpoint /workspace/outputs/exp_xxx_0601/epoch_40.pt \
    --mode all          # extract → knn → auc
```

### 分步评估

```bash
# 1. 提取 embedding（7 卡并行，约 25 分钟/实验）
bash scripts/eval/launch_eval.sh --checkpoint xxx.pt --mode extract --total-gpus 7

# 2. KNN 评估（3 任务：WorldCover/JRC Water/Dynamic World）
python scripts/eval/knn_eval.py \
    --embedding-file out/eval/patch_embeddings.npz --output-dir out/eval/knn/ \
    --device npu:0 --backend pytorch --k 5

# 3. AUC 评估
python scripts/eval/auc_eval.py \
    --config configs/config.yaml --checkpoint xxx.pt --device npu:0
```

**AUC 目标**: > 0.7 及格，> 0.8 良好，> 0.85 优秀。

**注意事项**:
- `ASCEND_RT_VISIBLE_DEVICES=X` 时 PyTorch 内必须用 `npu:0`。
- WorldCover 标签是 ESA 编码（10,30,40...），`knn_eval.py` 内已自动映射到 0-based。
- JRC Water `knn_eval.py` 已过滤 `label >= num_classes` 的无效值。

### 周期完整下游评估（训练内嵌）
`train.py` 每 `eval_every` 个 epoch 会自动调用 `scripts/eval/run_periodic_eval.py`，执行像素级 kNN mIoU + 变化检测 AUC 等完整评估，结果写入 `eval_epoch_{N}.json`。

---

## 数据状态

### 训练数据路径

```
/workspace/xuannv/data_raw/harbin/scenes/            ← 哈尔滨主数据（424 patches）
    ├── s2/          (~22帧/patch，云筛选后)
    ├── s1/
    ├── landsat/
    ├── dem/
    ├── worldcover/
    ├── dynamic_world/
    └── jrc_water/

/workspace/xuannv/data_raw/haidian/scenes/           ← 海淀数据（320 patches，多区域训练用）
/workspace/xuannv/statistics/harbin/{source}_stats.json   ← 哈尔滨归一化统计量
/workspace/xuannv/statistics/haidian/{source}_stats.json  ← 海淀归一化统计量
```

**注意**: `/workspace/xuannv/data_raw/` 与 `/workspace/xuannv/statistics/` 为本地目录（已纳入 `.gitignore`），并非 `/workspace/raw/` 共享路径。训练配置中的 `manifest_path` 均指向 `data_raw/` 下的子目录。

### 各源帧数（哈尔滨）

| 源 | Patch 数 | 均值帧数 | 备注 |
|----|---------|---------|------|
| S2（云筛选后） | 424 | 22.0 | 冬季 7.4% fallback |
| S1 | 424 | 42.7 | |
| Landsat | 407 | 17 个 patch 缺失，已优雅处理 | 47.9 |

### 变化检测标注时间窗口

哈尔滨变化检测标注全部为 **2025 年**，对应窗口：

| shapefile | before 窗口 | after 窗口 |
|-----------|------------|-----------|
| `june.shp` | 2025-04 | 2025-06 |
| `aug.shp` | 2025-06 | 2025-08 |
| `September.shp` | 2025-08 | 2025-09 |
| `October.shp` | 2025-09 | 2025-10 |

---

## 开发规范

### Python 代码风格
- 所有 Python 文件顶部使用 `from __future__ import annotations`。
- 类型注解完整（PEP 484）。
- 模块内注释使用中文。
- 设备选择统一走 `src.utils.device.get_device`。
- Checkpoint 统一走 `src.utils.checkpoint.load_checkpoint / save_checkpoint`。
- 训练脚本顶部设置 `torch.set_num_threads(4)`。
- 使用 `black` / `ruff` 风格（虽未在 CI 中强制，但保持一致的缩进与引号）。

### 实验输出目录命名
格式：`{前缀}_{版本}_{实验名}_{MMDD}`

| 前缀 | 含义 |
|------|------|
| `exp` | 主实验 |
| `quick` | 快速短训（< 20 epoch） |
| `ablation` | 消融实验 |
| `base` | 基线 |

示例：`exp_v13_temporal_contrastive_0601`

**禁止**: `xuannv_v2_expA`（项目名前缀重复）、无日期后缀。

### 文档命名
- 文件名必须包含日期后缀，格式 `YYYYMMDD`，例如 `BUG_FIX_20260527.md`。
- 完成的计划文档归档到 `archive/docs/legacy/`。

---

## 测试策略

本项目为研究型代码库，**无 pytest/unittest 自动化测试套件**。测试依靠以下机制：

1. **冒烟测试** (`scripts/test_smoke.py`): 验证月度采样数据集 + 模型前向 + 损失计算能完整跑通。每次重大修改后应运行。
2. **快速验证配置**: `configs/config_v27_quick_diag.yaml` 等 `quick_diag` 配置，使用 `max_steps_per_epoch: 20` 在数分钟内验证训练流程。
3. **max_patches 采样**: 在 `DataConfig` 中设置 `max_patches: 10` 可快速验证数据集逻辑。
4. **AUC / KNN 评估**: 作为端到端集成测试，验证 embedding 质量是否退化。
5. **Profiling**: `scripts/profile_train_step.py` 用于定位单卡 step 耗时瓶颈。

**测试执行示例**:
```bash
# 冒烟测试
python scripts/test_smoke.py

# 快速训练验证（20 step）
torchrun --nproc_per_node=2 scripts/train/train.py \
    --config configs/config_v27_quick_diag.yaml --save-every 1
```

---

## 部署与运行架构

### 训练部署
- **分布式**: `torchrun` + `hccl` 后端，推荐 4–8 卡 NPU。
- **内存**: 数据集预加载后约 26–32 GB/实验（缓存文件位于 `/workspace/outputs/.cache_shared/`）。
- **持久化**: Checkpoint 保存到 `/workspace/outputs/{experiment_name}/`。

### 推理部署
- 单卡即可运行 embedding 提取与评估。
- `scripts/eval/launch_eval.sh` 支持 7 卡并行提取（每卡处理 1/7 patches）。
- 推理必须显式设置 `ASCEND_RT_VISIBLE_DEVICES=X`，且 PyTorch 内使用 `npu:0`。

### 数据预处理部署
- `preprocessing/run.py` 为独立入口，不依赖训练环境初始化。
- 配置为 JSON 格式（`preprocessing/configs/harbin.json` 等）。
- 输出到 `data_raw/` 和 `statistics/` 目录下，与训练代码解耦。

---

## 安全与运维注意事项

1. **Git 安全**: 禁止直接 `git push --force` 或 `git reset --hard`；所有修改通过常规 commit 提交到 `v12-clean-dynamic` 分支。
2. **数据安全**: `data_raw/` 和 `statistics/` 为本地数据目录，受 `.gitignore` 保护；训练脚本不应意外覆盖。
3. **NPU 资源**: 训练前必须执行 `npu-smi info` 确认空闲卡；禁止在未确认的情况下占用全部 8 卡。
4. **缓存管理**: 数据集缓存（`dataset_cache_*.pt`）会累积占用大量磁盘空间，定期清理旧实验缓存。
5. **Checkpoint 清理**: `DDPv13Trainer.save_checkpoint` 会自动保留最近 3 个普通 checkpoint 和 3 个 best checkpoint，无需手动干预。

---

## 已知 Bug 记录

### Bug 1: `imshow(origin='lower')` 导致图像上下翻转

`rasterio.read()` 的 row=0 对应影像最北端，`imshow` 默认 `origin='upper'` 显示正确；改成 `origin='lower'` 会上下颠倒。

```python
# 正确
ax.imshow(rgb, extent=[left, right, bottom, top])  # 不加 origin='lower'
```

### Bug 2: Shapefile CRS 为 None 导致标注映射错误

哈尔滨 shapefile 原始 CRS 为 `None`，必须显式设置再重投影，否则 AUC 接近随机。

```python
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")
if gdf.crs.to_epsg() != 32652:
    gdf = gdf.to_crs(epsg=32652)
```

### Bug 3: 变化检测验证使用错误时间窗口

使用 2023/2024 固定窗口时，S2 数据（全为 2025 年）无法匹配，AUC ≈ 0.51（接近随机）。必须按标注 shapefile 对应的 2025 年月份配对（见上方"变化检测标注时间窗口"表格）。

---

*最后更新: 2026-06-06*
