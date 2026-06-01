<!-- AGENTS.md — xuannv 项目 AI 编码代理上下文 -->

## ⚠️ 强制规则（每次任务开始前必读）

1. **与用户交流必须使用中文回复**
2. **每次修改后必须执行**: `git add -A && git commit -m "描述" && git push origin main`（当前分支 `v12-clean-dynamic`）
3. **禁止使用 nohup 运行训练**，必须使用 `tmux`（nohup 会在会话断开时 kill torchrun DDP 进程）
4. **启动训练/推理前检查 NPU 占用**: `npu-smi info`
5. **所有文件操作限制在 `/workspace/xuannv/` 内**
6. **`archive/` 目录为废弃代码，只读参考，不得修改**
7. **训练出现 NaN/Inf**: 先检查 loss weight 是否过高，不要删除 checkpoint
8. **不要修改 `manifest_path`**: 已指向 `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered`
9. **不要修改 `filter_2025_monthly`**: 当前所有配置均为 `false`
10. **全量训练必须使用全部 424 个 patch**，`data.num_samples: 424`

---

## 项目概述

**xuannv**（包名 `xuannv`，版本 `0.1.0`）是 AlphaEarth Foundations (AEF) 的独立改进版，核心目标：

- 生成一次 embedding 能满足多个下游任务（变化检测、土地覆盖分类、水体识别等）
- 解决 embedding 坍缩，提升时间敏感性

**核心设计决策**:
- 输入只有 `S2`、`S1`、`Landsat` 三类时序图像；`DEM`/`WorldCover`/`Dynamic World`/`JRC Water` 仅作重建目标
- 训练时 `VMFBottleneck` skip L2 norm，在 pre-norm 空间计算反坍缩损失；推理时恢复标准 L2 + VMF 噪声
- 不重叠双窗口 + 时序对比损失提升时间区分能力

**硬件**: 8 × Huawei Ascend 910B4 NPU，`hccl` 后端，`torch 2.1.0 + torch_npu 2.1.0.post18`，conda 环境 `xuannv`（Python 3.11）

---

## 技术栈与构建系统

### 依赖管理
- **构建工具**: `setuptools>=61.0`（`pyproject.toml` 定义）
- **安装命令**: `pip install -e .`（在 `/workspace/xuannv` 下执行）
- **核心依赖**: `torch>=2.0`, `numpy`, `rasterio`, `geopandas`, `pyyaml`
- **运行环境**: conda `xuannv`，Python 3.11.15

### NPU 适配要点
- 所有 `.cuda()` → `.npu()`
- 所有 `torch.cuda` → `torch.npu`
- 所有 `torch.autocast(device_type="cuda")` → `torch.autocast(device_type="npu")`
- DDP 后端 `backend="nccl"` → `backend="hccl"`
- 所有训练/推理脚本必须 `import torch_npu`
- 设备选择统一走 `src.utils.device.get_device`

---

## 代码组织结构

```
xuannv/
├── pyproject.toml              # 包配置（setuptools，极简）
├── configs/
│   └── config.yaml             # 活跃训练配置（Round 9 基线）
├── src/                        # 核心源码（包名 xuannv，但实际通过 sys.path 直接 import src）
│   ├── config.py               # YAML → dataclass 配置加载（支持 _base_ 继承）
│   ├── data/
│   │   ├── dataset.py          # HarbinPatchDataset（月度采样，3输入/7目标，内存预加载）
│   │   ├── transforms.py       # 归一化、TIFF 读取、时间戳解析
│   │   ├── builder.py          # DataLoader 工厂（DistributedSampler）
│   │   └── multi_region_dataset.py  # 多区域混合数据集
│   ├── models/
│   │   ├── model.py            # AEFModel 主模型 + AEFOutput dataclass
│   │   ├── bottleneck.py       # VMFBottleneck（skip_l2_norm_training 核心）
│   │   ├── blocks.py           # STPEncoder / SpaceOperator / TimeOperator（手动 MHA）
│   │   ├── sensor_encoders.py  # SensorEncoderBank（多源独立 stem）
│   │   ├── decoders.py         # ContinuousDecoder / CategoricalDecoder（逐像素 MLP）
│   │   ├── heads.py            # ChangeDetectionHeadV1/V2/V3（轻量下游头）
│   │   ├── downstream_heads.py # PixelMLPHead / PixelConvHead
│   │   └── time_encoding.py    # TimeCode / WindowCode / RelativeTimeCode
│   ├── training/
│   │   ├── trainer.py          # DDPv13Trainer（唯一活跃训练器）
│   │   ├── losses.py           # 全部损失函数（raw_uniformity, VICReg, temporal 等）
│   │   ├── optimizer.py        # AdamW + cosine warmup 调度
│   │   ├── loops.py            # 重建损失辅助
│   │   ├── memory_bank.py      # EmbeddingMemoryBank
│   │   └── vicreg_loss.py      # VICReg 损失
│   ├── downstream/
│   │   └── heads.py            # SegmentationHead / ClassificationHead / ChangeDetectionHeadSimple
│   ├── inference/
│   │   └── engine.py           # 统一推理引擎（load_backbone, extract_embedding_for_month）
│   └── utils/
│       ├── checkpoint.py       # load/save_checkpoint（支持多 key 回退）
│       ├── device.py           # get_device（优先 NPU）
│       └── logging.py          # 日志工具
├── scripts/
│   ├── train/
│   │   └── train.py            # 训练入口（torchrun DDP）
│   ├── eval/
│   │   ├── launch_eval.sh      # 一站式评估启动（extract→knn→auc）
│   │   ├── extract_embeddings.py  # embedding 批量提取（支持分片并行）
│   │   ├── knn_eval.py         # KNN 下游评估（WorldCover/JRC Water/Dynamic World）
│   │   ├── auc_eval.py         # 变化检测 AUC 评估
│   │   ├── pipeline.py         # 完整评估流水线
│   │   ├── evaluate_cd_v2.py   # CD Head 评估
│   │   └── fewshot_*.py        # Few-Shot 评估脚本
│   ├── preprocessing/
│   │   ├── compute_statistics.py   # 计算 mean/std 统计量
│   │   └── filter_cloudy_frames.py # S2 云筛选
│   ├── test_smoke.py           # 冒烟测试（验证月度采样+前向+损失）
│   ├── monitor_training.py     # 训练监控脚本
│   └── ablation/               # 消融实验脚本
├── preprocessing/              # 数据预处理独立模块
│   ├── run.py                  # 预处理统一入口
│   ├── pipelines/              # 流水线编排
│   ├── downloaders/            # GEE / PlanetaryComputer / 本地导入
│   ├── processors/             # S2/S1/Landsat 后处理
│   ├── utils/                  # geo / tiff / logging 工具
│   └── viz/                    # 可视化
├── archive/                    # 废弃代码（只读参考，不得修改）
├── docs/                       # 项目文档（含 BUG_FIX_LOG.md）
└── data/                       # 运行时数据输出（change_masks, embeddings, labels）
```

---

## 模型关键接口

**`AEFModel.forward`** 主要参数:
`source_frames`, `source_timestamps_ms`, `source_frame_mask`, `source_input_mask`, `source_type_ids`, `valid_start_ms`, `valid_end_ms`, `target_relative_time`, `target_metadata`, `skip_decoder`, `dual_window`

**`AEFOutput`** 关键字段:
`embedding_map`, `embedding`, `pre_norm_embedding`, `pre_norm_map`, `reconstructions`, `dual_pre_w2`, `patch_id_logits`

**双窗口编码**:
- `encode_dual_window(...)` → `(emb_w1, emb_w2, pre_w1, pre_w2)`
- `encode_dual_window_explicit_diff(...)` → 额外返回 `change_score, diff_feat`

**推理引擎**:
```python
from src.inference.engine import load_backbone
model, dataset, cfg = load_backbone(
    config_path="configs/config.yaml",
    checkpoint_path="/workspace/outputs/.../epoch_40.pt",
    device="npu:0",
)
```

---

## 常用命令速查

```bash
# 环境激活
conda activate xuannv
cd /workspace/xuannv

# 启动训练（tmux 内执行）
torchrun --nproc_per_node=4 scripts/train/train.py \
    --config configs/config.yaml --save-every 20 --warmup-epochs 10

# 恢复训练
torchrun --nproc_per_node=4 scripts/train/train.py \
    --config configs/config.yaml --resume /workspace/outputs/.../epoch_best_xxx.pt

# 软重启（跨域迁移：加载权重，重置训练进度）
torchrun --nproc_per_node=4 scripts/train/train.py \
    --config configs/config.yaml --soft-restart /workspace/outputs/.../epoch_40.pt

# 冒烟测试
python scripts/test_smoke.py

# 变化检测 AUC 评估
python scripts/eval/auc_eval.py \
    --config configs/config.yaml --checkpoint /workspace/outputs/.../epoch_40.pt

# KNN 下游评估（需先提取 embedding）
python scripts/eval/knn_eval.py \
    --embedding-file /path/to/patch_embeddings.npz --output-dir out/ --device npu:0

# 一站式评估（提取 + KNN + AUC）
bash scripts/eval/launch_eval.sh \
    --checkpoint /workspace/outputs/.../epoch_40.pt --mode all

# 提取 embedding（7 卡并行）
bash scripts/eval/launch_eval.sh \
    --checkpoint /workspace/outputs/.../epoch_40.pt --mode extract --total-gpus 7

# 统计数据计算
python scripts/preprocessing/compute_statistics.py  # 哈尔滨默认路径

# 云筛选
python scripts/preprocessing/filter_cloudy_frames.py \
    --max-per-month 2 --cloud-threshold 0.3 --workers 16

# 预处理流水线
python preprocessing/run.py \
    --config preprocessing/configs/harbin.json \
    --steps patchify,download,cloud_filter,process,statistics

# 清理数据集缓存（约 26-32 GB/实验）
find /workspace/outputs -name "dataset_cache_*.pt" -delete

# Git 同步
git add -A && git commit -m "描述" && git push origin v12-clean-dynamic
```

---

## 训练系统

### 训练入口

- **活跃脚本**: `scripts/train/train.py`
- **活跃训练器**: `src/training/trainer.py`（类名 `DDPv13Trainer`，保留旧名以兼容 checkpoint）
- **活跃配置**: `configs/config.yaml`

### 损失函数速查 (`src/training/losses.py`)

| 函数 | 用途 |
|------|------|
| `raw_uniformity_loss` | 欧氏空间 uniformity，自适应 t=2/D |
| `batch_uniformity_loss_l2` | L2 空间 uniformity + 空间采样 + 维度 Dropout |
| `hyperspherical_uniformity_loss` | 球面 uniformity，直接防方向坍缩 |
| `pairwise_cosine_diversity_loss` | 均值两两余弦，坍缩时梯度非零 |
| `decorrelation_loss` | Barlow Twins 去相关 |
| `variance_regularizer` / `covariance_loss` | VICReg 方差/协方差正则 |
| `bottleneck_orthogonality_loss` | Conv1×1 权重正交约束 |
| `temporal_contrastive_loss` | 双窗口 hinge loss（纯斥力） |
| `gap_aware_temporal_cosine_loss` | 根据时间 gap 动态设定 target margin |
| `reconstruction_loss` | 掩码 L1（连续）/ CE（分类） |
| `inter_patch_infonce_loss` | Inter-Patch InfoNCE (NT-Xent) |
| `erank_maximization_loss` | SVD 奇异值熵最大化 |
| `coding_rate_loss` | MCR² log-det 优化 |
| `latent_mim_loss` | 潜在空间掩码预测（LMIM） |

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

调试 AUC 低时，优先检查：① `temporal_contrastive_loss` 是否生效 ② 双窗口数据是否正确生成 ③ `raw_unif` 是否正常

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

**AUC 目标**: > 0.7 及格，> 0.8 良好，> 0.85 优秀

**注意事项**:
- `ASCEND_RT_VISIBLE_DEVICES=X` 时 PyTorch 内必须用 `npu:0`
- WorldCover 标签是 ESA 编码（10,30,40...），`knn_eval.py` 内已自动映射到 0-based
- JRC Water `knn_eval.py` 已过滤 `label >= num_classes` 的无效值

---

## 数据状态

### 训练数据路径

```
/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/  ← 当前使用
    ├── s2/          (~22帧/patch，云筛选后)
    ├── s1 → ../harbin_scenes/s1
    ├── landsat → ../harbin_scenes/landsat
    ├── dem → ../harbin_scenes/dem
    ├── worldcover → ...
    └── ...

/workspace/statistics/harbin_scenes/{source}_stats.json     ← 归一化统计量（7个文件）
```

### 各源帧数

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

- 所有 Python 文件顶部使用 `from __future__ import annotations`
- 类型注解完整（PEP 484）
- 模块内注释使用中文
- 设备选择统一走 `src.utils.device.get_device`
- Checkpoint 统一走 `src.utils.checkpoint.load_checkpoint / save_checkpoint`
- 训练脚本顶部设置 `torch.set_num_threads(4)`
- 使用 `black` / `ruff` 风格（虽未在 CI 中强制，但保持一致的缩进与引号）

### 实验输出目录命名

格式：`{前缀}_{版本}_{实验名}_{MMDD}`

| 前缀 | 含义 |
|------|------|
| `exp` | 主实验 |
| `quick` | 快速短训（< 20 epoch） |
| `ablation` | 消融实验 |
| `base` | 基线 |

示例：`exp_v13_temporal_contrastive_0601`

**禁止**：`xuannv_v2_expA`（项目名前缀重复）、无日期后缀

### 文档命名

- 文件名必须包含日期后缀，格式 `YYYYMMDD`，例如 `BUG_FIX_20260527.md`
- 完成的计划文档归档到 `archive/docs/legacy/`

---

## 测试策略

本项目为研究型代码库，**无 pytest/unittest 自动化测试套件**。测试依靠以下机制：

1. **冒烟测试** (`scripts/test_smoke.py`): 验证月度采样数据集 + 模型前向 + 损失计算能完整跑通。每次重大修改后应运行。
2. **快速验证配置**: `configs/config_v27_quick_diag.yaml` 等 `quick_diag` 配置，使用 `max_steps_per_epoch: 20` 在数分钟内验证训练流程。
3. **max_patches 采样**: 在 `DataConfig` 中设置 `max_patches: 10` 可快速验证数据集逻辑。
4. **AUC / KNN 评估**: 作为端到端集成测试，验证 embedding 质量是否退化。

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
- **分布式**: `torchrun` + `hccl` 后端，推荐 4–8 卡 NPU
- **内存**: 数据集预加载后约 26–32 GB/实验（缓存文件位于 `/workspace/outputs/.cache_shared/`）
- **持久化**: Checkpoint 保存到 `/workspace/outputs/{experiment_name}/`

### 推理部署
- 单卡即可运行 embedding 提取与评估
- `scripts/eval/launch_eval.sh` 支持 7 卡并行提取（每卡处理 1/7 patches）
- 推理必须显式设置 `ASCEND_RT_VISIBLE_DEVICES=X`，且 PyTorch 内使用 `npu:0`

### 数据预处理部署
- `preprocessing/run.py` 为独立入口，不依赖训练环境初始化
- 输出到 `/workspace/raw/` 和 `/workspace/statistics/`，与训练代码解耦

---

## 安全与运维注意事项

1. **Git 安全**: 禁止直接 `git push --force` 或 `git reset --hard`；所有修改通过常规 commit 提交到 `v12-clean-dynamic` 分支。
2. **数据安全**: `/workspace/raw/` 和 `/workspace/statistics/` 为共享只读数据目录，训练脚本不应写入。
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

*最后更新: 2026-06-01*
