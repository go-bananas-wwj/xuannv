# AGENTS.md — aef_reference 项目上下文

> **目录**: `/workspace/xuannv/aef_reference/`
> **最后更新**: 2026-06-08
> **活跃分支**: `v12-clean-dynamic`

---

## 零、隔离原则（⚠️ 最高优先级）

1. **`aef_reference/` 是独立工作空间**。虽然物理上位于 `/workspace/xuannv/` 下，但其源码、训练、输出全部隔离在此目录内部。
2. **严禁使用父目录 `/workspace/xuannv/src/` 下的任何代码**（包括但不限于 `src/aef/`、`src/models/`、`src/training/` 等）。
3. **严禁修改父目录 `/workspace/xuannv/src/` 下的任何文件**。所有模型、数据集、损失、训练器、工具类的修改必须在本目录内完成。
4. **`train.py` 只能 import 本目录内的模块或标准库 / 第三方库**。若现有 `train.py` 仍通过 `sys.path.insert` 引用外部 `src.aef.*`，应在后续迭代中逐步迁移到本目录内部实现。
5. 本目录已有的 `src/alphaearth/` 为本工作空间的参考实现，可自由修改以适配需求；`src/utils/` 为本工作空间专用工具。

---

## 一、项目概述

### 1.1 这是什么项目

`aef_reference/` 是 **海淀区多源遥感嵌入底座** 的独立训练工作空间。它基于 **AlphaEarth Foundations (AEF)** 架构进行改进和适配，所有代码自包含，不依赖外部 `src/`。

**核心目标**：
- 在海淀区 320 个 patch 的多源遥感数据上，训练一个输出 **64 维 embedding** 的深度学习模型
- 通过 **困难的多源重建任务** 迫使模型学习高质量的图像语义表征
- 同时 **蒸馏 AEF 官方预训练 embedding** 作为辅助监督，加速收敛并提升质量
- 最终输出的 64D embedding 供下游任务使用（变化检测、地物分类等）

### 1.2 与主项目的关系

```
/workspace/xuannv/
├── src/                        # ← 父目录源码，本工作空间禁止使用、禁止修改
│   ├── aef/
│   ├── models/
│   └── training/
└── aef_reference/              # ← 本独立工作空间
    ├── train.py                # 训练入口（仅 import 本目录内模块或 pip 包）
    ├── src/alphaearth/         # 本工作空间的核心/参考源码（可自由修改）
    ├── src/utils/              # 本工作空间专用工具
    ├── outputs/                # 训练输出隔离在此
    └── AGENTS.md               # 本文件
```

**关键区别**：
- `src/alphaearth/`：本工作空间的核心实现（原参考代码），面向海淀区 5 源数据，NPU 设备，可自由修改
- `src/utils/`：本工作空间专用工具函数
- **任何情况下都不要修改父目录 `src/aef/` 或 `src/models/` 等文件来影响本工作空间**

### 1.3 解决什么问题

遥感数据在空间、时间、模态上高度冗余，普通 mask 策略（如 MAE 的 75% patch mask）对于多模态数据仍然过于简单。本项目通过：
- **5 源异构输入**（SAR + 多光谱 + 高分辨率），增加重建难度
- **时间窗口筛选**，聚焦特定季节的数据分布
- **AEF 蒸馏对齐**，让模型学习到经过大规模预训练验证的表征空间

### 1.4 要达到的效果

| 指标 | 目标 |
|------|------|
| Embedding 维度 | 64D 空间向量 |
| 与 AEF 对齐度 | Student PCA RGB 空间结构接近 AEF Official |
| 重建能力 | 能重建 5 源输入 + DEM/WorldCover/DynamicWorld/JRC Water |
| 下游可用性 | 64D embedding 可直接用于 kNN / 微调等下游任务 |

---

## 二、技术栈与运行环境

### 2.1 硬件

- 8 × Huawei Ascend 910B4 NPU
- DDP 后端：`hccl`（华为集合通信库）
- 必须设置 `ASCEND_LAUNCH_BLOCKING=1` 规避 SDMA 竞态（见第七节）

### 2.2 软件依赖

- **运行环境**: `conda activate xuannv`，Python 3.11.15
- **PyTorch**: torch 2.1.0 + torch_npu 2.1.0.post18
- **核心库**: einops, rasterio, geopandas, numpy, matplotlib, tqdm, scipy, pandas
- **完整依赖列表**: 见主项目 `requirements.txt`（`aef_reference/` 下无独立的 requirements）

### 2.3 构建与安装

本工作空间独立运行，不依赖主项目的 `pyproject.toml`。直接安装所需依赖即可：

```bash
cd /workspace/xuannv/aef_reference
conda activate xuannv
pip install -r requirements.txt
```

若 `requirements.txt` 中未列全，可手动补充：`torch>=2.0`, `torch-npu`, `einops`, `rasterio`, `geopandas`, `numpy`, `matplotlib`, `tqdm`, `scipy`, `pandas`。

---

## 三、代码组织结构

### 3.1 活跃代码（实际运行依赖）

这些文件位于本工作空间内，在 `train.py` 运行时被直接加载，**仅修改本目录下的文件才会影响训练行为**（严禁触碰父目录 `src/aef/`）：

| 文件 | 职责 |
|------|------|
| `aef_reference/train.py` | 活跃训练入口：DDP 初始化、Dataset/DataLoader 构建、Trainer 启动 |
| `aef_reference/src/alphaearth/architecture/aef_module.py` | AlphaEarthFoundations 模型定义（含 TemporalSummarizer、TimePooling） |
| `aef_reference/src/alphaearth/architecture/encoder.py` | STPEncoder（三通路编码器） |
| `aef_reference/src/alphaearth/architecture/decoder.py` | VonMisesFisherDecoder（隐式解码器） |
| `aef_reference/src/alphaearth/architecture/encoder_utils.py` | IndividualSourceEncoder、SinusoidalTimeEncoding、SummaryPeriodEncoder |
| `aef_reference/src/alphaearth/data.py` / `data_olmoearth.py` | 数据集定义与 collate_fn（可在此基础上改造为 5 源海淀数据集） |
| `aef_reference/src/alphaearth/loss_function.py` | AEFLoss（重建 + uniformity + consistency + distill + 多种备用正则） |
| `aef_reference/src/alphaearth/training.py` | Trainer（单卡基础实现；DDP/EMA/可视化等改进需在本目录内完成） |

**注意**：若 `train.py` 当前仍通过 `sys.path.insert` 引用外部 `src/aef/`，后续迭代应逐步将所需模块复制/重构到 `aef_reference/src/alphaearth/` 或新建子包中，确保完全隔离。

### 3.2 参考代码（原始实现，修改不影响 train.py）

位于 `aef_reference/src/alphaearth/`，面向 OlmoEarth 数据集和 CUDA 环境：

| 文件 | 职责 |
|------|------|
| `src/alphaearth/__init__.py` | 包导出 |
| `src/alphaearth/architecture/` | 原始 AEF 架构（aef_module, encoder, decoder, STPBlock, stp_operators, encoder_utils, laplacian_pyramid_exchange） |
| `src/alphaearth/data.py` | 合成数据 / NPZ 数据集（AEFDataset, AEFNPZDataset） |
| `src/alphaearth/data_olmoearth.py` | OlmoEarth 预训练数据集（tar 包读取） |
| `src/alphaearth/loss_function.py` | 原始 AEFLoss（无 distill） |
| `src/alphaearth/training.py` | 原始 Trainer（单卡、无 EMA、无可视化） |
| `src/alphaearth/run_train.py` | 合成数据冒烟测试入口 |
| `src/alphaearth/run_train_olmoearth_dataset.py` | OlmoEarth 训练入口 |

### 3.3 模型架构要点

- **STP Encoder**: 3 条并行通路
  - Space Operator: ViT-like 空间自注意力，1/16L 分辨率
  - Time Operator: 时间轴自注意力，1/8L 分辨率
  - Precision Operator: 3×3 卷积，1/2L 分辨率
- **TemporalSummarizer**: 单 query 多头时间注意力 → 投影到 64D
  - **训练时 skip L2 norm**（保留幅度信息，pre-norm 空间计算反坍缩损失）
  - 推理时恢复 L2 归一化
- **VonMisesFisherDecoder**: 以 embedding 为均值方向，拼接 geometry + timecode 后逐像素 MLP 解码
- **Teacher-Student 扰动**: 随机源丢弃、帧丢弃、半周期截断（向量化实现，避免 NPU→CPU 同步）

---

## 四、训练配置

### 4.1 超参数

| 参数 | 值 |
|------|-----|
| batch_size | 2 per GPU |
| grad_accum_steps | 2 |
| 等效 batch | 2 × 8 × 2 = 32 |
| lr | 1e-4 |
| warmup_steps | 2000 |
| max_steps | 100000 |
| optimizer | AdamW |
| weight_decay | 0.05 |
| save_every | 500 step |
| eval_every | 500 step |
| log_every | 50 step |
| distill_warmup_steps | 1000 |
| EMA momentum | 0.996 |
| gradient clipping max_norm | 1.0 |

### 4.2 模型配置

```python
input_sources = {
    "s1": 2,
    "s2": 6,
    "tianyi_sar": 1,
    "landsat": 6,
    "planet": 4,
}
decode_sources = {
    "s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4,
    "dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1,
}
per_source_latent = 32
model_size = "small"
```

### 4.3 启动命令

```bash
cd /workspace/xuannv
conda activate xuannv
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=29500 \
  aef_reference/train.py \
  --batch-size 2 \
  --max-steps 100000 \
  --save-every 500 \
  --eval-every 500 \
  --distill-warmup-steps 1000 \
  --grad-accum-steps 2 \
  --log-every 50 \
  --seed 42
```

**禁止用 nohup**，必须用 `tmux`：
```bash
tmux new-session -d -s aef_train -c /workspace/xuannv
tmux send-keys -t aef_train 'conda activate xuannv' Enter
tmux send-keys -t aef_train 'torchrun --nproc_per_node=8 aef_reference/train.py --batch-size 2 --max-steps 100000 --save-every 500 --eval-every 500 --distill-warmup-steps 1000 --grad-accum-steps 2 --log-every 50 --seed 42' Enter
```

### 4.4 恢复训练

```bash
torchrun --nproc_per_node=8 aef_reference/train.py \
  --resume aef_reference/outputs/aef_distill_seed42/step_000500_seed42.pt \
  ...（其他参数保持一致）
```

**注意**：3 源 checkpoint（s1/s2/landsat）**不能 resume** 到 5 源模型（缺少 tianyi_sar 和 planet 的 stem 参数）。

---

## 五、数据源与路径

### 5.1 输入源（5 源）

| 源名 | 通道数 | 空间分辨率 | 数据路径 |
|------|--------|-----------|----------|
| `s1` (Sentinel-1) | 2 (VH+VV) | ~10m | `data_raw/haidian/scenes/{patch_id}/s1/` |
| `s2` | 6 | 10m | `data_raw/haidian/scenes/{patch_id}/s2/` |
| `tianyi_sar` (天仪SAR) | 1 | ~3m | `data_raw/haidian/scenes/{patch_id}/tianyi_sar/` |
| `landsat` | 6 | 30m → 上采样到 10m | `data_raw/haidian/scenes/{patch_id}/landsat/` |
| `planet` | 4 | ~3-5m | `data_raw/beijing/planetscene/{patch_id}/` |

- `s1` 和 `tianyi_sar` 是两个**独立的 SAR 源**，同时保留
- Planet 数据不在 patch 目录下，单独存放在 `beijing/planetscene/`
- 各源独立做 Patch Embedding（不共享权重）

### 5.2 重建目标（解码源）

| 目标 | 输入数据（标签） | 模型预测输出 | 损失类型 | 权重 |
|------|-----------------|-------------|---------|------|
| `s1` | (T, H, W, 2) | (B, H, W, 2) | L1 | 1.0 |
| `s2` | (T, H, W, 6) | (B, H, W, 6) | L1 | 1.0 |
| `tianyi_sar` | (T, H, W, 1) | (B, H, W, 1) | L1 | 1.0 |
| `landsat` | (T, H, W, 6) | (B, H, W, 6) | L1 | 1.0 |
| `planet` | (T, H, W, 4) | (B, H, W, 4) | L1 | 1.0 |
| `dem` | 1 通道（高程值） | 1 通道（回归值） | L1 | 0.05 |
| `worldcover` | **1 通道（类别索引）** | **11 通道（类别 logits）** | CrossEntropy | 0.5 |
| `dynamic_world` | **1 通道（类别索引）** | **9 通道（类别 logits）** | CrossEntropy | 0.5 |
| `jrc_water` | 1 通道（水体概率/掩码） | 1 通道（回归值） | L1 | 0.3 |

**说明**：
- WorldCover 和 Dynamic World 作为**输入标签**时，本质上是单通道的类别索引图像（每个像素一个整数类别值），和 DEM/JRC Water 一样是静态数据
- 模型为了做分类预测，输出端会生成 `num_classes` 通道的 logits，再与单通道标签通过 CrossEntropy 比较

### 5.3 其他关键路径

| 用途 | 路径 |
|------|------|
| 海淀场景数据 | `data_raw/haidian/scenes/{patch_id}/` |
| Planet 数据 | `data_raw/beijing/planetscene/{patch_id}/` |
| 统计量（mean/std） | `statistics/haidian/{source}_stats.json` |
| AEF 官方 embedding | `data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy` |
| 训练输出 | `aef_reference/outputs/aef_distill_seed{seed}/` |
| 可视化输出 | `aef_reference/outputs/aef_distill_seed{seed}/visualizations/` |
| 时间映射缓存 | `aef_reference/.cache/temporal_mapping.json`（若需要，请建在本目录内） |

### 5.4 时间筛选

**范围**：`2025-12-01` 至 `2026-04-30`

- 各源只保留该时间窗口内的帧
- Planet 数据原本就约 6 帧（2025-12 ~ 2026-04），筛选后可能剩 3-6 帧
- 时间筛选在 `HaidianAEFDataset._load_source_frames()` 中实现，通过文件名 `YYYYMMDD.tif` 过滤

---

## 六、训练策略

### 6.1 两阶段蒸馏

| 阶段 | 条件 | Distill 权重 | Recon 权重 | 目的 |
|------|------|-------------|-----------|------|
| **Stage 1: distill_align** | step ≤ 1000 | 5.0 | 0.1 | 先对齐 AEF 表征空间 |
| **Stage 2: normal** | step > 1000 | 0.2 | 1.0 | 再学习重建细节 |

### 6.2 EMA Teacher

- 始终启用（无论单卡/多卡）
- EMA 更新公式：`p_ema = 0.996 * p_ema + 0.004 * p`
- EMA 模型用于覆盖 teacher_embeddings，不直接参与梯度更新

### 6.3 损失函数组成

| 损失 | 默认权重 | 计算空间 |
|------|---------|----------|
| reconstruction | 1.0（stage1 时 0.1） | 像素空间 |
| uniformity (batch) | 0.05 | L2-normed embedding |
| consistency | 0.02 | L2-normed teacher/student |
| distill (AEF 对齐) | 0.2 / 5.0 | cosine distance |
| clip (text) | 0.001 | L2-normed image/text |

备用损失（默认权重 0.0，可通过修改代码启用）：raw_uniform, variance, covariance, decorr, erank, coding_rate, magnitude

---

## 七、可视化方案

### 7.1 触发时机

每 `save_every=500` step，对 **5 个指定 patch** 各生成一张可视化图。

### 7.2 指定 Patch

```python
viz_patch_ids = ["patch_000036", "patch_000069", "patch_000091", "patch_000120", "patch_000150"]
```

如果 patch 在 val set 中不存在，自动从 train set 中搜索（通过 `dataset[idx]` 遍历）。

### 7.3 布局（2 行）

```
┌─────────────────────────────────────────────────────────┐
│ Row 0: [S1] [S2] [TIANYI_SAR] [LANDSAT[30m→10m]] [PLANET] │
├─────────────────────────────────────────────────────────┤
│ Row 1: [Student PCA RGB] [AEF PCA RGB] [|Student-AEF|]   │
└─────────────────────────────────────────────────────────┘
```

**第一行细节**：
- 每个源取该 patch 时间中点最近的有效帧
- Planet：跳过 collate 填充帧（全 0 帧），取第一个 `abs.max() > 0.001` 的帧
- Landsat：标题标注 `[30m→10m]` 提示低分辨率上采样
- 天仪 SAR：单通道显示为**灰度**（不是红色）
- 每通道单独 min-max 归一化后取前 3 通道作为 RGB

**第二行细节**：
- **PCA 统一基**：以 AEF embedding 做 SVD，Student 用同一套基投影，保证颜色空间可比
- 差异热图：`np.abs(student_rgb - aef_rgb).mean(axis=-1)`，hot colormap

**输出文件名**：`viz_step_{step:06d}_{patch_id}_seed{seed}.png`

---

## 八、已知坑与注意事项

### 8.1 NPU 多卡 SDMA 竞态（致命）

**现象**：4 卡/8 卡训练时，`aclnnInplaceAddcdiv` 或 `aclnnInplaceAdd` 触发 `fftsplus sdma error`，TBE 子进程崩溃。

**根因**：CANN 8.5.1 多卡异步执行时的 SDMA 硬件竞态条件。

**解决**：`train.py` 已内置 `os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")`，强制 NPU op 同步执行。

**代价**：同步模式比异步慢约 20-30%，但完全稳定。

### 8.2 可视化 tensor detach（致命）

**现象**：`_visualize()` 中调用 `.cpu().numpy()` 时报错 `Can't call numpy() on Tensor that requires grad`。

**根因**：NPU + DDP 环境下，`@torch.no_grad()` 装饰器可能不完全生效。

**解决**：所有从模型输出取出的 tensor 必须显式 `.detach()`。

### 8.3 Planet 填充帧

**现象**：Planet 每 patch 约 6 帧，但 collate_fn 统一填充到 `max_t=16`，frame 6-15 全为 0。

**解决**：可视化时跳过填充帧（检查 `frame.abs().max() > 0.001`）。

### 8.4 Landsat 低分辨率

**现象**：Landsat 原始 30m 分辨率，上采样到 128×128 后空间分布不均匀（某些行非零像素少）。

**注意**：这是数据本身特性，非 bug。可视化标题标注 `[30m→10m]`。

### 8.5 Checkpoint 兼容性

**重要**：3 源 checkpoint（s1/s2/landsat）**不能 resume** 到 5 源模型（缺少 tianyi_sar 和 planet 的 stem 参数）。

**当前状态**：之前的 500 step 3 源权重已废弃，5 源训练需**从头开始**。

### 8.6 单通道显示

**注意**：1 通道数据（如 tianyi_sar）在 `_tensor_to_rgb()` 中复制为灰度图，不要显示为红色（只填 R 通道）。

### 8.7 各源独立投影头固定（待实现）

**需求**：参考 OlmoEarth 论文，各源独立的投影头（stem）权重建议固定（`requires_grad=False`），使每次投影一致，更好训练 STP 层。

**状态**：待训练启动前确认是否实现。

---

## 九、实验历史与当前状态

### 9.1 实验 1：3 源蒸馏（已废弃）

| 项目 | 内容 |
|------|------|
| 时间 | 2026-06-08 之前 |
| 输入源 | s1, s2, landsat（3 源） |
| 训练步数 | ~500 step |
| 中断原因 | ① NPU SDMA 竞态崩溃 ② 可视化 `.numpy()` 无 detach 报错 |
| 遗留权重 | `aef_reference/outputs/aef_distill_seed42/step_000500_seed42.pt` |
| 可用性 | ❌ 不能 resume 到 5 源模型 |

### 9.2 当前状态（5 源改造完成，待训练）

| 项目 | 状态 |
|------|------|
| 输入源扩展 | ✅ 5 源（s1/s2/tianyi_sar/landsat/planet） |
| 时间筛选 | ✅ 2025-12-01 ~ 2026-04-30 |
| 损失函数 | ✅ 新增 tianyi_sar/planet 配置 |
| 可视化重设计 | ✅ 2 行布局 + 5 指定 patch |
| NPU SDMA 修复 | ✅ ASCEND_LAUNCH_BLOCKING=1 |
| 投影头固定 | ⏳ 待确认是否实现 |
| **训练状态** | ⏳ **等待用户确认后启动** |

---

## 十、开发规范

### 10.1 Git 提交规范

- **分支**：`v12-clean-dynamic`
- **禁止**：推送到 `main`
- **每次修改后必须执行**：`git add -A && git commit -m "描述" && git push origin v12-clean-dynamic`

### 10.2 代码风格

- 所有 Python 文件顶部使用 `from __future__ import annotations`
- 类型注解完整（PEP 484）
- 模块内注释使用中文
- 设备选择：活跃代码统一使用 `npu`，参考代码使用 `cuda`（不要混改）
- 训练脚本顶部设置 `torch.set_num_threads(4)`

### 10.3 文件修改范围

- **所有文件操作严格限制在 `/workspace/xuannv/aef_reference/` 内**
- **`archive/` 目录（若存在）为废弃代码，只读参考，不得修改**
- **严禁修改 `/workspace/xuannv/src/` 下的任何文件**（包括 `src/aef/`、`src/models/`、`src/training/` 等父目录源码）
- 若发现 `train.py` 仍引用外部 `src.aef.*`，应在本目录内创建替代模块并修改 import 路径，而不是去改父目录

### 10.4 测试与验证

本项目为研究型代码库，**无 pytest/unittest 自动化测试套件**。验证方式：

1. **合成数据冒烟测试**：
   ```bash
   cd /workspace/xuannv/aef_reference
   python -m alphaearth.run_train
   ```
   （验证单卡前向 + 损失 + 优化器步进能否跑通）

2. **OlmoEarth 子集训练**：
   ```bash
   python -m alphaearth.run_train_olmoearth_dataset \
       --data_dir ./data/olmoearth_pretrain_dataset/10_landsat_monthly \
       --batch_size 4 --max_steps 100
   ```

3. **可视化预览**：运行 `train.py` 前，可先通过 `outputs/viz_preview/` 下的历史图片确认可视化布局是否符合预期。

4. **NPU 占用检查**：训练前执行 `npu-smi info` 确认卡空闲。

---

## 十一、项目决策记录

### 11.1 数据源决策

- **S1 和天仪 SAR 同时保留**：两者是不同卫星（Sentinel-1 vs 天仪），物理特性不同，都作为独立输入源。
- **Planet 保留**：高分辨率（3-5m），虽然帧数少但信息密度高。

### 11.2 时间筛选决策

- **2025-12 ~ 2026-04**：聚焦冬季到春季的时段，Planet 数据恰好覆盖该窗口。

### 11.3 可视化决策

- **5 个指定 patch**：36, 69, 91, 120, 150 — 分散在空间上，覆盖不同地物类型。
- **PCA 统一基**：以 AEF 做 SVD，Student 用同一套基投影，差异图才有意义。
- **Planet 跳过填充帧**：避免显示全黑图。

### 11.4 训练策略决策

- **两阶段**：先蒸馏对齐（distill=5.0），再重建（recon=1.0）
- **distill_warmup_steps=1000**：足够让 embedding 空间对齐后再学重建

---

*本文档由 AI 编码代理根据实际项目内容整理，供后续开发和维护参考。*
