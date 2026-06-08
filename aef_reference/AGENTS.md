# AGENTS.md — aef_reference 项目上下文

> **目录**: `/workspace/xuannv/aef_reference/`
> **最后更新**: 2026-06-08

---

## 一、项目概述

### 1.1 这是什么项目

`aef_reference/` 是 **海淀区多源遥感嵌入底座** 的训练代码目录，基于 **AlphaEarth Foundations (AEF)** 架构进行改进和适配。

**核心目标**：
- 在海淀区 320 个 patch 的多源遥感数据上，训练一个输出 **64 维 embedding** 的深度学习模型
- 通过 **困难的多源重建任务** 迫使模型学习高质量的图像语义表征
- 同时 **蒸馏 AEF 官方预训练 embedding** 作为辅助监督，加速收敛并提升质量
- 最终输出的 64D embedding 供下游任务使用（变化检测、地物分类等）

### 1.2 解决什么问题

遥感数据在空间、时间、模态上高度冗余，普通 mask 策略（如 MAE 的 75% patch mask）对于多模态数据仍然过于简单。本项目通过：
- **5 源异构输入**（SAR + 多光谱 + 高分辨率），增加重建难度
- **时间窗口筛选**，聚焦特定季节的数据分布
- **AEF 蒸馏对齐**，让模型学习到经过大规模预训练验证的表征空间

### 1.3 要达到的效果

| 指标 | 目标 |
|------|------|
| Embedding 维度 | 64D 空间向量 |
| 与 AEF 对齐度 | Student PCA RGB 空间结构接近 AEF Official |
| 重建能力 | 能重建 5 源输入 + DEM/WorldCover/DynamicWorld/JRC Water |
| 下游可用性 | 64D embedding 可直接用于 kNN / 微调等下游任务 |

---

## 二、模型配置

### 2.1 输入源（5 源）

| 源名 | 通道数 | 空间分辨率 | 数据路径 |
|------|--------|-----------|----------|
| `s1` (Sentinel-1) | 2 (VH+VV) | ~10m | `data_raw/haidian/scenes/{patch_id}/s1/` |
| `s2` | 6 | 10m | `data_raw/haidian/scenes/{patch_id}/s2/` |
| `tianyi_sar` (天仪SAR) | 1 | ~3m | `data_raw/haidian/scenes/{patch_id}/tianyi_sar/` |
| `landsat` | 6 | 30m → 上采样到 10m | `data_raw/haidian/scenes/{patch_id}/landsat/` |
| `planet` | 4 | ~3-5m | `data_raw/beijing/planetscene/{patch_id}/` |

**注意**：
- `s1` 和 `tianyi_sar` 是两个**独立的 SAR 源**，同时保留
- Planet 数据不在 patch 目录下，单独存放在 `beijing/planetscene/`
- 各源独立做 Patch Embedding（不共享权重）

### 2.2 重建目标（解码源）

在 5 源输入基础上，模型可额外重建以下辅助目标。**这些目标是否参与训练、权重如何分配，可视具体实验需求灵活配置，不强制全部启用。**

| 目标 | 输入数据（标签） | 模型预测输出 | 损失类型 |
|------|-----------------|-------------|---------|
| `dem` | 1 通道（高程值） | 1 通道（回归值） | L1 |
| `worldcover` | **1 通道（类别索引）** | **11 通道（类别 logits）** | CrossEntropy |
| `dynamic_world` | **1 通道（类别索引）** | **9 通道（类别 logits）** | CrossEntropy |
| `jrc_water` | 1 通道（水体概率/掩码） | 1 通道（回归值） | L1 |

**说明**：
- WorldCover 和 Dynamic World 作为**输入标签**时，本质上是单通道的类别索引图像（每个像素一个整数类别值），和 DEM/JRC Water 一样是静态数据
- 模型为了做分类预测，输出端会生成 `num_classes` 通道的 logits，再与单通道标签通过 CrossEntropy 比较
- 上表中的"模型预测输出"列才是训练代码里 `decoder_channels[source]` 配置的数值

### 2.3 时间筛选

**范围**：`2025-12-01` 至 `2026-04-30`

- 各源只保留该时间窗口内的帧
- Planet 数据原本就约 6 帧（2025-12 ~ 2026-04），筛选后可能剩 3-6 帧
- 时间筛选在 `HaidianAEFDataset._load_source_frames()` 中实现，通过文件名 `YYYYMMDD.tif` 过滤

---

## 三、与 AEF 的对齐

### 3.1 架构对齐

- 输出 **64D embedding**，与 AEF 官方模型维度一致
- 使用 `VMFBottleneck` + `VonMisesFisherDecoder` 架构
- `per_source_latent=32`，5 源共 160D 输入到 Transformer

### 3.2 蒸馏对齐

- 加载 AEF 官方预计算的 64D embedding 作为软目标
- embedding 文件路径：`data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy`
- 蒸馏损失：cosine distance 或 L2 distance 对齐 Student 和 AEF embedding

### 3.3 两阶段训练策略

| 阶段 | 条件 | Distill 权重 | Recon 权重 | 目的 |
|------|------|-------------|-----------|------|
| **Stage 1: distill_align** | step ≤ 1000 | 5.0 | 0.1 | 先对齐 AEF 表征空间 |
| **Stage 2: normal** | step > 1000 | 0.2 | 1.0 | 再学习重建细节 |

---

## 四、训练配置

### 4.1 硬件

- 8 × Huawei Ascend 910B4 NPU
- 后端：`hccl`（华为集合通信库）
- 必须设置 `ASCEND_LAUNCH_BLOCKING=1` 规避 SDMA 竞态（见"已知坑"）

### 4.2 超参数

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
tmux send-keys -t aef_train 'torchrun --nproc_per_node=8 ...' Enter
```

---

## 五、可视化方案

### 5.1 触发时机

每 `save_every=500` step，对 **5 个指定 patch** 各生成一张可视化图。

### 5.2 指定 Patch

```python
viz_patch_ids = ["patch_000036", "patch_000069", "patch_000091", "patch_000120", "patch_000150"]
```

如果 patch 在 val set 中不存在，自动从 train set 中搜索。

### 5.3 布局（2 行）

```
┌─────────────────────────────────────────────────────────┐
│ Row 0: [S1] [S2] [TIANYI_SAR] [LANDSAT[30m→10m]] [PLANET] │
├─────────────────────────────────────────────────────────┤
│ Row 1: [Student PCA RGB] [AEF PCA RGB] [|Student-AEF|]   │
└─────────────────────────────────────────────────────────┘
```

**细节**：
- **第一行**：每个源取该 patch 时间中点最近的有效帧
  - Planet：跳过 collate 填充帧（全 0 帧），取第一个 `abs.max() > 0.001` 的帧
  - Landsat：标题标注 `[30m→10m]` 提示低分辨率上采样
  - 天仪 SAR：单通道显示为**灰度**（不是红色）
  - 每通道单独 min-max 归一化后取前 3 通道作为 RGB

- **第二行**：
  - **PCA 统一基**：以 AEF embedding 做 SVD，Student 用同一套基投影，保证颜色空间可比
  - 差异热图：`np.abs(student_rgb - aef_rgb).mean(axis=-1)`，hot colormap

**输出文件名**：`viz_step_{step:06d}_{patch_id}_seed{seed}.png`

---

## 六、关键文件路径

| 用途 | 路径 |
|------|------|
| **训练入口** | `aef_reference/train.py` |
| **模型定义** | `src/aef/architecture/aef_module.py` |
| **数据集** | `src/aef/data/haidian_dataset.py` |
| **collate_fn** | `src/aef/data/haidian_dataset.py` (同文件) |
| **损失函数** | `src/aef/loss_function.py` |
| **训练器** | `src/aef/training.py` |
| **海淀场景数据** | `data_raw/haidian/scenes/{patch_id}/` |
| **Planet 数据** | `data_raw/beijing/planetscene/{patch_id}/` |
| **统计量** | `statistics/haidian/{source}_stats.json` |
| **AEF 官方 embedding** | `data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy` |
| **训练输出** | `aef_reference/outputs/aef_distill_seed{seed}/` |
| **可视化输出** | `aef_reference/outputs/aef_distill_seed{seed}/visualizations/` |
| **本项目 AGENTS** | `aef_reference/AGENTS.md` (本文件) |

---

## 七、已知坑与注意事项

### 7.1 NPU 多卡 SDMA 竞态（致命）

**现象**：4 卡/8 卡训练时，`aclnnInplaceAddcdiv` 或 `aclnnInplaceAdd` 触发 `fftsplus sdma error`，TBE 子进程崩溃。

**根因**：CANN 8.5.1 多卡异步执行时的 SDMA 硬件竞态条件。

**解决**：`train.py` 已内置 `os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")`，强制 NPU op 同步执行。

**代价**：同步模式比异步慢约 20-30%，但完全稳定。

### 7.2 可视化 tensor detach（致命）

**现象**：`_visualize()` 中调用 `.cpu().numpy()` 时报错 `Can't call numpy() on Tensor that requires grad`。

**根因**：NPU + DDP 环境下，`@torch.no_grad()` 装饰器可能不完全生效。

**解决**：所有从模型输出取出的 tensor 必须显式 `.detach()`。

### 7.3 Planet 填充帧

**现象**：Planet 每 patch 约 6 帧，但 collate_fn 统一填充到 `max_t=16`，frame 6-15 全为 0。

**解决**：可视化时跳过填充帧（检查 `frame.abs().max() > 0.001`）。

### 7.4 Landsat 低分辨率

**现象**：Landsat 原始 30m 分辨率，上采样到 128×128 后空间分布不均匀（某些行非零像素少）。

**注意**：这是数据本身特性，非 bug。可视化标题标注 `[30m→10m]`。

### 7.5 Checkpoint 兼容性

**重要**：3 源 checkpoint（s1/s2/landsat）**不能 resume** 到 5 源模型（缺少 tianyi_sar 和 planet 的 stem 参数）。

**当前状态**：之前的 500 step 3 源权重已废弃，5 源训练需**从头开始**。

### 7.6 单通道显示

**注意**：1 通道数据（如 tianyi_sar）在 `_tensor_to_rgb()` 中复制为灰度图，不要显示为红色（只填 R 通道）。

### 7.7 投影头权重固定（待实现）

**需求**：参考 OlmoEarth 论文，各源独立的投影头（stem）权重建议固定（`requires_grad=False`），使每次投影一致，更好训练 STP（空间-时间处理）层。

**状态**：待训练启动前确认是否实现。

---

## 八、实验历史与当前状态

### 8.1 实验 1：3 源蒸馏（已废弃）

| 项目 | 内容 |
|------|------|
| 时间 | 2026-06-08 之前 |
| 输入源 | s1, s2, landsat（3 源） |
| 训练步数 | ~500 step |
| 中断原因 | ① NPU SDMA 竞态崩溃 ② 可视化 `.numpy()` 无 detach 报错 |
| 遗留权重 | `aef_reference/outputs/aef_distill_seed42/step_000500_seed42.pt` |
| 可用性 | ❌ 不能 resume 到 5 源模型 |

### 8.2 当前状态（5 源改造完成，待训练）

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

## 九、项目决策记录

### 9.1 数据源决策

- **S1 和天仪 SAR 同时保留**：两者是不同卫星（Sentinel-1 vs 天仪），物理特性不同，都作为独立输入源。
- **Planet 保留**：高分辨率（3-5m），虽然帧数少但信息密度高。

### 9.2 时间筛选决策

- **2025-12 ~ 2026-04**：聚焦冬季到春季的时段，Planet 数据恰好覆盖该窗口。

### 9.3 可视化决策

- **5 个指定 patch**：36, 69, 91, 120, 150 — 分散在空间上，覆盖不同地物类型。
- **PCA 统一基**：以 AEF 做 SVD，Student 用同一套基投影，差异图才有意义。
- **Planet 跳过填充帧**：避免显示全黑图。

### 9.4 训练策略决策

- **两阶段**：先蒸馏对齐（distill=5.0），再重建（recon=1.0）
- ** distill_warmup_steps=1000**：足够让 embedding 空间对齐后再学重建

---

## 十、Git 提交规范

- **分支**：`v12-clean-dynamic`
- **禁止**：推送到 `main`
- **每次修改后**：`git add -A && git commit -m "描述" && git push origin v12-clean-dynamic`

---

*本文档由 AI 编码代理根据用户要求和项目实际情况整理，供后续开发和维护参考。*
