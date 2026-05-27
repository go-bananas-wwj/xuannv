# xuannv — Agent 上下文

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

# 清理数据集缓存（约 26-32 GB/实验）
find /workspace/outputs -name "dataset_cache_*.pt" -delete

# Git 同步
git add -A && git commit -m "描述" && git push origin v12-clean-dynamic
```

---

## 项目简介

**xuannv**（包名 `xuannv`）是 AlphaEarth Foundations (AEF) 的独立改进版，核心目标：

- 生成一次 embedding 能满足多个下游任务（变化检测、土地覆盖分类、水体识别等）
- 解决 embedding 坍缩，提升时间敏感性

**核心设计决策**:
- 输入只有 `S2`、`S1`、`Landsat` 三类时序图像；`DEM`/`WorldCover`/`Dynamic World`/`JRC Water` 仅作重建目标
- 训练时 `VMFBottleneck` skip L2 norm，在 pre-norm 空间计算反坍缩损失；推理时恢复标准 L2 + VMF 噪声
- 不重叠双窗口 + 时序对比损失提升时间区分能力

**硬件**: 8 × Huawei Ascend 910B4 NPU，`hccl` 后端，`torch 2.1.0 + torch_npu 2.1.0.post18`，conda 环境 `xuannv`（Python 3.11）

---

## 代码结构

```
xuannv/
├── configs/
│   └── config.yaml             # 活跃训练配置（Round 9 基线）
├── src/
│   ├── config.py               # YAML → dataclass 配置加载
│   ├── data/
│   │   ├── dataset.py          # HarbinPatchDataset（3输入/7目标，内存预加载）
│   │   ├── transforms.py       # 归一化、TIFF 读取、时间戳解析
│   │   └── builder.py          # DataLoader 工厂
│   ├── models/
│   │   ├── model.py            # AEFModel 主模型
│   │   ├── bottleneck.py       # VMFBottleneck
│   │   ├── blocks.py           # STPBlock（三路径，手动 MHA）
│   │   ├── sensor_encoders.py  # 多源独立 stem
│   │   ├── decoders.py         # ContinuousDecoder / CategoricalDecoder
│   │   ├── heads.py            # ChangeDetectionHead
│   │   ├── downstream_heads.py # PixelMLPHead / PixelConvHead
│   │   └── time_encoding.py    # TimeCode / WindowCode
│   ├── training/
│   │   ├── trainer.py          # DDPv13Trainer（唯一活跃训练器）
│   │   ├── losses.py           # 全部损失函数
│   │   ├── optimizer.py        # AdamW + cosine warmup
│   │   ├── loops.py            # 重建损失辅助
│   │   ├── memory_bank.py      # Memory Bank
│   │   └── vicreg_loss.py      # VICReg 损失
│   ├── inference/
│   │   └── engine.py           # 统一推理引擎
│   └── utils/
│       ├── checkpoint.py       # load/save_checkpoint
│       ├── device.py           # get_device（优先 NPU）
│       └── logging.py          # 日志工具
├── scripts/
│   ├── train/
│   │   └── train.py            # 训练入口（torchrun）
│   ├── eval/
│   │   ├── launch_eval.sh      # 一站式评估启动（--mode extract|knn|auc|all）
│   │   ├── extract_embeddings.py  # embedding 批量提取（支持分片并行）
│   │   ├── knn_eval.py         # KNN 下游评估
│   │   ├── auc_eval.py         # 变化检测 AUC 评估
│   │   ├── pipeline.py         # 完整评估流水线
│   │   ├── evaluate_cd_v2.py   # CD Head 评估
│   │   ├── evaluate_mlp_v2.py  # MLP 下游评估
│   │   └── fewshot_*.py        # Few-Shot 评估脚本
│   ├── preprocessing/
│   │   ├── compute_statistics.py   # 计算 mean/std 统计量
│   │   └── filter_cloudy_frames.py # S2 云筛选
│   └── test_smoke.py           # 冒烟测试
├── archive/                    # 废弃代码（只读参考）
├── docs/                       # 项目文档
└── data/                       # 运行时数据输出
```

---

## 模型关键接口

**`AEFModel.forward`** 主要参数:
`source_frames`, `source_timestamps_ms`, `source_frame_mask`, `source_input_mask`, `source_type_ids`, `valid_start_ms`, `valid_end_ms`, `target_relative_time`, `target_metadata`, `skip_decoder`, `dual_window`

**`AEFOutput`** 关键字段:
`embedding_map`, `embedding`, `pre_norm_embedding`, `pre_norm_map`, `reconstructions`

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
| `decorrelation_loss` | Barlow Twins 去相关 |
| `variance_regularizer` / `covariance_loss` | VICReg 方差/协方差正则 |
| `bottleneck_orthogonality_loss` | Conv1×1 权重正交约束 |
| `temporal_contrastive_loss` | 双窗口 hinge loss |
| `temporal_cosine_pixel_loss` | 像素级 cosine 时序损失 |
| `pixel_temporal_info_nce_loss` | 像素级 Anti-Diagonal InfoNCE |
| `gap_aware_temporal_cosine_loss` | 根据时间 gap 动态设定 target |
| `reconstruction_loss` | 掩码 L1（连续）/ CE（分类） |

### 训练监控指标

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `raw_unif` | -4.0 ~ -1.0 | **> -0.5 持续 5 epoch → embedding 坍缩，训练失败** |
| `pre_unif` | 接近 `raw_unif` | 差距 > 0.5 → 空间不一致 |
| `recon` | < 0.3 | warmup 后 > 0.5 → 检查数据路径 |
| `var_reg` | ≈ 0 | > 0.5 → 方差坍缩 |
| `orth` | < 0.3 | > 0.5 → 权重不正交 |
| `decorr` | < 1.0 | > 2.0 → 强相关 |

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
| Landsat | 407 | 47.9 | 17 个 patch 缺失，已优雅处理 |

### 变化检测标注时间窗口

哈尔滨变化检测标注全部为 **2025 年**，对应窗口：

| shapefile | before 窗口 | after 窗口 |
|-----------|------------|-----------|
| `june.shp` | 2025-04 | 2025-06 |
| `aug.shp` | 2025-06 | 2025-08 |
| `September.shp` | 2025-08 | 2025-09 |
| `October.shp` | 2025-09 | 2025-10 |

---

## 命名与文档规范

### 实验输出目录

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

### 开发规范

- `from __future__ import annotations`，类型注解完整
- 模块内注释使用中文
- 设备选择统一走 `src.utils.device.get_device`
- Checkpoint 统一走 `src.utils.checkpoint.load_checkpoint / save_checkpoint`
- 训练脚本顶部设置 `torch.set_num_threads(4)`

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

*最后更新: 2026-05-27*
