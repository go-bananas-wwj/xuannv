# xuannv — Copilot Instructions

## 强制规则

- **与用户交流必须使用中文回复**
- 每次修改后执行: `git add -A && git commit -m "描述" && git push origin main`（当前分支 `v12-clean-dynamic`）
- **禁止使用 `nohup` 运行训练**，必须使用 `tmux`（nohup 会在会话断开时 kill torchrun DDP 进程）
- 启动训练/推理前检查 NPU 占用: `npu-smi info`
- 所有文件操作限制在 `/workspace/xuannv/` 内
- `archive/` 目录为废弃代码，只读参考，不得修改
- 训练出现 NaN/Inf：先检查 loss weight 是否过高，不要删除 checkpoint
- 不要修改 `manifest_path`（已指向 `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered`）
- 不要修改 `filter_2025_monthly`（当前所有配置均为 `false`）
- 全量训练必须使用全部 424 个 patch：`data.num_samples: 424`

## 环境与常用命令

```bash
conda activate xuannv
cd /workspace/xuannv

# 冒烟测试
python scripts/test_smoke.py

# 启动训练（tmux 内）
tmux new-session -d -s train -c /workspace/xuannv
tmux send-keys -t train 'conda activate xuannv && export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3' Enter
tmux send-keys -t train 'torchrun --nproc_per_node=4 scripts/train/train.py --config configs/config.yaml --save-every 20 --warmup-epochs 10' Enter

# 恢复训练
torchrun --nproc_per_node=4 scripts/train/train.py --config configs/config.yaml --resume /workspace/outputs/.../epoch_best_xxx.pt

# 一站式评估（提取 embedding → KNN → AUC）
bash scripts/eval/launch_eval.sh --checkpoint /workspace/outputs/.../epoch_40.pt --mode all

# 分步评估
bash scripts/eval/launch_eval.sh --checkpoint xxx.pt --mode extract --total-gpus 7
python scripts/eval/knn_eval.py --embedding-file out/eval/patch_embeddings.npz --output-dir out/eval/knn/ --device npu:0 --backend pytorch --k 5
python scripts/eval/auc_eval.py --config configs/config.yaml --checkpoint xxx.pt --device npu:0

# 清理数据集缓存（约 26-32 GB/实验）
find /workspace/outputs -name "dataset_cache_*.pt" -delete
```

> **`ASCEND_RT_VISIBLE_DEVICES=X` 时 PyTorch 内必须用 `npu:0`**

## 项目架构

xuannv 是 AlphaEarth Foundations (AEF) 的改进版，核心目标：解决 embedding 坍缩 + 提升时间敏感性，使单次推理的 embedding 能支撑变化检测、土地覆盖分类、水体识别等多个下游任务。

**硬件**: 8 × Huawei Ascend 910B4 NPU，`hccl` 后端，`torch 2.1.0 + torch_npu 2.1.0.post18`，conda 环境 `xuannv`（Python 3.11）

### 数据流

```
输入源（带时间戳）: S2 (6ch) + S1 (2ch) + Landsat (6ch)
    ↓ SensorEncoderBank（每个源独立 stem）
    ↓ STPEncoder（时序 + 空间 Transformer）
    ↓ VMFBottleneck（训练时 skip L2，推理时恢复 L2 + VMF 噪声）
    ↓ embedding_map [B, D, H, W] + pre_norm_map
    ↓ ContinuousDecoder / CategoricalDecoder（各源独立 decoder）
重建目标（不输入 encoder）: DEM + WorldCover + Dynamic World + JRC Water
```

### 反坍缩核心机制

训练时 `VMFBottleneck(skip_l2_norm_training=True)` 在 pre-norm 欧氏空间计算四件套损失，避免 L2 Norm 的梯度屏障（L2 Jacobian 在坍缩态秩降为 D-1）。推理时恢复标准 L2 + VMF 噪声保证 embedding 在球面上。

### 时间敏感性机制

训练时 50% 概率随机裁剪 `valid_period` 为 4–24 帧窗口（`temporal_window_augmentation`）。不重叠双窗口 + 时序对比损失（hinge loss）强化 before/after 区分能力。

## 关键文件

| 文件 | 职责 |
|------|------|
| `configs/config.yaml` | 活跃训练配置（Round 9 基线） |
| `src/models/model.py` | `AEFModel` 主模型 + `AEFOutput` 数据类 |
| `src/models/bottleneck.py` | `VMFBottleneck`（核心反坍缩改进） |
| `src/models/blocks.py` | `STPBlock`（三路径手动 MHA） |
| `src/training/trainer.py` | `DDPv13Trainer`（唯一活跃训练器） |
| `src/training/losses.py` | 全部损失函数 |
| `src/data/dataset.py` | `HarbinPatchDataset`（3 输入/7 目标，内存预加载） |
| `src/inference/engine.py` | 统一推理引擎 `load_backbone` |
| `src/config.py` | YAML → dataclass 配置加载 |

## 模型接口

```python
# 推理引擎（推荐用法）
from src.inference.engine import load_backbone
model, dataset, cfg = load_backbone(
    config_path="configs/config.yaml",
    checkpoint_path="/workspace/outputs/.../epoch_40.pt",
    device="npu:0",
)

# 双窗口变化检测
emb_w1, emb_w2, pre_w1, pre_w2 = model.encode_dual_window(...)
change_score, diff_feat = model.encode_dual_window_explicit_diff(...)  # 额外返回
```

`AEFOutput` 关键字段：`embedding_map [B,D,H,W]`、`embedding [B,D]`、`pre_norm_embedding [B,D]`、`pre_norm_map [B,D,H,W]`、`reconstructions`

## 训练监控

| 指标 | 正常范围 | 异常 → 原因 |
|------|----------|------------|
| `raw_unif` | -4.0 ~ -1.0 | **> -0.5 持续 5 epoch → embedding 坍缩，立即报告** |
| `pre_unif` | 接近 `raw_unif` | 差距 > 0.5 → 空间不一致 |
| `recon` | < 0.3 | warmup 后 > 0.5 → 检查数据路径 |
| `var_reg` | ≈ 0 | > 0.5 → 方差坍缩 |
| `orth` | < 0.3 | > 0.5 → 权重不正交 |
| `decorr` | < 1.0 | > 2.0 → 强相关 |

调试 AUC 低时，依次检查：① `temporal_contrastive_loss` 是否生效 ② 双窗口数据是否正确生成 ③ `raw_unif` 是否正常

**AUC 目标**: > 0.7 及格，> 0.8 良好，> 0.85 优秀

## 数据路径

```
/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/   ← manifest_path（勿修改）
/workspace/statistics/harbin_scenes/{source}_stats.json      ← 归一化统计量（7 个文件）
/workspace/outputs/{experiment_name}/                        ← 训练输出
```

变化检测标注时间窗口（全部为 2025 年，不得使用 2023/2024 固定窗口）：

| shapefile | before | after |
|-----------|--------|-------|
| `june.shp` | 2025-04 | 2025-06 |
| `aug.shp` | 2025-06 | 2025-08 |
| `September.shp` | 2025-08 | 2025-09 |
| `October.shp` | 2025-09 | 2025-10 |

## 开发规范

- 所有 Python 文件使用 `from __future__ import annotations`，类型注解完整
- 模块内注释使用中文
- 设备选择统一走 `src.utils.device.get_device`
- Checkpoint 统一走 `src.utils.checkpoint.load_checkpoint / save_checkpoint`
- 训练脚本顶部设置 `torch.set_num_threads(4)`
- 实验输出目录格式：`{前缀}_{版本}_{实验名}_{MMDD}`，例如 `exp_v13_temporal_contrastive_0601`（前缀：`exp` / `quick` / `ablation` / `base`）
- 文档文件名必须包含日期后缀 `YYYYMMDD`，完成后归档到 `archive/docs/legacy/`

## 已知 Bug

**Bug 1 — `imshow(origin='lower')` 图像上下翻转**：`rasterio.read()` row=0 对应最北端，不加 `origin='lower'` 即正确。

**Bug 2 — Shapefile CRS 为 None**：哈尔滨 shapefile 原始 CRS 为 `None`，必须显式设置后重投影，否则 AUC 接近随机：
```python
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")
if gdf.crs.to_epsg() != 32652:
    gdf = gdf.to_crs(epsg=32652)
```

**Bug 3 — 变化检测使用错误时间窗口**：使用 2023/2024 固定窗口时，S2 数据（全为 2025 年）无法匹配，AUC ≈ 0.51（接近随机）。必须按上方表格使用 2025 年月份配对。
