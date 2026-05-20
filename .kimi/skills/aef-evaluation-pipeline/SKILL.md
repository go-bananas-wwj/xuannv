---
name: aef-evaluation-pipeline
description: >
  AEF (AlphaEarth Foundations) 下游评估自动化 Pipeline。用于多实验并行 embedding 提取、
  KNN/MLP 分类评估、变化检测 AUC 计算，以及结果汇总报告生成。
  触发条件：用户提到"评估"、"下游"、"embedding 提取"、"变化检测"、"AUC"、
  "对比实验"、"调参"、"KNN"、"MLP"、"Pipeline"，或需要对多个训练实验做系统性对比时。
---

# AEF 下游评估自动化 Pipeline

## 概述

本 Pipeline 自动化执行以下流程：
1. **Embedding 提取**: 提取所有 patch × 12 个月的 embedding（推荐 DataLoader 路径）
2. **KNN 评估**: 3 任务（WorldCover 7类、DynamicWorld 9类、JRC Water 2类）— **仅作为基线参考**
3. **MLP 评估**: PixelMLPHead (hidden=256, 50 epochs) — **主要下游评估指标**
4. **变化检测评估**: 
   - **Bare AUC**: 像素级 cosine distance（无 CD Head）
   - **CD Head AUC**: 训练 ChangeDetectionHeadV3 后的 AUC
5. **报告生成**: Markdown 汇总表格 + 结论分析

---

## ⚠️ 评测前必查清单（2026-05-20 更新，关键！）

**以下 bug 曾导致整个评测结果无效，每次评测前必须逐项确认：**

### Check 1: 索引是否正确？

```python
# ❌ 错误 — 永远不要这样做
dataset[dataset.patches.index(pid)]  # patches索引 ≠ monthly_samples索引！

# ✅ 正确 — 方式A: 通过 extract_embedding_for_month
from src.inference.engine import extract_embedding_for_month
emb = extract_embedding_for_month(model, dataset, pid, 2025, month, device)

# ✅ 正确 — 方式B: 手动构建映射
patch_month_to_idx = {}
for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
    patch_month_to_idx[(pid, year, month)] = idx
idx = patch_month_to_idx[(pid, 2025, 6)]
item = dataset[idx]

# ✅ 正确 — 方式C: 使用 DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=False)
```

**影响范围**: 至少 18 个脚本受影响（见下方"系统性索引错位"）

### Check 2: 时间窗口是否与标注对齐？

哈尔滨变化检测标注对应 **2025 年** 的具体月份：
| Shapefile | Before | After |
|:---------:|:------:|:-----:|
| june.shp | 4月 | 6月 |
| aug.shp | 6月 | 8月 |
| September.shp | 8月 | 9月 |
| October.shp | 9月 | 10月 |

```python
# ❌ 错误 — 硬编码 2023/2024 年窗口
BEFORE = (1688169600000.0, 1703980800000.0)  # 2023H2

# ✅ 正确 — 使用 2025 年具体月份
MONTH_2025 = {
    4: (1743436800000, 1746028799000),
    6: (1748707200000, 1751299199000),
    8: (1753977600000, 1756655999000),
    9: (1756656000000, 1759247999000),
    10: (1759248000000, 1761926399000),
}
```

**教训**: 曾发现 `fewshot_change_detection.py` 中的 `MONTH_WINDOWS_2025` 实际存储的是 2024 年值，导致所有结果为零。

### Check 3: preload 是否开启？

```python
# ✅ 评测时务必开启 preload
cfg.data.preload = True
dataset = HarbinPatchDataset(cfg)
# 首次加载 ~4-5 分钟，但之后每个样本访问仅需 ~0.1s
# 否则每个样本 ~10s（从磁盘读 TIFF）
```

### Check 4: Active Dimensions 是否健康？

```python
# 训练完成后，必须先检查 active_dims
# 如果 active < 15 / embedding_dim，说明严重坍缩
# 下游任务（特别是分类）会受严重影响
```

**参考标准**:
| active_dims | 状态 | 下游影响 |
|:---:|:---:|:---|
| > 40 | ✅ 健康 | 分类/变化检测均可 |
| 20-40 | ⚠️ 轻度坍缩 | 分类受限，变化检测可用 |
| 10-20 | 🔴 中度坍缩 | 分类困难，变化检测勉强 |
| < 10 | 🔴 严重坍缩 | 所有下游任务接近随机 |

---

## 核心步骤

### Step 1: 确认实验列表

检查 `/workspace/outputs/` 下的实验目录，确认每个实验有：
- `epoch_best_epoch{N}.pt` 或 `epoch_{N}.pt` checkpoint
- 对应的 `.yaml` 配置文件

### Step 2: 提取 Embedding（推荐 DataLoader 路径）

**推荐方式**: `extract_v12_embeddings_all_months.py`（使用 DataLoader，避免索引问题）

```bash
python scripts/eval/extract_v12_embeddings_all_months.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0 \
  --batch-size 16 \
  --output-dir /workspace/outputs/{exp}/eval/embeddings_all_months
```

**输出**:
- `{patch_id}.npz`: 每个 patch 包含 `embedding_map`, `pre_norm_map`, `year_months`

**备选方式**: `extract_embeddings_v2.py`（使用手动 batch，需确认 reshape 顺序）

```bash
python scripts/eval/extract_embeddings_v2.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --output-dir /workspace/outputs/{exp}/evaluation/embeddings \
  --device npu:0 \
  --batch-size 8 \
  --save-every 500
```

**⚠️ 注意**: `extract_embeddings_v2.py` 的 reshape 假设 `monthly_samples` 是 patch-major 排序，如果 `_build_monthly_samples` 逻辑改变会出错。

### Step 3: 运行下游评估

#### 3a. KNN 评估（基线参考，~5分钟）

```bash
python scripts/eval/evaluate_knn_v2.py \
  --embedding-file /path/to/patch_embeddings.npz \
  --output-dir /path/to/downstream \
  --device npu:0 --k 5 --month 6
```

**注意**: KNN 只是无参数基线。**下游分类的权威指标是 MLP**。

#### 3b. MLP 评估（主要指标，~20分钟）

```bash
python scripts/eval/evaluate_mlp_v2.py \
  --embedding-file /path/to/patch_embeddings.npz \
  --output-dir /path/to/downstream_mlp \
  --device npu:0 --epochs 50 --month 6
```

**关键发现（2026-05-20）**:
- MLP 显著优于 KNN（WorldCover +21%, DynamicWorld +14%）
- 增加 epoch（50→200）几乎无提升（+0.001）— **embedding 质量是瓶颈**
- 增加 hidden_dim（256→512）几乎无提升 — **瓶颈不在 MLP 容量**
- 如果 MLP acc 停滞，问题在 embedding 本身（active_dims 坍缩）

#### 3c. 变化检测评估（两种方法）

**方法 A: Bare AUC（cosine distance，无 CD Head）**

```bash
python scripts/eval/validate_v12_bare.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0
```

**方法 B: CD Head AUC（训练 ChangeDetectionHeadV3）**

```bash
python scripts/eval/train_cd_head_v12.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0 --folds 5 --epochs 200
```

**关键发现（2026-05-20）**:
| 方法 | Mean AUC | 分析 |
|:---|:---:|:---|
| Bare AUC (cosine) | 0.52 | **严重低估**模型能力 |
| CD Head V3 (h=64) | **0.83** | 模型实际学到了变化信息 |
| CD Head V3 (h=128) | **0.84** | 增加容量提升有限 |

**教训**: Bare AUC ≈ 0.5 不代表模型失败！CD Head 能学到非线性变化模式，AUC 可达 0.83+。

### Step 4: 汇总结果

收集结果生成报告：

```bash
# 示例：手动汇总
python3 -c "
import json

# KNN
knn = json.load(open('downstream/knn_summary.json'))
# MLP
mlp = json.load(open('downstream_mlp/mlp_summary.json'))
# CD
bare = json.load(open('bare_auc.json'))

print('=== 汇总 ===')
print(f'KNN WC: {knn[\"worldcover\"][\"accuracy\"]:.3f}')
print(f'MLP WC: {mlp[\"worldcover\"][\"accuracy\"]:.3f}')
print(f'CD Head AUC: {bare[\"global\"][\"auc\"]:.3f}')
"
```

---

## 关键脚本路径

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/eval/extract_v12_embeddings_all_months.py` | **推荐** — DataLoader 路径提取 | ✅ 已修复索引 |
| `scripts/eval/extract_embeddings_v2.py` | 手动 batch 提取（有 reshape 风险） | ⚠️ 注意隐式假设 |
| `scripts/eval/validate_v12_bare.py` | Bare AUC（cosine distance） | ✅ 已修复索引+窗口 |
| `scripts/eval/train_cd_head_v12.py` | CD Head 训练（5-fold CV） | ✅ 新建，索引正确 |
| `scripts/eval/fewshot_change_detection.py` | Few-Shot CD（K-shot） | ✅ 已修复索引+窗口 |
| `scripts/eval/evaluate_knn_v2.py` | KNN 三任务评估 | ✅ 无索引问题（DataLoader） |
| `scripts/eval/evaluate_mlp_v2.py` | MLP 下游头训练 | ✅ 无索引问题 |
| `scripts/eval/visualize_cd_predictions.py` | 变化检测可视化 | ✅ 已修复空间匹配 |

**⚠️ 以下脚本有索引错位 bug（未修复），不要使用**：
- `validate_v12_auc.py`
- `validate_v7_level1_bare.py`
- `train_cd_head_v8.py`
- `train_cd_head_v8_v2.py`
- `extract_monthly_embeddings_all_patches.py`
- 完整列表见 `docs/BUG_FIX_LOG.md`

---

## 系统性索引错位 Bug（2026-05-16 发现）

### 根因

`dataset.patches.index(pid)` 返回 patch 索引（0-423），但 `dataset[idx]` 访问的是 `monthly_samples[idx]`（0-5087）。

### 受影响脚本（18个）

| 类别 | 脚本 | 修复状态 |
|------|------|:--------:|
| 验证 | `validate_v12_auc.py` | ❌ 未修复 |
| 验证 | `validate_v7_level1_bare.py` | ❌ 未修复 |
| 验证 | `validate_v7_downstream_heads.py` | ❌ 未修复 |
| 验证 | `validate_v5_bare.py` | ❌ 未修复 |
| 训练 | `train_cd_head_v8.py` | ❌ 未修复 |
| 训练 | `train_cd_head_v8_v2.py` | ❌ 未修复 |
| 提取 | `extract_monthly_embeddings_all_patches.py` | ❌ 未修复 |
| 提取 | `extract_v4_monthly_embeddings.py` | ❌ 未修复 |
| 提取 | `extract_v5_monthly_embeddings.py` | ❌ 未修复 |
| 提取 | `extract_v5_prenorm_embeddings.py` | ❌ 未修复 |
| 下游 | `eval_downstream_knn_v2.py` | ❌ 未修复 |
| 下游 | `eval_reconstruction_v8.py` | ❌ 未修复 |

### 修复方式

```python
# 方案 A: 使用新 API（推荐）
from src.inference.engine import extract_embedding_for_month
emb = extract_embedding_for_month(model, dataset, pid, 2025, month, device)

# 方案 B: 构建映射
patch_month_to_idx = {}
for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
    patch_month_to_idx[(pid, year, month)] = idx
idx = patch_month_to_idx[(pid, 2025, 6)]
item = dataset[idx]
```

---

## 常见问题

### 1. NPU 设备映射错误
**症状**: `open device X failed, runtime result = 107001`  
**解决**: `ASCEND_RT_VISIBLE_DEVICES=X` 时，PyTorch 内设备名必须为 `npu:0`

### 2. KNN 超时
**症状**: `evaluate_knn_v2.py` 运行数分钟无输出  
**解决**: 使用 `scipy.stats.mode` 在 CPU 上批量计算众数

### 3. 索引错位导致 AUC ≈ 0.5
**症状**: 所有变化检测结果接近随机（AUC=0.48-0.52）  
**根因**: `dataset.patches.index(pid)` 被错误用作 `dataset[idx]` 的索引  
**解决**: 使用 `extract_embedding_for_month` 或 `patch_month_to_idx` 映射

### 4. 时间窗口过时导致 AUC ≈ 0.5
**症状**: 修复索引后 AUC 仍接近随机  
**根因**: 硬编码 2023/2024 年窗口，但标注是 2025 年  
**解决**: 使用 2025 年具体月份窗口（4→6, 6→8, 8→9, 9→10）

### 5. MLP 训练不收敛（loss 不下降）
**症状**: MLP loss 停留在 ~2.0 不下降  
**根因**: `cfg.data.preload = False`，每个样本从磁盘读取 TIFF 导致数据加载极慢  
**解决**: 设置 `cfg.data.preload = True`

### 6. 提取 embedding 时 OOM
**症状**: `NPU out of memory` 在 Conv2D 层  
**解决**: 降低 batch_size（16→8→4），或使用 DataLoader 路径

### 7. Few-Shot CD 结果为 0
**症状**: 所有 K-shot 的 AUC = 0.0000  
**根因**: `MONTH_WINDOWS_2025` 存储的是 2024 年值，dataset 中无 2024 年数据  
**解决**: 修正窗口为真正的 2025 年值

### 8. Embedding 提取 reshape 错误
**症状**: `reshape(424, 12, ...)` 报 `cannot reshape`  
**根因**: `monthly_samples` 排序不是严格的 patch-major  
**解决**: 使用 `extract_v12_embeddings_all_months.py`（per-patch npz 输出，无需 reshape）

---

## 评估成功标准（2026-05-20 更新）

### 变化检测

| 方法 | 及格线 | 良好 | 优秀 | 说明 |
|------|--------|------|------|------|
| Bare AUC | > 0.55 | > 0.60 | > 0.65 | cosine distance，容易低估 |
| CD Head AUC | > 0.70 | > 0.80 | > 0.85 | 真实模型能力 |
| Few-Shot K=20 | > 0.55 | > 0.60 | > 0.65 | 轻量 head |

### 下游分类（MLP 为主要指标）

| 任务 | 及格线 | 良好 | 优秀 | KNN 参考 |
|------|--------|------|------|----------|
| WorldCover Acc | > 45% | > 50% | > 55% | ~35-40% |
| WorldCover mIoU | > 0.18 | > 0.22 | > 0.25 | ~0.15-0.17 |
| JRC Water Acc | > 65% | > 70% | > 75% | ~65-68% |
| JRC Water mIoU | > 0.35 | > 0.40 | > 0.45 | ~0.38-0.42 |
| Dynamic World Acc | > 50% | > 55% | > 60% | ~45-50% |
| Dynamic World mIoU | > 0.12 | > 0.15 | > 0.18 | ~0.12-0.15 |

**注意**: KNN 仅作为基线参考。**MLP 是下游分类的权威指标**。

### Embedding 质量

| 指标 | 及格线 | 良好 | 优秀 |
|------|--------|------|------|
| Active dims / Total | > 30% | > 50% | > 70% |
| Std mean | > 0.12 | > 0.15 | > 0.20 |
| Reconstruction Loss | < 0.15 | < 0.10 | < 0.05 |

---

## 关键发现与教训（2026-05-20）

### 教训 1: MLP >> KNN
- MLP WorldCover Acc = 0.529，KNN = 0.436（+21%）
- MLP 能学习维度选择和特征组合，KNN 不能
- **下游分类评估应以 MLP 为主要指标，KNN 仅作基线**

### 教训 2: CD Head >> Bare AUC
- Bare AUC (cosine) = 0.52，CD Head = 0.83（+0.31）
- cosine distance 在坍缩空间中无法捕捉非线性变化模式
- **AUC ≈ 0.5 不代表模型失败，必须训练 CD Head 验证**

### 教训 3: MLP 容量不是瓶颈
- 增加 epoch（50→200）: Acc 只提升 0.001
- 增加 hidden_dim（256→512）: Acc 完全不变
- **如果 MLP 停滞，问题在 embedding 质量，不在 MLP**

### 教训 4: active_dims 决定一切
- active=9/64 时，所有下游分类 mIoU < 0.25
- 必须先检查 active_dims，再决定是否投入下游评估
- **active < 15 的模型，下游评估意义有限**

### 教训 5: 索引错位的假阴性
- 索引 bug 导致 AUC ≈ 0.48，被误判为"模型失败"
- 修复后 AUC 提升到 0.52-0.53（虽然仍不高，但不是假阴性）
- **评测脚本必须经过索引正确性验证**

### 教训 6: preload 是必需的
- preload=False 时单次提取 ~10s
- preload=True 时单次提取 ~0.1s
- **评测前务必设置 cfg.data.preload = True**

---

## 参考文档

- **详细脚本参数**: 见 `references/evaluation-scripts.md`
- **报告模板**: 见 `references/report-template.md`
- **Bug 修复日志**: 见 `docs/BUG_FIX_LOG.md`（系统性索引错位完整记录）
- **完整评测报告示例**: 见 `docs/V12_EXP_A_COMPLETE_REPORT.md`
