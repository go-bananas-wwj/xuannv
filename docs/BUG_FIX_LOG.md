# Bug 修复日志

## 2026-05-08: JRC Water no-data (-128) 被错误参与 Reconstruction Loss

### 问题描述

Backbone 训练中，JRC Water 作为 target source（loss_type=0，MSE重建），其 no-data 值 `-128` 被当作有效数据参与 z-score 归一化和 reconstruction loss。

### 数据影响

| Patch | no-data (-128) 比例 |
|-------|---------------------|
| patch_000000 | 23.4% |
| patch_000001 | 41.4% |
| patch_000002 | 50.1% |
| patch_000004 | 94.3% |
| **平均** | **~30-40%** |

### 根因

1. `normalize_data()` 对 jrc_water 走 z-score 路径（不在 `CATEGORICAL_SOURCES` 中）
2. z-score 后 -128 → -1.26，被模型当作"低水位"学习
3. `loops.py` 的 reconstruction loss 只 mask `NaN`，不 mask `-128`

### 修复

**文件**: `src/data/transforms.py:normalize_data()`

```python
# 0. JRC Water: no-data (-128) → NaN, 不参与 reconstruction loss
if source_name == "jrc_water":
    data = data.astype(np.float32)
    data[data == -128] = np.nan
```

**原理**: `loops.py:64` 已有 `valid = ~torch.isnan(tgt_valid)`，NaN 自动被 mask。

### 影响范围

- **Backbone 训练**: JRC Water 重建更干净，约 1/3 无效监督信号被移除
- **下游任务**: 无影响（下游脚本自己重新加载 jrc_water 标签）

### 相关文件（不涉及但检查过）

- `scripts/train_v5_downstream_*.py`: 7 个文件有相同的 jrc_water 映射 bug（`<= 0` 覆盖 `-128`），属于下游任务，不在 backbone 训练中

---

---

## 2026-05-08: JRC Water no-data 修复完整记录

### 问题演进

**Phase 1 — 原始 Bug**：
- JRC Water 中 `-128` 是 no-data（占比 30-40%）
- 被当作有效值做 z-score：`-128 → -1.26`
- 模型被迫学习"预测 -1.26"来表示无数据区域

**Phase 2 — 首次修复引入新 Bug**：
- 在 `normalize_data()` 中把 `-128 → NaN`
- **严重问题**：JRC Water 尺寸 43×43，需 bilinear resize 到 128×128
- NaN 在 bilinear resize 中**灾难性扩散**（测试：4×4 中 2×2 NaN 块 → 8×8 中 49/64 像素变 NaN）
- 如果 30% no-data，resize 后可能 70%+ 像素变 NaN

**Phase 3 — 最终正确修复**：

### 修改内容

| 文件 | 修改 |
|------|------|
| `src/data/transforms.py` | 撤销 normalize_data 中的 `-128 → NaN` |
| `src/data/dataset.py` `_preload_all` | jrc_water read_tif 后 `-128 → NaN`，然后 normalize → pad → cache |
| `src/data/dataset.py` `__getitem__` cache 分支 | 从 cache 读取后，`has_nan=True` → nearest resize |
| `src/data/dataset.py` `__getitem__` non-cache 分支 | jrc_water read_tif 后 `-128 → NaN`，normalize，`has_nan=True` → nearest resize |
| `src/data/dataset.py` `_resize_to_target` | 新增 `has_nan` 参数，含 NaN 时用 nearest 避免扩散 |
| `src/data/dataset.py` 顶层 `_preload_patch` | 同 `_preload_all` 处理 |
| `statistics/harbin/jrc_water_stats.json` | 重新计算，排除 -128：mean=-32.76→27.20, std=75.53→30.07 |

### 关键设计决策

1. **NaN 在 resize 前设置**：read_tif 后（43×43）、normalize 前设置 NaN，resize 时用 nearest
2. **nearest resize 不扩散 NaN**：NaN 只映射到对应区域，不污染相邻有效像素
3. **loops.py 自动 mask NaN**：`valid = ~torch.isnan(tgt_valid)` 无需改动
4. **stats 重新计算**：排除 -128 后 mean/std 更准确

### 验证

```python
# PyTorch bilinear vs nearest resize 对 NaN 的处理对比
# Input: 4x4 with 2x2 NaN block in center
# Bilinear → 8x8: 49/64 pixels = NaN (76%扩散！)
# Nearest → 8x8: 16/64 pixels = NaN (25%，仅直接映射区域)
```

---

## 2026-05-08: 分类目标重建 Loss 的 one-hot 解码错误（致命）

### 问题描述

`src/training/loops.py:compute_recon_loss()` 中，分类目标（worldcover、dynamic_world）的重建 loss 提取类别索引的方式错误。

### 错误代码

```python
# 原代码（错误）
tgt_cls = tgt[batch_mask, 0].long()  # 取 one-hot 的第一个通道
```

### 错误分析

`tgt` 是 one-hot 编码 `[N, C, H, W]`，原代码取 `第 0 通道` 作为类别索引：

| 实际类别 | one-hot | `tgt[:,0]` | 模型被教导预测 | 正确 |
|----------|---------|------------|---------------|------|
| 类别 0 | `[1,0,0,...]` | 1 | **1** | 0 |
| 类别 1 | `[0,1,0,...]` | 0 | **0** | 1 |
| 类别 2 | `[0,0,1,...]` | 0 | **0** | 2 |
| ... | ... | 0 | **0** | ... |

**后果**：
- 类别 0 被错误教导预测为类别 1
- 类别 1-N **全部**被错误教导预测为类别 0
- worldcover + dynamic_world 的重建监督**完全错乱**

### 修复

```python
# 新代码（正确）
tgt_onehot = tgt[batch_mask]  # [N_valid, C, H, W]
tgt_cls = tgt_onehot.argmax(dim=1).long()  # [N_valid, H, W]
has_data = tgt_onehot.sum(dim=1) > 0.5  # 排除 no-data
valid_pixels = has_data & (tgt_cls >= 0) & (tgt_cls < num_classes)
```

### 影响评估

- **所有历史训练**（V1-V7）都受此 bug 影响
- worldcover/dynamic_world 的重建 loss 完全错误
- 但模型仍可从 S2/S1/Landsat/dem/jrc_water 的重建中学到特征
- bottleneck classification loss（weight=0.03）使用了正确标签，提供了部分正确监督

### 为什么之前没发现

1. 分类 loss 只是 7 个 target 中的 2 个
2. 没有单独的 worldcover/dynamic_world 重建质量监控
3. 总 loss 的下降趋势掩盖了分类 loss 的异常


---

## 2026-05-16: 系统性索引错位 Bug — `dataset.patches.index()` vs `dataset[idx]`

### 问题描述

`HarbinPatchDataset` 的 `__getitem__` 访问的是 **月度样本索引** (`monthly_samples[idx]`)，而非 **patch 索引** (`patches[idx]`)。

```python
# dataset.py:910-954
def __len__(self): 
    return len(self.monthly_samples)  # 例如 5088

def __getitem__(self, idx):
    patch_id, year, month = self.monthly_samples[idx]  # idx 是月度样本索引！
```

但大量评测脚本错误地用 `dataset.patches.index(pid)`（patch 在 patches 列表中的索引，0-423）作为 `dataset[idx]` 的索引。这导致：
- **idx 0-11**: 碰巧对应 patch_0 的 12 个月（运气对）
- **idx 12+**: 对应完全不同的 patch 和月份（严重错误）

### 影响评估

| 场景 | 后果 |
|------|------|
| 第1个 patch (idx=0) | 正确访问 patch_0 的 1 月份 |
| 第2个 patch (idx=1) | 错误访问 patch_0 的 2 月份（不是 patch_1！） |
| 第13个 patch (idx=12) | 错误访问 patch_1 的 1 月份 |
| 第424个 patch (idx=423) | 错误访问 patch_35 的 3 月份 |

**对于 AUC 验证：**
- 提取的 embedding 来自**错误的 patch 和错误的月份**
- 变化检测 AUC 实际上是「随机 patch 组合」的 AUC，而非真实 before/after 对比
- **之前报告的 AUC≈0.48 全部是假阴性**，不代表模型真实能力

### 受影响的文件（18个）

**验证脚本（索引错位 + 时间窗口过时）：**

| 文件 | 行号 | 问题 |
|------|:----:|------|
| `validate_v12_auc.py` | L165 | `pidx = dataset.patches.index(pid)` → `dataset[pidx]` |
| `validate_v12_bare.py` | L165 | ✅ **已修复**（使用 `patch_month_to_idx` 映射） |
| `validate_v7_level1_bare.py` | L49 | `item = dataset[dataset.patches.index(patch_id)]` |
| `validate_v7_downstream_heads.py` | L71 | `item = dataset[dataset.patches.index(patch_id)]` |
| `validate_v5_bare.py` | L82-83 | 通过 `extract_embedding_map` 间接使用 |
| `validate_v10_bare.py` | L144 | `extract_embedding(idx, ...)` 中 idx 来自 `patches.index` |
| `validate_aef_bare.py` | L97 | `idx = dataset.patches.index(patch_id)` |

**CD Head 训练脚本：**

| 文件 | 行号 | 问题 |
|------|:----:|------|
| `train_cd_head_v8.py` | L276 | `dataset.patches.index(pid)` → `extract_embedding_map` |
| `train_cd_head_v8_v2.py` | L72 | `idx = dataset.patches.index(pid)` → `extract_fn` |
| `fewshot_change_detection.py` | L424 | `dataset.patches.index(pid)` → `extract_embedding_map` |
| `full_evaluation_pipeline.py` | L93 | `idx = dataset.patches.index(pid)` → `extract_fn` |

**Embedding 提取脚本：**

| 文件 | 行号 | 问题 |
|------|:----:|------|
| `extract_embeddings_for_knn.py` | L22 | `idx = dataset.patches.index(pid)` |
| `extract_monthly_embeddings_all_patches.py` | L69 | `pidx = dataset.patches.index(pid)` → `extract_embedding_map` |
| `extract_v4_monthly_embeddings.py` | L54 | `pidx = dataset.patches.index(pid)` → `extract_embedding_map` |
| `extract_v5_monthly_embeddings.py` | L54 | `pidx = dataset.patches.index(pid)` → `extract_embedding_map` |
| `extract_v5_prenorm_embeddings.py` | L46 | `pidx = dataset.patches.index(pid)` → `extract_embedding_map` |

**下游评估脚本：**

| 文件 | 行号 | 问题 |
|------|:----:|------|
| `comprehensive_downstream_eval.py` | L315 | `item_idx = dataset.patches.index(pid)` |
| `eval_downstream_knn_v2.py` | L42 | `idx = dataset.patches.index(pid)` |
| `eval_reconstruction_v8.py` | L54 | `idx = dataset.patches.index(pid)` |

**API 级别问题：**

| 文件 | 行号 | 问题 |
|------|:----:|------|
| `src/inference/engine.py` | L103 | `extract_embedding_map` 文档说 "patch_idx: patch 索引"，但实现做 `dataset[patch_idx]`（实际是月度样本索引） |

### 未受影响的文件（正确用法）

| 文件 | 正确原因 |
|------|----------|
| `extract_embeddings_v2.py` | 使用 `DataLoader`，通过 `__getitem__` 正确迭代 |
| `extract_v12_embeddings_all_months.py` | 使用 `DataLoader`，正确 |
| `extract_v12_embedding_diagnostics_v2.py` | 直接使用 `dataset[idx]`，idx 是月度样本索引 |
| `validate_v12_bare.py` (修复后) | 使用 `patch_month_to_idx` 映射 |

### 修复方案

**方案 A：每个脚本独立修复（推荐用于紧急修复）**

像 `validate_v12_bare.py` 一样，构建 `patch_month_to_idx` 映射：
```python
patch_month_to_idx = {}
for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
    patch_month_to_idx[(pid, year, month)] = idx

# 使用时
idx = patch_month_to_idx[(patch_id, year, month)]
item = dataset[idx]  # 正确！
```

**方案 B：修复 API（推荐长期方案）**

修改 `src/inference/engine.py:extract_embedding_map`：
```python
def extract_embedding_map(model, dataset, patch_id, year, month, ...):
    """接受 patch_id + year + month，内部查找正确索引."""
    for idx, (pid, y, m) in enumerate(dataset.monthly_samples):
        if pid == patch_id and y == year and m == month:
            batch = dataset[idx]
            break
```

### 为什么之前没发现

1. **第一个 patch 碰巧对**：`dataset[0]` 确实是 patch_0，开发者测试时只用 patch_0，没发现问题
2. **AUC≈0.5 被解释为"模型不行"**：实际上模型可能是对的，只是评测用了错误数据
3. **没有 eyeball 验证**：没有人工抽查 "这个 patch 的这个月份 embedding 是否合理"
4. **DataLoader 路径是对的**：训练时用的 DataLoader 是正确的，所以训练本身不受影响
5. **索引命名误导**：`patches.index(pid)` 返回的确实是 patch 索引，让人误以为可以直接用于 dataset 索引

### 额外问题：时间窗口过时

以下脚本硬编码了 **2023/2024 年** 的时间窗口，但哈尔滨变化检测标注对应 **2025 年**：

| 脚本 | 硬编码窗口 |
|------|-----------|
| `validate_v12_auc.py` | 2023H2 vs 2024H2 |
| `validate_v7_level1_bare.py` | 2023H2 vs 2024H2 |
| `validate_v7_downstream_heads.py` | 2023H2 vs 2024H2 |
| `validate_v5_bare.py` | 2023H2 vs 2024H2 |
| `validate_v10_bare.py` | 2023H2 vs 2024H2 |
| `train_cd_head_v8.py` | 2023H2 vs 2024H2 |
| `train_cd_head_v8_v2.py` | 2023H2 vs 2024H2 |
| `full_evaluation_pipeline.py` | 2023H2 vs 2024H2 |
| `extract_embeddings_multigpu.py` | 2023H2 vs 2024H2 |

**后果**：即使索引正确，这些脚本也在对比**没有标注覆盖的时间窗口**，AUC 依然无意义。

### 教训

1. **永远不要用 `dataset.patches.index()` 作为 `dataset[]` 的索引**：两者语义完全不同
2. **评测脚本必须用 DataLoader 或显式映射**：避免手动索引
3. **时间窗口必须与标注对齐**：硬编码时间窗口前必须确认标注的对应时间
4. **第一个 patch 测试是陷阱**：只测 patch_0 会隐藏索引错位问题
5. **API 文档必须与实现一致**：`extract_embedding_map` 的文档误导了所有调用者
