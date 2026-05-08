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
| `statistics/harbin_scenes/jrc_water_stats.json` | 重新计算，排除 -128：mean=-32.76→27.20, std=75.53→30.07 |

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

