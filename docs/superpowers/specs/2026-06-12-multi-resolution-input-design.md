# 多分辨率多源输入改造设计文档

> 项目: xuannv  
> 日期: 2026-06-12  
> 目标: 引入天仪 SAR (~3m) 与 Planet (~3m) 高分辨率数据源，同时保持 S2/S1/Landsat 各自真实 GSD，不再统一 resize 到 128×128。所有源对齐到同一地理范围 (1280m×1280m)，模型输出仍保持 10m GSD (64×64) 的统一 embedding。

---

## 1. 背景与问题

### 1.1 当前状态

当前 xuannv 训练 pipeline 对所有输入源执行统一 resize：

- `src/data/transforms.py::read_tif(path, image_size=128)` 把所有 TIFF resize 到 128×128。
- `src/models/sensor_encoders.py::SensorEncoderBank` 强假设所有源共享同一个 `(H, W)`。
- `src/models/blocks.py::STPEncoder` 内部按 `H/8`、`H/4`、`H` 做固定比例下采样。

这导致：
- 天仪 SAR (~3m) 被 downsample 到 effective 10m，丢失高分辨率细节。
- Planet 数据尚未接入训练。
- Landsat (30m) 被 upsample 到 128×128，引入插值伪影。

### 1.2 用户核心要求

1. 不对齐到 128×128，而是对齐到同一地理位置大小。
2. 天仪 SAR 与 Planet 作为高分辨率源保留真实分辨率。
3. 输出 embedding 仍用于变化检测与分类。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 地理对齐 | 所有源覆盖同一 1280m×1280m 地理范围。 |
| 分辨率保真 | 每源按自身 GSD 输出对应像素数：S2/S1=128×128 (10m)，Landsat=43×43 (30m)，天仪/Planet=427×427 (~3m)。 |
| 统一输出 | `embedding_map` 仍保持 64×64 (10m GSD)。 |
| 下游兼容 | kNN、AUC、变化检测等下游脚本尽量不变。 |
| 可落地 | 在现有 AEF 架构上最小改动实现。 |

---

## 3. 参考论文与技术路线

### 3.1 AlphaEarth Foundations (AEF)

AEF 原论文描述："each resampled to a common grid"，输出 10m 分辨率 embedding。本方案与之思想一致：输入保留各自 GSD，在编码后对齐到公共网格。

### 3.2 AnySat (CVPR 2025)

- **scale-adaptive patch encoder**：不同分辨率源切成不同数量的 patch token，每个 token 独立编码到统一维度。
- **JEPA 训练框架**：在 latent space 做预测，对像素级扰动更鲁棒。
- **借鉴点**：不要强制所有源同像素尺寸；可用 per-source patch/token 化。

### 3.3 Scale-MAE (ICCV 2023)

- **GSD-based positional encoding (GSDPE)**：把地面采样距离编码进位置编码。
- **Laplacian pyramid decoder**：多尺度重建。
- **借鉴点**：可在位置编码中加入 GSD 信息；decoder 可按源输出不同分辨率。

### 3.4 TerraFM (2025)

- **modality-specific patch embedding**：每源有独立的 patch embedding。
- **cross-attention fusion**：不同源 token 序列在空间位置对齐后做 cross-attention。
- **借鉴点**：per-source stem 后统一投影到公共 latent 空间再融合。

### 3.5 SatMAE++

- 多尺度重建 + upsampling blocks。
- **借鉴点**：HR 源（天仪/Planet）的重建目标可以输出到自身高分辨率。

---

## 4. 方案设计

### 4.1 总体架构

```
输入层（每源保持真实GSD）:
  S2/S1    128×128  (10m)
  Landsat   43×43   (30m)
  天仪SAR  427×427  (~3m)
  Planet   427×427  (~3m)
       ↓
SensorEncoderBank (per-source adaptive stem):
  - 每源独立卷积 stem，stride/layers 可配置
  - S2/S1:    stride=2, 1层  → 64×64
  - Landsat:  stride=1, 1层  → 43×43 → bilinear → 64×64
  - 天仪/Planet: stride=2, 3层 → ~53×53 → bilinear → 64×64
  - 每源可选 1×1 align projection
       ↓
公共空间分辨率对齐: 全部 64×64
       ↓
STPEncoder: 现有逻辑基本不变
       ↓
Bottleneck: 64×64 embedding_map (10m GSD)
       ↓
Decoder: per-source 输出到各自目标分辨率
```

### 4.2 数据层改造

#### 4.2.1 配置新增

```yaml
data:
  patch_size_m: 1280
  source_gsd:
    s2: 10
    s1: 10
    landsat: 30
    tianyi_sar: 3
    planet: 3
  # 或由 GSD 自动计算：source_image_sizes
```

#### 4.2.2 读取函数改造

新增 `read_tif_aligned(path, dst_crs, dst_bounds, dst_shape, resampling)`：

- 使用 `rasterio.warp.reproject` 或 `WarpedVRT`。
- 指定 `dst_transform = from_bounds(*dst_bounds, *dst_shape)`。
- 统一 CRS、统一 bounds、输出指定 shape（即不同 GSD）。
- 连续/光学/SAR 用 `bilinear`；分类数据用 `nearest`；含 NaN 用 `nearest`。

#### 4.2.3 返回结构

`__getitem__` 返回 `List[Tensor]`，每个源独立 `[T, C, H_s, W_s]`，替代当前统一 6D tensor。

#### 4.2.4 Collate

新增 custom `collate_fn`：保持 list 结构，batch 内同源同 shape（按统一 GSD 重投影保证）。

### 4.3 模型层改造

#### 4.3.1 SensorEncoderBank

- 新增 `_SOURCE_TYPE_ID` 映射：`tianyi_sar=5`, `planet=6`。
- 每源 `SensorEncoder` 支持 `stem_stride`、`stem_layers` 配置。
- `forward` 接收 `List[Tensor]`，逐源编码后 `F.interpolate` 到 `common_spatial_size=(64,64)`。
- 每源可选轻量 align projection（1×1 conv + BN + GELU）。

#### 4.3.2 STPEncoder

若 `common_spatial_size = 64×64`，**基本无需改动**：
- Space 路径：`64//8 = 8`
- Time 路径：`64//4 = 16`
- Precision 路径：`64`

> 注意：若未来 common_size 不是 8 的倍数，跨尺度交换会引入 ±1px 偏差。本设计强制 common_size=64。

#### 4.3.3 Decoder

- `target_sources` 配置增加 `out_resolution`。
- Decoder 输出后按目标分辨率插值。
- `AEFOutput.reconstructions` 从 dense tensor 改为 `List[Tensor]`。

### 4.4 损失函数改造

`src/training/loops.py::compute_recon_loss` 改为遍历 list：

```python
for t_idx, (pred_t, tgt_t) in enumerate(zip(predictions, targets)):
    # pred_t: [B, C_i, H_i, W_i], tgt_t: [B, C_i, H_i, W_i]
    ...
```

### 4.5 位置编码（可选增强）

参考 Scale-MAE，在 spatial position encoding 中融入 GSD：

```python
pos_enc = base_pos_enc + gsd_embedding(gsd)
```

初期可先不加，作为后续优化。

---

## 5. 关键风险与缓解

| 风险 | 等级 | 说明 | 缓解 |
|------|------|------|------|
| 天仪SAR实际不是3m | **高** | 当前磁盘数据已被 resize 到 10m/128px | 重新预处理 `preprocessing/downloaders/local_importer.py` 天仪 SAR；或直接按 10m 处理 |
| 427×427 显存爆炸 | **高** | HR 源像素是 S2 的 11 倍 | 激进 stem 下采样（stride=8/16）、限制 max_frames、减小 batch_size |
| 数据 pipeline 大改 | **高** | dense tensor → List[Tensor]，collate/增强/缓存都要改 | 分阶段：先模型层跑通，再优化数据层 |
| Landsat 43×43 非2幂次 | 中 | 不能多次 stride=2 | stride=1 保持 43，再 bilinear 到 64 |
| Landsat 上采样伪影 | 中 | 43→64 是 upsample | align 模块后加平滑卷积；或用 bicubic |
| Decoder 变分辨率 | 中 | reconstructions 改 list | 同步改 loss 与评估 |
| checkpoint 不兼容 | 中 | 新增 per-source 参数 | 从头训练 |
| 教师-学生一致性 | 低 | 空间增强需同步 | 数据侧保证 per-source 同步 flip |

---

## 6. 分阶段实施计划

### 阶段 1：数据真实性验证

1. 扫描 `data_raw/haidian/scenes` 与 `data_raw/harbin_scenes` 各源 TIFF 的 metadata。
2. 确认天仪 SAR 真实 GSD、Planet 数据可用性。
3. 决定是否需要重新预处理天仪 SAR。

### 阶段 2：数据层最小改造

1. `src/config.py` 新增 `source_gsd` / `patch_size_m`。
2. `src/data/transforms.py` 新增 `read_tif_aligned`。
3. `src/data/dataset.py` 按 per-source shape 读取。
4. 返回结构改为 `List[Tensor]`，新增 custom `collate_fn`。
5. 更新缓存 key。

### 阶段 3：模型层改造

1. `src/models/sensor_encoders.py`：per-source stem + common size 对齐。
2. `src/models/model.py`：decoder 输出变分辨率 list。
3. `src/training/loops.py`：loss 支持 list。

### 阶段 4：快速验证

1. 用 `config_v27_quick_diag.yaml` + 小批量数据跑 smoke test。
2. 确认不 OOM、能收敛、重建损失正常下降。

### 阶段 5：完整训练

1. 新配置 `configs/config_multires_v1.yaml`。
2. 从头训练 80 epoch。
3. 评估重建质量与下游指标。

---

## 7. 预期收益

1. **HR 细节保留**：天仪/Planet 不再被粗暴 downsample，提升对细小地物的感知。
2. **Landsat 保真**：避免 30m→10m 上采样的伪影。
3. **与 AEF 原论文对齐**：真正体现 "resample to common grid" 的思想。
4. **下游兼容**：embedding_map 仍为 64×64@10m，变化检测/kNN 评估基本不变。

---

## 8. 待决策事项

| 事项 | 选项 | 建议 |
|------|------|------|
| 天仪 SAR 是否重新预处理为 3m | A) 重新预处理；B) 保持 10m | 先按 B 做模型架构验证，再决定是否投入 A |
| common_spatial_size | 64×64 / 32×32 / 可配置 | 64×64（与现有 STPEncoder 兼容） |
| HR 源重建目标分辨率 | A) 427×427；B) 64×64；C) 多尺度 | A 最符合用户要求，但计算成本高 |
| 是否加入 GSD-aware position encoding | 是 / 否 | 否（阶段1），后续作为优化 |

---

## 9. 相关文件清单

### 需修改

- `src/config.py`
- `src/data/transforms.py`
- `src/data/dataset.py`
- `src/data/builder.py`
- `src/data/multi_region_dataset.py`
- `src/models/sensor_encoders.py`
- `src/models/model.py`
- `src/training/loops.py`
- `preprocessing/downloaders/local_importer.py`（若重新预处理天仪 SAR）

### 需新增

- `configs/config_multires_v1.yaml`
- `scripts/eval/evaluate_reconstruction.py` 更新（支持变分辨率 target）
- 可能的新测试脚本

---

## 10. 结论

本设计采用 **per-source adaptive stem + common spatial resolution alignment** 路线，在保留 AEF 架构主体（STPEncoder、Bottleneck、输出分辨率）的前提下，实现多分辨率多源输入。这是当前代码库上风险与收益最平衡的落地方案，与 AEF 原论文及 AnySat/Scale-MAE/TerraFM 等前沿工作方向一致。

建议先以**保持天仪 SAR 为 10m**的方式做最小可行验证，确认模型层改动正确后，再决定是否投入重新预处理高分辨率 SAR 数据。
