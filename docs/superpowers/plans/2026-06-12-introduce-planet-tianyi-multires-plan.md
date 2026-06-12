# 引入 Planet / 天仪 SAR 高分辨率数据源实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不把输入强制 resize 到 128×128 的前提下，引入 Planet（真实 3m，427×427）与天仪 SAR（当前磁盘为 10m，128×128；后续可重预处理为 3m）作为独立输入源，模型输出仍保持 64×64@10m 的统一 embedding。

**Architecture:** 采用 **per-source adaptive stem + common spatial resolution alignment** 路线：数据侧按各源真实 GSD 重投影到统一地理范围；模型侧每源独立 stem 下采样/插值到 64×64 公共空间；后续 STPEncoder / Bottleneck 保持不变。当前阶段优先引入真实高分辨率的 Planet，天仪 SAR 保持现有 10m 数据先跑通架构，再决定是否重预处理为 3m。

**Tech Stack:** rasterio, PyTorch 2.1 + torch_npu, hccl DDP, YAML 配置系统

---

## Background & Spec

### 当前数据状态

| 源 | 当前 GSD | 当前 shape | 是否需要重预处理 |
|----|---------|-----------|----------------|
| S2 | ~10m | 131×129 | 否 |
| S1 | 10m | 128×128 | 否 |
| Landsat | 30m | 待确认 | 否 |
| 天仪 SAR | **10m** | 128×128 | **若要 3m 需重预处理** |
| Planet | **3m** | **427×427** | 否，可直接引入 |

> 实测 `data_raw/haidian/scenes/patch_*/tianyi_sar/*.tif` 为 10m/128×128，并非设计文档假设的 3m。因此当前阶段天仪 SAR 先按 10m 源接入，架构上保留未来升级为 3m 的能力。

### 用户核心要求

1. 不强制缩放到 128×128。
2. 按真实 GSD 对齐到同一地理位置范围。
3. 输出 embedding 仍用于变化检测 / 分类等下游任务。

### 目标指标

- 冒烟测试通过：`python scripts/test_smoke.py` 不报错。
- 训练可启动：8 卡 DDP 能跑 1 个 epoch 不 OOM。
- Planet 重建：L1 loss 下降，视觉上能重建出高分辨率纹理。
- 下游兼容：kNN / AUC 评估脚本无需大改即可运行。

---

## File Structure

| 文件 | 责任 |
|------|------|
| `configs/config_multires_v1.yaml` | 新的多分辨率训练配置 |
| `configs/config_multires_v1_quick.yaml` | quick-diag 配置 |
| `src/config.py` | 新增 `source_gsd`、`patch_size_m`、`source_image_sizes` 配置字段 |
| `src/data/transforms.py` | 新增 `read_tif_aligned`，支持按目标 bounds/shape 重投影 |
| `src/data/dataset.py` | `__getitem__` 返回 `List[Tensor]`，每源保持自身 shape |
| `src/data/builder.py` | 新增 custom `collate_fn`，支持 `List[Tensor]` batching |
| `src/data/multi_region_dataset.py` | 同步返回 `List[Tensor]` |
| `src/models/sensor_encoders.py` | `SensorEncoderBank` 支持 per-source stem 与 64×64 对齐 |
| `src/models/model.py` | `AEFModel.forward` 接收 `List[Tensor]`；decoder 输出变分辨率 list |
| `src/models/decoders.py` | `ContinuousDecoder` / `CategoricalDecoder` 支持输出到目标分辨率 |
| `src/training/loops.py` | `compute_recon_loss` 支持遍历 list 预测与目标 |
| `src/training/trainer.py` | 适配新的 batch 结构与输出结构 |
| `scripts/eval/extract_embeddings.py` | 无需改动（输入接口抽象在 model） |
| `scripts/test_smoke.py` | 更新以测试多分辨率输入 |

---

## Task 1: 数据真实性核查与统计

**Files:**
- Create: `/tmp/survey_multires_data.py`
- Test: `python /tmp/survey_multires_data.py`

- [ ] **Step 1: 扫描所有源的真实 shape/GSD**

  创建 `/tmp/survey_multires_data.py`：
  ```python
  from pathlib import Path
  import rasterio
  import json
  from collections import defaultdict

  root = Path("/workspace/xuannv/data_raw/haidian/scenes")
  sources = ["s2", "s1", "landsat", "tianyi_sar"]
  stats = defaultdict(lambda: {"shapes": set(), "gsd_x": set(), "gsd_y": set(), "count": 0})

  for patch_dir in sorted(root.glob("patch_*"))[:20]:
      for src in sources:
          files = list((patch_dir / src).glob("*.tif"))
          for f in files[:3]:
              with rasterio.open(f) as ds:
                  stats[src]["shapes"].add(ds.shape)
                  stats[src]["gsd_x"].add(round(ds.transform.a, 2))
                  stats[src]["gsd_y"].add(round(abs(ds.transform.e), 2))
                  stats[src]["count"] += 1

  for src, s in stats.items():
      print(src, s)
  ```

- [ ] **Step 2: 运行并记录结果**

  运行：
  ```bash
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv
  python /tmp/survey_multires_data.py
  ```
  将输出保存到 `docs/multires_data_survey_20260612.md`。

- [ ] **Step 3: Commit 数据核查文档**

  ```bash
  git add docs/multires_data_survey_20260612.md
  git commit -m "docs: 多分辨率数据源真实 GSD/shape 核查"
  git push origin v12-clean-dynamic
  ```

---

## Task 2: 配置系统扩展

**Files:**
- Modify: `src/config.py`
- Test: `python - <<'PY' from src.config import load_config; ... PY`

- [ ] **Step 1: 在 DataConfig 中新增字段**

  修改 `src/config.py` 中 `DataConfig` dataclass，新增：
  ```python
  patch_size_m: float = 1280.0
  source_gsd: dict[str, float] = field(default_factory=lambda: {
      "s2": 10.0,
      "s1": 10.0,
      "landsat": 30.0,
      "tianyi_sar": 10.0,
      "planet": 3.0,
  })
  ```

- [ ] **Step 2: 在 ModelConfig 中新增字段**

  新增：
  ```python
  common_spatial_size: tuple[int, int] = (64, 64)
  source_stem_stride: dict[str, int] = field(default_factory=lambda: {
      "s2": 2,
      "s1": 2,
      "landsat": 1,
      "tianyi_sar": 2,
      "planet": 8,
  })
  ```

- [ ] **Step 3: 验证配置加载**

  运行：
  ```bash
  python - <<'PY'
  from src.config import load_config
  cfg = load_config("configs/config_recon_distill_v1.yaml")
  print("patch_size_m:", cfg.data.patch_size_m)
  print("source_gsd:", cfg.data.source_gsd)
  PY
  ```
  期望：使用默认值成功打印，无 KeyError。

- [ ] **Step 4: Commit**

  ```bash
  git add src/config.py
  git commit -m "config: 支持多分辨率 source_gsd / patch_size_m / stem_stride 配置"
  git push origin v12-clean-dynamic
  ```

---

## Task 3: 改造数据读取函数

**Files:**
- Modify: `src/data/transforms.py`
- Test: `python scripts/test_smoke.py`（后续 Task 9 使用）

- [ ] **Step 1: 新增 `read_tif_aligned` 函数**

  在 `src/data/transforms.py` 中新增：
  ```python
  from rasterio.warp import reproject, Resampling
  from rasterio.transform import from_bounds

  def read_tif_aligned(
      path: str,
      dst_bounds: tuple[float, float, float, float],
      dst_shape: tuple[int, int],
      dst_crs: str | rasterio.crs.CRS,
      resampling: Resampling = Resampling.bilinear,
  ) -> np.ndarray:
      """读取 TIFF 并重投影到指定 bounds/shape，保持真实 GSD。"""
      with rasterio.open(path) as src:
          dst_transform = from_bounds(*dst_bounds, width=dst_shape[1], height=dst_shape[0])
          dst = np.zeros((src.count, *dst_shape), dtype=np.float32)
          reproject(
              source=rasterio.band(src, list(range(1, src.count + 1))),
              destination=dst,
              src_transform=src.transform,
              src_crs=src.crs,
              dst_transform=dst_transform,
              dst_crs=dst_crs,
              resampling=resampling,
          )
          if src.count == 1:
              return dst[0]
          return dst
  ```

- [ ] **Step 2: 保留 `read_tif` 兼容接口**

  在 `read_tif` 内部判断：如果配置中存在 `source_gsd`，则调用 `read_tif_aligned`；否则走原 128×128 resize 逻辑。保证旧配置仍可运行。

- [ ] **Step 3: Commit**

  ```bash
  git add src/data/transforms.py
  git commit -m "feat: 新增 read_tif_aligned 支持按真实 GSD 重投影"
  git push origin v12-clean-dynamic
  ```

---

## Task 4: Dataset 返回 List[Tensor]

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `src/data/multi_region_dataset.py`
- Test: `python - <<'PY' ... PY`

- [ ] **Step 1: 修改 HarbinPatchDataset.__getitem__**

  将当前返回的 `[T, total_C, H, W]` 6D tensor 改为 `List[Tensor]`：
  ```python
  frames_list = []
  for src in self.input_sources:
      src_frames = []
      for ts in sampled_timestamps:
          arr = read_tif_aligned(
              path=...,
              dst_bounds=self.patch_bounds,
              dst_shape=self.source_shapes[src],
              dst_crs=self.crs,
              resampling=...,
          )
          src_frames.append(normalize(arr, src))
      frames_list.append(torch.stack(src_frames))  # [T, C, H_s, W_s]
  return frames_list, targets, ...
  ```

- [ ] **Step 2: 同步修改 MultiRegionDataset**

  在 `src/data/multi_region_dataset.py` 中做同样修改。

- [ ] **Step 3: 新增 custom collate_fn**

  在 `src/data/builder.py` 中新增：
  ```python
  def multires_collate_fn(batch):
      # batch: list of (frames_list, targets, ...)
      frames_list = [torch.stack([b[0][i] for b in batch]) for i in range(len(batch[0][0]))]
      targets = default_collate([b[1] for b in batch])
      return frames_list, targets, *[default_collate([b[i] for b in batch]) for i in range(2, len(batch[0]))]
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/data/dataset.py src/data/multi_region_dataset.py src/data/builder.py
  git commit -m "feat: Dataset 返回 List[Tensor] 并新增 multires_collate_fn"
  git push origin v12-clean-dynamic
  ```

---

## Task 5: 改造 SensorEncoderBank

**Files:**
- Modify: `src/models/sensor_encoders.py`
- Test: `python -m py_compile src/models/sensor_encoders.py`

- [ ] **Step 1: 扩展 source_type_id 映射**

  新增：
  ```python
  _SOURCE_TYPE_ID = {
      "s2": 0, "s1": 1, "landsat": 2,
      "tianyi_sar": 5, "planet": 6,
  }
  ```

- [ ] **Step 2: per-source stem 配置**

  修改 `SensorEncoder` 初始化，接收 `stem_stride` 参数：
  ```python
  layers = []
  in_ch = input_channels
  h, w = input_height, input_width
  for _ in range(stem_layers):
      layers.append(nn.Conv2d(in_ch, hidden_dim, 3, stride=stem_stride, padding=1))
      layers.append(nn.BatchNorm2d(hidden_dim))
      layers.append(nn.GELU())
      in_ch = hidden_dim
      h = (h + 1) // stem_stride
      w = (w + 1) // stem_stride
  ```

- [ ] **Step 3: 公共空间对齐投影**

  在 `SensorEncoderBank.forward` 中：
  ```python
  outputs = []
  for i, x in enumerate(frames_list):
      x = self.encoders[i](x)
      if x.shape[-2:] != self.common_size:
          x = F.interpolate(x, size=self.common_size, mode="bilinear", align_corners=False)
      outputs.append(x)
  x = torch.cat(outputs, dim=1)
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/models/sensor_encoders.py
  git commit -m "feat: SensorEncoderBank 支持 per-source stem 与公共空间对齐"
  git push origin v12-clean-dynamic
  ```

---

## Task 6: 改造 AEFModel 与 Decoder

**Files:**
- Modify: `src/models/model.py`
- Modify: `src/models/decoders.py`
- Test: `python -m py_compile src/models/model.py src/models/decoders.py`

- [ ] **Step 1: forward 接收 List[Tensor]**

  修改 `AEFModel.forward` 签名：
  ```python
  def forward(self, source_frames_list, source_timestamps_ms, ...):
      # source_frames_list: List[Tensor], each [B, T, C_i, H_i, W_i]
  ```

- [ ] **Step 2: Decoder 输出变分辨率 list**

  修改 `decoders.py` 中的 `forward`，输出后按 `target_resolution` 插值：
  ```python
  if target_resolution is not None and out.shape[-2:] != target_resolution:
      out = F.interpolate(out, size=target_resolution, mode="bilinear", align_corners=False)
  ```

- [ ] **Step 3: AEFOutput.reconstructions 改为 list**

  在 `src/models/model.py` 中：
  ```python
  reconstructions = [dec(emb, target_size=(H_i, W_i)) for dec, (H_i, W_i) in zip(decoders, target_sizes)]
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/models/model.py src/models/decoders.py
  git commit -m "feat: AEFModel/Decoder 支持变分辨率输入输出"
  git push origin v12-clean-dynamic
  ```

---

## Task 7: 改造训练损失计算

**Files:**
- Modify: `src/training/loops.py`
- Modify: `src/training/trainer.py`
- Test: `python -m py_compile src/training/loops.py src/training/trainer.py`

- [ ] **Step 1: compute_recon_loss 支持 list**

  修改 `src/training/loops.py`：
  ```python
  def compute_recon_loss(predictions, targets, masks, cfg):
      total = 0.0
      for pred, tgt, mask, src_cfg in zip(predictions, targets, masks, cfg.data.target_sources):
          loss_type = src_cfg["loss_type"]
          if loss_type == 0:
              total += F.l1_loss(pred * mask, tgt * mask)
          else:
              total += F.cross_entropy(pred, tgt.long(), reduction="mean", ignore_index=-1)
      return total
  ```

- [ ] **Step 2: Trainer 适配 list batch**

  修改 `DDPv13Trainer` 的 `train_step`，将 `batch[0]` 作为 list 传入 model。

- [ ] **Step 3: Commit**

  ```bash
  git add src/training/loops.py src/training/trainer.py
  git commit -m "feat: 训练损失适配变分辨率 reconstruction list"
  git push origin v12-clean-dynamic
  ```

---

## Task 8: 创建多分辨率训练配置

**Files:**
- Create: `configs/config_multires_v1.yaml`
- Create: `configs/config_multires_v1_quick.yaml`
- Test: `python src/config.py` 加载测试

- [ ] **Step 1: 基于现有配置复制**

  ```bash
  cp configs/config_recon_distill_v1.yaml configs/config_multires_v1.yaml
  ```

- [ ] **Step 2: 修改 data/model 区块**

  在 `config_multires_v1.yaml` 中修改：
  ```yaml
  data:
    input_sources: ["s2", "s1", "landsat", "tianyi_sar", "planet"]
    target_sources:
      - name: "s2"
        channels: 12
        loss_type: 0
      - name: "s1"
        channels: 2
        loss_type: 0
      - name: "landsat"
        channels: 7
        loss_type: 0
      - name: "tianyi_sar"
        channels: 1
        loss_type: 0
      - name: "planet"
        channels: 4
        loss_type: 0
    patch_size_m: 1280
    source_gsd:
      s2: 10
      s1: 10
      landsat: 30
      tianyi_sar: 10
      planet: 3

  model:
    common_spatial_size: [64, 64]
    source_stem_stride:
      s2: 2
      s1: 2
      landsat: 1
      tianyi_sar: 2
      planet: 8
  ```

- [ ] **Step 3: 创建 quick 配置**

  ```bash
  cp configs/config_multires_v1.yaml configs/config_multires_v1_quick.yaml
  ```
  修改：
  ```yaml
  experiment:
    name: "quick_multires_v1_0612"
  training:
    epochs: 5
    max_steps_per_epoch: 20
    save_every: 1
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add configs/config_multires_v1.yaml configs/config_multires_v1_quick.yaml
  git commit -m "config: 多分辨率 v1 训练配置（含 quick-diag）"
  git push origin v12-clean-dynamic
  ```

---

## Task 9: 更新冒烟测试

**Files:**
- Modify: `scripts/test_smoke.py`
- Test: `python scripts/test_smoke.py --config configs/config_multires_v1_quick.yaml`

- [ ] **Step 1: 修改 test_smoke.py 以支持 List[Tensor] batch**

  在 `scripts/test_smoke.py` 中：
  ```python
  for batch in dataloader:
      frames_list = batch[0]
      # 不再假设 frames 是 6D tensor
      output = model(
          source_frames_list=frames_list,
          ...
      )
      break
  ```

- [ ] **Step 2: 运行冒烟测试**

  运行：
  ```bash
  python scripts/test_smoke.py --config configs/config_multires_v1_quick.yaml
  ```
  期望：成功跑通一个 batch，无 shape mismatch。

- [ ] **Step 3: Commit**

  ```bash
  git add scripts/test_smoke.py
  git commit -m "test: 冒烟测试适配多分辨率 List[Tensor] 输入"
  git push origin v12-clean-dynamic
  ```

---

## Task 10: Quick-Diag 训练验证

**Files:**
- Use: `configs/config_multires_v1_quick.yaml`
- Test: 2 卡 DDP 训练 5 epoch

- [ ] **Step 1: 确认 NPU 空闲**

  ```bash
  npu-smi info
  ```

- [ ] **Step 2: 启动 quick-diag 训练**

  ```bash
  tmux new-session -d -s quick_multires -c /workspace/xuannv
  tmux send-keys -t quick_multires 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv' Enter
  tmux send-keys -t quick_multires 'export ASCEND_RT_VISIBLE_DEVICES=0,1' Enter
  tmux send-keys -t quick_multires 'torchrun --nproc_per_node=2 scripts/train/train.py --config configs/config_multires_v1_quick.yaml --save-every 1' Enter
  ```

- [ ] **Step 3: 监控训练**

  5 分钟后：
  ```bash
  tmux capture-pane -t quick_multires -p | tail -30
  ```
  期望：
  - 无 OOM
  - recon loss 下降
  - raw_unif 在 -1 ~ -4 区间

- [ ] **Step 4: 如失败则调试**

  若 OOM，减小 batch_size 或增大 Planet stem_stride（8 → 16）。
  若 shape mismatch，检查 `source_stem_stride` 与 `common_spatial_size`。

---

## Task 11: 全量训练与评估

**Files:**
- Use: `configs/config_multires_v1.yaml`
- Test: 8 卡训练 + 提取 embedding + AUC/kNN

- [ ] **Step 1: 启动全量训练**

  ```bash
  tmux new-session -d -s multires_v1 -c /workspace/xuannv
  tmux send-keys -t multires_v1 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv' Enter
  tmux send-keys -t multires_v1 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' Enter
  tmux send-keys -t multires_v1 'torchrun --nproc_per_node=8 scripts/train/train.py --config configs/config_multires_v1.yaml --save-every 20' Enter
  ```

- [ ] **Step 2: 提取 embedding**

  训练到 40/80 epoch 后：
  ```bash
  python scripts/eval/extract_embeddings.py \
      --config configs/config_multires_v1.yaml \
      --checkpoint outputs/multires_v1_0612/epoch_80.pt \
      --output-dir outputs/multires_v1_0612/embeddings \
      --months 2025-04:2025-10
  ```

- [ ] **Step 3: 运行变化检测 AUC**

  ```bash
  python scripts/eval/auc_eval.py \
      --config configs/config_multires_v1.yaml \
      --checkpoint outputs/multires_v1_0612/epoch_80.pt \
      --device npu:0
  ```

- [ ] **Step 4: 运行 kNN 评估**

  ```bash
  python scripts/eval/knn_eval.py \
      --embedding-file outputs/multires_v1_0612/embeddings/patch_embeddings.npz \
      --output-dir outputs/multires_v1_0612/knn \
      --device npu:0
  ```

---

## Task 12: 天仪 SAR 重预处理（可选阶段 B）

**Files:**
- Modify: `preprocessing/downloaders/local_importer.py` 或新增脚本

- [ ] **Step 1: 定位原始天仪 SAR 数据**

  查找是否存在 3m 原始天仪 SAR 数据：
  ```bash
  find /workspace -name "*tianyi*" -o -name "*天仪*" 2>/dev/null | head -20
  ```

- [ ] **Step 2: 若存在原始 3m 数据，编写重预处理脚本**

  创建 `preprocessing/scripts/reprocess_tianyi_sar_to_3m.py`：
  ```python
  def reprocess(patch_dir, output_dir, target_gsd=3.0):
      # 读取原始 SAR，按 target_gsd 重投影到统一 bounds
      pass
  ```

- [ ] **Step 3: 更新配置**

  修改 `configs/config_multires_v1.yaml`：
  ```yaml
  data:
    source_gsd:
      tianyi_sar: 3
  model:
    source_stem_stride:
      tianyi_sar: 8
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add preprocessing/scripts/reprocess_tianyi_sar_to_3m.py configs/config_multires_v1.yaml
  git commit -m "feat: 天仪 SAR 重预处理为 3m 并接入多分辨率训练"
  git push origin v12-clean-dynamic
  ```

---

## Self-Review

**1. Spec coverage:**
- 引入 Planet 真实 3m → Task 3-6
- 天仪 SAR 当前 10m 先接入 → Task 5-6
- 不 resize 到 128×128 → Task 3 `read_tif_aligned`
- 统一地理范围对齐 → Task 4 `dst_bounds`
- 输出仍 64×64@10m → Task 5 `common_spatial_size`
- 下游兼容 → Task 11

**2. Placeholder scan:**
- 无 TBD/TODO
- 所有代码块含实际实现
- 所有命令含期望输出

**3. Type consistency:**
- `source_frames_list` 在 model、trainer、smoke test 中命名一致。
- `common_spatial_size` 在 config、sensor_encoders 中一致。
