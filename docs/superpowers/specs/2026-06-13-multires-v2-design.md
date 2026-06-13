# multires v2 训练设计文档

> 状态：待实现  
> 作者：Kimi Code CLI  
> 日期：2026-06-13  
> 项目：xuannv（分支 `v12-clean-dynamic`）  
> 依赖：v1 多分辨率改造已完成（tag `multires_v1_baseline`）

---

## 1. 背景与目标

### 1.1 v1 存在的问题

`exp_multires_v1_0612` 已跑完全量 80 epoch，但出现明显 embedding 坍缩：

| 指标 | 数值 | 正常范围 |
|------|------|----------|
| `erank` | 8.88 / 64 | > 32 |
| `active_dims` | 极低 | 64/64 |
| `std_mean` | 低 | > 0.60 |

根源：v1 配置为了“先跑通”把所有反坍缩/时序/蒸馏损失权重设为 0，模型只靠重建损失优化，embedding 退化成少数维度。

### 1.2 v2 目标

1. **引入 Planet 3m 高分辨率数据**作为第 5 路输入，与现有 S2/S1/Landsat/天仪 SAR 10m 数据一起训练；
2. **重新打开反坍缩、时序对比、AEF 教师蒸馏**等损失，把 embedding 打散并拉回语义空间；
3. 训练窗口聚焦 **2025-12 ~ 2026-04**，与海淀区 2026 标注评估对齐；
4. 目标：erank > 32，变化检测 AUC 与 6 类语义评估较 v1 全面提升。

---

## 2. 数据层设计

### 2.1 输入源

| 源 | 分辨率 | 尺寸 | 通道 | 备注 |
|----|--------|------|------|------|
| `s2` | 10m | 128×128 | 6 | 已存在 |
| `s1` | 10m | 128×128 | 2 | 已存在 |
| `landsat` | 30m（重采样到 128×128） | 128×128 | 6 | 已存在 |
| `tianyi_sar` | 10m | 128×128 | 1 | 已存在，10m 无 3m 可用 |
| `planet` | 3m | 427×427 | 4 | 新增，位于 `data_raw/beijing/planetscene` |

### 2.2 Planet 数据接入

Planet 数据实际路径：

```text
/workspace/xuannv/data_raw/beijing/planetscene/patch_000000/20251209.tif
...
```

训练代码通过 `manifest_path` 下的 `patch_*/<source>/` 目录读取输入源。因此通过**符号链接**把 Planet 挂到海淀场景目录：

```bash
for pid in $(ls /workspace/xuannv/data_raw/haidian/scenes); do
    src="/workspace/xuannv/data_raw/beijing/planetscene/$pid"
    dst="/workspace/xuannv/data_raw/haidian/scenes/$pid/planet"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
    fi
done
```

### 2.3 统计量

Planet 统计量已存在：`/workspace/xuannv/statistics/haidian/planet_stats.json`（4 通道 mean/std）。无需重新计算。

### 2.4 AEF 教师嵌入

路径：`/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches/`，共 320 个 `.npy`，shape `(64, 128, 128)`。配置中设置 `data.aef_embed_dir` 即可启用蒸馏。

### 2.5 训练时间窗口

配置已支持 `valid_start_ms` / `valid_end_ms` 与 temporal window augmentation。v2 配置将窗口设为 2025-12-01 ~ 2026-04-30，使训练/评估时间一致。

---

## 3. 配置设计

基于 `configs/config_multires_v1.yaml` 派生新配置 `configs/config_multires_v2.yaml`。

### 3.1 关键字段变更

```yaml
experiment:
  name: exp_multires_v2_0613
  output_dir: /workspace/xuannv/outputs/exp_multires_v2_0613

data:
  # 输入源扩展为 5 路
  num_input_sources: 5
  input_sources:
    - s2
    - s1
    - landsat
    - tianyi_sar
    - planet

  source_gsd:
    s2: 10
    s1: 10
    landsat: 30
    tianyi_sar: 10
    planet: 3

  source_channels:
    s2: 6
    s1: 2
    landsat: 6
    tianyi_sar: 1
    planet: 4

  # 重建目标：4 路 10m + planet（连续值）
  num_target_sources: 5
  target_sources:
    - name: s2
      loss_type: 0
      sensor_src: s2
      out_channels: 6
    - name: s1
      loss_type: 0
      sensor_src: s1
      out_channels: 2
    - name: landsat
      loss_type: 0
      sensor_src: landsat
      out_channels: 6
    - name: tianyi_sar
      loss_type: 0
      sensor_src: tianyi_sar
      out_channels: 1
    - name: planet
      loss_type: 0
      sensor_src: planet
      out_channels: 4

  aef_embed_dir: /workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches

model:
  num_sensor_types: 16  # 已包含 planet id=6
  source_stem_stride:
    s2: 2
    s1: 2
    landsat: 1
    tianyi_sar: 2
    planet: 4          # 427 -> ~107，再插值到 64×64
  source_stem_layers:
    s2: 1
    s1: 1
    landsat: 1
    tianyi_sar: 1
    planet: 2          # 高分辨率源加深一层 stem 降低后续计算量
  source_channels:
    s2: 6
    s1: 2
    landsat: 6
    tianyi_sar: 1
    planet: 4
```

### 3.2 训练策略（课程学习）

```yaml
training:
  epochs: 100
  lr: 0.0001
  lr_min: 0.000001
  weight_decay: 0.01
  grad_clip_norm: 1.0
  gradient_accumulation_steps: 4
  warmup_epochs: 10

  # 课程学习：前 20 epoch 先推散 embedding，再逐步加回复建与蒸馏
  use_curriculum: true
  curriculum_epochs: 20
  curriculum_recon_weight: 0.1
  reconstruction_weight: 1.0
  curriculum_pre_norm_uniform_weight: 3.0
  pre_norm_uniform_weight: 0.5
  curriculum_erank_loss_weight: 1.0
  erank_loss_weight: 0.2
  curriculum_start_weight: 0.3
  curriculum_end_weight: 1.0

  # 反坍缩/时序损失
  use_pre_norm_uniform: true
  pre_norm_uniform_weight: 0.5
  use_spatial_vicreg: true
  variance_weight: 0.25
  covariance_weight: 0.001
  vicreg_min_std: 1.0
  temporal_contrastive_weight: 0.1
  inter_patch_infonce_weight: 0.1
  inter_patch_infonce_temperature: 0.1

  # AEF 教师蒸馏
  aef_spatial_distill_weight: 0.5
  aef_global_distill_weight: 0.25

  # 源重建权重：planet 权重略低，避免 427×427 重建主导显存/梯度
  source_recon_weights: [1.0, 1.0, 1.0, 1.0, 0.5]

  save_every: 10
  save_best_balanced: true
  eval_every: 10
  max_steps_per_epoch: 200
```

---

## 4. 模型/代码改动

### 4.1 已有能力

v1 改造已支持：
- `List[Tensor]` 多分辨率输入；
- `SensorEncoderBank` per-source stem + 公共空间插值；
- `AEFModel` per-source decoder + 插值到目标分辨率；
- 训练损失适配 `List[Tensor]` 重建。

### 4.2 本次无需修改模型结构

仅通过配置即可加入 planet。若 427×427 重建导致显存或速度问题，再考虑：
- 把 planet `source_stem_stride` 从 4 提升到 7；
- 把 planet 重建权重降到 0.25 或关闭 planet 作为 target。

### 4.3 新增脚本

1. `scripts/preprocessing/link_planet_to_haidian.sh`：创建 planet 软链；
2. `configs/config_multires_v2.yaml`：v2 训练配置；
3. `configs/config_multires_v2_quick.yaml`：quick-diag 配置（epoch=2，max_steps=20）。

---

## 5. 训练与评估流程

### 5.1 训练前检查

```bash
npu-smi info
bash scripts/preprocessing/link_planet_to_haidian.sh
python scripts/test_smoke.py --config configs/config_multires_v2_quick.yaml
```

### 5.2 快速验证

```bash
torchrun --nproc_per_node=2 scripts/train/train.py \
    --config configs/config_multires_v2_quick.yaml --save-every 1
```

### 5.3 全量训练

```bash
tmux new-session -d -s train -c /workspace/xuannv
tmux send-keys -t train 'conda activate xuannv' Enter
tmux send-keys -t train 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' Enter
tmux send-keys -t train 'torchrun --nproc_per_node=8 scripts/train/train.py --config configs/config_multires_v2.yaml --save-every 10' Enter
```

### 5.4 评估

```bash
bash scripts/eval/launch_eval.sh \
    --checkpoint /workspace/xuannv/outputs/exp_multires_v2_0613/epoch_100.pt \
    --mode all
```

海淀区 2026 标注评估：

```bash
python scripts/eval/evaluate_haidian2026_labels.py \
    --embedding-file out/eval/patch_embeddings.npz \
    --second-embedding-file out/eval/patch_embeddings.npz \
    --second-month 2025-12 \
    --output-dir out/eval_haidian2026_v2/ \
    --device npu:0
```

---

## 6. 成功标准

| 指标 | v1 结果 | v2 目标 |
|------|---------|---------|
| `erank` | 8.88 | > 32 |
| `raw_unif` | > -0.5 | -4.0 ~ -1.0 |
| `active_dims` | 极低 | 64/64 |
| 海淀区 2026 农用地变化 AUC | 0.82 | > 0.85 |
| 海淀区 2026 施工工地 AUC | 0.66 | > 0.75 |

---

## 7. 风险与应对

| 风险 | 应对 |
|------|------|
| Planet 427×427 重建显存爆炸 | 先设 `planet` stem_stride=4/layers=2，若 OOM 改为 stride=7 或关闭 planet target；|
| 反坍缩损失导致训练不稳定 | 使用课程学习，前 20 epoch 低 recon、高 uniform/erank；|
| AEF 蒸馏与反坍缩目标冲突 | AEF 权重随课程从 0.3  ramp 到 1.0；|
| 评估时无 2025-12 对应帧 | 用双时相提取脚本，缺失月份自动 fallback 到最近帧；|

---

## 8. 下一步

1. 用户审批本设计文档；
2. 使用 `writing-plans` skill 生成实现计划；
3. 按实现计划创建软链、配置文件、快速验证、全量训练。
