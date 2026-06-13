# multires v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v1 多分辨率架构基础上加入 Planet 3m 作为第 5 路输入，通过课程学习重新打开反坍缩/时序/AEF 蒸馏损失，启动并验证 v2 训练。

**Architecture:** 复用 v1 已有的 `List[Tensor]` 多分辨率输入、`SensorEncoderBank` per-source stem 与公共空间插值、per-source decoder 插值到目标分辨率等能力；仅新增 Planet 软链、修改配置、跑通训练与评估。

**Tech Stack:** Python 3.11, PyTorch 2.1 + torch_npu, ModelScope SDK（仅用于下载检查）, rasterio, tmux, NPU (hccl)。

---

## File Map

| 文件 | 作用 |
|------|------|
| `scripts/preprocessing/link_planet_to_haidian.sh` | 为 320 个海淀 patch 创建指向 Planet 数据的符号链接 |
| `configs/config_multires_v2.yaml` | v2 全量训练配置（5 源 + 课程学习 + 反坍缩/蒸馏） |
| `configs/config_multires_v2_quick.yaml` | v2 快速验证配置（2 epoch / 20 steps） |
| `scripts/test_smoke.py` | 冒烟测试（无需修改，验证 5 源前向+损失） |
| `scripts/train/train.py` | 训练入口（无需修改） |
| `scripts/eval/launch_eval.sh` | 一站式评估（无需修改） |
| `scripts/eval/evaluate_haidian2026_labels.py` | 海淀区 2026 标注下游评估（无需修改） |

---

### Task 1: 创建并执行 Planet 软链脚本

**Files:**
- Create: `scripts/preprocessing/link_planet_to_haidian.sh`

- [ ] **Step 1: 写入软链脚本**

```bash
#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/workspace/xuannv/data_raw/beijing/planetscene"
DST_ROOT="/workspace/xuannv/data_raw/haidian/scenes"

if [ ! -d "$SRC_ROOT" ]; then
    echo "ERROR: Planet source not found: $SRC_ROOT"
    exit 1
fi

linked=0
skipped=0
for src_dir in "$SRC_ROOT"/patch_*; do
    [ -d "$src_dir" ] || continue
    pid=$(basename "$src_dir")
    dst_dir="$DST_ROOT/$pid/planet"
    if [ -L "$dst_dir" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    if [ -e "$dst_dir" ]; then
        echo "WARNING: $dst_dir already exists and is not a symlink; skipping"
        continue
    fi
    ln -s "$src_dir" "$dst_dir"
    linked=$((linked + 1))
done

echo "Linked $linked patches, skipped $skipped existing symlinks."
```

- [ ] **Step 2: 设置可执行权限并运行**

```bash
cd /workspace/xuannv
chmod +x scripts/preprocessing/link_planet_to_haidian.sh
bash scripts/preprocessing/link_planet_to_haidian.sh
```

**Expected output:**

```text
Linked 320 patches, skipped 0 existing symlinks.
```

- [ ] **Step 3: 抽样验证软链**

```bash
ls -l /workspace/xuannv/data_raw/haidian/scenes/patch_000000/planet | head -5
```

**Expected:** `planet -> /workspace/xuannv/data_raw/beijing/planetscene/patch_000000`

- [ ] **Step 4: Commit**

```bash
git add scripts/preprocessing/link_planet_to_haidian.sh
git commit -m "preprocessing: add script to symlink Planet 3m into haidian scenes"
```

---

### Task 2: 创建 v2 全量训练配置

**Files:**
- Create: `configs/config_multires_v2.yaml`
- Reference: `configs/config_multires_v1.yaml`

- [ ] **Step 1: 写入配置**

```yaml
# ============================================================
# xuannv 多分辨率输入 v2 配置
# 目标: 引入 Planet 3m + 课程学习反坍缩 + AEF 蒸馏
# 创建时间: 2026-06-13
# ============================================================

experiment:
  name: exp_multires_v2_0613
  seed: 42
  output_dir: /workspace/xuannv/outputs/exp_multires_v2_0613

data:
  dataset_type: harbin_patches
  manifest_path: /workspace/xuannv/data_raw/haidian/scenes
  stats_dir: /workspace/xuannv/statistics/haidian
  batch_size: 1
  num_workers: 0
  image_size: 128
  max_frames: 24
  num_samples: 320
  num_classes: 11
  input_dim: 6
  metadata_dim: 4

  use_multires: true
  patch_size_m: 1280
  source_gsd:
    s2: 10
    s1: 10
    landsat: 30
    tianyi_sar: 10
    planet: 3

  num_input_sources: 5
  input_sources:
    - s2
    - s1
    - landsat
    - tianyi_sar
    - planet

  source_channels:
    s2: 6
    s1: 2
    landsat: 6
    tianyi_sar: 1
    planet: 4

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

  random_target_frame: true
  spatial_augmentation: true
  temporal_window_augmentation: true
  temporal_window_prob: 0.5
  temporal_window_min_frames: 4
  temporal_window_max_frames: 24
  preload: false

  recon_mask_ratio: 0.85
  recon_mask_patch_size: 16

  filter_2025_monthly: false

  aef_embed_dir: /workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches
  use_dual_teacher: false
  teacher_embed_dir: null
  olmoearth_tokens_root: null

model:
  input_dim: 6
  stem_dim: 128
  precision_dim: 128
  time_dim: 256
  space_dim: 256
  embedding_dim: 64
  time_code_dim: 64
  window_code_dim: 64
  relative_time_code_dim: 16
  num_blocks: 8
  num_heads: 8
  vmf_kappa: 8000.0
  bottleneck_noise_scale: 0.02
  metadata_dim: 4
  num_sensor_types: 16
  gradient_checkpointing: true
  per_source_decoders: true
  decoder_hidden_mult: 1
  skip_l2_norm_training: true
  use_distill_head: false

  common_spatial_size: [64, 64]
  source_stem_stride:
    s2: 2
    s1: 2
    landsat: 1
    tianyi_sar: 2
    planet: 4
  source_stem_layers:
    s2: 1
    s1: 1
    landsat: 1
    tianyi_sar: 1
    planet: 2
  source_channels:
    s2: 6
    s1: 2
    landsat: 6
    tianyi_sar: 1
    planet: 4

training:
  epochs: 100
  gradient_accumulation_steps: 4
  lr: 0.0001
  lr_min: 0.000001
  weight_decay: 0.01
  grad_clip_norm: 1.0

  reconstruction_weight: 1.0
  consistency_weight: 0.0
  classification_weight: 0.0

  use_pre_norm_uniform: true
  pre_norm_uniform_weight: 0.5
  batch_uniformity_weight: 0.0
  use_spatial_vicreg: true
  variance_weight: 0.25
  covariance_weight: 0.001
  vicreg_min_std: 1.0
  decorrelation_weight: 0.0
  orthogonality_weight: 0.0
  coding_rate_weight: 0.0
  erank_loss_weight: 0.2
  temporal_gap_aware_weight: 0.0
  temporal_contrastive_weight: 0.1

  aef_spatial_distill_weight: 0.5
  aef_global_distill_weight: 0.25

  olmoearth_spatial_distill_weight: 0.0
  olmoearth_global_distill_weight: 0.0

  inter_patch_infonce_weight: 0.1
  inter_patch_infonce_temperature: 0.1
  inter_variance_weight: 0.0
  inter_variance_min_std: 0.3
  inter_decorr_weight: 0.0
  temporal_cosine_pixel_weight: 0.0
  patch_id_loss_weight: 0.0
  lmim_weight: 0.0
  hyperspherical_uniform_weight: 0.0
  spherical_variance_weight: 0.0
  temporal_magnitude_weight: 0.0
  pixel_change_supervision_weight: 0.0
  change_consistency_weight: 0.0
  memory_bank_size: 512
  text_contrastive_weight: 0.0
  vicreg_weight: 0.0
  dino_weight: 0.0
  koleo_weight: 0.0
  pixel_temporal_info_nce_weight: 0.0
  l2_temporal_weight: 0.0
  ct_reconstruction_weight: 0.0
  aux_classification_weight: 0.0
  bottleneck_cls_weight: 0.0
  encoder_uniform_weight: 0.0
  aef_batch_uniformity_weight: 0.0

  source_recon_weights: [1.0, 1.0, 1.0, 1.0, 0.5]

  teacher_momentum: 0.996
  teacher_dropout_rate: 0.1
  student_frame_drop_rate: 0.5
  student_source_drop_rate: 0.3
  student_front_drop_prob: 0.15
  student_back_drop_prob: 0.15

  kappa_start: 8000.0
  kappa_end: 8000.0
  kappa_warmup_epochs: 0

  lr_schedule: cosine_no_restart
  warmup_epochs: 10
  recon_warmup_epochs: 0

  max_steps_per_epoch: 200

  save_every: 10
  save_best_balanced: true
  eval_every: 10
  early_stop_patience: 100

  use_curriculum: true
  curriculum_epochs: 20
  curriculum_recon_weight: 0.1
  curriculum_pre_norm_uniform_weight: 3.0
  curriculum_erank_loss_weight: 1.0
  curriculum_start_weight: 0.3
  curriculum_end_weight: 1.0

  decoder_cond_dropout: 0.6
  temporal_magnitude_temperature: 0.1
  temporal_use_spatial_weight: true

evaluation:
  knn_k: 5
  bootstrap_samples: 100
```

- [ ] **Step 2: 验证 YAML 可解析**

```bash
cd /workspace/xuannv
/root/miniconda3/envs/xuannv/bin/python3 -c "from src.config import load_config; cfg = load_config('configs/config_multires_v2.yaml'); print(cfg.experiment.name)"
```

**Expected output:**

```text
exp_multires_v2_0613
```

- [ ] **Step 3: Commit**

```bash
git add configs/config_multires_v2.yaml
git commit -m "config: add multires v2 full training config (Planet 3m + curriculum)"
```

---

### Task 3: 创建 v2 快速验证配置

**Files:**
- Create: `configs/config_multires_v2_quick.yaml`
- Modify: `configs/config_multires_v2.yaml`（复制后修改）

- [ ] **Step 1: 生成 quick 配置**

```bash
cd /workspace/xuannv
cp configs/config_multires_v2.yaml configs/config_multires_v2_quick.yaml
```

- [ ] **Step 2: 用 sed 修改关键字段**

```bash
cd /workspace/xuannv
sed -i 's/name: exp_multires_v2_0613/name: exp_multires_v2_0613_quick/' configs/config_multires_v2_quick.yaml
sed -i 's/epochs: 100/epochs: 2/' configs/config_multires_v2_quick.yaml
sed -i 's/max_steps_per_epoch: 200/max_steps_per_epoch: 20/' configs/config_multires_v2_quick.yaml
sed -i 's/save_every: 10/save_every: 1/' configs/config_multires_v2_quick.yaml
sed -i 's/eval_every: 10/eval_every: 1/' configs/config_multires_v2_quick.yaml
sed -i 's/save_best_balanced: true/save_best_balanced: false/' configs/config_multires_v2_quick.yaml
```

- [ ] **Step 3: 验证 quick 配置**

```bash
/root/miniconda3/envs/xuannv/bin/python3 -c "from src.config import load_config; cfg = load_config('configs/config_multires_v2_quick.yaml'); print(cfg.experiment.name, cfg.training.epochs, cfg.training.max_steps_per_epoch)"
```

**Expected output:**

```text
exp_multires_v2_0613_quick 2 20
```

- [ ] **Step 4: Commit**

```bash
git add configs/config_multires_v2_quick.yaml
git commit -m "config: add multires v2 quick-diag config"
```

---

### Task 4: 冒烟测试验证 5 源前向与损失

**Files:**
- Test: `scripts/test_smoke.py`（已有，无需修改）

- [ ] **Step 1: 运行冒烟测试**

```bash
cd /workspace/xuannv
/root/miniconda3/envs/xuannv/bin/python3 scripts/test_smoke.py --config configs/config_multires_v2_quick.yaml
```

**Expected output:** 无报错，最后打印 `Smoke test passed.` 或类似成功信息。

- [ ] **Step 2: 检查关键日志**

```bash
grep -E "(ERROR|FAIL|Smoke test passed)" /tmp/smoke_test.log 2>/dev/null || echo "check stdout above"
```

- [ ] **Step 3: Commit（如有测试脚本改动）**

若 `test_smoke.py` 需要改动，按实际改动提交；否则无需提交。

---

### Task 5: 快速训练验证（2 epoch / 20 steps）

**Files:**
- Use: `scripts/train/train.py`

- [ ] **Step 1: 检查 NPU 占用**

```bash
npu-smi info
```

**Expected:** 至少 2 张卡空闲。

- [ ] **Step 2: 启动 2 卡 quick-diag 训练**

```bash
cd /workspace/xuannv
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 scripts/train/train.py \
    --config configs/config_multires_v2_quick.yaml \
    --save-every 1
```

**Expected:** 2 个 epoch 跑完，loss 无 NaN/Inf，`raw_unif` 进入 -4 ~ -1 区间，`erank` 明显高于 v1。

- [ ] **Step 3: 检查日志关键指标**

```bash
ls -1 /workspace/xuannv/outputs/exp_multires_v2_0613_quick/*.log | tail -1 | xargs tail -30
```

**Expected:** 最后几行包含 epoch 2 step 20，无 `nan`/`inf`。

- [ ] **Step 4: 清理 quick 输出（可选，节省空间）**

```bash
rm -rf /workspace/xuannv/outputs/exp_multires_v2_0613_quick
```

---

### Task 6: 全量 100 epoch 训练

**Files:**
- Use: `scripts/train/train.py`

- [ ] **Step 1: 检查 8 卡全部空闲**

```bash
npu-smi info
```

- [ ] **Step 2: 创建 tmux 会话并启动训练**

```bash
tmux new-session -d -s multires_v2 -c /workspace/xuannv
tmux send-keys -t multires_v2 'conda activate xuannv' Enter
tmux send-keys -t multires_v2 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' Enter
tmux send-keys -t multires_v2 'torchrun --nproc_per_node=8 scripts/train/train.py --config configs/config_multires_v2.yaml --save-every 10' Enter
tmux detach -t multires_v2
```

- [ ] **Step 3: 验证训练已启动**

```bash
sleep 30
tmux capture-pane -t multires_v2 -p | tail -20
```

**Expected:** 看到 `epoch=0 step=0` 日志，无 OOM/NaN。

- [ ] **Step 4: 等待训练完成**

全量训练约 10~14 小时（视 8 卡实际速度）。完成后自动通知或定时查看：

```bash
tmux capture-pane -t multires_v2 -p | tail -30
```

---

### Task 7: 评估

**Files:**
- Use: `scripts/eval/launch_eval.sh`
- Use: `scripts/eval/evaluate_haidian2026_labels.py`

- [ ] **Step 1: 一站式评估（extract + knn + auc）**

```bash
cd /workspace/xuannv
bash scripts/eval/launch_eval.sh \
    --checkpoint /workspace/xuannv/outputs/exp_multires_v2_0613/epoch_100.pt \
    --mode all
```

**Expected:** 输出 `out/eval/knn/` 与 `out/eval/auc/` 结果。

- [ ] **Step 2: 海淀区 2026 标注双时相评估**

```bash
cd /workspace/xuannv
python scripts/eval/evaluate_haidian2026_labels.py \
    --embedding-file out/eval/patch_embeddings.npz \
    --second-embedding-file out/eval/patch_embeddings.npz \
    --second-month 2025-12 \
    --output-dir out/eval_haidian2026_v2/ \
    --device npu:0
```

**Expected:** 输出每个类别的 Linear/MLP AUC、IoU、BAcc。

- [ ] **Step 3: 汇总结果并提交**

将结果记录到 `docs/multires_v2_results_202606XX.md`，并 commit：

```bash
git add docs/multires_v2_results_202606XX.md
git commit -m "docs: add multires v2 evaluation results"
```

---

## Self-Review

**Spec coverage:**
- Planet 软链接入 → Task 1
- v2 配置（5 源、课程学习、反坍缩、蒸馏） → Task 2
- quick-diag 配置 → Task 3
- 冒烟测试/快速验证 → Task 4/5
- 全量训练 → Task 6
- 下游评估 → Task 7

**Placeholder scan:** 无 TBD/TODO；所有配置字段、脚本路径、命令均已给出。

**Type consistency：**
- 配置中 `source_recon_weights` 长度 5 与 `target_sources` 一致；
- `source_channels` / `source_stem_stride` / `source_stem_layers` 均包含 planet；
- 训练器读取的 `curriculum_*` 字段与现有 `src/training/trainer.py` 命名一致。
