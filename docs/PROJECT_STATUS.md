# 玄女 (XuanNv) 双教师蒸馏训练 — 项目状态总览

> 更新时间: 2026-06-04  
> 当前阶段: Round 5 / 双教师 (AEF + OlmoEarth) 蒸馏训练

---

## 1. 项目目标

训练一个自监督遥感表征学习模型，同时蒸馏两个教师模型的知识：

- **Teacher 1: AEF (AlphaEarth Foundations)** — 64D 嵌入，直接从源图像编码
- **Teacher 2: OlmoEarth** — 768D 嵌入，大规模 Earth Observation 预训练模型

学生模型输出 64D embedding，通过投影头对齐 768D 空间，用于下游变化检测任务。

---

## 2. 硬件与环境

| 项目 | 配置 |
|------|------|
| NPU | 8 × Ascend 910B4 |
| DDP Backend | hccl |
| 训练环境 | `xuannv` (torch 2.1.0 + torch_npu) |
| Token 生成环境 | `olmoearth` (torch 2.7.1 + NPU) |

**环境分离原因**: `xuannv` 环境的 torch 2.1.0 缺少 `DeviceMesh`/`fully_shard`，无法运行最新 `olmoearth_pretrain` 代码，因此 token 生成必须在 `olmoearth` 环境完成。

---

## 3. 数据概况

### 3.1 区域

| 区域 | Patches | 数据根目录 | 统计目录 |
|------|---------|-----------|----------|
| 海淀 (haidian) | 320 | `/workspace/xuannv/data_raw/haidian/scenes` | `/workspace/statistics/haidian` |
| 哈尔滨 (harbin) | 424 | `/workspace/xuannv/data_raw/harbin_newarea_olmoearth` | (manifest 中配置) |
| **合计** | **744** | — | — |

### 3.2 月度样本

- 时间跨度: 2025-01 ~ 2026-05 (17 个月)
- 月度样本数: **12,008** (744 patches × 17 months，部分月份缺失)
- Batch size: 4 per GPU
- 有效 batch: 4 × 8 = 32
- 每 epoch steps: ~375

### 3.3 教师 Token

| 教师 | 路径 | 月份数 | Patches |
|------|------|--------|---------|
| OlmoEarth Haidian | `/workspace/outputs/olmoearth_haidian/` | 17 | 320 |
| OlmoEarth Harbin | `/workspace/outputs/olmoearth_harbin/` | 17 | 424 |
| AEF Haidian | `/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches/` | — | 320 |
| AEF Harbin | `/workspace/xuannv/data_raw/harbin/aef_embeddings/harbin_2025_patches/` | — | 424 |

**OlmoEarth Token 加载**: 17 个月，12,648 patches，~19.89GB (fp16 内存常驻)

**Token 键格式**: `year * 100 + month` (如 `202501`, `202512`, `202601`)，避免 2025-01 与 2026-01 冲突。

---

## 4. 已完成的工作

### 4.1 Token 生成 (已完成)

- [x] 使用 8 卡 DDP 生成 Haidian + Harbin 全部 17 个月 (2025-01 ~ 2026-05) 的 OlmoEarth spatial tokens
- [x] 2026 年 token 存储在 `2026/MM/` 子目录下
- [x] 每个 patch 输出 `(32, 32, 768)` fp16 spatial tokens
- [x] 创建了 mosaic 可视化脚本验证 token 质量 (`scripts/visualize_olmoearth_spatial_mosaic.py`)

### 4.2 代码修复 (已完成)

- [x] **Critical**: 修复 `multi_region_dataset.py` 中 `_preload_teacher_tokens` 的缩进错误（token 从未被加载）
- [x] **Critical**: 修复 year-month 键冲突（`_teacher_tokens` 键从月份数字改为 `year*100+month`）
- [x] **Critical**: 修复 2026 目录被 `isdigit()` 误识别为月份的问题（`"2026".isdigit() == True`）
- [x] **Critical**: 修复 `train.py` 缺少 `import json`（eval 结果解析崩溃）
- [x] **Critical**: 修复 `_load_teacher_tokens` 返回键不一致导致的 DataLoader collate 崩溃（`teacher_global_emb` 有时缺失）
- [x] 添加 `reconstruction_weight=0.05`
- [x] 硬化 eval 逻辑：kNN `all_gather_object` deadlock 防护 + subprocess 600s timeout 包装
- [x] 修改 `print_interval=1`，每个 step 都打印日志

### 4.3 训练配置

```yaml
experiment:
  name: exp_dual_teacher_v1_aef_olmoearth
  output_dir: /workspace/outputs/exp_dual_teacher_v1

training:
  epochs: 40
  eval_every: 10
  batch_size: 4        # per GPU
  lr: 0.0001
  warmup_epochs: 10
  
  # 蒸馏权重 (curriculum: epoch 0 实际为 30%)
  aef_spatial_distill_weight: 2.5
  aef_global_distill_weight: 1.0
  olmoearth_spatial_distill_weight: 1.0
  olmoearth_global_distill_weight: 0.5
  
  # 反坍缩
  pre_norm_uniform_weight: 1.0
  variance_weight: 0.2
  covariance_weight: 0.05
  erank_loss_weight: 0.3
  
  # 重建
  reconstruction_weight: 0.05
  source_recon_weights: [1.0, 1.0, 1.0, 0.02, 0.1]
  
  # 分类
  classification_weight: 0.03
```

---

## 5. 当前问题 (严重)

### 5.1 🚨 问题 1: 训练速度极慢

| 指标 | 数值 | 正常值 | 偏差 |
|------|------|--------|------|
| 每 epoch 时间 | **~8865s (~2.5h)** | ~10-20min | **慢 10-15 倍** |
| 每 step 时间 | **~23.6s** | ~1-3s | **慢 8-10 倍** |
| 预计总训练时间 | **~100 小时** | ~6-12 小时 | 不可接受 |

**根因假设 (按可能性排序)**:

1. **`torch.linalg.svdvals` (erank 计算) 在 NPU 上极慢** — erank 损失和诊断每 step 都做 SVD
2. **`_all_gather_diff` 跨卡聚合开销大** — `torch.distributed.nn.functional.all_gather` 在 hccl 上可能有同步开销
3. **`encode_frames` 中的 `sys.stdout.write(".")`** — 每 forward 一次就触发一次 CPU 同步
4. **`torch.npu.empty_cache()` 每 100 step** — 可能导致 NPU 同步和内存碎片化
5. **DataLoader collate / 内存拷贝** — 虽然数据预加载，但教师 token 是 fp16，可能涉及类型转换开销

**诊断方法**: 已创建 `scripts/profile_train_step.py` 进行单卡分阶段 profiling。

### 5.2 🚨 问题 2: Embedding 空间严重坍缩

| 阶段 | erank | 评价 |
|------|-------|------|
| Step 0 | 6.3 | 偏低 |
| Step 4 | 10.2 | 短暂好转 |
| Epoch 1 | 10.2 | — |
| Epoch 2 | 6.3 | 下降 |
| Step 226 (Epoch 3) | **3.7-3.8** | **严重坍缩** |

**目标**: erank > 32 (embedding_dim=64 的一半)  
**现状**: erank < 4，意味着 64 维空间中实际只用了不到 4 个有效维度。

**根因**: 双教师蒸馏损失主导了训练。AEF spatial cosine distance 从 1.05 → 0.15（cosine similarity 从 -0.05 → 0.85），学生正在紧密模仿教师方向，导致方向坍缩。反坍缩机制（uniformity + VICReg + erank）权重不足以对抗强大的蒸馏拉力。

### 5.3 问题 3: 无 Checkpoint / Eval 输出

- `save_every=5`，但还没有新的 checkpoint 被保存（只有旧的 `epoch_best_miou0.2889_ep10.pt`）
- `eval_every=10`，Epoch 10 才会运行第一次 eval
- 需要关注 eval 是否会因为 OOM 或 timeout 失败

---

## 6. 待办事项 (TODO)

### 6.1 紧急 (P0)

- [ ] **Profiling**: 运行 `scripts/profile_train_step.py` 找出每 step 23.6s 的瓶颈
- [ ] **修复速度**: 根据 profiling 结果，禁用/优化最慢的操作
- [ ] **修复坍缩**: 调整超参数，降低蒸馏权重 / 增加反坍缩权重
- [ ] **重启训练**: 应用修复后重启 8 卡 DDP 训练

### 6.2 高优先级 (P1)

- [ ] 监控 Epoch 10 的 eval 结果（kNN mIoU + FullEval）
- [ ] 如果 erank 持续 < 10，考虑完全禁用双教师蒸馏，先训练 base model
- [ ] 评估是否需要生成缺失的 `emb_all.npz`（当前大部分月份缺失 global embedding）
- [ ] 检查 `source_recon_weights` 中 DEM=0.02 和 WorldCover=0.1 是否过低

### 6.3 中优先级 (P2)

- [ ] AEF 嵌入 mosaic 可视化（用户之前提过但未完成）
- [ ] 训练完成后生成下游变化检测评估
- [ ] 对比单教师 vs 双教师的 distill 效果

---

## 7. 关键文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练配置 | `configs/config_dual_teacher_v1.yaml` | 主配置文件 |
| 多区域清单 | `configs/multi_region_manifest.json` | Haidian + Harbin 数据配置 |
| 训练入口 | `scripts/train/train.py` | DDP 训练启动脚本 |
| Trainer | `src/training/trainer.py` | 训练循环、损失计算、eval |
| 模型 | `src/models/model.py` | AEFModel + distill head |
| 数据集 | `src/data/dataset.py` | HarbinPatchDataset |
| 多区域数据集 | `src/data/multi_region_dataset.py` | MultiRegionPatchDataset |
| DataLoader | `src/data/builder.py` | DataLoader 构建 |
| Profiling | `scripts/profile_train_step.py` | 单卡 step 耗时分析 |
| 评估脚本 | `scripts/eval/run_periodic_eval.py` | 下游任务评估 |
| Token 生成 | `scripts/distill/generate_olmoearth_tokens_ddp.py` | 8卡 DDP token 生成 |
| Mosaic 可视化 | `scripts/visualize_olmoearth_spatial_mosaic.py` | OlmoEarth token 空间可视化 |
| 训练日志 | `/workspace/outputs/exp_dual_teacher_v1/train_*.log` | 实时日志 |
| Checkpoint | `/workspace/outputs/exp_dual_teacher_v1/epoch_*.pt` | 模型检查点 |
| 数据集缓存 | `/workspace/outputs/.cache_shared/` | 预加载数据缓存 (12.2GB) |

---

## 8. 历史 Checkpoint

| Checkpoint | mIoU | Epoch | 说明 |
|------------|------|-------|------|
| `epoch_best_miou0.2889_ep10.pt` | 0.2889 | 10 | 上一轮训练的最佳 |

---

## 9. 快速命令参考

```bash
# 查看训练状态
tmux attach -t dual_teacher_v1
tail -f /workspace/outputs/exp_dual_teacher_v1/train_*.log

# NPU 状态
npu-smi info

# 启动训练
cd /workspace/xuannv && source activate xuannv
torchrun --nproc_per_node=8 scripts/train/train.py --config configs/config_dual_teacher_v1.yaml

# Profiling
cd /workspace/xuannv && source activate xuannv
python scripts/profile_train_step.py

# 杀死训练
ps aux | grep train.py | awk '{print $2}' | xargs kill -9 2>/dev/null
```

---

## 10. 备注

- **不要**在训练运行时修改 `dataset.py` 或 `multi_region_dataset.py`，DataLoader worker 会 fork 进程，代码修改不会生效直到重启
- `teacher_global_emb` 目前大部分月份缺失（`emb_all.npz` 不存在），`_load_teacher_tokens` 已返回 zero tensor 作为 fallback
- `find_unused_parameters=True` 在 DDP 中开启，因为部分损失（如 distill）在 warmup 阶段权重为 0 时梯度图为空
- `skip_l2_norm_training=True` 时 bottleneck 输出 pre-norm embedding，uniformity 在欧氏空间计算
