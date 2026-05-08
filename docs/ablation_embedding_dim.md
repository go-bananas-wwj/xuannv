# Embedding 维度消融实验计划

## 背景

当前 V7 Phase1 使用 **128-dim embedding**，源自 V4 的升级决策（给下游任务更多学习空间）。原版 AlphaEarth 论文使用 **64-dim**。

用户希望系统性地评估更低维度（8/16/32/64）的表现，为模型压缩和边缘部署提供数据支撑。

---

## 实验目标

1. **量化维度对 backbone 训练质量的影响**：VICReg/KoLeo/重建损失是否随维度变化
2. **量化维度对下游任务的影响**：变化检测 AUC、KNN 分类准确率
3. **确定信息瓶颈临界点**：维度降到多少时性能断崖式下跌
4. **为模型压缩提供依据**：最佳性价比维度

---

## 实验设计

### 对比组

| 组 | Embedding Dim | 来源 | 预期角色 |
|----|--------------|------|---------|
| A (基线) | **128** | 当前 V7 配置 | 上限参照 |
| B | **64** | 原版 AEF 论文标准 | 论文对齐 |
| C | **32** | 减半 | 压缩测试 |
| D | **16** | 四分之一 | 极限压缩 |
| E | **8** | 八分之一 | 信息瓶颈测试 |

### 固定变量（必须一致）

所有组的以下参数必须完全相同：

```yaml
# 数据
data:
  batch_size: 2
  image_size: 128
  max_frames: 32
  manifest_path: /workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered
  preload: true

# 模型（除 embedding_dim 外）
model:
  stem_dim: 128
  precision_dim: 256
  time_dim: 256
  space_dim: 256
  num_blocks: 10
  num_blocks_disable_space: 2
  num_heads: 8
  vmf_kappa: 2000.0
  bottleneck_noise_scale: 0.02
  skip_l2_norm_training: true

# 训练
training:
  epochs: 400
  gradient_accumulation_steps: 8
  lr: 0.0001
  weight_decay: 0.05
  grad_clip_norm: 1.0
  
  vicreg_weight: 2.0
  vicreg_lambda_var: 1.0
  vicreg_lambda_cov: 0.04
  koleo_weight: 0.5
  vicreg_gamma: 1.0
  
  teacher_momentum: 0.996
  reconstruction_weight: 1.0
  consistency_weight: 0.03
  classification_weight: 0.03
  
  recon_warmup_epochs: 10
  warmup_epochs: 20
  lr_schedule: cosine_no_restart
  lr_min: 0.000001
  
  save_every: 20
  checkpoint_interval: 20
```

### 需调整变量

| 参数 | 128-dim (基线) | 64-dim | 32-dim | 16-dim | 8-dim | 调整理由 |
|------|---------------|--------|--------|--------|-------|---------|
| `model.embedding_dim` | 128 | 64 | 32 | 16 | 8 | 实验变量 |
| `training.vicreg_lambda_cov` | 0.04 | **0.02** | **0.01** | **0.005** | **0.003** | 协方差惩罚与维度成反比，低维需要更小的惩罚 |
| `training.koleo_weight` | 0.5 | **0.5** | **0.5** | **0.3** | **0.2** | 低维球面更小，KoLeo 惩罚过强会排斥最近邻 |
| `experiment.name` | `v7_dim128` | `v7_dim64` | `v7_dim32` | `v7_dim16` | `v7_dim8` | 区分输出目录 |

> **理论依据**：VICReg 的协方差正则化 `lambda_cov * off_diag_cov` 中，`off_diag_cov` 的数值范围与维度 `D` 相关。低维时相同的 `lambda_cov` 会导致过强的惩罚，需要按比例缩放。
>
> KoLeo 在低维球面上，最近邻距离天然更小，过大的 `koleo_weight` 会把样本推得过散，破坏局部结构。

---

## 下游任务评估协议

每个维度训练完成后，必须运行以下评估（使用冻结 backbone）：

### 1. 变化检测 AUC（核心指标）

```bash
python scripts/eval/validate_v7_level1_bare.py \
    --checkpoint /workspace/outputs/{exp_name}/epoch_best_*.pt \
    --config configs/ablation/{config_name}.yaml
```

- 评估 105 个光学标注 patch
- 指标：macro AUC、per-patch AUC 分布

### 2. KNN 分类（WorldCover 标签）

```bash
python scripts/eval/benchmark_knn_classification.py \
    --checkpoint /workspace/outputs/{exp_name}/epoch_best_*.pt \
    --k 5 --bootstrap 100
```

- 用 embedding 做 KNN 分类（不训练任何 head）
- 指标：Top-1 / Top-5 准确率、混淆矩阵

### 3. Embedding 空间质量分析

```bash
python scripts/eval/analyze_v7_embedding_quality.py \
    --checkpoint /workspace/outputs/{exp_name}/epoch_best_*.pt
```

- 均匀性（uniformity）
- 对齐性（alignment）
- 维度有效利用率（variance per dim）
- t-SNE 可视化（对比不同维度的聚类效果）

### 4. 下游 Head 训练对比

对每个维度，冻结 backbone，训练 ChangeDetectionHead：

```bash
python scripts/train/train_cd_head.py \
    --backbone /workspace/outputs/{exp_name}/epoch_best_*.pt \
    --epochs 100 --lr 0.001
```

对比收敛速度和最终 AUC。

---

## 训练资源估算

| 维度 | 模型参数量 | 每 epoch 时间 (8卡) | 总训练时间 | 显存/卡 |
|------|-----------|-------------------|-----------|--------|
| 128 | ~120M | ~8 min | ~53h | ~28GB |
| 64 | ~110M | ~7 min | ~47h | ~26GB |
| 32 | ~105M | ~6.5 min | ~43h | ~24GB |
| 16 | ~102M | ~6 min | ~40h | ~22GB |
| 8 | ~101M | ~6 min | ~40h | ~22GB |

> 参数量差异主要来自：
> - `to_embedding` Conv2d: C×D
> - Decoder 输入层: D×hidden
> - Classification heads: D×num_classes
>
> 低维减少的参数有限（~5-15%），主要收益在 decoder 计算量和下游 Head 参数量。

---

## 实验执行顺序

**推荐并行策略**：

由于 8 卡 NPU 一次只能跑一个 8 卡 DDP 训练，建议：

1. **Phase 1（当前）**：完成 128-dim 基线训练
2. **Phase 2（并行 4+4）**：
   - 卡 0-3: 64-dim 训练
   - 卡 4-7: 32-dim 训练
3. **Phase 3（并行 4+4）**：
   - 卡 0-3: 16-dim 训练
   - 卡 4-7: 8-dim 训练

总时间：~53h + max(47h, 43h) + max(40h, 40h) = **~130h（约 5.5 天）**

---

## 结果记录模板

| 维度 | VICReg | Recon | PreUnif | CD AUC | KNN Top-1 | 均匀性 | 有效维度 |
|------|--------|-------|---------|--------|-----------|--------|---------|
| 128 | | | | | | | |
| 64 | | | | | | | |
| 32 | | | | | | | |
| 16 | | | | | | | |
| 8 | | | | | | | |

---

## 预期假设

1. **64-dim vs 128-dim**：AUC 差异 < 0.02（收益递减）
2. **32-dim**：AUC 可能下降 0.02-0.05，但模型显著更小
3. **16-dim**：AUC 可能下降 0.05-0.10，开始出现信息瓶颈
4. **8-dim**：AUC 可能断崖式下跌 > 0.15，无法有效编码遥感时序信息

**关键问题**：32-dim 是否是性价比拐点？

---

## 配置文件清单

需要创建以下配置文件：

```
configs/ablation/
├── v7_dim128.yaml   # 基线（copy 当前 v7）
├── v7_dim64.yaml    # 64-dim + 调整 lambda_cov
├── v7_dim32.yaml    # 32-dim + 调整 lambda_cov + koleo
├── v7_dim16.yaml    # 16-dim + 调整 lambda_cov + koleo
└── v7_dim8.yaml     # 8-dim + 调整 lambda_cov + koleo
```

---

## 自动化脚本

建议创建 `scripts/ablation/run_dim_ablation.sh`：

```bash
#!/bin/bash
# 自动化运行所有维度消融实验

DIMS=(128 64 32 16 8)
for dim in "${DIMS[@]}"; do
    echo "===== Training ${dim}-dim ====="
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        torchrun --nproc_per_node=8 \
        scripts/train/train_ddp_v7.py \
        --config configs/ablation/v7_dim${dim}.yaml \
        --save-every 20 \
        --wandb-project xuannv-backbone-ablation \
        --wandb-run-name v7_dim${dim}
    
    echo "===== Evaluating ${dim}-dim ====="
    python scripts/eval/validate_v7_level1_bare.py \
        --checkpoint /workspace/outputs/v7_dim${dim}/epoch_best_*.pt \
        --config configs/ablation/v7_dim${dim}.yaml
    
    python scripts/eval/analyze_v7_embedding_quality.py \
        --checkpoint /workspace/outputs/v7_dim${dim}/epoch_best_*.pt
done
```

---

*创建时间: 2026-05-08*
*关联: V7 Phase1 训练、V4 评估报告、AlphaEarth 论文*
