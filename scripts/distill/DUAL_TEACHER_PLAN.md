# 双教师蒸馏实施计划 (AEF + OlmoEarth)

## 1. 方案概述

### 1.1 核心思路

从两个全球预训练模型同时蒸馏知识：
- **Teacher 1**: AEF (AlphaEarth Foundations, 64D) — Google DeepMind, 2025
- **Teacher 2**: OlmoEarth Base (768D) — Allen AI, 2024

**为什么双教师能解决数据少导致的坍塌？**

| 问题 | 单 OlmoEarth | 双教师 (AEF + OlmoEarth) |
|------|-------------|------------------------|
| 维度不匹配 | 64D vs 768D，需投影头 | AEF 64D vs 64D，直接对齐 |
| 蒸馏压力 | 过大 (weight=5.0) → 压扁 | AEF 分担压力，更稳定 |
| 知识来源 | 单一模型 | 两个全球模型互补 |
| erank 保护 | 弱 | AEF 保持学生空间结构 |

### 1.2 论文依据

1. **MTARD (2023)**: 多教师对抗蒸馏，Entropy-Based Balance + Normalization Loss Balance
2. **MT-BERT**: 多教师 hidden loss + distillation loss，不同教师编码互补知识
3. **RdimKD (2023)**: 降维蒸馏范式 — "强迫学生学到老师的全部信息是次优的"
4. **Projector Ensemble (NeurIPS 2022)**: 投影头解耦"分类任务"和"蒸馏任务"

---

## 2. 损失权重设计

### 2.1 权重配置

```yaml
# Teacher 1: AEF (64D 直接对齐) — 主要教师
aef_spatial_distill_weight: 2.5    # 空间蒸馏 (高，因为维度匹配)
aef_global_distill_weight: 1.0     # 全局蒸馏

# Teacher 2: OlmoEarth (768D 投影对齐) — 辅助教师
olmoearth_spatial_distill_weight: 1.0   # 空间蒸馏 (中，需投影头)
olmoearth_global_distill_weight: 0.5    # 全局蒸馏

# 防坍缩保护
erank_loss_weight: 0.3             # 双教师下降低

# 自监督辅助
pre_norm_uniform_weight: 1.0       # 降低，为蒸馏让路
variance_weight: 0.2               # 降低
covariance_weight: 0.05            # 降低

# 变化检测主任务
classification_weight: 0.03        # 保持弱约束
```

### 2.2 Curriculum 学习 (前 10 epoch)

```
Epoch 0-10: 蒸馏权重从 30% → 100% 渐进
Epoch 0:    aef_sp=0.75, olmo_sp=0.30  (轻量蒸馏，先让 backbone 稳定)
Epoch 5:    aef_sp=1.50, olmo_sp=0.60  (逐步增加)
Epoch 10+:  aef_sp=2.50, olmo_sp=1.00  (全量蒸馏)
```

**设计理由**: 先让学生建立稳定的 64D 表示空间，再逐步引入教师信号。

---

## 3. 文件修改清单

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `scripts/distill/download_aef_embeddings.py` | AEF 下载完整版 (Source Cooperative + GEE) |
| `scripts/distill/download_aef_simple.py` | AEF 下载简化版 (仅 GEE) |
| `configs/config_dual_teacher_v1.yaml` | 双教师蒸馏配置文件 |
| `scripts/train/train_dual_teacher.sh` | 训练启动脚本 |

### 3.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/config.py` | 添加 `aef_spatial_distill_weight`, `aef_global_distill_weight`, `use_curriculum` 等 |
| `src/data/dataset.py` | 添加 `_preload_aef_embeddings()` 和 `_load_aef_embedding()` |
| `src/training/trainer.py` | 双教师蒸馏损失计算 + Curriculum + 精简日志 |
| `scripts/train/train.py` | 精简 Epoch 日志 |

---

## 4. 数据准备流程

### 4.1 AEF 嵌入下载

```bash
# 步骤 1: GEE 认证 (如未认证)
earthengine authenticate

# 步骤 2: 下载海淀区 AEF 2024 嵌入
python scripts/distill/download_aef_simple.py \
    --region haidian \
    --year 2024 \
    --action download

# 步骤 3: 从 Google Drive 下载导出的文件，然后裁剪到 patch 级别
python scripts/distill/download_aef_simple.py \
    --region haidian \
    --action crop \
    --aef-tif /path/to/downloaded/aef_haidian_2024.tif
```

### 4.2 预期输出

```
/workspace/raw/aef_embeddings/
├── harbin_2024_patches/
│   ├── patch_0001.npy    # (64, H, W) float32
│   ├── patch_0002.npy
│   └── ...
└── haidian_2024_patches/
    ├── patch_0001.npy
    └── ...
```

### 4.3 OlmoEarth 嵌入 (已有)

```
/workspace/outputs/olmoearth_haidian/
├── 04/
│   ├── spatial_tokens.npz   # (N, 32, 32, 768) fp16
│   └── emb_all.npz          # (N, 768) fp32
├── 06/
└── ...
```

---

## 5. 训练启动

### 5.1 检查数据

```bash
# 检查 AEF 嵌入
ls /workspace/raw/aef_embeddings/haidian_2024_patches/*.npy | wc -l
# 应输出 320 (海淀区 patches 数量)

# 检查 OlmoEarth 嵌入
ls /workspace/outputs/olmoearth_haidian/
```

### 5.2 启动训练

```bash
cd /workspace/xuannv
bash scripts/train/train_dual_teacher.sh
```

### 5.3 监控指标

精简后的日志只显示：

**Step 日志** (每 epoch ~10 条):
```
[Step  10/80] total=1.234 recon=0.123 cls=0.045 var=0.234 cov=0.012 l2unif=0.567 erank=8.2 aef=[sp=0.234,gl=0.123] olmo=[sp=0.345,gl=0.234] lr=0.000100
```

**Epoch 日志**:
```
[12:34:56] Epoch 005/040 | total=1.234 recon=0.123 cls=0.045 var=0.234 cov=0.012 l2unif=0.567 erank=8.2 aef=[sp=0.234,gl=0.123] olmo=[sp=0.345,gl=0.234] lr=0.000100 | time=45.2s elapsed=3m ETA=2h15m
[Eval] epoch 5: kNN acc=0.4567 mIoU=0.2345
```

**关键监控指标**:
| 指标 | 目标 | 说明 |
|------|------|------|
| `erank` | > 6.0 | 防止坍塌 |
| `aef_sp` | 0.1-0.5 | AEF 空间蒸馏损失 |
| `olmo_sp` | 0.1-0.5 | OlmoEarth 空间蒸馏损失 |
| `mIoU` | > 0.30 | 下游变化检测 |

---

## 6. 故障排除

### 6.1 AEF 嵌入下载失败

**问题**: GEE 导出任务失败或无法访问
**解决**: 
- 检查 GEE 认证: `earthengine authenticate`
- 使用 Source Cooperative 直接下载 (需要 `aef-loader` 包)
- 或者手动从 Google Earth Engine 下载

### 6.2 erank 仍然坍塌

**可能原因**:
1. 蒸馏权重仍然过高
2. AEF 嵌入质量有问题

**解决**:
```yaml
# 降低蒸馏权重
aef_spatial_distill_weight: 1.5    # 从 2.5 降低
olmoearth_spatial_distill_weight: 0.5  # 从 1.0 降低

# 增加 erank 保护
erank_loss_weight: 0.5

# 延长 curriculum
use_curriculum: true
curriculum_epochs: 15
curriculum_start_weight: 0.1
```

### 6.3 mIoU 不提升

**可能原因**:
1. 变化检测头训练不足
2. 蒸馏过度，学生只学会了模仿教师

**解决**:
```yaml
# 增加分类权重
classification_weight: 0.05

# 降低 OlmoEarth 权重，提高 AEF 权重
# (AEF 更贴近变化检测任务)
```

---

## 7. 预期效果

| 指标 | 单 OlmoEarth | 双教师 (目标) |
|------|-------------|--------------|
| erank | 4.7 (坍塌) | > 8.0 (稳定) |
| mIoU | 0.2864 | > 0.30 |
| 训练稳定性 | 差 (OOM) | 好 |
| 收敛速度 | 慢 | 快 |

---

## 8. 后续优化方向

1. **动态权重调整**: 根据每个教师的"可靠性"动态调整权重
2. **三教师蒸馏**: 加入 Prithvi-EO-2.0 或 SatCLIP
3. **AEF 时序融合**: 使用多年 AEF 嵌入 (2022-2024) 增强时序理解
4. **对比学习**: 在 AEF 和 OlmoEarth 之间做对比，学习更好的对齐
