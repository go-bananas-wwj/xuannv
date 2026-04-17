# AEF_qwen 变化检测能力升级计划 V2（按用户意见修订）

> **核心目标**：让 backbone 的 embedding 对**相邻月份（1-2 个月间隔）**的细微变化敏感，同时为每个变化类别配备独立的轻量任务头，最终让 per-patch AUC 的"长尾低分"问题得到显著改善。
> 
> **关键约束**：
> - 只能用 **GPU6 单卡训练**（不支持多卡 DDP）。
> - CD Head 保持**轻量**，参考 AlphaEarth 原始论文的"轻量下游头"设计理念。
> - 下游任务以**相邻月份变化监测**为主（如 2025-04→2025-06，间隔约 2 个月）。
>
> **现状速览**（基于 `benchmark_summary.json`）：
> | 指标 | Backbone 裸测 | CD Head V3+OHEM |
> |------|--------------|-----------------|
> | Mean AUC | 0.505 | 0.755 |
> | Median AUC | 0.490 | 0.917 |
> | 关键问题 | 全面失效 | 两极分化严重：部分 patch AUC≈0.999，部分≈0.0 |

---

## 一、当前核心问题诊断

### 1.1 Embedding 对"短间隔变化"不敏感
当前 `validate_v2.py` 使用跨季度大窗口（2023Q3-Q4 vs 2024Q3-Q4），但实际下游是**相邻月份**（2 个月间隔）。短间隔内变化信号更微弱，模型更容易被季节性噪声淹没。

### 1.2 Per-Patch 性能两极分化
从 `benchmark_summary.json` 可见：
- 高分 patch（AUC > 0.95）：变化区域大、语义特征明显
- 低分 patch（AUC < 0.2）：变化区域小、或主要由"季节性差异"主导

这说明模型**不能区分"真实变化"和"季节性伪变化"**。

### 1.3 单卡训练的资源限制
当前 `trainer.py` 基于 DDP 设计。单卡训练时：
- `batch_size=2` 保持不变（受显存限制）。
- 梯度累积（`gradient_accumulation_steps=4`）可以模拟大 batch。
- 但 **batch-based uniformity / decorrelation 在小 batch 下不稳定**（参考 IConE, 2026）。

---

## 二、Backbone（Embedding 模型）升级

### 2.1 跨时相掩码重建（Cross-Temporal Masked Reconstruction）⭐ 新增

**动机**：让模型学会"利用另一个时间窗口的信息来补全当前窗口的缺失部分"，从而被迫理解两个窗口之间的差异。

**方案**：
```
输入:
  - w1: 完整输入帧序列
  - w2: 空间掩码后的输入帧序列（随机 mask 30% 空间 patch）

目标:
  - 从 w1 + masked(w2) 重建 w2 的完整内容

实施细节:
  1. 在 dataset.py 中，为 w2 生成随机空间掩码 [H, W]（训练时）
  2. 将掩码区域置零或填充均值
  3. 模型输出 w2 的重建图（通过现有的 per_source_decoders）
  4. Loss: L1(重建, 原始) 只在掩码区域计算
  5. 若 w1 和 w2 内容高度相似（无变化），重建容易；若有真实变化，
     模型必须利用 w1 的上下文来推断 w2 的 masked 区域
```

**为什么对短间隔变化检测有效**：
- 相邻月份的地物通常高度相似，**真实变化区域往往是局部的**。
- 掩码重建强迫模型关注"哪些局部区域在两个月份之间不一致"。
- 这与下游变化检测的像素级判别目标完全一致。

---

### 2.2 EMA Teacher + DINO/iBOT 自蒸馏

**动机**：单卡小 batch 场景下，batch statistics 极不稳定。EMA Teacher 提供稳定、低噪声的目标，是防止训练崩溃的关键。

**单卡适配方案**：

#### A. EMA Teacher 模型
```python
class SingleGPUTrainer:
    def __init__(self, cfg):
        self.model = AEFModel(cfg).to("cuda:6")
        self.teacher = copy.deepcopy(self.model).to("cuda:6").eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
    def update_teacher(self, momentum=0.996):
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
                p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)
```

#### B. DINO Head（图像级）
- 在 `VMFBottleneck` 之后添加轻量 projector：`embedding_dim → 256 → 65536`（prototype 数量）。
- Teacher 输出经过 centering + softmax：`P_teacher = softmax((logits_t - center) / tau)`
- Student 输出经过 softmax：`P_student = softmax(logits_s / tau)`
- Loss: `L_dino = -sum(P_teacher * log(P_student))`
- Center 用 EMA 更新：`center = momentum * center + (1-momentum) * batch_mean`

**单卡注意**：prototype 数可以减小到 16384 或 4096，降低显存。

#### C. iBOT Head（Patch 级）
- 对 `embedding_map [B, D, H, W]` 的每个空间位置添加共享 MLP head。
- 只对被掩码的空间位置计算 loss（与 2.1 的掩码策略共享）。
- Teacher 提供被掩码位置的目标分布，Student 学习匹配。

**预期效果**：
- Patch-level iBOT 提升空间精细度，直接帮助像素级变化检测。
- Teacher 的稳定性补偿了小 batch 的统计噪声。

---

### 2.3 反坍缩正则 → VICReg + KoLeo（小 batch 适配版）

**动机**：当前 `raw_uniformity + decorrelation + variance + orth` 在 batch_size=2 时极不稳定（`decorr=29.7` 说明数值爆炸）。VICReg 的 variance 和 covariance 项在小 batch 下同样会退化（IConE, 2026）。

**单卡适配方案**：

#### A. 用 Teacher-Student Pair 替代 Batch Statistics
标准 VICReg 需要两个视图（z1, z2）的 batch statistics。单卡 batch=2 时，改为：
```python
def vicreg_teacher_student(emb_student, emb_teacher):
    # emb: [B, D] — B=2, 但 teacher/student 各提供一个视图
    # 1. Invariance: MSE(emb_student, emb_teacher)
    inv = F.mse_loss(emb_student, emb_teacher)
    
    # 2. Variance: 跨所有样本（student + teacher 合并）
    z_all = torch.cat([emb_student, emb_teacher], dim=0)  # [2B, D]
    std = torch.sqrt(z_all.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std))
    
    # 3. Covariance: 同样基于合并样本
    z_all = z_all - z_all.mean(dim=0, keepdim=True)
    cov = (z_all.T @ z_all) / (z_all.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / cov.shape[0]
    
    return inv + 1.0 * var + (1.0/25.0) * cov_loss
```

**关键改进**：将 student 和 teacher 的 embedding 合并计算 statistics，有效 batch 从 2 提升到 4，缓解小 batch 退化。

#### B. KoLeo 正则化
```python
def koleo_loss(x):
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    # 排除自身
    eye = torch.eye(dists.shape[0], device=dists.device)
    dists = dists + eye * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
```
- 强制 batch 内（student + teacher 合并后）最近邻距离最大化。
- 不依赖 covariance 矩阵，对 batch size 极不敏感。

#### C. Expander（可选）
- 在计算 VICReg/KoLeo 前，加一个轻量 MLP：`128 → 512 → 128`。
- expander 只在训练时活跃，不影响 inference embedding。

---

### 2.4 架构深化与数据采样策略（单卡适配）

#### A. STP Blocks 增加（8 → 12）
- 原版 AEF 有 15 个 blocks，当前 8 个。考虑到单卡显存，增至 **12 个**。
- Block 1-2 禁用 Space 路径（只做 Precision + Time）。
- **显存控制**：开启 `gradient_checkpointing`（当前已支持），以时间换空间。

#### B. 双窗口采样策略：面向"相邻月份变化"
当前下游任务是相邻月份（2 个月间隔），而非跨季度。采样策略需调整：

**修改 `dataset.py` 的双窗口逻辑**：
```python
# 当前：随机裁剪 valid_period，可能只相差几天
# 新策略：

STRATEGY = "adjacent_month"  # 新增模式

if STRATEGY == "adjacent_month":
    # 1. 从 ts_sorted 中选择两个相邻月份的中点作为窗口中心
    # 2. w1 = [month_i_start, month_i_end]
    # 3. w2 = [month_{i+1}_start, month_{i+1}_end]
    # 4. 保证两个窗口的帧数 >= min_frames
    
    # 例如：2025-04-15 ~ 2025-04-30 vs 2025-06-01 ~ 2025-06-15
    # 间隔约 1-2 个月，与下游任务对齐
```

**训练-验证对齐**：
- 训练时的 w1/w2 间隔应与验证时一致（当前 validate_v2 用季度窗口，与月度训练不匹配）。
- 建议新增 `validate_monthly.py`，使用与训练相同的相邻月份窗口，作为更真实的评估。

#### C. 时序窗口增强（Temporal Window Augmentation）升级
当前 `temporal_window_augmentation` 随机裁剪 `valid_period`。针对相邻月份场景：
- 以月份边界为锚点，随机扰动窗口起止 ±7 天。
- 保证 w1 和 w2 不会重叠（至少间隔 7 天）。
- 这模拟了"不同采集时间"带来的变化不确定性。

---

## 三、变化检测头（CD Head）升级 —— 轻量 + 类特定

### 3.1 设计原则：对齐 AlphaEarth 原始论文
AlphaEarth 论文中强调：**"为每个下游任务训练一个轻量线性或浅层 MLP 头"**，保持 backbone 通用、head 专用。

当前 MultiClassChangeDetectionHead 的问题是：
- 3 个类别共享同一个 encoder（2 层 ResBlock + ECA）。
- construction 样本占主导，demolition 和 land_conversion 被淹没。
- 不同变化类型的空间模式差异大（建筑工地扩散 vs 道路延伸），共享 encoder 难以兼顾。

### 3.2 类特定轻量头（Class-Specific Lightweight Heads）

**方案**：每个类别一个独立的轻量 head，共享输入特征但独立参数。

```python
class ClassSpecificCDHead(nn.Module):
    """类特定轻量变化检测头。
    
    设计参考 AlphaEarth: 每个下游任务一个轻量头。
    总参数量与 V3 相当，但每类独立优化。
    """
    NUM_CLASSES = 3
    CLASS_NAMES = ["construction", "demolition", "land_conversion"]
    
    def __init__(self, embedding_dim=128, hidden_dim=32):
        super().__init__()
        # 轻量差异编码器（所有类别共享，只算一次）
        in_dim = embedding_dim * 2
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),  # 1x1 降维
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        
        # 每类独立的轻量 head（2层 conv）
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            for _ in range(self.NUM_CLASSES)
        ])
        
    def forward(self, emb_before, emb_after):
        # 共享差异特征
        feat = torch.cat([emb_before, emb_after], dim=1)
        feat = self.diff_encoder(feat)
        
        # 每类独立预测
        logits = torch.cat([head(feat) for head in self.heads], dim=1)
        return logits  # [B, 3, H, W]
```

**优势**：
- **参数量**：`diff_encoder (2*128*32) + 3 * (32*32*3*3 + 32*1)` ≈ 8K + 3*9K ≈ 35K，极其轻量。
- **独立优化**：每类可用不同采样策略和损失权重，避免 construction 主导。
- **可扩展**：新增类别只需添加一个 head，不影响已有类别。

### 3.3 每类独立损失与采样

```python
# 训练时，每类独立计算 OHEM Focal Loss
losses = []
class_weights = [0.5, 2.0, 1.5]  # construction 降权, demolition 提权

for c in range(3):
    logits_c = logits[:, c]  # [B, H, W]
    target_c = mc_target[:, c]
    loss_c = ohem_focal_bce_loss(
        logits_c, target_c, 
        ohem_ratio=0.25, 
        alpha=0.25 * class_weights[c],  # 每类不同 focal alpha
        gamma=2.0
    )
    losses.append(loss_c)

total_loss = sum(losses) / 3
```

### 3.4 端到端微调（End-to-End Fine-tuning）

**两阶段策略**：
1. **Stage 1（冻结 backbone）**：快速训练 ClassSpecificCDHead，找到每类最佳初始化。
2. **Stage 2（解冻 backbone 最后 2 个 STP blocks）**：
   - 以极低学习率（backbone LR = 1e-6, head LR = 1e-4）联合微调。
   - 让 backbone 的时序特征向"相邻月份变化检测"方向适应。
   - 由于 head 极轻量，主要计算开销在 backbone，单卡可承受。

---

## 四、单卡训练适配清单

| 项目 | 当前状态 | 修改方案 |
|------|---------|---------|
| DDP | `torchrun --nproc_per_node=3` | 改为单卡 `python train_single_gpu.py` |
| DistributedSampler | 使用 | 改为普通 `Shuffle=True` DataLoader |
| Batch size | 2 per GPU × 3 = 6 effective | 保持 2，gradient_accumulation=8 模拟 batch=16 |
| Teacher EMA | 无 | 在 `SingleGPUTrainer` 中实现 |
| 显存优化 | gradient_checkpointing 已开 | 保持开启，batch_size 不超过 2 |
| 学习率 | 5e-5 (batch=6) | 调整为 2e-5 (batch=2, accum=8 等效 batch=16) |

---

## 五、实施路线图（单卡 GPU6）

### Phase 0：单卡训练基础设施（1 天）
1. 新建 `scripts/train/train_single_gpu.py`，移除 DDP 依赖。
2. 调整 `DDPTrainer` 或新建 `SingleGPUTrainer`，支持 EMA Teacher。
3. 验证单卡训练能正常跑通一个 epoch。

### Phase 1：双窗口采样对齐下游（2 天）
1. 修改 `dataset.py`：
   - 添加 `adjacent_month` 双窗口模式。
   - w1/w2 以月份为锚点，间隔 1-2 个月。
2. 新建 `scripts/eval/validate_monthly.py`：
   - 使用与训练相同的相邻月份窗口评估 backbone。
   - 作为比 `validate_v2.py` 更真实的 AUC 基准。

### Phase 2：跨时相掩码重建 + EMA Teacher（3-4 天）
1. 实现 Cross-Temporal Masked Reconstruction loss。
2. 添加 EMA Teacher 和 DINO/iBOT head。
3. 单卡训练 50-100 epochs，观察 `validate_monthly.py` AUC。

### Phase 3：损失升级 → VICReg + KoLeo（2 天）
1. 用 Teacher-Student pair 实现小 batch VICReg。
2. 添加 KoLeo 正则化。
3. 训练对比：新损失组合 vs 旧四件套。

### Phase 4：类特定轻量 CD Head（2 天）
1. 实现 `ClassSpecificCDHead`。
2. Stage 1：冻结 backbone，训练 head。
3. Stage 2：解冻最后 2 个 STP blocks，端到端微调。

### Phase 5：架构深化 STP 12 blocks（2 天）
1. `num_blocks` 8 → 12。
2. Block 1-2 禁用 Space path。
3. 重新训练并评估。

---

## 六、关键论文索引（按模块）

| 模块 | 论文/方法 | 来源 |
|------|-----------|------|
| 掩码重建 | MAE: Masked Autoencoders | He et al., CVPR 2022 |
| 跨时相重建 | SSLCD: Self-Supervised Learning Pretraining for RS CD | Wan et al., IEEE TGRS 2025 |
| Teacher-Student | DINO: Emerging Properties in Self-Supervised ViTs | Caron et al., ICCV 2021 |
| Patch 预测 | iBOT: Image BERT Pre-Training with Online Tokenizer | Zhou et al., ICLR 2022 |
| 小 batch 反坍缩 | IConE: Batch Independent Collapse Prevention | arXiv 2026 |
| VICReg | Variance-Invariance-Covariance Regularization | Bardes et al., 2021 |
| KoLeo | Kozachenko-Leonenko Entropy Estimator (in DINOv2) | Sablayrolles et al., 2018 |
| 轻量下游头 | AlphaEarth Foundations (original paper) | Brown et al., 2025 |
| 相邻月份变化 | Remote Sensing Change Detection via Weak Temporal Supervision | arXiv 2026 |

---

*文档修订时间：2025-04-18*
*建议下一步：从 Phase 0（单卡训练基础设施）或 Phase 1（相邻月份双窗口采样）开始。*
