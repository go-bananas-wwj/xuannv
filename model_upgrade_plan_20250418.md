# xuannv_embdding 模型升级计划（V4）

> 目标：解决嵌入时间敏感度不足（AUC≈0.5）与潜在的空间-时序表示分离问题，使模型具备真正的变化检测能力。
> 
> 参考：AlphaEarth Foundations 原始设计、DINOv2 / iBOT、CACo (CVPR 2023)、VICReg、KoLeo 正则化。

---

## 一、当前问题诊断（基于 v2 epoch_499 结果）

| 指标 | 数值 | 诊断 |
|------|------|------|
| `validate_v2.py` AUC | ~0.49 | 接近随机，模型对时间变化不敏感 |
| `dist_v1 / dist_v2` | ~0.0299 | 双窗口 embedding 差异极小，时间信号被淹没 |
| `decorr` | 29.7 | Barlow Twins 去相关项数值过大，可能不稳定 |
| `temporal` | -1.96 | 时序对比损失有梯度，但强度不足以改变表征空间结构 |
| `consist` | 0.0 | 教师-学生一致性损失已关闭，缺失原版核心机制 |
| `recon` | 2.81 | 重建任务仍在学习，但与时间敏感度无明显关联 |

**核心结论**：
1. **时序信号被空间/语义信号淹没**：模型学会了“这是什么地物”，但没学会“这个地物随时间怎么变”。
2. **缺失 Patch-level 自监督目标**：仅有全局/像素级重建和全局 uniformity，没有 DINOv2/iBOT 风格的局部 patch 对比/预测。
3. **反坍缩损失组合不够协调**：raw_uniformity + decorrelation + variance + orthogonality 四件套虽然防止了完全坍缩，但没有像 VICReg 那样明确分离 invariance / variance / covariance，也没有 KoLeo 来保证球面均匀分布。
4. **Teacher-Student EMA 蒸馏缺失**：原版 AEF 和 DINO 都依赖 EMA teacher 提供稳定的目标，当前 consistency=0 意味着 student 没有从 teacher 的完整视图中学习鲁棒性。

---

## 二、升级方向总览

```
┌─────────────────────────────────────────────────────────────────┐
│  方向 A：引入 EMA Teacher + DINO 风格自蒸馏（稳定性与判别性）      │
│  方向 B：引入 iBOT 风格 Patch-level 掩码预测（空间精细度）        │
│  方向 C：升级反坍缩正则 → 完整 VICReg + KoLeo（几何结构）        │
│  方向 D：设计“变化检测预训练任务”（时间敏感度）                  │
│  方向 E：架构深化（STP blocks 从 8→12/15，更接近原版）           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、Phase 1：基线修复与诊断增强（1-2 天）

### 3.1 修复当前损失的数值不稳定
**问题**：`decorr=29.7` 说明 Barlow Twins 的互相关矩阵在当前 pre-norm 特征上量纲不对。

**行动**：
- 将 `decorrelation_loss` 改为标准 **VICReg covariance loss**：
  - 计算 batch 内零均值 embedding 的协方差矩阵 `C = Z^T Z / (N-1)`
  - 惩罚 `sum(C_off_diag^2)`（不强制对角线为 1）
  - 这样可以避免 Barlow Twins 中 `(c.diagonal() - 1)^2` 在幅度变化大时的数值爆炸。
- 或者，将 decorrelation 计算从 pre-norm 空间转移到 **L2-normalized 空间**，因为相关性对幅度不敏感。

### 3.2 增加训练监控
**行动**：
- 在 `trainer.py` 的每个 epoch 结束时，输出：
  - 双窗口 cosine similarity 的均值/标准差
  - pre-norm embedding 的范数均值
  - 各 source decoder 的重建 PSNR
- 这能帮助我们判断 temporal loss 是否真的在缩小 `cos_sim`。

### 3.3 简化当前配置做快速验证
**行动**：
- 临时关闭 `decorr` 和 `orth`，仅保留 `raw_unif + var_reg + temporal`，观察 AUC 是否变化。
- 如果 AUC 仍 ~0.5，说明问题不在反坍缩四件套的冲突，而在更深层的架构/任务设计。

---

## 四、Phase 2：引入 EMA Teacher + DINO 蒸馏（核心升级，3-5 天）

### 4.1 为什么需要 Teacher-Student？
- **原版 AEF**：用 teacher（完整输入）和 student（dropout/扰动输入）的 consistency loss 强制 embedding 对缺失数据鲁棒。
- **DINOv2**：用 EMA teacher 提供稳定、低噪声的目标，student 学习匹配 teacher 的输出分布，这是目前自监督视觉表征的最强范式之一。

### 4.2 实施方案

#### A. 添加 EMA Teacher 模型
```python
# 在 DDPTrainer.__init__ 中
self.teacher_model = copy.deepcopy(self.model).eval()
for p in self.teacher_model.parameters():
    p.requires_grad = False

# 每个 step 后更新 EMA
@torch.no_grad()
def update_teacher(self, momentum=0.996):
    for param_t, param_s in zip(self.teacher_model.parameters(), self.model.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
```

#### B. 设计 DINO Loss（图像级）
- 对 **全局 embedding**（`embedding_vector`）添加一个可学习的 `DINOHead`（3层 MLP + softmax）。
- Teacher 和 Student 分别用各自的 head。
- Loss 为 cross-entropy：
  ```
  L_dino = - Σ P_teacher * log(P_student)
  ```
- Teacher 输出需要 **centering + sharpening**：
  - `center` 用 EMA 更新：`center = momentum * center + (1-momentum) * batch_mean`
  - `P_teacher = softmax((logits_teacher - center) / tau)`

#### C. 重启 consistency loss
- 当前 consistency loss 是 `1 - cosine_similarity`。
- 升级：改为 DINO 风格的分布匹配（比单纯 cosine 更强）。
- 或者保留 cosine consistency，但让它作用于 teacher-student 的 pre-norm embedding。

### 4.3 预期效果
- Teacher 提供稳定目标，减少 student 对 batch statistics 的依赖。
- 对于小 batch size（=2），DINO 的 centering 机制比纯 batch-based uniformity 更稳定。
- 为后续 iBOT patch-level loss 打下基础。

---

## 五、Phase 3：引入 iBOT 风格 Patch-level 预测（空间精细度，3-5 天）

### 5.1 为什么需要 Patch-level？
- 当前模型输出 `embedding_map [B, D, H, W]`，但除了 pixel_temporal_info_nce（采样 16 个像素）外，**没有任何针对局部 patch 的自监督目标**。
- DINOv2 / iBOT 证明：patch-level 的掩码预测能显著提升空间语义精度，这对变化检测至关重要（我们需要像素级变化概率）。

### 5.2 实施方案

#### A. Masked Patch Modeling
1. 在输入图像帧上随机掩码一部分空间 patch（类似 MAE / iBOT）。
2. Student 接收掩码后的输入；Teacher 接收完整输入（通过 EMA）。
3. 对 `embedding_map` 上的每个空间位置，添加一个 `iBOTHead`（3层 MLP）。
4. Student 只计算被掩码位置的 iBOT loss，Teacher 提供这些位置的目标分布。

#### B. 简化版：Spatial Contrastive Learning
如果 iBOT 实现复杂，可以先做简化版：
- **Random Spatial Cropping Consistency**：对同一个时间窗口，做两个不同空间裁剪，要求它们的重叠区域 embedding 一致，非重叠区域通过上下文推断。
- 或者 **Patch Retrieval Loss**：从 `embedding_map` 中随机采样 patch，用 InfoNCE 对比同一地物不同时间的 patch vs 不同地物的 patch。

#### C. 与 Temporal 结合
- **时序 + 掩码联合任务**：
  - 窗口 w1：完整输入
  - 窗口 w2：掩码输入（随机 drop 帧或掩码空间 patch）
  - 要求 student 从 w2 的掩码输入恢复 w1 的对应 patch embedding（由 teacher 提供）
  - 这样模型必须依赖时序上下文来填补缺失信息，自然增强时间敏感性。

---

## 六、Phase 4：升级反坍缩正则 → VICReg + KoLeo（几何结构，2-3 天）

### 6.1 完整 VICReg 损失
当前代码有 VICReg 的雏形（variance + decorrelation），但缺少明确的 **invariance** 项，且 decorrelation 实现不够标准。

**升级方案**：
```python
def vicreg_loss(z1, z2):
    # z1, z2: [N, D] — 同一 patch 的两个增强视图（或 teacher/student 输出）
    
    # 1. Invariance (对齐)
    inv = F.mse_loss(z1, z2)
    
    # 2. Variance (每个维度标准差不低于 gamma)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
    
    # 3. Covariance (去相关)
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
    cov_loss = (cov_z1.pow(2).sum() - cov_z1.diagonal().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov_z2.pow(2).sum() - cov_z2.diagonal().pow(2).sum()) / z2.shape[1]
    
    return inv + 1.0 * var + (1.0/25.0) * cov_loss
```

**关键修改**：
- VICReg 需要 **两个视图** 的 embedding（z1, z2）。
- 我们的自然来源：teacher vs student 的全局 embedding，或者同一窗口的两个不同空间增强。
- 这样可以把 `raw_uniformity`、`decorr`、`variance`、`consistency` 四者统一替换为更简洁、理论上更完善的 VICReg。

### 6.2 KoLeo 正则化（DINOv2 核心技巧）
KoLeo（Kozachenko-Leonenko 熵估计）鼓励 batch 内 embedding 均匀分布。

```python
def koleo_loss(x):
    # x: [N, D], L2 normalized
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    # 排除自身
    dists = dists + torch.eye(dists.shape[0], device=dists.device) * 1e6
    # 最近邻距离的负对数平均
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
```

**作用**：
- 当 batch 内 embedding 聚集时，最近邻距离小，loss 大（惩罚聚集）。
- 当 embedding 均匀分散时，loss 小（奖励分散）。
- 这是比 uniformity loss 更直接的几何分散正则，且对 batch size 较小的场景也比较稳定（只依赖最近邻）。

### 6.3 球面约束的重新思考
当前训练时跳过 L2 norm，但 VICReg 和 KoLeo 都是在欧氏空间工作。

**建议**：
- 保留 `skip_l2_training=True`（避免梯度屏障）。
- 但在计算 VICReg / KoLeo 时，可以先用一个 **可学习的 projector/expander**（类似 VICReg 的 expander）将 pre-norm 特征映射到更高维空间（如 1024 或 4096），再计算正则。
- projector 只在训练时使用，推理时丢弃，不影响 checkpoint 兼容性。

---

## 七、Phase 5：设计变化检测预训练任务（时间敏感度，2-3 天）

### 7.1 当前 temporal loss 的问题
- `temporal_contrastive_loss`（hinge loss）和 `temporal_info_nce_loss`（anti-diagonal）都在试图让 w1 和 w2 的 embedding **不同**。
- 但问题在于：如果模型只是学会了给不同窗口加上一个全局偏移（global time bias），那么变化检测仍然无效——真正的变化应该是**空间异质**的。

### 7.2 升级方案：显式变化预测任务

#### A. Pixel-level Change Prediction（像素级变化预测）
**任务定义**：
- 输入：w1 和 w2 的 embedding map `E1, E2` [B, D, H, W]
- 目标：在预训练阶段，利用重建目标（如 S2 图像）直接生成一个**伪变化图**。
- 具体来说，对连续目标（S2/S1/Landsat），计算两个时间窗口内最近帧的像素级差异 `|I1 - I2|`。
- 添加一个轻量的 `ChangeDecoder`（2层 conv），从 `|E1 - E2|` 和 `concat(E1, E2)` 预测伪变化图。
- Loss：MSE 或 L1。

**为什么有效**：
- 这直接监督模型学习“哪些像素随时间变了”，与下游变化检测目标完全一致。
- 不需要人工标注，伪标签来自输入数据本身。

#### B. Temporal Warping / Frame Reordering
- 随机打乱输入帧的时间顺序，要求模型预测正确的时间序列（类似视频中的顺序预测）。
- 或者，从 w1 中删除一帧，要求模型从 w2 的上下文恢复该帧的 embedding（时序插值）。

#### C. 双窗口数据增强策略（参考 CACo）
- **Change-Aware Sampling**：确保 w1 和 w2 之间至少包含一个已知的季节性转折点（如春耕、收割）或至少 6 个月的时间间隔。
- 当前随机裁剪窗口可能导致 w1 和 w2 只相差几天，变化信号太弱。
- 修改 `dataset.py`：
  - 强制 w1 和 w2 的最小时间间隔 >= 3 个月（或 >= 6 个月）。
  - 优先选择包含不同季节的窗口对。

---

## 八、Phase 6：架构深化（STP Blocks 扩展，1-2 天）

### 8.1 原版 AEF 架构
根据 `ml4rs.com` 文档，原版 AEF 有 **15 个 STP blocks**。
- Block 1-2：只有 Precision 和 Precision+Time（没有 Space）
- Block 3-15：完整的 Space + Time + Precision

### 8.2 当前代码
当前所有 8 个 blocks 都是 Space+Time+Precision，且数量只有 8 个。

### 8.3 升级方案
- **将 num_blocks 从 8 增加到 12 或 15**。
- **前 2 个 block 禁用 Space 路径**，只做 Precision (+ Time)，让模型先学习低级特征（边缘、纹理），再学习空间上下文。
- 这需要在 `model.py` 的 block 初始化中做条件判断。

**注意**：增加 blocks 会显著增加计算量和内存。如果 GPU 资源有限（H100 80GB × 8），可以先实验 12 blocks。

---

## 九、推荐执行顺序与优先级

| 优先级 | 阶段 | 预期投入 | 预期收益 |
|--------|------|----------|----------|
| **P0** | Phase 4A：像素级变化预测任务 | 2-3 天 | **最直接提升 AUC**，与下游任务对齐 |
| **P0** | Phase 5：强制 w1/w2 最小 3-6 个月间隔 | 0.5 天 | 保证变化信号强度 |
| **P1** | Phase 2：EMA Teacher + DINO 蒸馏 | 3-5 天 | 大幅提升训练稳定性与表征质量 |
| **P1** | Phase 4B/C：时序插值/Frame Reordering | 2-3 天 | 增强时间推理能力 |
| **P2** | Phase 3：iBOT Patch-level | 3-5 天 | 提升空间精细度 |
| **P2** | Phase 4：VICReg + KoLeo | 2-3 天 | 更优雅地替代反坍缩四件套 |
| **P3** | Phase 6：STP blocks 12/15 | 1-2 天 | 更接近原版 AEF，但成本高 |

---

## 十、快速实验建议（本周可验证）

如果想在 **1-2 天内** 快速验证方向，建议做以下两个最小化实验：

### 实验 A：强制大间隔双窗口 + Pixel Change Prediction
1. 修改 `dataset.py`：
   - 强制 `w1` 落在 2023Q3-Q4，`w2` 落在 2024Q3-Q4（与 validate_v2 的窗口一致）。
   - 或者至少保证 `|center_w2 - center_w1| >= 6个月`。
2. 添加一个 2 层 conv 的 `PseudoChangeDecoder`，预测 `|S2_w1 - S2_w2|` 的像素级差异。
3. 训练 20-50 epochs，观察 AUC 变化。

### 实验 B：DINO Teacher + VICReg 替换现有正则
1. 添加 EMA teacher（momentum=0.996）。
2. 对 teacher/student 的全局 embedding 计算 VICReg（inv + var + cov）。
3. 暂时关闭 `raw_unif`、`decorr`、`var_reg`、`orth`、`consist`。
4. 保留 reconstruction 和 temporal。
5. 训练 20-50 epochs，观察 `pre_unif` 和 `enc_unif` 是否改善，以及 AUC 变化。

---

## 十一、参考论文与技术栈

1. **AlphaEarth Foundations** (Brown et al., 2025) — 原始架构与损失设计
2. **DINOv2** (Oquab et al., 2024) — EMA teacher、KoLeo、patch-level iBOT
3. **iBOT** (Zhou et al., 2022) — 自蒸馏掩码 patch 预测
4. **CACo** (Mall et al., CVPR 2023) — 卫星图像变化感知采样与对比学习
5. **VICReg** (Bardes et al., 2021) — 方差-不变性-协方差正则化
6. **SatMAE / Scale-MAE** — 遥感掩码自编码器
7. **SiamMAE / VideoMAE** — 时序掩码建模

---

*计划生成时间：2025-04-18*
*下一步：可选择上述任一 Phase 进入实施。*
