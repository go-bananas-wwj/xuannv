# AEF_qwen 变化检测能力升级计划（Embedding 模型 + CD Head 双轨升级）

> **核心目标**：让 backbone 产出的 embedding **天然具备时间敏感度**，同时让 CD Head 具备**更强的双时相特征交互能力**，最终使变化检测 AUC 突破 0.85+（主模型）/ 0.90+（分类头）。
> 
> **现状速览**：
> - `validate_v2.py`（backbone 裸测）AUC ≈ **0.49** → embedding 本身几乎无法区分前后时相变化
> - CD Head（V3+OHEM，冻结 backbone）5-Fold CV AUC ≈ **0.836** → 头部在努力学习，但受限于 backbone  embedding 质量
> - 当前 CD Head 为浅层 CNN（2 个残差块 + ECA），**无显式双时相 cross-attention**
>
> **参考来源**：AlphaEarth Foundations 原始架构、DINOv2/iBOT、BIT (IEEE TGRS)、BiFA (IEEE TGRS 2024)、MSCCANet (IEEE TGRS 2024)、ChangeFormer、CDMamba (IEEE TGRS 2025)、PT-Former (IEEE TGRS 2024)。

---

## 第一部分：Embedding 模型（Backbone）升级

### 问题 1：Embedding 对时间变化不敏感
**现象**：双窗口 embedding 的 cosine similarity 接近 0.99（dist≈0.03），时序信号被空间语义信号淹没。

### 升级路径 A：预训练任务显式面向“变化检测”

#### A1. 像素级伪变化预测（Pseudo Change Map Prediction）
**灵感**：SSLCD (TGRS 2025)、DAE-DFN (Remote Sensing 2025) 中的跨时相重建。

**方案**：
- 在预训练阶段，不只做单时相重建，而是添加一个轻量 `ChangeDecoder`。
- 输入：`|E_before - E_after|` 与 `concat(E_before, E_after)`。
- 目标：同一 patch 在两个窗口中最近帧的像素级绝对差异 `|S2_before - S2_after|`（作为伪标签）。
- Loss：L1 + SSIM（结构相似性），让模型直接学习“哪里变了”。

**为什么有效**：这与下游变化检测目标**完全一致**，且不需要人工标注。

#### A2. 跨时相掩码重建（Cross-Temporal Masked Reconstruction）
**灵感**：MAE、iBOT、RECM (TGRS 2024)。

**方案**：
- 对窗口 w2 的输入帧做**随机空间掩码**（mask 30% patch）。
- 要求模型同时利用 w2 的可见 patch 和 w1 的完整信息，重建 w2 的 masked 区域。
- 这样模型必须学会“哪些地物随时间变了、哪些没变”才能正确重建。

#### A3. 时序顺序判别（Temporal Order/Frame Interpolation）
**方案**：
- 随机打乱输入帧顺序，要求模型预测正确时间序列。
- 或从 w1 删除一帧，要求模型从 w2 插值恢复该帧的 embedding。

---

### 升级路径 B：引入 EMA Teacher + DINO/iBOT 自蒸馏

**灵感**：DINOv2 (Oquab et al., 2024)、iBOT (Zhou et al., 2022)、原版 AEF Teacher-Student。

**现状**：当前 `consistency_weight=0.0`，且无任何 EMA Teacher 机制。这是自监督训练稳定性的一大缺失。

**方案**：
1. **EMA Teacher 模型**：
   ```python
   self.teacher = copy.deepcopy(model).eval()
   # 每个 step 后用 momentum=0.996 更新
   ```
2. **DINO Head（图像级）**：
   - 在全局 embedding 上加 3 层 MLP projector → softmax。
   - Student 匹配 Teacher 的分布：`L_dino = -Σ P_teacher * log(P_student)`
   - Teacher 输出需做 centering + sharpening。
3. **iBOT Head（Patch 级）**：
   - 对 `embedding_map` 的每个空间位置加 MLP head。
   - Student 只预测被掩码的 patch，目标来自 Teacher 的对应位置。
4. **重启 Consistency Loss**：
   - 不仅是 cosine similarity，而是 DINO 分布匹配 + cosine 双管齐下。

**预期效果**：
- Teacher 提供稳定、低噪声的监督信号。
- Patch-level iBOT 强迫模型学习精细空间语义，直接利好像素级变化检测。

---

### 升级路径 C：反坍缩正则升级 → 完整 VICReg + KoLeo

**灵感**：VICReg (Bardes et al., 2021)、DINOv2 KoLeo regularizer。

**现状**：当前 `raw_uniformity + decorrelation + variance + orthogonality` 四件套虽然防止了完全坍缩，但 `decorr=29.7` 数值过大，组合不够协调。

**方案**：
1. **完整 VICReg Loss**：
   - 需要 **两个视图**（teacher vs student，或同一窗口的两个空间增强）。
   - 三项：Invariance（MSE 对齐）+ Variance（每维 std≥1）+ Covariance（off-diagonal 惩罚）。
   - 直接替换现有四件套，损失更简洁、理论更完善。
2. **KoLeo 正则化**：
   ```python
   def koleo_loss(x):
       x = F.normalize(x, p=2, dim=-1)
       dists = torch.cdist(x, x, p=2) + eye * 1e6
       nn_dists = dists.min(dim=1)[0]
       return -torch.log(nn_dists + 1e-8).mean()
   ```
   - 强制 batch 内 embedding 的最近邻距离最大化，防止聚集。
   - 对 batch size 小（如 2）的场景比 uniformity loss 更稳定。
3. **Expander/Projector**：
   - 在计算 VICReg/KoLeo 前，加一个可学习的 MLP expander（128 → 1024/4096）。
   - expander 只在训练时使用，不影响 inference embedding 维度，保持 checkpoint 兼容。

---

### 升级路径 D：架构深化与数据采样策略

#### D1. STP Blocks 增加（8 → 12/15）
**依据**：原版 AEF 有 15 个 STP blocks，当前只有 8 个。

**方案**：
- `num_blocks` 增至 12 或 15。
- **Block 1-2 禁用 Space 路径**（只做 Precision + Time），先学习低级特征（边缘、纹理），再引入空间注意力。

**风险**：参数量和显存增加。H100 80GB × 8 可支撑，但需监控。

#### D2. 强制大间隔双窗口采样
**现状**：随机裁剪窗口可能只相差几天，变化信号极弱。

**方案**：
- 修改 `dataset.py`，强制 `|center_w2 - center_w1| >= 6个月`。
- 优先选择跨越不同季节的窗口对（如春耕 vs 秋收）。
- 参考 **CACo (CVPR 2023)** 的 Change-Aware Sampling 思想。

---

## 第二部分：变化检测头（CD Head）升级

### 问题 2：当前 CD Head 缺乏双时相深度交互
**现状**：V3 头只是将 `|e1-e2|, e1*e2, e1, e2` 拼接后用 2 层 ResBlock + ECA 处理。这本质上是**早期融合（early fusion）**，没有利用 Transformer 的 cross-attention 来建模“哪里变了”的长程依赖。

**文献共识（2024-2025 TGRS）**：
- **BIT** (Chen et al., TGRS 2022)：用 Transformer encoder 对双时相 token 序列建模上下文关系。
- **BiFA** (Zhang et al., TGRS 2024)：提出 Bitemporal Feature Alignment，用隐式神经表示 + flow field 对齐双时相特征。
- **MSCCANet** (Zhang et al., TGRS 2024)：多级级联 Cross-Attention 层次网络。
- **ChangeFormer** (Bandara et al., 2022)：Siamese Transformer + MLP decoder。
- **CDMamba** (Zhang et al., TGRS 2025)：将 Mamba 的 selective scan 引入变化检测，替代 Transformer 的二次复杂度。

---

### 升级路径 E：引入 Cross-Attention / Transformer 交互模块

#### E1. BIT-Style CD Head（轻量 Cross-Attention）
**方案**：
```python
class BITCDHead(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, num_heads=4):
        # 1. 将 e1, e2 分别投影为 token 序列 [B, HW, C]
        # 2. 用 Transformer Encoder 做双向 cross-attention
        #    - 以 e1 为 query，e2 为 key/value
        #    - 以 e2 为 query，e1 为 key/value
        # 3. 交互后的特征投影回空间，预测变化图
```
**优势**：
- 直接建模双时相像素间的长程依赖。
- 抑制伪变化（pseudo-changes），增强真实变化区域响应。

#### E2. ChangeFormer-Style MLP Decoder Head
**方案**：
- 若从 backbone 提取**多级特征**（如 block 4, 8, 12 的输出），可构建层次化 CD Head。
- 每一级分别做差异图 `|feat1 - feat2|`，再上采样融合。
- 顶部用轻量 MLP decoder 输出最终变化概率。

**实施方式**：
- 修改 `AEFModel` 的 `forward`，可选返回中间层 `summary_map`（从 STP block 中间抽取）。
- CD Head 接收多尺度特征对，做级联差异融合。

#### E3. 差异增强 + 通道-空间注意力
**方案**：
- 在现有 V3 基础上，将首个 `Conv2d` 替换为 **Difference Enhancement Module**：
  - 不只拼接 `|diff|, mul, e1, e2`，而是额外计算 `channel-wise diff attention` 和 `spatial diff attention`。
- 插入 **CBAM** 或 **Coordinate Attention** 替代 ECA，同时建模通道和空间重要性。

---

### 升级路径 F：引入 Mamba / State Space Model（前沿方向）

**灵感**：CDMamba (IEEE TGRS 2025)、RS-Mamba (2024)、ConMamba (TGRS 2024)。

**为什么用 Mamba**：
- Transformer 的 self-attention 复杂度为 O(N²)，对于高分辨率遥感图（128×128=16384 tokens）计算昂贵。
- Mamba 的 selective scan 能做到 **O(N)** 复杂度，同时保留长程依赖建模能力。

**方案**：
1. **Mamba-based CD Head**：
   - 将 `embedding_map` 展平为 1D 序列 `[B, HW, D]`。
   - 通过若干层 `Mamba Block`（如 `mamba_ssm` 库的 `Mamba` 或 `vim` 实现）做双时相交互。
   - 再投影回 2D 空间预测变化图。
2. **Hybrid CNN-Mamba Head**：
   - 先用 3×3 CNN 提取局部差异特征，再用 Mamba 建模全局上下文，最后上采样输出。

**优势**：
- 参数量小、推理快、长程建模能力强。
- 尤其适合需要大感受野的变化检测任务。

---

### 升级路径 G：损失函数与训练策略优化

#### G1. 从 Binary 升级到 Multi-Class + 边界感知
**现状**：MultiClassHead 的 CV AUC 仅 0.747，不如 Binary V3 的 0.836。

**分析**：多类别标注不平衡（construction 远多于 demolition）。

**方案**：
- **类别重加权 Focal Loss**：给 demolition 和 land_conversion 更高权重。
- **边界感知 Dice Loss**：惩罚变化区域边界的模糊性。
- **Deep Supervision**：在 CD Head 的中间层也输出变化图，用辅助损失监督深层梯度。

#### G2. 联合训练 Backbone + Head（End-to-End Fine-tuning）
**现状**：CD Head 训练时 backbone 完全冻结。这意味着 backbone 不会针对变化检测任务做任何适应。

**方案**：
- **两阶段微调**：
  1. Stage 1：冻结 backbone，训练 CD Head（快速收敛）。
  2. Stage 2：**解冻 backbone 最后 2-4 个 STP blocks**，以极低学习率（如 1e-5）端到端微调整个模型。
- 这能让 backbone 的时序特征向“变化检测”方向微调，是提升 AUC 的关键一步。

#### G3.  hard negative mining + 对比学习（Head 级）
- 在 embedding 空间，将**无变化像素对**作为负样本、**变化像素对**作为正样本，计算 InfoNCE。
- 这能让 CD Head 不仅学分类，还学“变化”与“不变”的 embedding 判别边界。

---

## 第三部分：推荐实施路线图

### 第一阶段（P0，本周）：快速验证核心假设
**目标**：用最小成本验证“backbone 还是 head 是主要瓶颈”。

| 任务 | 操作 | 预期结果 |
|------|------|----------|
| **实验 1** | 强制 backbone 输出 validate_v2 用的季度窗口（6 个月间隔），重新提取 monthly embedding，再训 CD Head | 若 CD Head AUC 提升到 0.88+，说明**窗口间隔**是关键 |
| **实验 2** | 端到端微调 backbone 最后 2 个 STP blocks + CD Head V3 | 若 AUC 提升 >0.03，说明**冻结 backbone** 是瓶颈 |
| **实验 3** | 临时关闭 decorr/orth，仅保留 raw_unif + var + temporal，观察 backbone 训练 20 epoch | 若 validate_v2 AUC 仍 ~0.5，确认问题在**任务设计**而非正则冲突 |

### 第二阶段（P1，2-3 周）：Backbone 核心升级
1. **实现 EMA Teacher + DINO/iBOT**（参考 DINOv2 源码）
2. **添加 Pixel-level Pseudo Change Prediction 任务**
3. **升级损失为 VICReg + KoLeo**
4. **强制大间隔双窗口采样**

### 第三阶段（P2，1-2 周）：CD Head 升级
1. **实现 BIT-Style Cross-Attention Head**
2. **实现 ChangeFormer-Style 多尺度融合 Head**（若已提取中间特征）
3. **尝试 Mamba-based Head**（可选，视算力和库安装情况而定）
4. **端到端微调完整 pipeline**

### 第四阶段（P3，1 周）：架构深化与消融
1. **STP blocks 8 → 12/15**
2. **Block 1-2 禁用 Space path**
3. **系统消融各组件对 validate_v2 AUC 的贡献**

---

## 第四部分：关键论文与技术栈索引

| 方向 | 论文/方法 | 来源 |
|------|-----------|------|
| Teacher-Student 蒸馏 | DINOv2: Learning Robust Visual Features without Supervision | Oquab et al., 2024 |
| Patch-level 掩码预测 | iBOT: Image BERT Pre-Training with Online Tokenizer | Zhou et al., 2022 |
| 变化感知采样 | CACo: Change-Aware Sampling and Contrastive Learning | Mall et al., CVPR 2023 |
| 反坍缩正则 | VICReg: Variance-Invariance-Covariance Regularization | Bardes et al., 2021 |
| 球面分散正则 | KoLeo regularizer (in DINOv2) | Sablayrolles et al., 2018 |
| 双时相 Transformer | BIT: Bitemporal Image Transformer | Chen et al., IEEE TGRS 2022 |
| 特征对齐 | BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment | Zhang et al., IEEE TGRS 2024 |
| 多级 Cross-Attention | MSCCANet: A Multiscale Cascaded Cross-Attention Hierarchical Network | Zhang et al., IEEE TGRS 2024 |
| 纯 Transformer CD | ChangeFormer: A Transformer-Based Siamese Network for CD | Bandara et al., 2022 |
| Mamba for CD | CDMamba: Incorporating Local Clues Into Mamba for RS Image Binary CD | Zhang et al., IEEE TGRS 2025 |
| 自监督 CD 预训练 | SSLCD: A Self-Supervised Learning Pretraining Framework for RS Image CD | Wan et al., IEEE TGRS 2025 |
| 去噪 + 对比 | DAEDFN: Unified Denoising Autoencoder and Contrastive Learning Framework | Yang et al., Remote Sensing 2025 |

---

## 附录：当前 CD Head 性能基准（供对比）

| Head | 配置 | 5-Fold Mean AUC | Std |
|------|------|-----------------|-----|
| V2 | hidden=64, dropout=0.4 | 0.830 | 0.032 |
| V2 + OHEM | hidden=64, dropout=0.4 | 0.828 | 0.031 |
| **V3** | hidden=64, dropout=0.4 | **0.835** | 0.031 |
| **V3 + OHEM** | hidden=64, dropout=0.4 | **0.836** | 0.051 |
| MultiClass | hidden=64, dropout=0.4 | 0.747 | 0.063 |
| Backbone 裸测 (validate_v2) | — | **0.49** | — |

**结论**：CD Head 已经挖掘了冻结 backbone 的极限。下一步必须**升级 backbone 的时序表达能力**，同时**给 CD Head 引入 cross-attention / transformer 交互**。

---

*文档生成时间：2025-04-18*
*建议下一步：从“P0 实验 1（强制大间隔 + 端到端微调）”或“P1 EMA Teacher + VICReg”开始实施。*
