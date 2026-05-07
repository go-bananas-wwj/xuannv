# xuannv 模型升级训练计划 V7

> 生成日期: 2026-05-08
> 基准模型: V5 (aef_qwen_v5_mixed_scale, epoch_best_epoch161.pt)
> 目标: 解决 embedding 时间敏感度不足，建立对齐 AEF 论文的评估体系，系统性提升变化检测与下游任务性能

---

## 一、执行摘要

当前最佳资产为 **V5 模型**（CD Head AUC 0.9555, Bare AUC 0.486）。V6/V6.5 尝试通过增强 temporal loss 提升 backbone 时间敏感度，但均告失败（uniformity 崩溃、bare AUC 不升）。

本计划的核心思想是：
1. **停止在 V5 backbone 上继续训练** — V6 已证明 soft-restart + 新 temporal loss 会导致表征退化
2. **建立对齐 AEF 论文的标准评估体系** — 用 kNN/Linear Probe + Balanced Accuracy + Bootstrap 置信区间取代单一的 AUC
3. **双轨升级**: 
   - **Track A (数据+评估)**: 优化数据采样策略，建立严谨的 few-shot 评估基准
   - **Track B (模型)**: 从头训练 V7 backbone（不继承 V5 encoder），引入 EMA Teacher + VICReg + 强制大间隔双窗口
   - **Track C (CD Head)**: 在 V5 embedding 上快速验证新型 CD Head 架构，为 V7 做准备

---

## 二、当前状态诊断

### 2.1 数据准备现状

| 项目 | 状态 | 问题 |
|------|------|------|
| S2 数据 | 云筛选后 ~22 帧/patch | 357 个冬季月份 fallback（7.4%），质量差 |
| S1 数据 | ~42 帧/patch | 无缺失，但时间分辨率低（12天重访） |
| Landsat | ~47 帧/patch，17 个 patch 缺失 | 缺失 patch 代码已优雅处理 |
| 静态目标 | DEM/WorldCover/DynamicWorld/JRC Water | Symlink 正常，统计量已计算 |
| 训练标注 | 105 个建筑变化 polygon（哈尔滨） | 仅用于 CD Head 训练，无分类标注 |
| 缓存 | 27.7GB，已生成 | 首次加载 4.2 分钟，缓存加载 125 秒 |

**数据层面的核心问题**:
1. **双窗口采样随机性过强**: V5 的 `mixed_scale` 模式有 50% 概率采样短间隔（1-3月），变化信号极弱
2. **冬季数据质量差**: 357 个 fallback 月份集中在 2024-01、2024-02、2025-01，模型被迫学习低质量输入
3. **无真实分类标注用于 backbone 训练**: WorldCover 仅作为重建目标，分类头 `classification_weight=0.03` 几乎无监督信号

### 2.2 模型与训练现状

| 版本 | Uniformity | Bare AUC | CD Head AUC | 状态 |
|------|-----------|----------|-------------|------|
| V1 (基线) | ~-0.5 | ~0.49 | 未训练 | 已废弃 |
| V4 Official | -3.04 | 0.586 | 0.903 | 可用但非最佳 |
| **V5 (最佳)** | **-3.1** | **0.486** | **0.9555** | **生产基线** |
| V6 | -4.24→-2.72 (崩溃) | 0.489 | 未测试 | 已废弃 |
| V6.5 | -4.17→-3.28 (缓慢下降) | 0.479 | 未测试 | 已废弃 |

**关键结论**:
- **Uniformity 提升不带来 Bare AUC 改善**: 从 -3.1 到 -4.24，bare AUC 纹丝不动
- **Temporal Loss 与 Uniformity 天然互斥**: V6 的 naive cosine 3 个 epoch 即崩溃，V6.5 的 gap-aware 只能延缓
- **Soft Restart 是陷阱**: 保留 V5 encoder 权重使新 loss 无法重塑表征空间
- **CD Head 是变化检测的唯一有效路径**: V5 backbone + V3 CD Head = 0.9555 AUC，bare 方式永远无法突破 0.5

### 2.3 评估体系现状（与 AEF 论文对比）

| 维度 | AEF 论文标准 | 我们当前 | 差距 |
|------|-------------|---------|------|
| 评估协议 | 1-shot / 10-shot / max-trial | 仅 full training (CD Head) | 缺少低样本评估 |
| 分类器 | kNN (k=1,3) + Linear Probe | 只有 CNN Head / sklearn LR | 未评估 Linear Probe |
| 分类指标 | Balanced Accuracy (BA) | AUC (变化检测) / OA (分类) | 缺少 BA |
| 置信区间 | 1σ (68.27%), bootstrapping + k-folds | 无 | 统计严谨性不足 |
| 空间采样 | 最小 1.28km 间距 | 无控制 | 样本不独立 |
| 变化检测 | 无监督(BA=71.3%) + 监督(BA=78.4%) | AUC=0.9555 (CD Head) | 指标不对齐，无法横向对比 |
| 回归任务 | R², MAE | 无 | 未验证生物物理变量估计能力 |

---

## 三、历史失败根因分析（V5→V6→V6.5）

### 根因 1: 优化目标与评估目标错位

我们优化的指标是 `raw_uniformity`（目标 -4.0），但验证指标是 `bare AUC`（变化检测）。
- Uniformity 描述的是**空间分布**的分散程度
- Bare AUC 描述的是**时间变化**的区分能力
- 两者没有必然的单调关系 — V5/V6/V6.5 反复证明了这一点

**AEF 论文的启示**: AEF 的 batch uniformity objective 解决的是**空间分布均匀性**（防止坍缩），不是**时间变化区分度**。变化检测能力是重建任务 + 多源融合 + 大模型容量的自然涌现，而非单独优化 temporal loss 的结果。

### 根因 2: Temporal Loss 设计困境

| Loss 类型 | 问题 |
|-----------|------|
| Hinge (V5 temporal_magnitude) | 几乎无梯度（loss≈0），因为双窗口 cosine dist 始终 < threshold |
| Naive Cosine (V6 tc_pixel) | 强制 cos_sim→-1，无视时间 gap 大小，破坏空间语义一致性 |
| Gap-aware MSE (V6.5) | 更好但仍与 uniformity 冲突：uniformity 要求分散，temporal 要求某些对相似 |

**根本矛盾**: uniformity 要求所有 embedding 在球面上均匀分散；temporal loss 要求"相近时间的 embedding 相似"。这两个目标在高维空间中是可以共存的（时间上相近的位于同一子流形，不同子流形间均匀分散），但**当前损失函数没有显式建模这种流形结构**。

### 根因 3: Soft Restart 的 Encoder 偏见

V6/V6.5 均从 V5 checkpoint 软重启（保留 encoder，重置 bottleneck/decoder/head）。
- V5 encoder 经过 161 epochs 训练，embedding 结构根深蒂固
- 新的 temporal loss 试图改变这个结构，但 encoder 权重被保留
- 结果: temporal loss 与 encoder 结构冲突 → uniformity 下降，而非时间敏感度提升

### 根因 4: 评估指标单一化

过度依赖 `bare AUC` 一个指标判断 backbone 质量，忽略了：
- 语义分类能力（WorldCover BA）
- 低样本迁移能力（1-shot/10-shot）
- 重建质量（PSNR）
- Embedding 空间结构（均匀性 + 语义聚类质量）

---

## 四、参考 AEF 论文的评估体系设计

### 4.1 核心原则（对齐论文 Section 4 + 补充材料 S6/S18）

AEF 论文的评估设计目标：**模拟真实数据稀缺场景下的低样本迁移性能**。所有评估遵循以下原则：

1. **像素级标注**: 所有评估数据集提供像素级标签（非图像级）
2. **样本独立**: 训练/测试样本最小间距 1.28km，防止空间自相关
3. **低样本协议**: 必须测试 1-shot、10-shot、max-trial 三种样本量
4. **轻量迁移**: 只允许 kNN 和 Linear Probe（禁止端到端微调 backbone）
5. **统计严谨**: 报告 1σ 置信区间，通过 bootstrapping + k-folds 计算

### 4.2 我们可立即实施的评估体系

基于现有数据（哈尔滨 424 patches + 105 变化标注 + WorldCover/DynamicWorld/JRC Water），设计以下评估协议：

#### 协议 A: 变化检测评估（对齐 AEF Change Detection）

**直接监督（Supervised）— 对应 AEF "direct classification of change"**:
- 方法: 在 before/after embedding 对上训练 Linear Probe 或轻量 CNN Head
- 训练样本: 从 105 个标注中按 1/10/max 采样
- 指标: Balanced Accuracy (BA) — 而非 AUC
  - 原因: AEF 论文使用 BA，因为变化检测是类别极度不平衡的二分类（~1.8% positive），BA 比 AUC 更能反映模型对少数类的识别能力
- 置信区间: 5-fold cross-validation，报告 mean ± std

**无监督阈值化（Unsupervised）— 对应 AEF "unsupervised change detection"**:
- 方法: 计算 before/after embedding 的 cosine distance，用 Otsu/最佳阈值分割
- 指标: Balanced Accuracy (BA)
- 目的: 直接对比 AEF 的 71.3% BA

#### 协议 B: 地物分类评估（对齐 AEF Thematic Mapping）

**数据源**: WorldCover (11类) / DynamicWorld (9类) / JRC Water (2类)

**Few-shot 协议**:
1. 从每个类别中随机抽取 K 个像素作为训练样本（K = 1, 10, 100, max）
2. 用 **Linear Probe** 和 **kNN (k=1, 3)** 训练分类器
3. 在剩余像素上测试
4. 每个 K 值重复 5 次，报告 **Balanced Accuracy (BA)** 和 **Overall Accuracy (OA)**
5. 空间约束: 训练/测试像素最小间距 ≥ 32 像素（对应 320m，近似 1.28km 的局部版本）

**预期对标**:
- AEF LCMAP land cover (max-trial): BA = 78.4% (linear), 我们目标 > 60% (max-trial)
- AEF 10-shot: ~70% BA，我们目标 > 50% (10-shot)

#### 协议 C: Embedding 空间质量评估

| 指标 | 计算方法 | 正常范围 | 异常信号 |
|------|---------|----------|---------|
| Uniformity | `raw_uniformity_loss` | -4.0 ~ -1.0 | > -0.5 坍缩 |
| Silhouette Score | 用 WorldCover 标签计算 embedding 聚类质量 | > 0.3 | < 0.1 语义混乱 |
| NN Retrieval Accuracy | 对每个像素，检索最近邻是否同类 | > 60% | < 40% 局部结构差 |
| Temporal Consistency | 同一地点不同时间 embedding 的方差 | 适中 | 过大=不稳定，过小=无时间变化 |

### 4.3 评估工具实现计划

需要新建的脚本:
1. `scripts/eval/evaluate_fewshot_classification.py` — Few-shot BA 评估
2. `scripts/eval/evaluate_unsupervised_cd_ba.py` — 无监督变化检测 BA（替代 AUC）
3. `scripts/eval/evaluate_embedding_space.py` — Silhouette + NN Retrieval
4. `scripts/eval/evaluate_linear_probe.py` — Linear Probe 在所有下游任务上的统一评估

---

## 五、分阶段升级方案

### Phase 0: 评估体系重构（1-2 天，零风险）

**目标**: 建立严谨的评估基准，为后续所有改进提供可量化的对比依据。

| 任务 | 详情 | 产出 |
|------|------|------|
| P0-1 | 实现 `evaluate_fewshot_classification.py` | WorldCover/DynamicWorld/JRC 的 1/10/100/max-shot BA |
| P0-2 | 实现 `evaluate_unsupervised_cd_ba.py` | V5 backbone 无监督 CD BA（对比 AEF 71.3%） |
| P0-3 | 实现 `evaluate_embedding_space.py` | Silhouette Score, NN Retrieval Accuracy, t-SNE 可视化 |
| P0-4 | 在 V5 checkpoint 上跑完全部评估，建立基线报告 | `reports/v5_baseline_eval.json` |

**通过标准**: 
- 无监督 CD BA > 55%（V5 当前未知，若 < 50% 则确认 backbone 时间敏感度确实不足）
- WorldCover max-trial BA > 55%（Phase 3 下游头目标是 > 65%）
- Silhouette Score > 0.2

### Phase 1: 数据工程优化（2-3 天，低风险）

**目标**: 在不修改模型的前提下，通过数据采样策略提升训练信号质量。

#### P1-1: 强制大间隔双窗口采样（高优先级）

**问题**: V5 的 `mixed_scale` 允许 50% 概率采样短间隔（1-3月），导致大量训练对的变化信号极弱。

**方案**:
- 修改 `src/data/dataset.py`，在 `sample_dual_window` 中强制约束：
  ```python
  min_gap_ms = 6 * 30 * 24 * 3600 * 1000  # 6个月
  ```
- 同时确保 w1 和 w2 不重叠（`non_overlap=True`）
- 保留混合尺度概念，但将"短间隔"定义为 3-6 个月，"长间隔"定义为 6-18 个月
- 参考 **CACo (CVPR 2023)** 的 Change-Aware Sampling

**验证**: 训练 20 epochs，观察 temporal_magnitude_loss 是否不再≈0（因为有真正的变化信号了）。

#### P1-2: 冬季数据降级处理

**问题**: 357 个 fallback 月份（主要是 2025-01 的 272 个 patch）云量高、质量差。

**方案**:
- 在 `dataset.py` 的帧采样中，为 fallback 帧添加质量权重：
  ```python
  if is_fallback_frame:
      weight *= 0.5  # 降低 fallback 帧在重建损失中的权重
  ```
- 或在云筛选阶段，将冬季 fallback 帧直接排除（若 patch 冬季无清晰帧，则该月份不采样）

**验证**: 观察重建损失 `recon` 是否下降（因为去除了噪声大的目标）。

#### P1-3: 引入合成变化增强（可选）

**方案**: 在数据预处理阶段，对少量 S2 帧注入合成变化（如随机建筑掩码、植被替换），创造"已知变化"的训练对。

**风险**: 合成变化与真实变化的分布可能不一致，导致过拟合。建议仅作为辅助任务，权重 < 0.1。

### Phase 2: V7 Backbone 从头训练（核心升级，5-7 天，高风险高回报）

**核心决策**: 放弃 V5 encoder，**完全从头训练**新模型。理由:
1. V6 证明 soft-restart 无法摆脱 V5 encoder 的结构偏见
2. V5 的 uniformity -3.1 已足够好（CD Head 0.9555 证明下游可扩展性佳）
3. 但若想突破 bare AUC 0.5 的天花板，必须允许 encoder 从头学习新的时间敏感度

#### P2-1: 损失函数重构 — VICReg + KoLeo 替代反坍缩四件套

**现状问题**: `raw_uniformity + decorrelation + variance + orthogonality` 四件套虽然防止坍缩，但:
- `decorr=29.7` 数值过大，不稳定
- 四者之间没有明确的协同关系
- 缺少对 batch 内最近邻距离的直接约束

**方案**（参考 DINOv2 + VICReg 论文）:

```python
def vicreg_loss(z1, z2):
    """VICReg: 需要两个视图 (teacher vs student, 或同一窗口的两个增强)."""
    # 1. Invariance (对齐)
    inv = F.mse_loss(z1, z2)
    
    # 2. Variance (每维标准差 ≥ 1)
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
    
    return inv + lambda_var * var + lambda_cov * cov_loss

def koleo_loss(x):
    """KoLeo: 强制最近邻距离最大化."""
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    dists = dists + torch.eye(dists.shape[0], device=dists.device) * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
```

**关键修改**:
- VICReg 需要 **两个视图** 的 embedding。自然来源:
  - Teacher vs Student 的全局 embedding（推荐，与 EMA Teacher 联动）
  - 或同一窗口的两个不同空间增强
- 在 VICReg 计算前，添加 **可学习的 Projector/Expander**（128 → 1024/4096）， projector 只在训练时使用
- 用 `vicreg_loss` 统一替换 `raw_uniformity + decorrelation + variance`
- 用 `koleo_loss` 替换 `orthogonality_loss`（或对 pre_norm_embedding 额外加 KoLeo）

#### P2-2: 引入 EMA Teacher + DINO 自蒸馏

**现状问题**: V5 的 `consistency_weight=0.05` 是简单的 cosine similarity，没有 centering/sharpening，也不是真正的 DINO。

**方案**（参考 DINOv2）:

1. **EMA Teacher 模型**:
   ```python
   self.teacher_model = copy.deepcopy(model).eval()
   for p in self.teacher_model.parameters():
       p.requires_grad = False
   
   @torch.no_grad()
   def update_teacher(self, momentum=0.996):
       for param_t, param_s in zip(self.teacher_model.parameters(), self.model.parameters()):
           param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
   ```

2. **DINO Head（图像级）**:
   - 对全局 embedding 添加 3层 MLP projector + softmax
   - Teacher 输出需做 centering + sharpening:
     ```python
     center = momentum * center + (1-momentum) * batch_mean
     P_teacher = softmax((logits_teacher - center) / tau)
     ```
   - Loss: `L_dino = -Σ P_teacher * log(P_student)`

3. **Consistency Loss 升级**:
   - 保留 cosine consistency，但作用于 teacher-student 的 **pre_norm embedding**
   - 或直接用 DINO 分布匹配替代 cosine consistency

**预期效果**:
- Teacher 提供稳定、低噪声的监督信号
- 对于小 batch size（=2），DINO 的 centering 比纯 batch-based uniformity 更稳定
- 为后续 iBOT patch-level loss 打下基础

#### P2-3: 像素级伪变化预测任务（Change Decoder）

**动机**: 当前所有损失（reconstruction, uniformity, temporal）都没有直接监督模型学习"哪些像素随时间变了"。AEF 论文通过大规模重建任务隐式学习这一点，但我们数据量小（424 patches），需要更 explicit 的信号。

**方案**:

```python
class PseudoChangeDecoder(nn.Module):
    """轻量 decoder，从双窗口 embedding 预测像素级变化幅度."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )
    
    def forward(self, e1, e2):
        # e1, e2: [B, D, H, W]
        x = torch.cat([e1, e2], dim=1)
        return self.conv(x)  # [B, 1, H, W]
```

**伪标签生成**:
- 对连续目标（S2 6通道），计算 w1 和 w2 各自最近帧的像素级绝对差异: `|I_w1 - I_w2|`
- 对差异图做高斯平滑，作为伪变化幅度标签
- Loss: `L1(change_pred, pseudo_change_map)`

**为什么有效**:
- 直接监督模型学习"哪里变了"，与下游变化检测目标一致
- 不需要人工标注
- 权重建议: `change_prediction_weight = 0.3`

#### P2-4: Temporal Loss 重新设计 — 流形感知对比学习

**核心洞察**: uniformity 和 temporal loss 不必互斥。关键在于:
- **同一时间地点不同窗口**的 embedding 应该位于同一"语义流形"上（距离小）
- **不同地点**的 embedding 应该在球面上均匀分散
- **同一时间地点但相距很远窗口**的 embedding 应该在该流形上相距较远

**方案**: 放弃 naive cosine / gap-aware MSE，改用 **流形对比学习 (Manifold Contrastive Learning)**:

```python
def manifold_temporal_loss(e_w1, e_w2, pre_norm_w1, pre_norm_w2, gap_ms):
    """
    e_w1, e_w2: L2-normalized embedding [B, D]
    pre_norm_w1, pre_norm_w2: 原始幅度 embedding [B, D]
    gap_ms: 时间间隔（毫秒）
    """
    # 1. 语义一致性: 同一地点不同窗口的 L2 embedding 不应太远
    # 但允许有差异（变化存在），所以用 upper bound 而非 lower bound
    cos_sim = (e_w1 * e_w2).sum(dim=1)  # [-1, 1]
    
    # 目标: gap 小 → cos_sim 高（不变），gap 大 → cos_sim 低（可能变）
    # 但这不是强约束，因为"变化"不是必然的
    # 所以改用"对比"思想：
    
    # 2. 在 pre_norm 空间，要求同一地点的 w1/w2 的**相对幅度**反映变化强度
    # 用 pre_norm 的差分幅度作为变化的隐式信号
    diff_norm = torch.norm(pre_norm_w1 - pre_norm_w2, dim=1)
    
    # 3. 对比: 同一地点的 (w1, w2) 应该比不同地点的 (w1, w2') 更"相关"
    # InfoNCE: 正样本 = 同一地点的 w1-w2，负样本 = batch 内其他地点的 w1-w2
    # 但这不是标准的 instance discrimination，而是"地点-时间"联合判别
    
    # 简化实现：使用 pre_norm 差分的对比
    # 正样本对: (pre_w1[i], pre_w2[i]) for i in batch
    # 负样本对: (pre_w1[i], pre_w2[j]) for j != i
    logits_pos = -diff_norm  # [B]
    logits_neg = torch.cdist(pre_norm_w1, pre_norm_w2, p=2).diag()  # 不对，需要重新设计
    
    # 更简单的方案：
    # 只保留"同一地点双窗口 embedding 的 cosine similarity 不应完美为 1"的 soft 约束
    # 即：惩罚 cos_sim = 1（完全相同），但允许 cos_sim = 0.5~0.9
    loss = F.relu(cos_sim - 0.9).mean()  # 只有当 cos_sim > 0.9 才惩罚
    
    return loss
```

**更务实的方案**: 鉴于 V6 反复失败，**P2 阶段暂时不引入新的 temporal loss**。改为:
1. 依赖 **PseudoChangeDecoder** 提供时间敏感度信号
2. 依赖 **强制大间隔双窗口**（Phase 1）确保变化信号强度
3. 依赖 **VICReg + DINO** 提供稳定的反坍缩

#### P2-5: 架构微调 — STP Blocks 8→10，前2 block 禁用 Space

**现状**: 8 个 blocks 全部启用 Space+Time+Precision。

**AEF 论文描述**: 原版 AEF 有 15 个 STP blocks，Block 1-2 只有 Precision (+ Time)，Block 3+ 才有 Space。

**方案**:
- 将 `num_blocks` 从 8 增加到 **10**（12/15 风险高，先尝试 10）
- Block 1-2 禁用 Space 路径（`use_space=False`），只做 Precision + Time
- 这需要在 `src/models/blocks.py` 的 `STPBlock` 中添加条件控制

**预期效果**: 先学习低级特征（边缘、纹理、时序动态），再学习空间上下文，与标准 CNN 设计原则一致。

#### P2-6: V7 训练配置

```yaml
# configs/qwen_v7_from_scratch.yaml
experiment:
  name: aef_qwen_v7_from_scratch
  
data:
  window_mode: "non_overlap"  # 强制不重叠
  non_overlap_min_gap_ms: 15552000000  # 6个月
  
model:
  num_blocks: 10
  embedding_dim: 128
  
training:
  epochs: 400
  lr: 0.0001
  
  # VICReg 替代反坍缩四件套
  vicreg_weight: 1.0        # 统一替换 raw_unif + decorr + var
  vicreg_lambda_var: 1.0
  vicreg_lambda_cov: 0.04   # 1/25
  koleo_weight: 0.1         # 替代 orth
  
  # DINO
  dino_weight: 0.1
  dino_temperature: 0.07
  teacher_momentum: 0.996
  
  # Pseudo Change Decoder
  change_prediction_weight: 0.3
  
  # 保留 reconstruction
  reconstruction_weight: 1.0
  
  # 降低/移除冲突项
  temporal_magnitude_weight: 0.0   # 移除 hinge loss
  temporal_cosine_pixel_weight: 0.0 # 移除 naive cosine
  pixel_temporal_info_nce_weight: 0.0 # 移除 pixel InfoNCE
  uniformity_weight: 0.0           # 由 VICReg 替代
  decorrelation_weight: 0.0        # 由 VICReg 替代
  variance_weight: 0.0             # 由 VICReg 替代
  orthogonality_weight: 0.0        # 由 KoLeo 替代
```

### Phase 3: CD Head 与下游任务升级（3-4 天，与 Phase 2 并行）

#### P3-1: BIT-Style Cross-Attention CD Head

**现状**: V3 CD Head 是早期融合（concat + ResBlock），无显式双时相交互。

**方案**: 引入轻量 Cross-Attention:

```python
class BITCDHead(nn.Module):
    """轻量 BIT-style 变化检测头."""
    def __init__(self, embedding_dim=128, hidden_dim=64, num_heads=4):
        super().__init__()
        # 将 e1, e2 投影到 token 序列 [B, HW, C]
        self.proj = nn.Conv2d(embedding_dim, hidden_dim, 1)
        
        # 双向 Cross-Attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 差异融合 + 预测
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
    
    def forward(self, e1, e2):
        # e1, e2: [B, D, H, W]
        B, D, H, W = e1.shape
        
        x1 = self.proj(e1).flatten(2).permute(0, 2, 1)  # [B, HW, C]
        x2 = self.proj(e2).flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # 双向 cross-attention
        y1, _ = self.cross_attn(x1, x2, x2)  # e1 作为 query，e2 作为 kv
        y2, _ = self.cross_attn(x2, x1, x1)  # e2 作为 query，e1 作为 kv
        
        y1 = y1.permute(0, 2, 1).reshape(B, -1, H, W)
        y2 = y2.permute(0, 2, 1).reshape(B, -1, H, W)
        
        diff = torch.cat([y1, y2], dim=1)
        return self.decoder(diff)
```

**训练**: 冻结 V5 backbone，只训练 BITCDHead。对比 V3 CD Head 的 AUC。

#### P3-2: 多尺度 CD Head（ChangeFormer-Style）

若 Phase 2 中修改了 `AEFModel` 以返回中间层特征（block 4, 8 的 summary_map），则 CD Head 可接收多尺度特征对:
- 每级分别计算差异图 `|feat1 - feat2|`
- 上采样融合
- 顶部用轻量 MLP decoder 输出

#### P3-3: 端到端微调验证

在 BITCDHead 收敛后，尝试**解冻 backbone 最后 2 个 STP blocks**，以极低学习率（1e-5）端到端微调:
- 若 AUC 提升 > 0.02，说明 backbone 有微调空间
- 若 AUC 不提升或下降，立即停止（防止破坏预训练知识）

### Phase 4: 系统评估与消融（2-3 天）

| 实验 | 内容 | 目的 |
|------|------|------|
| 消融 A | V7 vs V5 backbone + 相同 CD Head | V7 backbone 是否优于 V5 |
| 消融 B | V7 with/without PseudoChangeDecoder | 伪变化任务的价值 |
| 消融 C | V7 with/without DINO Teacher | EMA Teacher 的价值 |
| 消融 D | V7 10 blocks vs 8 blocks | 架构深度的影响 |
| 消融 E | V7 embedding_dim 128 vs 64 | 维度对下游任务的影响 |

---

## 六、验证指标与通过标准

### 6.1 Phase 0（评估体系）通过标准

| 指标 | V5 基线 | 通过标准 |
|------|---------|---------|
| 无监督 CD BA | 未知（首次测量） | 完成测量即可 |
| WorldCover max BA | 未知 | > 55% |
| WorldCover 10-shot BA | 未知 | > 40% |
| Silhouette Score | 未知 | > 0.2 |

### 6.2 Phase 1（数据优化）通过标准

| 指标 | 当前 | 目标 |
|------|------|------|
| temporal_magnitude_loss | ~0（无梯度） | < -0.5（有梯度） |
| recon_loss | ~0.25 | < 0.20（冬季数据降级后） |

### 6.3 Phase 2（V7 Backbone）通过标准

| 指标 | V5 | V7 目标 | 说明 |
|------|-----|---------|------|
| raw_uniformity / vicreg | -3.1 | -3.5 ~ -4.0 | 不坍缩 |
| recon_loss | ~0.25 | < 0.25 | 重建质量不下降 |
| 无监督 CD BA | V5 基线 | **> V5 + 5pp** | 核心目标 |
| WorldCover max BA | V5 基线 | **> V5 + 5pp** | 语义表征提升 |

**停止条件**: 若训练 50 epochs 后 uniformity > -0.5 且持续不下降，立即报告并终止。

### 6.4 Phase 3（CD Head）通过标准

| Head | V3 基线 | 目标 |
|------|---------|------|
| BITCDHead (frozen backbone) | 0.9555 | > 0.9555 或验证更快收敛 |
| BITCDHead + end2end finetune | — | > 0.96 |

---

## 七、资源需求与风险

### 7.1 计算资源

| 阶段 | GPU 需求 | 时间 | 备注 |
|------|----------|------|------|
| Phase 0 | 1× NPU | 1-2 天 | 评估脚本，可单卡运行 |
| Phase 1 | 2-3× NPU | 2-3 天 | 数据修改 + 短训验证 |
| Phase 2 | 3× NPU DDP | 5-7 天 | V7 从头训练 400 epochs |
| Phase 3 | 1× NPU | 2-3 天 | CD Head 训练 |
| Phase 4 | 1-3× NPU | 2-3 天 | 消融实验 |

**总预算**: 约 2-3 周，8 张 NPU 空闲时可并行推进多个实验。

### 7.2 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| V7 从头训练 uniformity 仍不改善 | 中 | 高 | Phase 2 中保留 V5 作为 fallback，同时继续优化 CD Head |
| DINO Teacher 实现复杂、引入 NaN | 中 | 中 | 先简化实现（无 centering/sharpening），验证稳定后再加 |
| BITCDHead 参数量大，过拟合 | 低 | 中 | hidden_dim 从 64 开始，逐步增加 |
| 数据修改引入 bug | 低 | 高 | 修改后先用 `test_v6_launch.py` 模式验证 5 个 step |

### 7.3 需要用户确认/提供的资源

1. **标注数据扩展**: 当前 105 个 polygon 仅覆盖建筑变化。若用户有更多变化类型标注（如道路、植被），可显著增强 CD Head 和监督评估的可靠性。
2. **其他区域数据**: V7 从头训练需要更多数据。若用户有其他城市/区域的 patch 数据，可合并训练以提升泛化性。
3. **AEF 官方 embedding**: 若用户有权限获取哈尔滨区域的 AEF 官方 embedding，可用于直接对比验证（Level 2 评估）。

---

## 八、实施路线图（甘特图式）

```
Week 1
├── Day 1-2: Phase 0 — 实现评估脚本（fewshot BA / unsupervised BA / Silhouette）
├── Day 3-4: Phase 0 — V5 基线评估，生成报告
└── Day 5-7: Phase 1 — 数据采样修改（强制6月间隔 + 冬季降级）

Week 2
├── Day 1-2: Phase 2 — 实现 VICReg + KoLeo + DINO Teacher
├── Day 3-4: Phase 2 — 实现 PseudoChangeDecoder + 架构修改（10 blocks）
└── Day 5-7: Phase 2 — V7 从头训练启动（3 NPU DDP）

Week 3
├── Day 1-3: Phase 2 — V7 训练监控（目标 epoch 100-150 出最佳）
├── Day 4-5: Phase 3 — BITCDHead 实现 + 在 V5 embedding 上验证
└── Day 6-7: Phase 3 — 端到端微调实验

Week 4
├── Day 1-3: Phase 4 — 系统消融实验
└── Day 4-7: 报告撰写 + checkpoint 归档
```

---

## 九、下一步行动（待用户确认后立即执行）

### 选项 A: 保守路线（推荐）
- 只做 **Phase 0 + Phase 1 + Phase 3**
- 保留 V5 backbone，优化数据采样 + 升级 CD Head
- 风险最低，预期收益: CD Head AUC 0.9555 → ~0.97，bare BA 提升 5-10pp

### 选项 B: 激进路线
- 全量执行 **Phase 0-4**
- 从头训练 V7 backbone + 全新损失体系
- 风险高（可能不如 V5），但潜在收益大（bare BA 可能突破 60%）

### 选项 C: 最小可行路线
- 只做 **Phase 0**
- 建立评估基准，明确当前模型的真实能力边界
- 后续决策基于评估结果再做

---

## 附录: AEF 论文关键数据备忘

| 项目 | AEF 官方 | 我们当前 | 差距 |
|------|---------|---------|------|
| 训练数据 | 30亿观测，1.1%陆地 | 424 patches，哈尔滨 | 数据量差距极大 |
| 模型参数量 | 480M | ~57M (V5) | 8.4x |
| Embedding 维度 | 64 | 128 | 我们的更高 |
| STP Blocks | 15 | 8 | 我们的更少 |
| 无监督 CD BA | 71.3% | 未知 (~50%?) | 待验证 |
| 监督 CD BA | 78.4% | AUC=0.9555 (无法直接转 BA) | 指标不对齐 |
| 10-shot LC BA | ~78% | 未测试 | 待验证 |
| Max-trial LC BA | ~78% | 未测试 | 待验证 |

**关键认知**: 由于数据量和模型规模的根本差距，我们的模型在 bare backbone 性能上不应期望超越 AEF。我们的**差异化价值**在于：
1. **月度/季度 embedding**（AEF 只有年度）
2. **针对特定区域（哈尔滨）的优化**
3. **更强的 CD Head 适配能力**（V5 的 0.9555 已证明）

因此，升级目标不应是"超越 AEF"，而是"在有限数据和算力下，最大化 backbone 的时间敏感度与下游任务性能"。
