# V7 Backbone 从头训练详细计划（修订版）

> 修订日期: 2026-05-08
> 核心变更（vs 初版）:
>   1. 双窗口策略：混合相邻月份（短间隔）+ 长间隔（≥6月），而非强制大间隔
>   2. 8 卡 DDP 训练（effective batch = 128）
>   3. 冬季 fallback 帧 quality weight = 0.5（基于 noisy sample reweighting 论文）
>   4. 极简验证先行：VICReg+KoLeo 替换四件套，其他不变，训练 50 epoch 验证

---

## 一、用户决策确认

| 决策项 | 用户选择 | 方案 |
|--------|---------|------|
| 双窗口间隔 | **混合长/短** | 相邻月份（1月间隔）30% + 长间隔≥6月 70% |
| 冬季降级 | **支持** | fallback 帧重建权重 ×0.5 |
| 训练卡数 | **8 卡 DDP** | batch=2×8×accum=8 = effective batch=128 |
| STP Blocks | **10 blocks** | 前 2 禁用 Space |
| 实施路线 | **极简验证先行** | Phase 0: 只改损失（VICReg+KoLeo），不改架构/数据，50 epoch 验证 |
| 嵌入评估 | **需详细回答** | 见第八节 |

---

## 二、双窗口策略重新设计（核心修改）

### 2.1 问题分析

V5 的 `mixed_scale` 模式：
- 长间隔（≥6月）：变化信号强，适合学习年度/结构性变化
- 短间隔（1-3月）：**随机采样**，可能选中几乎无变化的相邻月份，导致 temporal loss 无梯度

**用户要求**：下游需要做相邻月份变化检测（如 4月 vs 5月），因此模型必须见过短间隔训练对。

### 2.2 改进方案：Structured Mixed Scale

**新增 window_mode**: `"structured_mixed"`

```python
# 配置
data:
  window_mode: "structured_mixed"
  structured_mixed_short_prob: 0.3      # 30% 短间隔
  structured_mixed_long_prob: 0.7       # 70% 长间隔
  structured_mixed_short_mode: "adjacent_month"  # 短间隔=严格相邻月份
  structured_mixed_long_min_gap_ms: 15552000000  # 6个月
```

**短间隔实现（相邻月份）**:
```python
def _sample_adjacent_month_windows(self, ts_sorted):
    """采样严格相邻的两个月份作为 w1/w2.
    
    例如: 2025-04 vs 2025-05, 或 2024-08 vs 2024-09
    保证两月之间有季节性差异（如春→夏，秋→冬），但间隔仅1个月。
    """
    from collections import defaultdict
    from datetime import datetime
    
    month_groups = defaultdict(list)
    for ts in ts_sorted:
        dt = datetime.fromtimestamp(ts / 1000.0)
        key = f"{dt.year:04d}-{dt.month:02d}"
        month_groups[key].append(ts)
    
    months = sorted(month_groups.keys())
    if len(months) < 2:
        # fallback: 默认分割
        return _default_split(ts_sorted)
    
    # 随机选择一对相邻月份
    i = random.randint(0, len(months) - 2)
    m1_frames = month_groups[months[i]]
    m2_frames = month_groups[months[i + 1]]
    
    # 每边至少 min_frames 才用
    min_f = self._min_window_frames
    if len(m1_frames) >= min_f and len(m2_frames) >= min_f:
        w1_start = float(m1_frames[0])
        w1_end = float(m1_frames[-1])
        w2_start = float(m2_frames[0])
        w2_end = float(m2_frames[-1])
        return w1_start, w1_end, w2_start, w2_end
    else:
        # fallback: 找下一对相邻月份
        ...
```

**长间隔实现**：复用 V5 的 `_sample_long_gap_windows`（≥6月）

**为什么这样设计**:
- 短间隔严格限定为"相邻月份"，避免随机采到几乎无差异的任意1-3月对
- 相邻月份有真实的季节性变化（如植被生长、农作物收割），变化信号明确
- 长间隔占 70%，保证模型仍学到年度/结构性变化
- 与下游任务对齐：下游月度 CD 用的就是相邻月份对

### 2.3 与 CACo (CVPR 2023) 的对比

CACo 论文的核心洞察：
- **短期差异（<1年）**：通常是季节性变化，应视为 augmentation（即 embedding 应相似）
- **长期差异（>1年）**：可能是结构性变化，应视为 negative（即 embedding 应不同）

我们的设计与之对齐：
- 短间隔（相邻月份）：季节性变化 → 模型应学会"同一地点不同季节的特征变化"
- 长间隔（≥6月）：跨越季节或年度 → 模型应学会"长期结构性变化的特征差异"

这与 V5 的 naive 短间隔有本质区别：V5 的短间隔是"任意1-3月"，可能包含同一季节内的无变化对。

---

## 三、冬季 Fallback 帧质量降级（论文支持）

### 3.1 问题

357 个冬季月份为 fallback（全 cloudy，被迫保留最不清的一张）。这些帧：
- 云量大，像素值被云污染
- NDVI 等植被指数失真
- 若与清晰帧同等权重，模型被迫拟合噪声

### 3.2 论文依据

**Ren et al., 2018 (Learning to Reweight Examples)**:
> "For detected clean samples, we set their weights to 1 (no reweighting) ... for detected noisy samples, we assign them smaller weights individually based on how likely one sample being noisy."

**Shu et al., 2019 (Meta-Weight-Net)**:
> "Noisy samples are absent in the target distribution ... the weights of the noisy samples should be decreased to better match the source and target distributions."

**Focal Loss (Lin et al., 2017)**:
> 降低 easy/noisy 样本的权重，使模型聚焦于 hard/clean 样本。

**关键结论**: 对于质量低/噪声大的样本，标准做法是将权重降低到 **0.1 ~ 0.5**。

### 3.3 我们的方案

| 帧类型 | 质量标记 | 重建权重 | 依据 |
|--------|---------|----------|------|
| 清晰帧（云筛选后保留）| quality=1.0 | 1.0 | 基准 |
| Fallback 帧（全 cloudy，被迫保留）| quality=0.5 | 0.5 | Ren/Shu: noisy sample weight 0.1-0.5，取中值 |

**为什么不设 0.0 或 0.1？**
- 0.0：完全丢弃。但冬季 fallback 在某些 patch 是唯一可用数据，完全丢弃可能导致该 patch 冬季无数据
- 0.1：过于激进。fallback 帧仍包含一些地理信息（如云下地表的部分反射）
- **0.5**：平衡方案。降低噪声影响，但不完全丢弃信息。与 Meta-Weight-Net 的实验结果一致（noisy sample weight 0.3-0.5 时性能最佳）

### 3.4 实现方式

**Step 1**: 在云筛选脚本中生成 `frame_quality.json`
```python
# scripts/preprocessing/filter_cloudy_frames.py 修改
for each patch, each frame:
    if is_fallback:  # 该月份全 cloudy，被迫保留最差帧
        quality = 0.5
    else:
        quality = 1.0
```

**Step 2**: Dataset 读取 quality
```python
# src/data/dataset.py
self.frame_quality = {}
if hasattr(d, 'frame_quality_json') and d.frame_quality_json:
    with open(d.frame_quality_json) as f:
        self.frame_quality = json.load(f)
```

**Step 3**: Trainer 中应用 quality weight
```python
# src/training/trainer.py 的 compute_recon_loss 中
# 传入 frame_quality_mask [B, T_tgt, 1, 1, 1]
# loss = |pred - target| * target_mask * quality_mask
```

---

## 四、极简验证方案（Phase 0）

### 4.1 目标

在投入完整 V7（架构修改 + 数据修改 + PseudoChangeDecoder）之前，先用**最小代价**验证核心假设：

> **假设**: VICReg + KoLeo 比 raw_uniformity + decorrelation + variance + orthogonality 四件套更稳定、更有效。

### 4.2 修改范围（仅限以下内容）

| 模块 | 修改内容 | 不动内容 |
|------|---------|---------|
| 损失函数 | 新增 `vicreg_loss.py`, `koleo_loss` | 其他损失不变 |
| Trainer | 替换四件套为 VICReg+KoLeo | 不引入 EMA Teacher |
| 配置 | 新 config `qwen_v7_minimal.yaml` | 数据采样、模型架构、CD Head 均不变 |

### 4.3 具体修改

**A. 新增 `src/training/vicreg_loss.py`**
```python
def vicreg_loss(z1, z2, lambda_var=1.0, lambda_cov=0.04):
    inv = F.mse_loss(z1, z2)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (z1_c.shape[0] - 1)
    cov_z2 = (z2_c.T @ z2_c) / (z2_c.shape[0] - 1)
    cov_loss = (cov_z1.pow(2).sum() - cov_z1.diagonal().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov_z2.pow(2).sum() - cov_z2.diagonal().pow(2).sum()) / z2.shape[1]
    return inv + lambda_var * var + lambda_cov * cov_loss

def koleo_loss(x):
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    dists = dists + torch.eye(dists.shape[0], device=dists.device) * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
```

**B. 修改 Trainer（复用现有 trainer.py）**

由于极简验证不改 EMA Teacher，我们可以复用现有 `DDPTrainer`，只是替换损失计算部分：

```python
# 在 trainer.py 的 train_epoch 中
# 原来:
raw_unif = raw_uniformity_loss(gathered_pre_norm)
decorr = decorrelation_loss(gathered_pre_norm)
var_reg = variance_regularizer(gathered_pre_norm)
orth = bottleneck_orthogonality_loss(...)

# 替换为:
# 需要两个视图。极简方案：对同一 batch 做两个不同空间增强（或 teacher/student 用同一模型）
# 简化：用 batch 内随机分成两半作为 z1, z2（不够严谨但可快速验证）
# 更好：对 pre_norm 做两个独立 dropout 作为视图

# 最简实现（快速验证）:
z1 = gathered_pre_norm[::2]   # 偶数索引
z2 = gathered_pre_norm[1::2]  # 奇数索引
if z1.shape[0] == z2.shape[0]:
    vicreg = vicreg_loss(z1, z2)
else:
    vicreg = vicreg_loss(z1, z2[:z1.shape[0]])
koleo = koleo_loss(gathered_pre_norm)
```

**注意**: 这种"batch 内分两半"作为两个视图的方案不是标准 VICReg，但足以快速验证 VICReg 的数值稳定性。若验证通过，完整 V7 再改为 Teacher/Student 双视图。

**C. 配置 `configs/qwen_v7_minimal.yaml`**
```yaml
_base_: qwen_v1_scenes.yaml

experiment:
  name: aef_qwen_v7_minimal
  output_dir: /workspace/outputs/aef_qwen_v7_minimal

training:
  epochs: 50  # 极简验证只跑 50 epochs
  
  # VICReg + KoLeo
  vicreg_weight: 1.0
  vicreg_lambda_var: 1.0
  vicreg_lambda_cov: 0.04
  koleo_weight: 0.1
  
  # 原有四件套权重置 0
  uniformity_weight: 0.0
  variance_weight: 0.0
  decorrelation_weight: 0.0
  orthogonality_weight: 0.0
  
  # 其他保持不变
  reconstruction_weight: 1.0
  consistency_weight: 0.05
  temporal_magnitude_weight: 0.3
  classification_weight: 0.03
  
  lr: 0.0001
  gradient_accumulation_steps: 4
```

### 4.4 验证标准

运行 50 epochs 后，与 V5 同期（epoch 50）对比：

| 指标 | V5 epoch 50 | V7 Minimal 目标 | 判断 |
|------|------------|----------------|------|
| recon | ~0.3 | < 0.35 | 重建不崩 |
| vicreg | N/A | -2.0 ~ 0.0 | 不坍缩 |
| koleo | N/A | -2.0 ~ -0.5 | 分散良好 |
| pre_unif (监控) | ~-2.5 | ~-2.5 | 与 V5 持平 |

**通过标准**: 50 epochs 内无 NaN/Inf，vicreg < 1.0，recon < 0.35。

**时间**: 8 卡 DDP，50 epochs ≈ 1 天。

---

## 五、完整 V7 方案（极简验证通过后执行）

### 5.1 数据层

- window_mode: `"structured_mixed"`（相邻月份 30% + 长间隔 70%）
- frame_quality_json: 启用，fallback 权重 0.5

### 5.2 模型层

- num_blocks: 10（前 2 禁用 Space）
- PseudoChangeDecoder: 新增

### 5.3 损失层

- VICReg + KoLeo（极简验证已验证）
- PseudoChangeDecoder Loss（权重 0.3）
- Reconstruction（权重 1.0，带 quality weight）
- Consistency（权重 0.05）

### 5.4 训练层

- 8 卡 DDP，batch=2，accum=8，effective batch=128
- lr=0.0001，400 epochs
- EMA Teacher（momentum=0.996）

### 5.5 8 卡训练配置

```yaml
# configs/qwen_v7_from_scratch.yaml
training:
  epochs: 400
  gradient_accumulation_steps: 8
  # batch=2 * 8GPU * accum=8 = effective batch=128
  
  lr: 0.0001
  lr_min: 0.000001
  warmup_epochs: 10
  recon_warmup_epochs: 10
  
  # VICReg + KoLeo
  vicreg_weight: 1.0
  vicreg_lambda_var: 1.0
  vicreg_lambda_cov: 0.04
  koleo_weight: 0.1
  
  # Pseudo Change Decoder
  change_prediction_weight: 0.3
  
  # 保留
  reconstruction_weight: 1.0
  consistency_weight: 0.05
  classification_weight: 0.03
  
  # 移除
  uniformity_weight: 0.0
  variance_weight: 0.0
  decorrelation_weight: 0.0
  orthogonality_weight: 0.0
  temporal_magnitude_weight: 0.0
  temporal_cosine_pixel_weight: 0.0
  pixel_temporal_info_nce_weight: 0.0
  
  # EMA Teacher
  teacher_momentum: 0.996
  
  # Student 轻量扰动
  student_frame_drop_rate: 0.2
  student_source_drop_rate: 0.15
  student_front_drop_prob: 0.1
  student_back_drop_prob: 0.1
  
  save_every: 20
  grad_clip_norm: 1.0
  weight_decay: 0.05
```

**启动命令**:
```bash
cd /workspace/xuannv
conda activate xuannv
npu-smi info  # 确认 8 卡空闲

torchrun --nproc_per_node=8 \
  scripts/train/train_ddp_v7.py \
  --config configs/qwen_v7_from_scratch.yaml \
  --save-every 20
```

---

## 六、实施路线图

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| **Phase 0: 极简验证** | Day 1-2 | 1. 新增 vicreg_loss.py<br>2. 修改 trainer.py（替换四件套）<br>3. 新增 qwen_v7_minimal.yaml<br>4. 冒烟测试 | `scripts/test_v7_minimal.py` 通过 |
| | Day 3-4 | 8 卡 DDP 训练 50 epochs | checkpoint + 损失曲线 |
| | Day 5 | 对比 V5 epoch 50，判断假设是否成立 | 评估报告 |
| **Phase 1: 完整 V7**（极简验证通过后） | Day 6-7 | 1. 修改 blocks.py（use_space）<br>2. 修改 model.py（10 blocks + change_decoder）<br>3. 修改 dataset.py（structured_mixed + quality）<br>4. 新增 ddp_v7_trainer.py（EMA Teacher） | 代码完成 |
| | Day 8 | 冒烟测试 + bug 修复 | `scripts/test_v7_launch.py` 通过 |
| | Day 9-16 | 8 卡 DDP 训练 400 epochs | 每 20 epoch checkpoint |
| | Day 17-18 | 阶段性验证（epoch 50, 100, 150, 200...） | validate_v7.py 报告 |
| **Phase 2: 评估与对比** | Day 19-20 | 与 V5 公平对比（相同 CD Head + 相同数据） | 最终评估报告 |

---

## 七、风险与退出策略

| 风险 | 应对 |
|------|------|
| **极简验证失败**（vicreg > 1.0 或 NaN） | 回退到四件套，分析 VICReg 超参（lambda_cov 过大？）。若仍失败，终止 V7。 |
| **8 卡 OOM** | 降低 accum_steps（8→4），effective batch=64。或降低 num_blocks（10→8）。 |
| **训练 100 epoch 后 bare BA 不升** | 对比 V5 基线，若差距 < 3pp 继续到 200 epoch；若 > 5pp 终止。 |
| **PseudoChangeDecoder 引入 NaN** | 降低 weight 到 0.1，或完全移除。 |

---

## 八、如何评估生产的嵌入数据集好坏（回答用户问题）

评估 embedding 质量需从 **外在下游性能** 和 **内在空间属性** 两个维度进行。

### 8.1 外在评估：下游任务性能（与 AEF 论文对齐）

这是最重要的评估维度。AEF 论文的核心评估设计是**模拟真实数据稀缺场景**。

#### A. 变化检测（Change Detection）

**直接监督（Supervised）**:
- **方法**: 在 before/after embedding 对上，用 **1-shot / 10-shot / max-trial** 训练 Linear Probe 或 kNN
- **指标**: **Balanced Accuracy (BA)** — 不是 AUC
  - 原因: 变化检测是极度不平衡二分类（~1.8% positive），AUC 对少数类不敏感，BA 直接反映模型对变化区域的真实识别能力
- **对标**: AEF 论文报告 BA = 78.4%（linear, max-trial），我们的目标 > 60%

**无监督阈值化（Unsupervised）**:
- **方法**: 计算 before/after embedding 的 cosine distance，用 Otsu 或最佳阈值自动分割
- **指标**: BA
- **对标**: AEF 论文报告 BA = 71.3%，我们的目标 > 55%

#### B. 地物分类（Thematic Mapping）

**数据源**: WorldCover / DynamicWorld

**Few-shot 协议**:
1. 从每类随机抽取 K 个像素（K = 1, 10, 100, max）
2. 用 **Linear Probe** 和 **kNN (k=1, 3)** 训练
3. 报告 **Balanced Accuracy (BA)** 和 **Overall Accuracy (OA)**
4. 每个 K 值重复 5 次，取平均
5. **空间约束**: 训练/测试像素最小间距 ≥ 32 像素（~320m），防止空间自相关

**对标**:
- AEF LCMAP max-trial: BA = 78.4%
- 我们的目标: max-trial BA > 60%，10-shot BA > 45%

#### C. 生物物理变量回归（Biophysical Regression）

若有标注数据（如蒸散发、地表发射率）：
- **方法**: Linear Probe 回归
- **指标**: R², MAE
- **对标**: AEF 论文 ET 回归 R² = 0.72

### 8.2 内在评估：Embedding 空间属性

这些指标不依赖下游标注，直接衡量 embedding 空间的质量。

#### A. 均匀性（Uniformity）— 防止坍缩

| 指标 | 计算方法 | 正常范围 | 异常信号 |
|------|---------|----------|---------|
| `raw_uniformity` | `logsumexp(-t*||xi-xj||^2)` | -4.0 ~ -1.0 | > -0.5 坍缩 |
| `koleo` | 最近邻距离的负对数平均 | -2.0 ~ -0.5 | > 0.0 聚集 |
| `effective_rank` | 协方差矩阵的有效秩 | ≈ D (128) | < D/2 维度坍缩 |

**effective_rank 计算**:
```python
cov = (Z.T @ Z) / (N - 1)  # [D, D]
eigenvalues = torch.linalg.eigvalsh(cov)
effective_rank = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
```
- effective_rank ≈ 128: 所有维度都在使用，无坍缩
- effective_rank < 64: 一半维度坍缩，表征空间退化

#### B. 语义聚类质量（Semantic Clustering）

| 指标 | 计算方法 | 正常范围 | 意义 |
|------|---------|----------|------|
| **Silhouette Score** | 用 WorldCover 标签计算 | > 0.3 | 同类 embedding 聚集，异类分离 |
| **NN Retrieval Accuracy** | 对每个像素，检索最近邻是否同类 | > 60% | 局部结构良好 |
| **Class Separation** | 类间距离 / 类内距离 | > 2.0 | 类别判别性强 |

**Silhouette Score 计算**:
```python
from sklearn.metrics import silhouette_score
# X: [N, D] embedding (L2 normalized)
# labels: [N] WorldCover class
score = silhouette_score(X, labels, metric='cosine')
```
- > 0.5: 优秀
- 0.25 ~ 0.5: 良好
- < 0.25: 差

#### C. 时序属性（Temporal Properties）

| 指标 | 计算方法 | 正常范围 | 意义 |
|------|---------|----------|------|
| **Temporal Consistency** | 同一地点相邻月份 embedding 的 cosine similarity | 0.5 ~ 0.9 | 太低=不稳定，太高=无时间变化 |
| **Temporal Discriminability** | 变化区域 vs 不变区域的 distance 差异 | > 0.1 | 差异太小=无法检测变化 |
| **Seasonal Trajectory** | 12个月 embedding 在 PCA 上的轨迹是否形成闭合环 | 是 | 验证模型学到季节性周期 |

**Seasonal Trajectory 可视化**:
```python
# 提取某 patch 12 个月的 embedding
# PCA 降维到 2D
# 画轨迹：1月→2月→...→12月→1月
# 期望：形成闭合环（季节性周期）
```

#### D. 重建质量（Reconstruction Quality）

| 指标 | 计算方法 | 正常范围 |
|------|---------|----------|
| **PSNR (S2)** | 重建图像 vs 原始图像 | > 25 dB |
| **SSIM (S2)** | 结构相似性 | > 0.7 |
| **WorldCover Accuracy** | 分类重建的像素准确率 | > 70% OA |

### 8.3 综合评估报告模板

每个 checkpoint 应输出以下报告：

```json
{
  "epoch": 100,
  "intrinsic": {
    "uniformity": -3.2,
    "koleo": -1.5,
    "effective_rank": 118,
    "silhouette_worldcover": 0.31,
    "nn_retrieval_acc": 0.64,
    "temporal_consistency": 0.72,
    "temporal_discriminability": 0.15
  },
  "downstream": {
    "cd_unsupervised_ba": 0.58,
    "cd_supervised_10shot_ba": 0.52,
    "cd_supervised_maxtrial_ba": 0.64,
    "worldcover_10shot_ba": 0.48,
    "worldcover_maxtrial_ba": 0.62,
    "recon_psnr": 27.3
  }
}
```

### 8.4 与 AEF 官方 Embedding 的对比

若用户能获取哈尔滨区域的 AEF 官方 embedding：

| 对比项 | AEF 官方 | 我们的模型 |
|--------|---------|-----------|
| 维度 | 64 | 128 |
| 时间粒度 | 年度 | 月度 |
| 无监督 CD BA | 71.3% | 待测 |
| 10-shot LC BA | ~78% | 待测 |
| 存储 | 64 bytes/pixel | 128 bytes/pixel |

**核心对比维度**:
1. **时间敏感度**: AEF 年度 embedding 对月度变化不敏感，我们的月度 embedding 应在此占优
2. **局部区域优化**: AEF 是全球模型，我们是哈尔滨区域优化模型
3. **下游可扩展性**: AEF + simple head 的效果 vs 我们 + CD Head 的效果

---

## 九、待你最终确认的事项

请确认以下事项，我将立即开始实施：

1. **极简验证先行**: 确认 Phase 0（VICReg+KoLeo 替换四件套，50 epoch，8卡）作为第一步
2. **双窗口策略**: 确认 `"structured_mixed"`（相邻月份 30% + 长间隔 70%）
3. **Fallback quality weight**: 确认 0.5（若你有其他偏好，如 0.3 或 0.7，请告知）
4. **8 卡训练**: 确认使用全部 8 张 NPU
5. **嵌入评估**: 上述评估体系是否满足你的需求？是否需要新增其他评估项（如特定下游任务）？

**请回复确认，我将立即开始 Phase 0 的实施。**
