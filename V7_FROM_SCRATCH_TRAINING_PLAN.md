# V7 Backbone 从头训练详细计划

> 目标: 在 V5 经验基础上，从头训练一个更高质量 embedding 的 backbone
> 核心假设: V5 encoder 结构已被 161 epochs 固化，soft-restart 无法引入新的时间敏感度
> 训练方式: 完全从头初始化，不加载任何 checkpoint
> 预计时间: 5-7 天 (400 epochs, 3×NPU DDP)

---

## 一、训练目标与成功标准

### 1.1 目标

生成一个 embedding 空间满足以下特性的 backbone:
1. **空间均匀性**: uniformity ≥ -3.5（不坍缩）
2. **时间敏感度**: 无监督变化检测 BA > 60%（V5 基线约 50-55%）
3. **语义表征**: WorldCover max-trial BA > 60%（V5 基线约 55%）
4. **重建质量**: reconstruction loss < 0.25（不劣于 V5）

### 1.2 与 V5 的关键差异

| 维度 | V5 | V7 |
|------|-----|-----|
| 初始化 | V4 checkpoint soft-restart | **完全随机初始化** |
| 反坍缩 | raw_unif + decorr + var + orth 四件套 | **VICReg + KoLeo** |
| 自蒸馏 | consistency (cosine, weight=0.05) | **DINO (EMA Teacher + centering + sharpening)** |
| 时序监督 | temporal_magnitude (hinge, ≈0) | **PseudoChangeDecoder (像素级伪变化预测)** |
| STP Blocks | 8 blocks, 全部 Space+Time+Precision | **10 blocks, 前2 block 禁用 Space** |
| 双窗口 | mixed_scale (50% 短间隔 1-3月) | **强制 ≥6 个月间隔** |
| 数据质量 | 所有帧等权重 | **冬季 fallback 帧重建权重 ×0.5** |

---

## 二、数据工程（无需重新生成缓存，运行时调整）

### 2.1 强制大间隔双窗口

**修改文件**: `src/data/dataset.py`

**当前问题**: V5 的 `mixed_scale` 有 50% 概率采样短间隔（1-3月），变化信号极弱，导致 temporal_magnitude_loss ≈ 0。

**修改点**: 在 `_sample_dual_windows` 中新增 `"large_gap_only"` 模式:

```python
# 在 __init__ 中
self.window_mode = getattr(d, "window_mode", "random_split")
self._min_gap_ms = getattr(d, "min_gap_ms", 6 * 30 * 24 * 3600 * 1000)  # 6个月

# 在 _sample_dual_windows 中
if self.window_mode == "large_gap_only" and self.training:
    return self._sample_long_gap_windows(ts_sorted, min_gap_ms=self._min_gap_ms)
```

**具体逻辑**:
- 训练时: 强制 w1 和 w2 的中心时间间隔 ≥ 6 个月
- 推理时: 按用户指定的 valid_period（不受此限制）
- 实现方式: 复用已有的 `_sample_long_gap_windows`，但移除 short gap 分支

**验证方法**: 训练 10 epochs 后检查 `temporal_magnitude_loss`（若保留的话）是否不再≈0。由于 V7 移除了 temporal_magnitude_loss，改为检查 **PseudoChangeDecoder 的 loss** 是否 > 0 且有下降趋势。

### 2.2 冬季 Fallback 帧降级

**修改文件**: `src/data/dataset.py`

**当前问题**: 357 个冬季月份为 fallback（全 cloudy，被迫保留最不清的一张），这些帧噪声大，重建损失被迫拟合噪声。

**修改点**: 在帧加载时标记 fallback 帧，在训练器中降低其重建权重:

```python
# dataset.py: _load_input_frames_impl 或 _preload_patch_worker 中
# 标记 fallback 帧 (可以在 filter_cloudy_frames.py 时生成标记文件)
# 或者在 dataset 中通过判断 cloud_score 阈值来推断

# 更简单的方案：在 dataset __getitem__ 返回中增加 frame_quality_mask
# 对于 fallback 月份的唯一帧，quality_mask = 0.5
```

**简化实现**（推荐）:
- 在 `scripts/preprocessing/filter_cloudy_frames.py` 中，为每个 patch 生成一个 `frame_quality.json`，记录每帧的质量等级（1.0=清晰, 0.5=fallback）
- 在 `dataset.py` 加载时读取该文件
- 在 `trainer.py` 的 `compute_recon_loss` 中，将 quality_mask 乘到重建损失上

**预期效果**: `recon_loss` 在 warmup 后应 < 0.25（因为不再强制拟合噪声）。

---

## 三、模型架构修改

### 3.1 STPBlock 支持条件 Space 路径

**修改文件**: `src/models/blocks.py`

**当前**: 所有 block 都包含 Space+Time+Precision 三条路径。

**目标**: 前 2 个 block 禁用 Space 路径，只做 Precision + Time。

**代码修改**:

```python
class STPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        time_code_dim: int = 64,
        use_space: bool = True,  # ★ 新增参数
    ) -> None:
        super().__init__()
        self.use_space = use_space
        
        # Precision path (始终启用)
        self.precision_conv = nn.Sequential(...)
        
        # Time path (始终启用)
        self.time_down = nn.Conv2d(...)
        self.time_attn = nn.MultiheadAttention(...)
        self.time_up = nn.ConvTranspose2d(...)
        
        # Space path (条件启用)
        if use_space:
            self.space_down = nn.Sequential(...)
            self.space_attn = nn.MultiheadAttention(...)
            self.space_up = nn.Sequential(...)
            fusion_in_ch = channels * 3
        else:
            fusion_in_ch = channels * 2
        
        # 融合层输入通道根据 use_space 调整
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        
    def forward(self, x, time_codes=None, frame_mask=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        
        # Precision path
        x_prec = self.precision_conv(x_flat)
        
        # Time path
        x_t = self.time_down(x_flat)
        ...  # 原有逻辑
        x_t_up = self.time_up(x_t)
        
        # Space path (条件)
        if self.use_space:
            x_s = self.space_down(x_flat)
            ...  # 原有逻辑
            x_s_up = self.space_up(x_s)
            fused = self.fusion(torch.cat([x_prec, x_t_up, x_s_up], dim=1))
        else:
            fused = self.fusion(torch.cat([x_prec, x_t_up], dim=1))
        
        residual = self.residual_norm(x_flat)
        out = fused + residual
        return out.reshape(B, T, C, H, W)
```

**验证方法**: 修改后运行 `scripts/test_v6_launch.py` 或写一个快速测试，确认:
- use_space=True: 输出 shape 正确，与原有行为一致
- use_space=False: 输出 shape 正确，参数量减少约 1/3

### 3.2 AEFModel 初始化 10 blocks（前2禁用Space）

**修改文件**: `src/models/model.py`

**代码修改**:

```python
# 在 __init__ 中
self.stp_blocks = nn.ModuleList()
for i in range(m.num_blocks):
    use_space = i >= 2  # 前 2 个 block 禁用 Space
    self.stp_blocks.append(
        STPBlock(
            channels=m.precision_dim,
            num_heads=m.num_heads,
            time_code_dim=m.time_code_dim,
            use_space=use_space,
        )
    )
```

**配置**:
```yaml
model:
  num_blocks: 10  # 8 → 10
```

**预期效果**: 
- 前 2 block 学习低级特征（边缘、纹理、时序动态），不受空间注意力干扰
- 后 8 block 学习空间上下文
- 参数量增加约 15%（因为前 2 block 更小），计算量增加约 10%

### 3.3 PseudoChangeDecoder（新增模块）

**新增文件**: `src/models/change_decoder.py`

**设计目标**: 从双窗口 embedding 预测像素级变化幅度，提供显式的时间敏感度监督信号。

```python
"""像素级伪变化预测 Decoder."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class PseudoChangeDecoder(nn.Module):
    """轻量 decoder，从双窗口 pre-norm embedding 预测像素级变化幅度.
    
    输入: e1, e2 — pre_norm_map [B, D, H, W]
    输出: change_map [B, 1, H, W] — 预测的像素级变化幅度 (0~1)
    
    伪标签生成:
        对 S2 最近帧计算绝对差异 |I_w1 - I_w2|，归一化到 0~1
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # 输入: concat(e1, e2, |e1-e2|) → [B, 3D, H, W]
            nn.Conv2d(embedding_dim * 3, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(e1 - e2)
        x = torch.cat([e1, e2, diff], dim=1)
        return self.conv(x)  # [B, 1, H, W]
```

**集成到 AEFModel**:

```python
# 在 AEFModel.__init__ 中
self.change_decoder = PseudoChangeDecoder(
    embedding_dim=m.embedding_dim,
    hidden_dim=getattr(m, "change_decoder_hidden", 64),
)

# 新增方法
@torch.no_grad()
def compute_pseudo_change_target(
    self,
    source_frames: torch.Tensor,
    source_frame_mask: torch.Tensor,
    valid_start_w1: torch.Tensor,
    valid_end_w1: torch.Tensor,
    valid_start_w2: torch.Tensor,
    valid_end_w2: torch.Tensor,
) -> torch.Tensor:
    """从原始 S2 帧生成伪变化标签 [B, 1, H, W]."""
    # 1. 找出 w1 和 w2 各自最近的 S2 帧
    # 2. 计算绝对差异
    # 3. 归一化到 0~1
    ...
```

**伪标签生成逻辑**（在 trainer 中实现，不放进 model forward）:
```python
# 对每个 batch:
# 1. 从 source_frames 中提取 S2 源 (source_type_ids 中 s2 对应的索引)
# 2. 在 w1 的 valid_period 内找最近帧，w2 同理
# 3. 计算 |I_w1 - I_w2| 的 mean across channels
# 4. 归一化: (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
# 5. 高斯平滑 (kernel=3, sigma=1) 作为最终标签
```

**损失**: `F.mse_loss(pred_change_map, pseudo_target)` 或 `F.l1_loss`

**权重**: `change_prediction_weight = 0.3`

---

## 四、损失函数重构

### 4.1 移除的损失

| 损失 | 移除原因 |
|------|---------|
| `raw_uniformity_loss` | 由 VICReg 统一替代 |
| `decorrelation_loss` | 由 VICReg covariance 替代 |
| `variance_regularizer` | 由 VICReg variance 替代 |
| `bottleneck_orthogonality_loss` | 由 KoLeo 替代 |
| `temporal_magnitude_loss` | hinge loss 几乎无梯度，失效 |
| `temporal_cosine_pixel_loss` | 与 uniformity 冲突，V6 已证明有害 |
| `pixel_temporal_info_nce_loss` | 计算昂贵，收益不明确 |

### 4.2 新增/保留的损失

#### A. VICReg Loss（核心反坍缩）

**新增文件**: `src/training/vicreg_loss.py`

```python
def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, 
                lambda_var: float = 1.0, lambda_cov: float = 0.04) -> torch.Tensor:
    """VICReg: Variance-Invariance-Covariance Regularization.
    
    Args:
        z1, z2: [N, D] — 同一 batch 的两个视图（teacher vs student）
        lambda_var: variance 项权重
        lambda_cov: covariance 项权重（论文推荐 1/25 = 0.04）
    """
    # 1. Invariance (对齐)
    inv = F.mse_loss(z1, z2)
    
    # 2. Variance (每维标准差 ≥ 1)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
    
    # 3. Covariance (去相关)
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (z1_c.shape[0] - 1)
    cov_z2 = (z2_c.T @ z2_c) / (z2_c.shape[0] - 1)
    cov_loss = (cov_z1.pow(2).sum() - cov_z1.diagonal().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov_z2.pow(2).sum() - cov_z2.diagonal().pow(2).sum()) / z2.shape[1]
    
    return inv + lambda_var * var + lambda_cov * cov_loss
```

**两个视图的来源**:
- Teacher: 完整输入（无 dropout）→ `model.forward(...)`
- Student: 扰动输入（frame drop + source drop）→ `model.forward(..., student_mode=True)`

**关键**: VICReg 的 invariance 项要求 z1 和 z2 是**同一输入的不同增强视图**。在我们的场景中:
- 若 teacher/student 输入差异太大（如 student drop 了 40% 帧），invariance 会过高，导致模型被迫对缺失数据鲁棒，但可能牺牲表征质量
- **折中方案**: student 只做轻量扰动（drop 10-20% 帧），而非 V5 的 40%

#### B. KoLeo Loss（几何分散）

```python
def koleo_loss(x: torch.Tensor) -> torch.Tensor:
    """Kozachenko-Leonenko 熵估计正则化.
    
    强制 batch 内 embedding 的最近邻距离最大化。
    比 uniformity loss 更直接，对小 batch size 更稳定。
    """
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    dists = dists + torch.eye(dists.shape[0], device=dists.device) * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()
```

**作用**: 当 batch 内 embedding 聚集时，最近邻距离小，loss 大（惩罚聚集）。

**与 VICReg 的关系**:
- VICReg 的 variance + covariance 保证全局分散和去相关
- KoLeo 保证局部（最近邻）不聚集
- 两者互补，理论上比 raw_uniformity + decorr + var + orth 更协调

#### C. DINO Loss（EMA Teacher 自蒸馏）

**新增模块**: `src/models/dino_head.py`

```python
class DINOHead(nn.Module):
    """DINO 投影头: 3层 MLP + softmax."""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.mlp(x), dim=-1)
```

**EMA Teacher 更新**:

```python
class EMATeacher:
    def __init__(self, student_model: nn.Module, momentum: float = 0.996):
        self.teacher = copy.deepcopy(student_model).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.momentum = momentum
        self.center = None
        self.center_momentum = 0.9
    
    @torch.no_grad()
    def update(self, student: nn.Module):
        for param_t, param_s in zip(self.teacher.parameters(), student.parameters()):
            param_t.data.mul_(self.momentum).add_(param_s.data, alpha=1 - self.momentum)
    
    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor):
        """DINO centering: 防止某个维度主导."""
        batch_center = teacher_logits.mean(dim=0)
        if self.center is None:
            self.center = batch_center
        else:
            self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center
    
    def get_teacher_probs(self, teacher_logits: torch.Tensor, temperature: float = 0.07):
        """Sharpening + centering."""
        logits = (teacher_logits - self.center) / temperature
        return F.softmax(logits, dim=-1)
```

**DINO Loss 计算**:
```python
# Student 前向 (扰动输入)
student_out = model(student_input)
student_logits = model.dino_head(student_out.pre_norm_embedding)  # [B, K]

# Teacher 前向 (完整输入)
with torch.no_grad():
    teacher_out = teacher_model(teacher_input)
    teacher_logits = teacher_model.dino_head(teacher_out.pre_norm_embedding)  # [B, K]
    ema_teacher.update_center(teacher_logits)
    teacher_probs = ema_teacher.get_teacher_probs(teacher_logits)

# Loss: cross-entropy
loss_dino = -(teacher_probs * student_logits).sum(dim=-1).mean()
```

**注意**: DINO 需要较大的 output_dim（如 65536）才能有效。但我们的 batch size 小（effective=48），可能需要调整为 4096 或 16384。

**简化方案**（推荐，降低实现复杂度）:
- 不用显式的 DINOHead 和 softmax，改用 **cosine similarity + InfoNCE**
- Teacher 提供目标 embedding，Student 学习匹配
- 即: `loss_dino = 1 - cosine_similarity(student_emb, teacher_emb).mean()`
- 但这与现有的 consistency_loss 几乎相同

**最终建议的简化 DINO**:
```python
# 不新增 DINOHead，直接对 pre_norm_embedding 做:
# Teacher: EMA 更新的完整模型
# Student: 当前模型（输入有扰动）
# Loss: MSE(student_pre_norm, teacher_pre_norm.detach()) + KoLeo(student_pre_norm)
# 这等价于 VICReg invariance + KoLeo，不需要显式 DINO head
```

考虑到实现复杂度和 NPU 兼容性，**V7 第一版采用简化方案**:
- **EMA Teacher**: momentum=0.996，逐参数更新
- **Invariance Loss**: `MSE(student_pre_norm, teacher_pre_norm.detach())`
- **KoLeo Loss**: 在 student_pre_norm 上计算
- **VICReg Loss**: 需要两个独立视图，可用 (teacher_pre_norm, student_pre_norm) 作为 z1, z2

这样三者合一，实现简洁:
```python
# 总反坍缩损失
vicreg = vicreg_loss(teacher_pre_norm, student_pre_norm, lambda_var=1.0, lambda_cov=0.04)
koleo = koleo_loss(student_pre_norm)
antcollapse_total = vicreg + 0.1 * koleo
```

### 4.3 损失组合（最终版）

```python
total_loss = (
    recon_weight * recon_loss                    # 重建 (权重 1.0)
    + vicreg_weight * vicreg_loss(...)           # VICReg (权重 1.0)
    + koleo_weight * koleo_loss(...)             # KoLeo (权重 0.1)
    + change_pred_weight * change_pred_loss      # 伪变化预测 (权重 0.3)
    + consistency_weight * consistency_loss      # Teacher-Student cosine (权重 0.05)
    + cls_weight * cls_loss                      # WorldCover 分类 (权重 0.03)
)
```

**移除项**: uniformity, decorrelation, variance, orthogonality, temporal_magnitude, temporal_cosine_pixel, pixel_temporal_info_nce

---

## 五、训练器设计（DDPv7Trainer）

### 5.1 新增文件: `src/training/ddp_v7_trainer.py`

基于 `src/training/trainer.py` 修改，核心差异:

1. **初始化 EMA Teacher**
2. **Student 扰动**: 轻量 frame drop（10-20%，而非 V5 的 40%）
3. **双 forward**: teacher 完整输入，student 扰动输入
4. **PseudoChangeDecoder 前向**: 对 student 的双窗口 embedding 预测变化
5. **损失计算**: VICReg + KoLeo + ChangePred + Recon

**关键代码结构**:

```python
class DDPv7Trainer:
    def __init__(self, ...):
        ...
        # EMA Teacher
        self.teacher_model = copy.deepcopy(self.model.module).eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_momentum = cfg.training.teacher_momentum  # 0.996
        
    def train_epoch(self, epoch, dataloader):
        for step, batch in enumerate(dataloader):
            # --- Teacher 前向 (完整输入) ---
            with torch.no_grad():
                teacher_out = self.teacher_model(...)
            
            # --- Student 前向 (扰动输入) ---
            student_batch = self._apply_student_perturbation(batch)
            student_out = self.model(**student_batch)
            
            # --- 重建损失 ---
            recon = compute_recon_loss(student_out.reconstructions, ...)
            
            # --- VICReg (teacher_pre_norm vs student_pre_norm) ---
            gathered_teacher = self._gather(teacher_out.pre_norm_embedding)
            gathered_student = self._gather(student_out.pre_norm_embedding)
            vicreg = vicreg_loss(gathered_teacher, gathered_student)
            
            # --- KoLeo (student_pre_norm) ---
            koleo = koleo_loss(gathered_student)
            
            # --- PseudoChangeDecoder ---
            if "valid_start_w1" in batch:
                emb_w1, emb_w2, pre_w1, pre_w2 = self.model.module.encode_dual_window(...)
                change_pred = self.model.module.change_decoder(pre_w1, pre_w2)
                pseudo_target = self._compute_pseudo_change_target(batch)
                change_loss = F.mse_loss(change_pred, pseudo_target)
            else:
                change_loss = 0.0
            
            # --- 总损失 ---
            total = (
                t.reconstruction_weight * recon
                + t.vicreg_weight * vicreg
                + t.koleo_weight * koleo
                + t.change_prediction_weight * change_loss
                + t.consistency_weight * consistency_loss(gathered_teacher.detach(), gathered_student)
            )
            
            total.backward()
            ...
            
            # 更新 EMA Teacher
            if (step + 1) % t.gradient_accumulation_steps == 0:
                self._update_teacher()
    
    @torch.no_grad()
    def _update_teacher(self):
        for param_t, param_s in zip(self.teacher_model.parameters(), self.model.module.parameters()):
            param_t.data.mul_(self.teacher_momentum).add_(param_s.data, alpha=1 - self.teacher_momentum)
    
    def _apply_student_perturbation(self, batch):
        """轻量扰动: 随机 drop 10-20% 帧."""
        # 比 V5 更轻，保证 student 和 teacher 输入差异不大
        ...
    
    def _compute_pseudo_change_target(self, batch):
        """从 S2 原始帧生成伪变化标签."""
        ...
```

### 5.2 训练入口脚本

**新增文件**: `scripts/train/train_ddp_v7.py`

基于 `scripts/train/train_ddp.py` 修改:
- 导入 `DDPv7Trainer`
- 支持 `--config configs/qwen_v7_from_scratch.yaml`
- 不支持 `--resume` V5 checkpoint（强制从头训练）

---

## 六、配置文件

**新增文件**: `configs/qwen_v7_from_scratch.yaml`

```yaml
# ============================================================
# AEF_qwen V7 — 从头训练
# 核心设计:
#   1. 完全随机初始化，不加载任何 checkpoint
#   2. VICReg + KoLeo 替代反坍缩四件套
#   3. EMA Teacher (momentum=0.996)
#   4. PseudoChangeDecoder (像素级伪变化预测)
#   5. 10 STP blocks (前2禁用 Space)
#   6. 强制双窗口间隔 ≥6 个月
# ============================================================

_base_: qwen_v1_scenes.yaml

experiment:
  name: aef_qwen_v7_from_scratch
  output_dir: /workspace/outputs/aef_qwen_v7_from_scratch

data:
  # 强制大间隔双窗口
  window_mode: "large_gap_only"
  min_gap_ms: 15552000000  # 6个月
  non_overlap_min_frames: 4
  non_overlap_max_frames: 12
  
  # 冬季 fallback 降级
  use_frame_quality_weight: true
  frame_quality_json: "/workspace/statistics/harbin_scenes/frame_quality.json"

model:
  num_blocks: 10
  embedding_dim: 128
  change_decoder_hidden: 64
  vmf_kappa: 50.0

training:
  epochs: 400
  
  # 梯度累积: batch=2 * 3GPU * accum=8 = effective batch=48
  gradient_accumulation_steps: 8
  
  # EMA Teacher
  teacher_momentum: 0.996
  
  # --- 核心损失权重 ---
  reconstruction_weight: 1.0
  consistency_weight: 0.05   # Teacher-Student cosine
  
  # ★ VICReg (替代 raw_unif + decorr + var)
  vicreg_weight: 1.0
  vicreg_lambda_var: 1.0
  vicreg_lambda_cov: 0.04   # 1/25
  
  # ★ KoLeo (替代 orth)
  koleo_weight: 0.1
  
  # ★ Pseudo Change Decoder
  change_prediction_weight: 0.3
  
  # 分类监督
  classification_weight: 0.03
  
  # --- 移除项 (权重=0) ---
  uniformity_weight: 0.0
  variance_weight: 0.0
  decorrelation_weight: 0.0
  orthogonality_weight: 0.0
  temporal_magnitude_weight: 0.0
  temporal_cosine_pixel_weight: 0.0
  pixel_temporal_info_nce_weight: 0.0
  
  # --- Student 扰动 (比 V5 更轻) ---
  student_frame_drop_rate: 0.2   # 0.4 → 0.2
  student_source_drop_rate: 0.15  # 0.25 → 0.15
  student_front_drop_prob: 0.1    # 0.15 → 0.1
  student_back_drop_prob: 0.1     # 0.15 → 0.1
  
  # --- 渐进 VMF Kappa ---
  kappa_start: 50.0
  kappa_end: 500.0
  kappa_warmup_epochs: 100
  
  # --- 学习率 ---
  lr: 0.0001
  lr_min: 0.000001
  warmup_epochs: 10
  recon_warmup_epochs: 10
  save_every: 20
  grad_clip_norm: 1.0
  weight_decay: 0.05
  
  # 最佳 checkpoint 选择
  save_best_balanced: true
  best_balanced_vicreg_min: -2.0  # VICReg loss 正常范围
  best_balanced_vicreg_max: 0.0
  early_stop_patience: 150
```

---

## 七、验证协议

### 7.1 训练过程监控（每 epoch）

| 指标 | 来源 | 正常范围 | 异常信号 |
|------|------|----------|---------|
| `recon` | 重建损失 | < 0.3 | > 0.5 数据问题 |
| `vicreg` | VICReg 总损失 | -2.0 ~ 0.0 | > 1.0 可能坍缩 |
| `vicreg_inv` | Invariance (MSE) | 0.0 ~ 1.0 | > 2.0 teacher/student 差异过大 |
| `vicreg_var` | Variance | 0.0 ~ 1.0 | > 2.0 方差坍缩 |
| `vicreg_cov` | Covariance | 0.0 ~ 0.5 | > 1.0 强相关 |
| `koleo` | KoLeo | -2.0 ~ -0.5 | > 0.0 embedding 聚集 |
| `change_pred` | 伪变化预测 | 0.05 ~ 0.2 | ≈0 无梯度，>0.5 预测失败 |
| `consist` | Teacher-Student cosine | 0.0 ~ 0.2 | > 0.5 差异过大 |

### 7.2 阶段性验证（每 50 epochs）

**验证脚本**: `scripts/eval/validate_v7.py`（基于 validate_v2.py 修改）

| 验证项 | 方法 | 指标 | V5 基线 | V7 目标 |
|--------|------|------|---------|---------|
| 无监督变化检测 | backbone bare cosine distance + Otsu | BA | ~50-55% | > 60% |
| 监督变化检测 | 1-shot Linear Probe on embedding | BA | 未测 | > 55% |
| 地物分类 | WorldCover max-trial Linear Probe | BA | ~55% | > 60% |
| Embedding 聚类 | Silhouette (WorldCover 标签) | Score | 未测 | > 0.25 |
| 重建质量 | S2 PSNR | dB | 未测 | > 25 |

### 7.3 与 V5 的公平对比

验证时必须控制变量:
- 使用 **相同的 CD Head 架构**（V3）
- 使用 **相同的训练数据划分**
- 使用 **相同的评估脚本**（只改 checkpoint 路径）
- 对比指标: CD Head AUC, Bare BA, WorldCover BA

---

## 八、启动命令与监控

### 8.1 启动训练

```bash
cd /workspace/xuannv
conda activate xuannv

# 检查 NPU 占用
npu-smi info

# 使用 NPU 0,1,2 进行 3 卡 DDP
torchrun --nproc_per_node=3 \
  scripts/train/train_ddp_v7.py \
  --config configs/qwen_v7_from_scratch.yaml \
  --save-every 20 \
  --warmup-epochs 10
```

### 8.2 训练监控（必须人工检查）

```bash
# 实时监控日志
tail -f /workspace/outputs/aef_qwen_v7_from_scratch/train.log
```

**关键检查点**:
- **Epoch 1-5**: vicreg 应该在 -1.0 ~ 0.0 之间，koleo 在 -1.5 ~ -0.5 之间。若 vicreg > 1.0 或 koleo > 0.0，立即报告。
- **Epoch 10**: change_pred loss 应该 < 0.15 且有下降趋势。若 ≈0，说明伪标签生成失败。
- **Epoch 50**: 运行 validate_v7.py，记录 bare BA 和 WorldCover BA。
- **Epoch 100-150**: 观察 vicreg 是否稳定，recon 是否 < 0.25。
- **Epoch 200+**: 若 bare BA 仍 < 55%，考虑提前终止（大概率无法超越 V5）。

### 8.3 自动恢复

使用 `watchdog.sh` 模式，崩溃后自动 resume 最新 checkpoint:
```bash
torchrun --nproc_per_node=3 \
  scripts/train/train_ddp_v7.py \
  --config configs/qwen_v7_from_scratch.yaml \
  --resume /workspace/outputs/aef_qwen_v7_from_scratch/epoch_best_xxx.pt
```

---

## 九、实施时间表

| 阶段 | 任务 | 文件 | 预计时间 | 依赖 |
|------|------|------|----------|------|
| Day 1 | 修改 STPBlock 支持 use_space | `src/models/blocks.py` | 2h | 无 |
| Day 1 | 修改 AEFModel 初始化 10 blocks | `src/models/model.py` | 1h | blocks.py |
| Day 1 | 新增 PseudoChangeDecoder | `src/models/change_decoder.py` | 2h | 无 |
| Day 1 | 修改 dataset 强制大间隔 | `src/data/dataset.py` | 1h | 无 |
| Day 2 | 新增 VICReg + KoLeo | `src/training/vicreg_loss.py` | 2h | 无 |
| Day 2 | 新增 DDPv7Trainer | `src/training/ddp_v7_trainer.py` | 4h | vicreg_loss.py |
| Day 2 | 新增训练入口脚本 | `scripts/train/train_ddp_v7.py` | 1h | ddp_v7_trainer.py |
| Day 2 | 新增配置文件 | `configs/qwen_v7_from_scratch.yaml` | 1h | 无 |
| Day 3 | 冒烟测试（5 steps） | `scripts/test_v7_launch.py` | 2h | 全部 |
| Day 3 | 修复 bug + 联调 | 多个文件 | 4h | 冒烟测试 |
| Day 4-10 | 启动训练 + 监控 | NPU 0,1,2 | 7天 | 联调通过 |
| Day 11 | 阶段性验证（epoch 50, 100, 150...） | `scripts/eval/validate_v7.py` | 1天 | checkpoint |

---

## 十、风险与退出策略

### 10.1 关键风险

| 风险 | 概率 | 应对 |
|------|------|------|
| V7 训练 50 epochs 后 vicreg > 1.0（坍缩） | 中 | 检查 learning rate 是否过高，或 KoLeo weight 是否过低。若无法修复，终止。 |
| V7 训练 100 epochs 后 bare BA < 55% | 中 | 对比 V5 基线，若差距 < 3pp，继续到 200 epoch；若差距 > 5pp，终止。 |
| PseudoChangeDecoder 导致 NaN | 低 | 降低 change_prediction_weight 到 0.1，或移除该损失。 |
| NPU OOM（10 blocks） | 低 | 若 OOM，将 num_blocks 改回 8，或降低 batch_size 到 1。 |
| EMA Teacher 与 DDP 冲突 | 低 | EMA Teacher 在每个 rank 上独立维护，不跨 GPU sync。 |

### 10.2 退出标准（满足任一即终止）

1. vicreg loss > 1.0 持续 10 个 epoch 不下降
2. bare BA < 50% 且训练 150 epochs 无改善
3. 出现 NaN/Inf 且 3 次重启后仍出现
4. 训练时间超过 10 天（资源预算）

---

## 十一、文件清单（新增/修改）

### 新增文件
- `src/models/change_decoder.py` — PseudoChangeDecoder
- `src/training/vicreg_loss.py` — VICReg + KoLeo
- `src/training/ddp_v7_trainer.py` — V7 DDP 训练器
- `scripts/train/train_ddp_v7.py` — V7 训练入口
- `configs/qwen_v7_from_scratch.yaml` — V7 配置
- `scripts/test_v7_launch.py` — V7 冒烟测试
- `scripts/eval/validate_v7.py` — V7 验证脚本

### 修改文件
- `src/models/blocks.py` — STPBlock 支持 use_space
- `src/models/model.py` — 10 blocks + change_decoder 集成
- `src/data/dataset.py` — 强制大间隔 + 冬季降级
- `src/config.py` — 新增 V7 配置字段（若需要）

---

## 十二、待你确认的关键决策

1. **是否接受完全从头训练（不加载 V5）？** 
   - 这意味着需要 5-7 天训练时间，且结果不确定
   - 替代方案: 在 V5 基础上只改数据采样 + CD Head（2-3 天，风险低）

2. **STP Blocks 10 是否可接受？**
   - 若 NPU OOM，需回退到 8 blocks
   - 10 blocks 比 8 增加约 10% 计算量

3. **PseudoChangeDecoder 是否必要？**
   - 它增加了实现复杂度，但提供了最直接的时间敏感度监督
   - 若你希望简化，可以移除它，只保留 VICReg + KoLeo + 强制大间隔

4. **是否需要我在实施前先写一个简化版验证（只用 VICReg + KoLeo，不改架构）？**
   - 可在 1-2 天内快速验证 VICReg 是否比四件套更好
   - 若验证通过，再投入完整 V7 训练

请确认以上计划，或提出修改意见。确认后我将按 Day 1 → Day 2 → ... 的顺序逐步实施。
