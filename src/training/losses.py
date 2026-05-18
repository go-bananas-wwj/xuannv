"""损失函数 — 核心改进: raw_uniformity 替代 batch_uniformity.

反坍缩四件套:
1. raw_uniformity_loss: 欧氏空间 uniformity, 自适应 t=2/D, 梯度永远非零
2. decorrelation_loss: Barlow Twins 去相关
3. variance_regularizer: VICReg 方差正则
4. bottleneck_orthogonality_loss: Conv1x1 权重正交约束

这些损失全部在 pre-norm (原始幅度) 空间计算，完全绕过 L2 Jacobian 梯度屏障。
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ────────────────────────────────────────────
# 1. Raw Uniformity Loss (核心改进)
# ────────────────────────────────────────────

def raw_uniformity_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """欧氏空间 uniformity loss — 无 L2 归一化, 梯度永远非零.

    原理:
    1. 标准化到零均值、全局单位方差 (适配任意特征尺度)
    2. 在标准化后的欧氏空间计算 RBF uniformity
    3. 自适应 t = 2/D, 使 exp(-t*||x-y||^2) 在正常范围

    与 batch_uniformity_loss 的关键区别:
    - 不通过 L2 归一化 → 没有 (I-uu^T)/||x|| Jacobian 的梯度杀死效应
    - 幅度差异被保留 → ||xi-xj|| > 0 → 梯度非零!
    - decorrelation 推各维度去相关 → 自然产生角度多样性

    良好分散时 (64-dim): loss ≈ -4
    完全坍缩时: loss ≈ 0
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    # 标准化: 零均值 + 全局单位方差
    z = embeddings - embeddings.mean(dim=0)
    global_std = z.std() + 1e-4
    z = z / global_std

    N = z.shape[0]
    D = z.shape[1]
    # 自适应 t: 对 D 维标准化特征, E[||xi-xj||^2] ≈ 2D
    t = 2.0 / D

    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    pair_mask = torch.triu(torch.ones(N, N, device=z.device, dtype=torch.bool), diagonal=1)
    sq_pdist_pairs = sq_pdist[pair_mask]
    loss = torch.logsumexp(-t * sq_pdist_pairs, dim=0) - math.log(sq_pdist_pairs.shape[0])
    return loss


# ────────────────────────────────────────────
# 1b. Batch Uniformity Loss — L2-normalized (V11 改进版)
# ────────────────────────────────────────────

def batch_uniformity_loss_l2(embeddings: torch.Tensor, dim_dropout: float = 0.1,
                                 max_samples: int = 512) -> torch.Tensor:
    """改进版 Batch Uniformity Loss — All-Pairs + 空间采样 + 维度 Dropout.

    V11.1 实现 (当前最佳):
    1. All-Pairs: 每个样本与 batch 中**所有其他样本**计算 |u_i · u_j|
    2. 空间采样: 对 embedding_map [B,D,H,W] 随机采样最多 max_samples 个空间位置，
       避免 OOM（全图 H*W=4096 时 all-pairs 矩阵达 16GB+）
    3. chunk 分批: 进一步降低峰值内存
    4. 维度 Dropout: 随机 mask 10% 维度后计算 dot product，增加鲁棒性
    5. 值域 [0, 1]: 0=均匀分布, 1=完全坍缩

    Args:
        embeddings: [B, D] 或 [B, D, H, W]
                   已经 L2 归一化，或在函数内部归一化
        dim_dropout: 维度 dropout 比率 (默认 0.1)
        max_samples: 最大采样空间位置数 (默认 512)

    Returns:
        scalar loss, 越小越好
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    x = embeddings

    # 处理 spatial embedding map [B, D, H, W]
    if x.dim() == 4:
        B, D, H, W = x.shape
        # 展平空间: [B, D, H*W]
        x = x.reshape(B, D, H * W)
        # 随机采样空间位置（避免 OOM）
        n_spatial = H * W
        if n_spatial > max_samples:
            indices = torch.randperm(n_spatial, device=x.device)[:max_samples]
            x = x[:, :, indices]
        # 转置为 [B, H*W, D] → 展平为 [B*H*W, D]
        x = x.permute(0, 2, 1).reshape(-1, D)
    elif x.dim() > 2:
        # 其他高维情况，展平所有非 batch 维度
        while x.dim() > 2:
            B = x.shape[0]
            D = x.shape[-1]
            x = x.reshape(-1, D)

    N, D = x.shape
    if N < 2:
        return x.new_tensor(0.0)

    # L2 归一化（确保在球面上）
    x = F.normalize(x, p=2, dim=-1)

    # 维度 Dropout: 随机 mask 部分维度，增加配对多样性
    if dim_dropout > 0 and x.requires_grad:
        mask = torch.empty_like(x).bernoulli_(1 - dim_dropout)
        if mask.sum() > 0:
            x_dropped = x * mask
            x_dropped = F.normalize(x_dropped, p=2, dim=-1)
        else:
            x_dropped = x
    else:
        x_dropped = x

    # All-Pairs: 使用 chunk 分批计算所有 i≠j 的 |u_i · u_j|
    chunk_size = min(1024, N)
    total_loss = x.new_tensor(0.0)
    n_pairs = 0

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_i = x_dropped[i_start:i_end]  # [chunk, D]
        sim = chunk_i @ x_dropped.T  # [chunk, N]
        # 排除对角线
        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            sim[local_i, global_i] = 0.0
        total_loss = total_loss + sim.abs().sum()
        n_pairs += (i_end - i_start) * N - (i_end - i_start)

    return total_loss / max(n_pairs, 1)


# ────────────────────────────────────────────
# 1c. Cyclic Shift Batch Uniformity (AEF 风格)
# ────────────────────────────────────────────

def batch_uniformity_cyclic_shift(embeddings: torch.Tensor) -> torch.Tensor:
    """Cyclic shift batch uniformity — AEF 原文风格.
    
    对 batch 中的 embeddings 做循环移位，最小化相邻样本的 dot product。
    比 all-pairs 更轻量，但要求 batch shuffle 良好。
    
    Args:
        embeddings: [B, D] 已 L2 归一化
    Returns:
        scalar loss, 越小越好
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    
    # 循环移位
    shifted = torch.roll(embeddings, shifts=1, dims=0)
    
    # 相邻样本的 dot product
    sim = (embeddings * shifted).sum(dim=-1)
    
    return sim.abs().mean()


# ────────────────────────────────────────────
# 2. Decorrelation Loss (Barlow Twins)
# ────────────────────────────────────────────

def decorrelation_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Barlow Twins 去相关损失: 惩罚维度间的相关性.

    计算 embedding 维度间的互相关矩阵，最小化非对角元素的平方和。
    同时鼓励对角元素接近 1（即每个维度有方差）。
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    # 标准化到零均值单位方差
    z = embeddings - embeddings.mean(dim=0)
    std = z.std(dim=0) + 1e-4
    z = z / std

    N = z.shape[0]
    c = (z.T @ z) / N  # 互相关矩阵 [D, D]

    D = c.shape[0]
    on_diag = (c.diagonal() - 1).pow(2).sum() / D
    off_diag = c.pow(2).sum() / D - c.diagonal().pow(2).sum() / D

    return on_diag + off_diag


# ────────────────────────────────────────────
# 3. Variance Regularizer (VICReg)
# ────────────────────────────────────────────

def variance_regularizer(embeddings: torch.Tensor, min_std: float = 1.0) -> torch.Tensor:
    """VICReg 风格方差正则: 约束每个维度的标准差不低于 min_std.

    使用 biased variance (unbiased=False)，与 VICReg 官方实现一致。
    当某维度方差坍缩（std < min_std）时产生惩罚，否则为 0。
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(min_std - std).mean()


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """VICReg covariance loss — 惩罚维度之间的相关性.

    计算 embedding 的协方差矩阵，惩罚非对角元素。
    与 L2 归一化完全兼容。
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    N = embeddings.shape[0]
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (emb_centered.T @ emb_centered) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()


# ────────────────────────────────────────────
# 4. Bottleneck Orthogonality Loss
# ────────────────────────────────────────────

def bottleneck_orthogonality_loss(weight: torch.Tensor) -> torch.Tensor:
    """Bottleneck 1x1 conv 权重正交约束.

    直接操作权重矩阵 W [out_dim, in_dim]，使行向量趋向正交。
    在坍缩时提供非零梯度（不依赖 embedding 激活值）。
    """
    W = weight.reshape(weight.shape[0], -1)  # [out_dim, in_dim]
    W_norm = F.normalize(W, p=2, dim=1)
    gram = W_norm @ W_norm.T  # [out_dim, out_dim], 对角线=1
    D = gram.shape[0]
    # 只惩罚非对角元素(维度间相关性)
    off_diag = (gram.pow(2).sum() - D) / (D * (D - 1))
    return off_diag


# ────────────────────────────────────────────
# 5. 标准损失函数
# ────────────────────────────────────────────

def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    target_loss_type: torch.Tensor | None = None,
    recon_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """掩码重建损失: L1 (连续) / CE (类别).
    
    Args:
        recon_mask: [H, W] 或 [B, H, W] — MAE风格，只对mask=1的区域计算loss
    """
    if target_mask.dim() == 1:
        target_mask = target_mask[:, None]

    # 简单的 L1 重建 (兼容原版)
    pixel_valid_mask = (~torch.isnan(target)).float()
    mask = target_mask[:, :, None, None, None].float() * pixel_valid_mask
    
    # V13-MAE: 应用重建掩码
    if recon_mask is not None:
        if recon_mask.dim() == 2:
            # [H, W] → [1, 1, 1, H, W]
            mask = mask * recon_mask[None, None, None, :, :]
        elif recon_mask.dim() == 3:
            # [B, H, W] → [B, 1, 1, H, W]
            mask = mask * recon_mask[:, None, None, :, :]
    
    diff = torch.abs(prediction - target) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff.sum() / denom


def consistency_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """教师-学生一致性损失: 1 - cosine_similarity."""
    teacher = F.normalize(teacher, dim=-1)
    student = F.normalize(student, dim=-1)
    return 0.5 * (1.0 - torch.sum(teacher * student, dim=-1)).mean()


def consistency_loss_spatial(teacher_map: torch.Tensor, student_map: torch.Tensor) -> torch.Tensor:
    """教师-学生 Spatial Map 一致性损失.
    
    Args:
        teacher_map: [B, D, H, W] — 必须是 L2-normalized（bottleneck 已做）
        student_map: [B, D, H, W] — 必须是 L2-normalized
    """
    # bottleneck 已做 L2 Norm，此处直接计算 cosine similarity，避免双重 normalize
    sim = torch.sum(teacher_map * student_map, dim=1)  # [B, H, W]
    
    # 1 - similarity = distance
    return 0.5 * (1.0 - sim).mean()


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """交叉熵分类损失."""
    return F.cross_entropy(logits, labels)


def pre_norm_uniformity_loss(features: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Pre-norm 方向性 uniformity: 在归一化后的 pre-norm 特征上计算.

    用于监控: 这个值应该和 raw_uniformity 一致 (如果训练正常).
    """
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    normed = F.normalize(features, p=2, dim=1)
    N = normed.shape[0]
    sq_pdist = torch.cdist(normed, normed, p=2).pow(2)
    pair_mask = torch.triu(torch.ones(N, N, device=normed.device, dtype=torch.bool), diagonal=1)
    sq_pairs = sq_pdist[pair_mask]
    return torch.logsumexp(-t * sq_pairs, dim=0) - math.log(sq_pairs.shape[0])


def directional_uniformity_loss(features: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """方向性 uniformity: 消除幅度后在球面上计算 uniformity.

    用于 encoder-level 的 uniformity 监控。
    """
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    # 消除幅度
    norms = features.norm(dim=1, keepdim=True).detach().clamp(min=1e-6)
    directions = features / norms
    # BN 放大微小方向差异
    mu = directions.mean(0, keepdim=True)
    std = directions.std(0, keepdim=True).clamp(min=1e-5)
    amplified = (directions - mu) / std
    amplified = F.normalize(amplified, p=2, dim=1)
    # uniformity
    N = amplified.shape[0]
    sq_pdist = torch.cdist(amplified, amplified, p=2).pow(2)
    pair_mask = torch.triu(torch.ones(N, N, device=amplified.device, dtype=torch.bool), diagonal=1)
    sq_pairs = sq_pdist[pair_mask]
    return torch.logsumexp(-t * sq_pairs, dim=0) - math.log(sq_pairs.shape[0])


def temporal_contrastive_loss(
    emb_w1: torch.Tensor,
    emb_w2: torch.Tensor,
    temperature: float = 0.05,
    target_margin: float = 0.2,
) -> torch.Tensor:
    """强时序对比损失 — 基于 global mean embedding 的 Hinge Loss.

    关键改进 (Rev B, 2025-04-15):
    - target_margin = 0.2: 要求不同窗口的 cos_sim ≤ 0.2 (角度 ≥ 78°)
      之前 -0.5 (120°) 太极端，难以达到
    - temperature = 0.05: 更尖锐的对比
    - Global mean: 梯度集中，不稀释空间信号
    """
    if emb_w1.shape[0] < 1:
        return emb_w1.new_tensor(0.0)

    B, D, H, W = emb_w1.shape
    # Global mean over spatial dimensions
    flat_w1 = emb_w1.reshape(B, D, -1).mean(dim=-1)  # [B, D]
    flat_w2 = emb_w2.reshape(B, D, -1).mean(dim=-1)
    flat_w1 = F.normalize(flat_w1, p=2, dim=-1)
    flat_w2 = F.normalize(flat_w2, p=2, dim=-1)

    cos_sim = (flat_w1 * flat_w2).sum(-1)  # [B]

    # Hinge loss: 只惩罚 cos_sim > target_margin 的部分
    loss = F.relu(cos_sim - target_margin).mean() / temperature
    return loss


def l2_temporal_contrastive_loss(
    emb_w1: torch.Tensor,
    emb_w2: torch.Tensor,
    temperature: float = 0.05,
    target_margin: float = 0.2,
) -> torch.Tensor:
    """L2 空间时序对比损失 — 直接在 L2-normalized embedding map 上计算.

    与 temporal_contrastive_loss 的区别:
    - 输入已经是 L2-normalized (如 encode_dual_window 返回的 emb_w1/emb_w2)
    - 确保推理时的 embedding 也有时间敏感性
    """
    if emb_w1.shape[0] < 1:
        return emb_w1.new_tensor(0.0)

    B, D, H, W = emb_w1.shape
    # 已经在 L2 空间，直接 global mean
    flat_w1 = emb_w1.reshape(B, D, -1).mean(dim=-1)  # [B, D]
    flat_w2 = emb_w2.reshape(B, D, -1).mean(dim=-1)

    cos_sim = (flat_w1 * flat_w2).sum(-1)  # [B]

    loss = F.relu(cos_sim - target_margin).mean() / temperature
    return loss


def temporal_cosine_pixel_loss(
    emb_w1: torch.Tensor,
    emb_w2: torch.Tensor,
    temperature: float = 0.05,
    use_spatial_weight: bool = False,
) -> torch.Tensor:
    """Pixel-level cosine temporal loss — 空间加权版本 (Round 9 fix).

    核心改进:
    - use_spatial_weight=True: 基于 embedding 自身差异计算自适应空间权重
    - 变化区域（差异大）被推得更远，未变化区域（差异小）保持相似
    - use_spatial_weight=False: 回退到全局平均（与之前兼容）
    """
    if emb_w1.shape[0] < 1:
        return emb_w1.new_tensor(0.0)

    B, D, H, W = emb_w1.shape
    # 每个像素位置独立 normalize: [B, D, H, W]
    f1 = F.normalize(emb_w1, p=2, dim=1)
    f2 = F.normalize(emb_w2, p=2, dim=1)
    # 逐像素 cos_sim: [B, H, W]
    cos_map = (f1 * f2).sum(dim=1)

    if use_spatial_weight:
        # ★ Round 9: 自适应空间加权
        # 基于 embedding 差异计算变化掩码
        with torch.no_grad():
            # 计算像素级 embedding 差异幅度 [B, H, W]
            diff = torch.abs(emb_w1 - emb_w2).mean(dim=1)
            # 归一化到 [0, 1]
            diff_min = diff.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
            diff_max = diff.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
            w = (diff - diff_min) / (diff_max - diff_min + 1e-8)
            
            # 变化区域（w 高）需要 cos_sim -> -1（正交）
            # 未变化区域（w 低）需要 cos_sim -> 1（相似）
            target_sim = 1.0 - 2.0 * w  # [B, H, W]
        
        # 加权 hinge loss: 变化区域需要更大的差异
        loss = F.relu(cos_map - target_sim).mean() / temperature
    else:
        # 回退到全局平均（旧行为）
        loss = cos_map.mean() / temperature
    return loss


def temporal_info_nce_loss(emb_w1: torch.Tensor, emb_w2: torch.Tensor,
                            temperature: float = 0.1) -> torch.Tensor:
    """Anti-Diagonal InfoNCE — 核心修复版本 (Global).

    关键洞察 (CACo / Change Detection):
    同一地点、不同时间的两个窗口应该产生**不同**的 embedding.
    因此我们把对角线 (i,i) 当作"负样本".
    """
    B, D, H, W = emb_w1.shape
    flat_w1 = F.adaptive_avg_pool2d(emb_w1, 1).view(B, D)  # [B, D]
    flat_w2 = F.adaptive_avg_pool2d(emb_w2, 1).view(B, D)

    flat_w1 = F.normalize(flat_w1, p=2, dim=-1)
    flat_w2 = F.normalize(flat_w2, p=2, dim=-1)

    # 当 batch_size=1 时，InfoNCE 退化为 0，改用 cosine similarity 替代
    if B <= 1:
        sim = (flat_w1 * flat_w2).sum(dim=-1)  # [-1, 1]，越高表示越相似
        return (1.0 - sim).mean()  # 希望相似度低 -> loss 高

    logits = (flat_w1 @ flat_w2.T) / temperature  # [B, B]
    labels = torch.arange(B, device=emb_w1.device)
    return F.cross_entropy(-logits, labels)


def pixel_temporal_info_nce_loss(
    emb_w1: torch.Tensor,
    emb_w2: torch.Tensor,
    temperature: float = 0.1,
    num_samples: int = 16,
) -> torch.Tensor:
    """Anti-Diagonal InfoNCE — 像素级版本.

    从空间 embedding map 中随机采样若干像素位置,
    对每个空间位置独立计算 Anti-Diagonal InfoNCE.

    这强制模型不能靠"全局时间偏移"作弊:
    它必须在每个局部空间位置上都让 w1 和 w2 产生差异,
    从而自然产生"某些区域差异大、某些区域差异小"的空间模式.
    """
    B, D, H, W = emb_w1.shape
    # [B, D, H, W] -> [B, H*W, D]
    f1 = emb_w1.reshape(B, D, -1).permute(0, 2, 1)
    f2 = emb_w2.reshape(B, D, -1).permute(0, 2, 1)

    hw = H * W
    if num_samples < hw:
        indices = torch.randperm(hw, device=emb_w1.device)[:num_samples]
        f1 = f1[:, indices, :]
        f2 = f2[:, indices, :]
    else:
        indices = torch.arange(hw, device=emb_w1.device)

    f1 = F.normalize(f1, p=2, dim=-1)
    f2 = F.normalize(f2, p=2, dim=-1)

    labels = torch.arange(B, device=emb_w1.device)
    loss = 0.0
    N = f1.shape[1]
    for i in range(N):
        logits = (f1[:, i, :] @ f2[:, i, :].T) / temperature
        loss += F.cross_entropy(-logits, labels)
    return loss / max(N, 1)


def gap_aware_temporal_cosine_loss(
    pre_w1: torch.Tensor,
    pre_w2: torch.Tensor,
    time_gap_ms: torch.Tensor,
    max_gap_ms: float = 6 * 30 * 24 * 3600 * 1000,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Gap-aware pixel-level temporal cosine loss.

    核心改进 (vs temporal_cosine_pixel_loss):
    - 根据时间 gap 大小设定不同的 target cos_sim
    - gap=0 → target=1 (相似), gap=max → target=-1 (相反)
    - 小间隔不过度推开, 大间隔允许充分差异
    - 保留渐变变化检测能力
    """
    if pre_w1.shape[0] < 1:
        return pre_w1.new_tensor(0.0)

    B, D, H, W = pre_w1.shape
    f1 = F.normalize(pre_w1, p=2, dim=1)
    f2 = F.normalize(pre_w2, p=2, dim=1)
    cos_map = (f1 * f2).sum(dim=1)  # [B, H, W]

    # 每个 batch 的 target: [B] → [B, 1, 1]
    time_gap_norm = torch.clamp(time_gap_ms / max_gap_ms, 0.0, 1.0)  # [B]
    target = 1.0 - 2.0 * time_gap_norm  # gap=0→1, gap=max→-1
    target_map = target.view(B, 1, 1).expand(B, H, W)

    loss = ((cos_map - target_map).pow(2)).mean() / temperature
    return loss


def temporal_magnitude_loss(
    pre_w1: torch.Tensor,
    pre_w2: torch.Tensor,
    time_gap_ms: torch.Tensor,
    max_gap_ms: float = 6 * 30 * 24 * 3600 * 1000,  # 默认 6 个月
    margin: float = 0.1,
) -> torch.Tensor:
    """时序幅度损失 — 约束 embedding distance ≤ time_gap_norm + margin.

    核心思想 (CaCo / 变化检测):
    同一地点不同时间的 embedding 距离应与真实时间间隔正相关。
    但只设上限约束 (hinge)，因为实际地表变化不是严格线性的。

    Args:
        pre_w1: [B, D, H, W] pre-norm embedding (window 1)
        pre_w2: [B, D, H, W] pre-norm embedding (window 2)
        time_gap_ms: [B] 两个窗口中心点之间的时间间隔 (毫秒)
        max_gap_ms: 归一化分母，将时间 gap 映射到 [0, 1]
        margin: 允许的松弛量

    Returns:
        scalar loss
    """
    if pre_w1.shape[0] < 1:
        return pre_w1.new_tensor(0.0)

    B, D, H, W = pre_w1.shape

    # Global mean over spatial dimensions -> [B, D]
    flat_w1 = pre_w1.reshape(B, D, -1).mean(dim=-1)
    flat_w2 = pre_w2.reshape(B, D, -1).mean(dim=-1)

    # Cosine similarity in pre-norm space
    flat_w1_norm = F.normalize(flat_w1, p=2, dim=-1)
    flat_w2_norm = F.normalize(flat_w2, p=2, dim=-1)
    cos_sim = (flat_w1_norm * flat_w2_norm).sum(dim=-1)  # [B]

    # Distance in [0, 1]: (1 - cos_sim) / 2
    dist = (1.0 - cos_sim) * 0.5  # [B]

    # Normalized time gap — 不 clamp，让长间隔的 threshold 可以 > 1 (自然不触发)
    time_gap_norm = time_gap_ms / max_gap_ms  # [B]

    # Hinge loss: 只惩罚 dist > time_gap_norm + margin 的部分
    loss = F.relu(dist - time_gap_norm - margin).mean()
    return loss


# ────────────────────────────────────────────
# 6b. Coding Rate Loss (MCR²)
# ────────────────────────────────────────────

def coding_rate_loss(embeddings: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Maximal Coding Rate Reduction (MCR²) Loss.

    基于率失真理论，最大化整体编码率：
    R(Z; eps) = 0.5 * log det(I + d/(N*eps²) * ZZ^T)

    对低秩构型（坍缩）施加无限惩罚，比VICReg的hinge variance更严格。

    Args:
        embeddings: [N, D] pre-norm embedding
        eps: 正则化参数，防止奇异

    Returns:
        scalar loss, 负值（最大化编码率）
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    N, D = embeddings.shape
    # 中心化
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    # 协方差矩阵
    cov = (emb_centered.T @ emb_centered) / N

    I = torch.eye(D, device=embeddings.device, dtype=embeddings.dtype)
    # 编码率
    logdet = torch.logdet(I + (D / (N * eps**2 + 1e-8)) * cov + 1e-6 * I)

    return -0.5 * logdet


# ────────────────────────────────────────────
# 7. Change Consistency Loss (V10)
# ────────────────────────────────────────────

def change_consistency_loss(
    change_score: torch.Tensor,
    img_w1: torch.Tensor,
    img_w2: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """变化一致性损失 — 让 bottleneck 预测的 change_score 与图像差异对齐.

    V10 核心: bottleneck 产生的 change_score [B, 1, H, W] 应该反映真实的图像变化。
    不需要标注，直接用 S2 图像的像素差异作为弱监督。

    Args:
        change_score: [B, 1, H, W] bottleneck 预测的变化概率 (sigmoid 后 0~1)
        img_w1: [B, C, H, W] 窗口1的代表图像
        img_w2: [B, C, H, W] 窗口2的代表图像
        threshold: 图像差异阈值，低于此值视为无变化

    Returns:
        scalar loss
    """
    if change_score is None or change_score.shape[0] < 1:
        return img_w1.new_tensor(0.0) if img_w1 is not None else torch.tensor(0.0)

    # 图像差异 (通道平均)
    img_diff = torch.abs(img_w1.mean(dim=1) - img_w2.mean(dim=1))  # [B, H, W]

    # 归一化到 [0, 1] (per-batch)
    img_diff_max = img_diff.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    img_diff_norm = img_diff / img_diff_max  # [B, H, W]

    # 构建变化掩码: 只监督"确实有图像变化"的区域
    change_mask = (img_diff_norm > threshold).float()
    if change_mask.sum() < 1.0:
        return change_score.new_tensor(0.0)

    # BCE Loss: change_score 应该预测图像差异
    # 用 img_diff_norm 作为 target (软标签，不是二值)
    # ★ NPU bf16 不支持 BCE，fallback 到 fp32
    loss = F.binary_cross_entropy(
        change_score.squeeze(1).float(), img_diff_norm.float(), reduction='none'
    )
    loss = (loss * change_mask).sum() / change_mask.sum().clamp(min=1.0)
    return loss


# ────────────────────────────────────────────
# 6. Pixel Change Supervision Loss (V9.5)
# ────────────────────────────────────────────

def pixel_change_supervision_loss(
    pre_w1: torch.Tensor,
    pre_w2: torch.Tensor,
    img_w1: torch.Tensor,
    img_w2: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """像素级变化监督 — 利用两期图像差异指导 embedding 差异.

    核心思想:
    - 两期图像差异大的区域 → embedding 差异也应大
    - 两期图像差异小的区域 → embedding 差异也应小
    - 不需要人工标注，直接用 S2 图像的像素差异作为弱监督信号

    Args:
        pre_w1: [B, D, H, W] pre-norm embedding map (window 1)
        pre_w2: [B, D, H, W] pre-norm embedding map (window 2)
        img_w1: [B, C_img, H, W] 窗口1的代表图像 (已归一化)
        img_w2: [B, C_img, H, W] 窗口2的代表图像 (已归一化)
        threshold: 图像差异阈值，低于此值视为"无变化"

    Returns:
        scalar loss
    """
    if pre_w1.shape[0] < 1 or img_w1 is None or img_w2 is None:
        return pre_w1.new_tensor(0.0)

    B, D, H, W = pre_w1.shape

    # 1. 计算 embedding 差异 (cosine distance → [0, 1])
    f1 = F.normalize(pre_w1, p=2, dim=1)
    f2 = F.normalize(pre_w2, p=2, dim=1)
    emb_cos = (f1 * f2).sum(dim=1)  # [B, H, W], range [-1, 1]
    emb_diff = (1.0 - emb_cos) * 0.5  # [B, H, W], range [0, 1]

    # 2. 计算图像差异 (逐像素绝对差)
    # img: [B, C, H, W] → 在通道维度求平均
    img_diff = torch.abs(img_w1.mean(dim=1) - img_w2.mean(dim=1))  # [B, H, W]

    # 3. 归一化到 [0, 1] (per-batch)
    img_diff_max = img_diff.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    img_diff_norm = img_diff / img_diff_max  # [B, H, W]

    # 4. 构建变化掩码: 只监督"确实有图像变化"的区域
    # 同时排除完全无变化的区域（避免强迫无变化区域产生虚假差异）
    change_mask = (img_diff_norm > threshold).float()

    # 如果整个 batch 都无显著变化，跳过
    if change_mask.sum() < 1.0:
        return pre_w1.new_tensor(0.0)

    # 5. 加权 MSE: 图像变化大的区域权重更高
    # weight = 1 + img_diff_norm → 变化区域权重 1~2x
    weights = 1.0 + img_diff_norm

    loss = (weights * change_mask * (emb_diff - img_diff_norm).pow(2)).sum()
    loss = loss / change_mask.sum().clamp(min=1.0)
    return loss
