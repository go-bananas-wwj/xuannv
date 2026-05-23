# 玄女底座 Embedding 时间判别力提升方案 —— 深度调研报告

> 调研时间: 2025-05-07
> 核心问题: V5 raw embedding AUC = 52.8% (接近随机)，embedding 严重坍缩
> 目标: raw embedding AUC > 70% (接近 AEF unsupervised 71.3%)

---

## 一、问题根因诊断

### 1.1 当前状态

| 指标 | V5 实测 | AEF (参考) | 差距 |
|------|---------|-----------|------|
| Raw Embedding AUC | 52.8% | 71.3% | -18.5% |
| Uniformity | -0.50 | ~-3.0 (推测) | 严重不足 |
| Mean Cos Sim | 0.765 | ~0.2 | 严重坍缩 |
| Separation | 0.0017 | >0.05 | 无分离 |

### 1.2 为什么时间判别力缺失？

通过代码分析和文献对比，识别出 5 个根本原因：

**原因 1: Temporal Contrastive Loss 设计缺陷**
- V5 使用双窗口 hinge loss (global mean)，但窗口时间 gap 可能不够大
- 没有 pixel-level 的时序约束，变化信息在 global pooling 中被平均掉了
- Loss weight 可能被 reconstruction loss 淹没

**原因 2: Uniformity Loss 失效**
- V5 使用 `raw_uniformity_loss` (t=2/D 自适应)，但 t 值过大导致梯度消失
- 没有 batch-level uniformity 约束（AEF 的核心机制）
- Skip L2 norm 训练导致 pre-norm 空间的 uniformity 无法传递到 L2 norm 后的推理空间

**原因 3: Reconstruction Loss 主导**
- S2/S1/Landsat 重建任务要求 embedding 编码"图像内容"
- 模型优先学习静态内容，时间变化被当作噪声忽略
- 重建损失权重 >> 时序损失权重

**原因 4: 数据局限性**
- 仅哈尔滨 424 个 patch，变化样本稀疏
- 缺乏大时间 gap (如 1-2 年) 的标注变化对
- 训练数据中"变化"像素占比极低 (<1%)

**原因 5: 缺少 Teacher-Student Consistency**
- AEF 使用 dual-network consistency 确保鲁棒性
- V5 没有这种机制，embedding 对输入 dropout 敏感

---

## 二、解决方案全景图

基于文献调研，解决方案分为 5 个层面，共 14 项具体措施：

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 训练目标重构 (Loss Engineering)                    │
│  Layer 2: 防坍缩机制 (Anti-Collapse)                         │
│  Layer 3: 数据增强 (Data Augmentation)                       │
│  Layer 4: 架构改进 (Architecture)                            │
│  Layer 5: 训练策略 (Training Strategy)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、Layer 1: 训练目标重构

### 3.1 引入 Batch Uniformity Objective (AEF 核心机制)

**原理**: AEF 的论文明确说明，batch uniformity 是防止坍缩的关键：

```
BatchUniformity = sum(|ui · ui'|)
```

其中 `ui'` 是 `ui` 在 batch 维度上循环移位后的向量。最小化这个值迫使 batch 中的 embedding 互相正交，从而均匀分布在球面上。

**AEF 配置**: weight = 0.05

**对 XuanNv 的改进**:
```python
def batch_uniformity_loss(embedding_map: torch.Tensor) -> torch.Tensor:
    """
    Args:
        embedding_map: [B, D, H, W] L2-normalized embedding map
    Returns:
        loss: scalar
    """
    B, D, H, W = embedding_map.shape
    # 将空间维度 flatten，每个像素是一个 embedding 向量
    vectors = embedding_map.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
    
    # Batch 内循环移位
    shifted = torch.roll(vectors, shifts=1, dims=0)
    
    # 计算绝对值点积，最小化 = 迫使正交
    uniformity = torch.mean(torch.abs(torch.sum(vectors * shifted, dim=1)))
    return uniformity
```

**预期效果**: uniformity 从 -0.50 提升到 -2.0 以上

### 3.2 Pixel-Level Temporal Contrastive Loss

**问题**: V5 的 temporal contrastive loss 在 global mean embedding 上计算，空间信息丢失。

**解决方案**: 在 pixel level 计算时序对比损失

```python
def pixel_temporal_contrastive_loss(emb_t1, emb_t2, gap_months):
    """
    Args:
        emb_t1, emb_t2: [B, D, H, W] 两个时间窗口的 embedding
        gap_months: float 时间间隔（月）
    """
    B, D, H, W = emb_t1.shape
    
    # 将空间维度 flatten
    f1 = emb_t1.reshape(B, D, -1)  # [B, D, H*W]
    f2 = emb_t2.reshape(B, D, -1)
    
    # 对每个像素位置计算 cosine similarity
    cos_sim = torch.sum(f1 * f2, dim=1)  # [B, H*W]
    
    # Gap-aware target: 时间间隔越大，期望的相似度越低
    # 1个月 -> target_sim = 0.9
    # 6个月 -> target_sim = 0.5
    # 12个月 -> target_sim = 0.2
    target_sim = max(0.1, 1.0 - gap_months / 15.0)
    
    # MSE loss 迫使相似度接近 target
    loss = F.mse_loss(cos_sim, torch.full_like(cos_sim, target_sim))
    return loss
```

**预期效果**: 迫使 embedding 在时间轴上产生可预测的变化

### 3.3 Anti-Diagonal InfoNCE (V6 已有，需增强)

**原理**: 对同一空间位置的不同时间 embedding，使用 InfoNCE 做对比学习。

**改进版本**:
```python
def anti_diagonal_temporal_infonce(emb_seq, temperature=0.1):
    """
    Args:
        emb_seq: [B, T, D, H, W] 多个时间点的 embedding
    """
    B, T, D, H, W = emb_seq.shape
    
    # 对每个空间位置 (h, w)，构建时间序列的对比学习
    emb_seq = emb_seq.permute(0, 3, 4, 1, 2)  # [B, H, W, T, D]
    
    loss = 0
    for t in range(T - 1):
        # 当前时间 t 和下一时间 t+1 是正样本对
        anchor = emb_seq[:, :, :, t, :]    # [B, H, W, D]
        positive = emb_seq[:, :, :, t+1, :] # [B, H, W, D]
        
        # 负样本：其他空间位置的同一时间点
        # [B, H, W, D] vs [B, H, W, D]
        sim_pos = torch.sum(anchor * positive, dim=-1) / temperature  # [B, H, W]
        
        # 负样本：所有其他空间位置
        anchor_flat = anchor.reshape(B, H*W, D)
        pos_flat = positive.reshape(B, H*W, D)
        
        sim_matrix = torch.bmm(anchor_flat, pos_flat.transpose(1, 2)) / temperature  # [B, HW, HW]
        
        # InfoNCE: 对角线为正样本
        logits = sim_matrix  # [B, HW, HW]
        labels = torch.arange(H*W, device=logits.device).unsqueeze(0).expand(B, -1)
        
        loss += F.cross_entropy(logits, labels)
    
    return loss / (T - 1)
```

### 3.4 显式变化监督 (Synthetic Change Supervision)

**灵感来源**: Changen2 使用扩散模型合成变化数据，XuanNv 可以简化实现。

**方案**: 在训练时，对同一个 patch 的两个不同时间点，构造"伪变化对"：
1. 对 t1 的图像做随机遮挡/替换，生成"变化后"的伪图像
2. 让模型学习：真实变化区域 -> embedding 差异大，未变化区域 -> embedding 差异小

```python
def synthetic_change_supervision_loss(model, patch_t1, patch_t2):
    """
    使用图像差异作为弱监督信号
    """
    # 计算图像层面的变化强度（不需要标注）
    img_diff = torch.abs(patch_t1 - patch_t2).mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # 获取 embedding
    emb_t1 = model.encode(patch_t1)
    emb_t2 = model.encode(patch_t2)
    
    # 计算 embedding 差异
    emb_diff = 1 - F.cosine_similarity(emb_t1, emb_t2, dim=1)  # [B, H, W]
    
    # 弱监督：embedding 差异应与图像差异正相关
    loss = F.mse_loss(emb_diff, img_diff.squeeze(1))
    return loss
```

---

## 四、Layer 2: 防坍缩机制

### 4.1 改进 Uniformity Loss (从 V5 的 raw_uniformity 到 batch_uniformity)

**V5 的问题**:
```python
# V5 的 raw_uniformity_loss
pairwise_dist = torch.cdist(vectors, vectors, p=2)
uniformity = torch.log(torch.mean(torch.exp(-t * pairwise_dist)))
# t = 2/D = 2/64 = 0.031，过小导致梯度消失
```

**改进方案 A**: 使用固定 t=2.0（而非自适应 t=2/D）
```python
def improved_uniformity_loss(vectors, t=2.0):
    """固定 t=2.0，确保梯度非零"""
    pairwise_dist = torch.cdist(vectors, vectors, p=2)
    return torch.log(torch.mean(torch.exp(-t * pairwise_dist)))
```

**改进方案 B**: 引入 CLOA (Contrastive Learning with Orthonormal Anchors)

来自论文 "CLOA: Contrastive Learning with Orthonormal Anchors" (2024):

```python
def cloa_loss(embeddings, labels, num_classes):
    """
    使用正交锚点防止坍缩
    Args:
        embeddings: [N, D]
        labels: [N] 类别标签（可用伪标签或地物类型）
    """
    # 初始化正交锚点
    anchors = torch.eye(num_classes, embeddings.shape[1], device=embeddings.device)
    
    # 回归损失：将 embedding 拉向对应锚点
    loss = 0
    for i in range(len(embeddings)):
        anchor = anchors[labels[i]]
        sim = F.cosine_similarity(embeddings[i:i+1], anchor.unsqueeze(0))
        loss += (1 - sim)
    
    return loss / len(embeddings)
```

**注意**: CLOA 需要类别标签。XuanNv 可以使用 WorldCover/DynamicWorld 作为伪标签。

### 4.2 引入 VICReg 方差-协方差约束

来自论文 "VICReg: Variance-Invariance-Covariance Regularization" (2022):

```python
def vicreg_loss(z1, z2):
    """
    Args:
        z1, z2: [B, D] 两个 augmented view 的 embedding
    """
    # Invariance: 正样本对靠近
    inv_loss = F.mse_loss(z1, z2)
    
    # Variance: 每个维度有方差
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    
    # Covariance: 不同维度去相关
    def off_diagonal(cov):
        n = cov.shape[0]
        return cov.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
    cov_loss = off_diagonal(cov_z1).pow_(2).sum() + off_diagonal(cov_z2).pow_(2).sum()
    
    loss = inv_loss + var_loss + cov_loss
    return loss
```

**预期效果**: 防止维度坍缩，保持 embedding 的多样性

### 4.3 使用 Whitening / PCA 预处理

**原理**: 在计算 temporal loss 之前，对 embedding 做白化，消除相关性。

```python
def whitening_loss(embedding_map):
    """
    鼓励 embedding 维度之间去相关
    """
    B, D, H, W = embedding_map.shape
    vectors = embedding_map.permute(0, 2, 3, 1).reshape(-1, D)
    
    # 计算协方差矩阵
    cov = torch.cov(vectors.T)
    
    # 惩罚非对角线元素
    off_diag = cov - torch.diag(torch.diag(cov))
    loss = torch.sum(off_diag ** 2)
    
    return loss
```

---

## 五、Layer 3: 数据增强

### 5.1 合成变化数据 (Inspired by Changen2)

**方案**: 不训练扩散模型，使用简单规则生成变化对：

1. **Object Removal**: 用 SAM3 分割建筑物，随机擦除 -> 模拟拆除
2. **Object Addition**: 复制邻近 patch 的建筑物 -> 模拟新建
3. **Spectral Shift**: 对农田做颜色抖动 -> 模拟作物变化
4. **Seasonal Mixing**: 不同季节的同一地点 -> 自然变化

```python
def generate_synthetic_change_pair(image, mask_objects):
    """
    Args:
        image: [C, H, W] 原始图像
        mask_objects: [K, H, W] K 个对象掩码
    """
    changed_image = image.clone()
    change_mask = torch.zeros(H, W)
    
    # 随机选择 1-3 个对象进行"变化"
    n_changes = random.randint(1, 3)
    selected = random.sample(range(K), n_changes)
    
    for idx in selected:
        obj_mask = mask_objects[idx]
        change_type = random.choice(['remove', 'shift', 'replace'])
        
        if change_type == 'remove':
            # 用背景均值填充
            changed_image[:, obj_mask] = image.mean(dim=(1,2), keepdim=True)
            change_mask[obj_mask] = 1
        elif change_type == 'shift':
            # 平移对象
            dx, dy = random.randint(-5, 5), random.randint(-5, 5)
            shifted = torch.roll(image * obj_mask, shifts=(dx, dy), dims=(1,2))
            changed_image = changed_image * (~obj_mask) + shifted
            change_mask[obj_mask] = 1
    
    return changed_image, change_mask
```

### 5.2 大时间 Gap 采样

**问题**: V5 的双窗口可能时间 gap 太小（如 1-2 个月），变化不明显。

**改进**: 强制采样大 gap 对：
- 短 gap (1-3 个月): 季节性变化
- 中 gap (3-6 个月): 农作物周期
- 长 gap (6-18 个月): 建筑/道路变化

```python
def sample_temporal_pair(available_dates, strategy='mixed'):
    """
    策略:
    - mixed: 30% 短 gap, 40% 中 gap, 30% 长 gap
    - long_only: 只采样 >6 个月的 gap
    """
    if strategy == 'long_only':
        # 只选择时间间隔 > 6 个月的对
        valid_pairs = [(d1, d2) for d1 in available_dates for d2 in available_dates 
                       if abs(d1 - d2) > 6 * 30 * 86400]
    
    return random.choice(valid_pairs)
```

### 5.3 引入外部变化检测数据集做预训练

**可用数据集**:
- LEVIR-CD: 637 对建筑变化
- WHU-CD: 两期航空影像
- SECOND: 语义变化检测
- S2Looking: 侧视建筑变化

**方案**: 在这些数据集上做阶段性预训练，学习"变化表示"，再迁移到 XuanNv。

---

## 六、Layer 4: 架构改进

### 6.1 引入 Teacher-Student Consistency (AEF 核心)

**AEF 的架构**:
- Teacher: 处理完整输入序列
- Student: 处理 dropout 后的序列（随机丢失 30-50% 的帧）
- Loss: `1 - (teacher_emb · student_emb) / 2`

**对 XuanNv 的改进**:
```python
class TeacherStudentConsistency(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.teacher = model
        self.student = copy.deepcopy(model)
        
        # Teacher 不更新梯度
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # EMA 更新 teacher
        self.ema_decay = 0.999
    
    def forward(self, batch):
        # Teacher: 完整输入
        with torch.no_grad():
            teacher_emb = self.teacher(batch)
        
        # Student: dropout 后的输入
        student_batch = self.dropout_frames(batch, ratio=0.3)
        student_emb = self.student(student_batch)
        
        # Consistency loss
        consistency = (1 - F.cosine_similarity(teacher_emb, student_emb)) / 2
        
        return consistency.mean()
    
    @torch.no_grad()
    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = self.ema_decay * t_param.data + (1 - self.ema_decay) * s_param.data
```

**预期效果**: embedding 对输入缺失鲁棒，减少过拟合静态内容

### 6.2 引入 Difference Module (MambaCD 思路)

来自论文 "M-CD: A Mamba-based Siamese Network for Remote Sensing Change Detection":M-CD 使用 Difference Module (DM) 显式学习两个时间的差异：

```python
class DifferenceModule(nn.Module):
    """
    显式建模时间差异
    """
    def __init__(self, dim):
        super().__init__()
        self.linear_pre = nn.Linear(dim, dim)
        self.linear_post = nn.Linear(dim, dim)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, pre_features, post_features):
        """
        Args:
            pre_features: [B, D, H, W]
            post_features: [B, D, H, W]
        """
        B, D, H, W = pre_features.shape
        
        # 线性变换
        pre = self.linear_pre(pre_features.permute(0,2,3,1)).permute(0,3,1,2)
        post = self.linear_post(post_features.permute(0,2,3,1)).permute(0,3,1,2)
        
        # 深度可分离卷积
        pre_flat = pre.reshape(B, D, -1)  # [B, D, HW]
        post_flat = post.reshape(B, D, -1)
        
        pre_conv = self.dw_conv(pre_flat).reshape(B, D, H, W)
        post_conv = self.dw_conv(post_flat).reshape(B, D, H, W)
        
        # 差异特征
        diff = pre_conv - post_conv
        diff = self.norm(diff.permute(0,2,3,1)).permute(0,3,1,2)
        
        return diff
```

**集成到 XuanNv**: 在 STP Encoder 后添加 Difference Module，输出差异特征作为 temporal embedding。

### 6.3 改进 VMF Bottleneck

**AEF 的配置**: κ = 8000 (固定 concentration)
**V5 的问题**: skip_l2_training 可能导致 pre-norm 和 post-norm 不一致

**改进方案**:
1. 训练时也不完全 skip L2，而是使用 soft L2:
   ```python
   embedding = pre_norm / (torch.norm(pre_norm, dim=-1, keepdim=True) + 1e-6)
   # 而不是完全 skip
   ```
2. 在 pre-norm 空间计算 temporal loss，在 L2-norm 空间计算 uniformity loss

---

## 七、Layer 5: 训练策略

### 7.1 课程学习 (Curriculum Learning)

**阶段 1 (Epoch 0-50)**: 重建为主
- 重建 loss weight = 1.0
- Temporal loss weight = 0.1
- 目标: 让模型先学会提取内容特征

**阶段 2 (Epoch 50-150)**: 引入时间
- 重建 loss weight = 0.5
- Temporal loss weight = 0.5
- 目标: 学习时间变化

**阶段 3 (Epoch 150-300)**: 强化时间判别力
- 重建 loss weight = 0.3
- Temporal loss weight = 0.7
- 目标: 让 temporal 信号主导

### 7.2 硬负样本挖掘 (Hard Negative Mining)

**原理**: 在 temporal contrastive learning 中，大部分负样本太容易区分，模型学不到东西。

**方案**: 只保留"困难"的负样本：
```python
def hard_negative_temporal_loss(emb_t1, emb_t2, neg_embs, margin=0.5):
    """
    Args:
        emb_t1: [D] anchor
        emb_t2: [D] positive (同一时间不同 augment)
        neg_embs: [N, D] 负样本
    """
    # 正样本距离
    pos_dist = 1 - F.cosine_similarity(emb_t1.unsqueeze(0), emb_t2.unsqueeze(0))
    
    # 计算所有负样本距离
    neg_dists = 1 - F.cosine_similarity(emb_t1.unsqueeze(0), neg_embs, dim=1)
    
    # 只保留"困难"负样本（距离 < pos_dist + margin）
    hard_negs = neg_dists[neg_dists < pos_dist + margin]
    
    if len(hard_negs) > 0:
        loss = torch.mean(F.relu(pos_dist - hard_negs + margin))
    else:
        loss = torch.tensor(0.0, device=emb_t1.device)
    
    return loss
```

### 7.3 混合精度与梯度裁剪

**防止训练不稳定**:
```python
# 使用 bfloat16 而非 float16（更稳定）
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(batch)

# 梯度裁剪
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 八、评估验证方案

### 8.1 实时监控指标

训练时应关注：

| 指标 | 健康范围 | 当前 V5 | V6 目标 |
|------|---------|---------|---------|
| `uniformity` | -3.5 ~ -1.5 | -0.50 | < -2.0 |
| `mean_cos_sim` | < 0.3 | 0.765 | < 0.4 |
| `temporal_separation` | > 0.05 | 0.0017 | > 0.05 |
| `raw_auc` | > 0.60 | 0.528 | > 0.60 |

### 8.2 阶段性验证

每 20 epoch 运行一次：
1. Embedding space diagnosis (uniformity, cos_sim)
2. Raw embedding AUC (20 annotated patches)
3. Downstream linear probe (WorldCover, JRC Water)

---

## 九、实施路线图

### Phase 1: 快速修复 (2-3 周)

| 任务 | 优先级 | 预期收益 |
|------|--------|---------|
| 引入 Batch Uniformity Loss | 🔴 最高 | uniformity -> -2.0 |
| 修复坐标 bug | 🔴 最高 | 评估准确 |
| Pixel-Level Temporal Loss | 🔴 最高 | raw AUC -> 0.55-0.60 |
| 改进 Uniformity t 参数 | 🟡 高 | 防止梯度消失 |

### Phase 2: 深度优化 (4-6 周)

| 任务 | 优先级 | 预期收益 |
|------|--------|---------|
| Teacher-Student Consistency | 🟡 高 | 鲁棒性提升 |
| VICReg 方差-协方差约束 | 🟡 高 | 防止维度坍缩 |
| 合成变化数据 | 🟡 高 | 增加变化样本 |
| 大时间 Gap 采样 | 🟢 中 | 增强时间判别力 |

### Phase 3: 架构升级 (6-8 周)

| 任务 | 优先级 | 预期收益 |
|------|--------|---------|
| Difference Module | 🟢 中 | 显式差异建模 |
| CLOA 正交锚点 | 🟢 中 | 类别可分离性 |
| 外部数据集预训练 | 🟢 中 | 泛化能力 |
| 全国数据扩展 | 🔵 低 | 长期价值 |

---

## 十、关键参考文献

1. **AEF**: Brown et al., "AlphaEarth Foundations", 2025. 核心参考: batch uniformity + teacher-student + VMF
2. **CLOA**: Li & Pimentel-Alarcon, "Contrastive Learning with Orthonormal Anchors", 2024. 核心参考: 正交锚点防坍缩
3. **VICReg**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization", 2022. 核心参考: 方差-协方差约束
4. **Changen2**: Chen et al., "Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model", 2024. 核心参考: 合成变化数据
5. **M-CD**: "A Mamba-based Siamese Network for Remote Sensing Change Detection", 2024. 核心参考: Difference Module
6. **SeCo/CACo**: 季节对比与变化感知对比学习
7. **S2Looking**: 度量学习方法 (contrastive/triplet loss)

---

## 十一、一句话总结

> **V5 的问题不是"没有时序对比损失"，而是"时序对比损失的强度、粒度和防坍缩机制都不足"。解决方案的核心是：用 batch uniformity 替代失效的 raw uniformity，用 pixel-level temporal loss 替代 global-mean temporal loss，用 teacher-student consistency 增强鲁棒性，用合成变化数据补充真实变化样本的稀缺。**
