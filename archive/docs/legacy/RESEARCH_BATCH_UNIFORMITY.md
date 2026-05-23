# Uniformity Loss 问题调研报告

## 一、Patch 空间分布与 Batch 组成

### 1.1 Patch ID 是严格空间有序的

```
Grid: 26 cols × 24 rows = 624 cells (实际 424 patches)
X step: 1280.0m  (东西向间隔 1.28km)
Y step: 1280.0m  (南北向间隔 1.28km)
```

| Patch ID | 坐标 (x, y) | 位置 |
|----------|------------|------|
| patch_000000 | (306797, 5069852) | 左上角第一行第一列 |
| patch_000001 | (308077, 5069852) | 第一行第二列 |
| patch_000004 | (311917, 5069852) | 第一行第五列 |
| patch_000419 | (306797, 5098012) | 倒数第二行第一列 |
| patch_000423 | (306797, 5099292) | 最后一行第一列 |

**结论**：`_discover_patches()` 返回 `sorted(patch_ids)`，所以 dataset 中的 patch 是按**从左到右、从上到下**的空间网格顺序排列的。

### 1.2 DistributedSampler 的 Shuffle 行为

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training)
```

- 训练时 `shuffle=True`，每个 epoch 会随机打乱整个 dataset
- 8 张卡各取 424/8 = 53 个 patch（不均匀分配）
- **每个 batch（batch_size=2）内的 2 个 patch 来自该 rank 的 53 个 patch 中的随机位置**

**问题**：即使 shuffle 了，424 个 patch 全部来自**哈尔滨同一个区域**，景观类型高度同质化（主要是农田+少量城市建筑）。一个 batch 内的 2 个 patch 很可能都是农田，uniformity loss 强迫它们"不相似"，与重建目标矛盾。

### 1.3 Uniformity Loss 的 Gather 行为

```python
embedding = student_out.embedding  # [2, 128] per GPU
if dist.is_initialized():
    gathered_emb = [torch.zeros_like(embedding) for _ in range(self.world_size)]
    dist.all_gather(gathered_emb, embedding)
    gathered_emb = torch.cat(gathered_emb, dim=0)  # [16, 128]
uniform = batch_uniformity_loss_l2(gathered_emb)
```

- **Effective batch = 16**（8 GPU × 2）
- 16 个样本来自哈尔滨不同位置，但景观类型可能高度重叠
- Uniformity 计算所有 `16×15/2 = 120` 对样本的 `|cos(θ)|` 平均

---

## 二、Uniformity Loss 数值分析

### 2.1 5 Epoch 指标

| Epoch | Recon | Consist | Uniform |
|-------|-------|---------|---------|
| 001 | 1.156 | 0.353 | 0.669 |
| 002 | 1.061 | 0.132 | 0.845 |
| 003 | 0.850 | 0.100 | 0.960 |
| 004 | 0.646 | 0.097 | 0.989 |
| 005 | 0.470 | 0.086 | **0.996** |

### 2.2 数值含义

`batch_uniformity_loss_l2` 的值域 `[0, 1]`：
- **0 = 完美分散**（所有样本对的 cosine similarity = 0，即互相正交）
- **1 = 完全坍缩**（所有样本对的 cosine similarity = 1，即完全同向）

**Uniform = 0.996 意味着**：几乎所有 120 对样本的夹角都在 **5° 以内**。

在 128 维球面上，随机均匀分布的期望夹角是 **90°**（cos = 0）。0.996 意味着 embedding 已经**完全坍缩到同一个方向**。

### 2.3 为什么 Uniformity 救不回来？

| 因素 | AEF 论文 | 我们的 V12 |
|------|----------|-----------|
| Batch Size | 2048 | 16 |
| 数据来源 | 全球 8.4M 序列 | 哈尔滨 424 patch |
| 景观多样性 | 沙漠/森林/城市/海洋/冰川... | 农田/农田/农田... |
| Uniformity 信号强度 | ~200万对样本 | 120对样本 |
| 信号强度比 | 基准 | **差 1.7 万倍** |

**根本原因**：
1. Batch 太小（16 vs 2048），uniformity 梯度极其微弱
2. 数据同质化，batch 内样本本就相似，uniformity 目标与重建目标冲突
3. Reconstruction weight（1.0）远大于 Uniformity weight（0.1），模型自然选择"坍缩来更好重建"

---

## 三、显存使用与 Batch Size 提升空间

### 3.1 当前显存

```
HBM-Usage: ~32300 / 65536 MB = 49.3%
Free: ~33 GB per NPU
```

### 3.2 理论提升

当前配置：batch_size=2, accum=4, 8卡 → effective batch=64

如果 batch_size=3：
- 显存增长约 1.5x（非线性，teacher-student + grad checkpointing）
- 预计显存使用 ~48GB/65GB = 73%，安全
- Effective batch = 3×8×4 = 96

如果 batch_size=4：
- 显存增长约 2x
- 预计显存使用 ~64GB/65GB = 98%，风险高（可能 OOM）
- Effective batch = 4×8×4 = 128

**建议**：batch_size=3 是安全选择。

---

## 四、Uniformity Loss 的算法缺陷

### 4.1 当前实现的问题

```python
# batch_uniformity_loss_l2
x = F.normalize(x, p=2, dim=-1)  # [N, D]
sim = chunk_i @ x_dropped.T      # [chunk, N]
loss = sim.abs().sum() / n_pairs # 所有对的 |cos(θ)| 平均
```

**缺陷 A**：对**所有样本对**一视同仁，没有区分"应该相似的样本"和"应该分散的样本"。

**缺陷 B**：在 128 维球面上，随机向量的期望内积是 0。但我们的数据是**地理相近的 patch**，它们本来就应该有相似的 embedding（比如都是农田）。强迫它们的 cosine similarity 接近 0 是不合理的。

**缺陷 C**：batch 太小（N=16），统计噪声极大。

### 4.2 正确的 Uniformity 应该怎么做？

参考 AEF 论文 S2.2.4 和后续研究（Wang & Isola 2020, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"）：

**Alignment**：相似的样本应该在球面上靠近（小角度）。
**Uniformity**：不相似的样本应该在球面上均匀分散（大角度）。

关键：**Uniformity 只应该惩罚"不相似的样本对"过于靠近**。对于"相似的样本对"（比如相邻农田），应该允许它们靠近。

当前实现缺少"相似性判断"机制，对所有样本对一视同仁地惩罚靠近。

---

## 五、可行的解决方案

### 方案 1：Memory Bank（推荐）

维护一个全局 embedding 队列，存储最近 K 个 batch 的 embedding。计算 uniformity 时，使用当前 batch + memory bank 中的 embedding。

- K=512 时，effective samples = 16 + 512 = 528
- 信号强度从 120 对 → `528×527/2 ≈ 14万` 对，提升 ~1100 倍
- 可以进一步从 memory bank 中**排除与当前 batch 地理相近的 patch**

### 方案 2：Hard Negative Mining

只对"最不相似的样本对"（hard negatives）计算 uniformity。具体做法：
- 先计算所有对的 cosine similarity
- 只惩罚 similarity > threshold 的对（即"本不该相似却相似了"的对）
- 对于 similarity 已经很低（< 0.3）的对，不惩罚

这样，相似的 patch（如相邻农田）允许靠近，只有"不相似却坍缩"的情况才被惩罚。

### 方案 3：提高 Batch Size + 调整权重

- batch_size: 2→3（显存安全）
- effective batch: 64→96
- 同时提高 uniformity weight: 0.1→0.3
- 降低 reconstruction weight: 1.0→0.5

### 方案 4：放弃 L2-normalized Uniformity，改用 Raw Uniformity

`raw_uniformity_loss` 在欧氏空间计算，不依赖 L2 归一化：
- 对 batch size 不敏感
- 不需要"不相似的样本"假设
- 但 V12 配置已设 `skip_l2_norm_training: false`，训练时也是 L2 空间

---

## 六、用户建议的可行性分析

### 建议 A："随机采样不同地区的 patch"

**可行性**：⭐⭐⭐⭐⭐（高）

当前 DistributedSampler 已经是随机 shuffle 的。但问题是：
1. 424 个 patch 全部来自哈尔滨，没有"不同地区"
2. 即使 shuffle，景观类型仍可能重叠

**改进方向**：
- 在 dataset 中增加**空间间隔采样**：每个 batch 中的 patch 至少相隔 N 个网格位置
- 或者：使用 memory bank 确保 uniformity 的负样本来自"远距离" patch

### 建议 B："提高 batch size"

**可行性**：⭐⭐⭐⭐（中高）

显存还有 33GB 空闲，batch_size=3 是安全的。
- 从 2→3：effective batch 64→96，uniformity 对数从 120→`24×23/2=276`
- 提升约 2.3 倍，但仍然远远不够（vs AEF 的 200万对）

**结论**：两个建议都有用，但**单独提高 batch size 不够**，需要配合 memory bank 或 hard negative mining。
