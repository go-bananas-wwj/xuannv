# Embedding 训练指标增强计划

> 针对用户反馈："Active dims 没有在训练中显示"、"不同派系之间的差异没有显示"

---

## 一、当前状况

| 项目 | 状态 |
|------|------|
| 训练脚本 | `ddp_v10_temporal_trainer.py` (最新) |
| 当前打印指标 | recon, consist, cls, uniform, var, decorr, orth, temporal, pixel_change, change_consist, lr |
| **缺失指标** | effective_rank, silhouette_score, nn_retrieval_acc, per_class_separation |
| 训练状态 | **当前无训练在进行** |

---

## 二、问题诊断

### 2.1 Active dims 缺失

**定义**: `effective_rank` = 协方差矩阵特征值的 `(sum)^2 / sum(square)`，衡量 embedding 空间有多少维度被实际使用。

**现状**: 训练脚本完全没有计算此指标。`analyze_v7_embedding_quality.py` 只计算了 `n_dim_low_std`（低 std 维度数），但没有计算有效秩。

**正常范围**: ≈ embedding_dim (64)，< 32 表示严重维度坍缩。

### 2.2 不同派系差异缺失

**定义**: "派系" = WorldCover 土地覆盖类别（11类：林地、灌木、草地、农田、建筑、裸地、冰雪、水体、湿地、红树林、苔藓）。

**现状**: 训练脚本没有计算：
- `silhouette_score`: 同类聚集、异类分离的程度
- `nn_retrieval_acc`: 最近邻是否同类的比例
- `per_class_separation`: 每类类内/类间距离比

**正常范围**: silhouette > 0.3 为良好，> 0.5 为优秀。

---

## 三、修改计划

### 3.1 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/training/ddp_v10_temporal_trainer.py` | 新增 `_compute_embedding_metrics()` 方法 |
| `scripts/train/train_ddp_v10.py` | 修改 epoch 总结打印格式 |

### 3.2 新增指标计算（每 5 个 epoch，rank 0 执行）

```python
def _compute_embedding_metrics(self, embeddings, worldcover_labels):
    """
    embeddings: [N, D] pre_norm_embedding (已 gather 到 rank 0)
    worldcover_labels: [N] patch-level WorldCover 众数类别
    
    返回:
    - effective_rank: float
    - silhouette_score: float
    - nn_retrieval_acc: float
    - per_class_separation: dict
    """
```

**计算逻辑**:
1. **effective_rank**: 计算 embedding 的协方差矩阵特征值，然后用 `(sum)^2 / sum(square)` 公式
2. **silhouette_score**: 使用 sklearn 的 `silhouette_score`（metric='cosine'），基于 WorldCover 标签
3. **nn_retrieval_acc**: L2 归一化后计算 pairwise cosine distance，对每个样本找最近邻，统计是否同类
4. **per_class_separation**: 计算每类 centroid，类间距离 / 类内 std

**性能控制**:
- 只在 rank 0 计算
- 每 5 个 epoch 计算一次
- 使用当前 epoch 已采样的 2-4 个 batch 的 embedding（不额外 forward）
- 异步计算，不阻塞训练

### 3.3 打印格式

每 5 个 epoch 新增一行：
```
[Epoch 5] Metrics: eff_rank=58.3/64, silhouette=0.42, nn_acc=0.67, sep=2.1
```

其中：
- `eff_rank=58.3/64`: 有效秩 58.3，embedding_dim=64
- `silhouette=0.42`: silhouette score（>0.3 良好）
- `nn_acc=0.67`: 最近邻检索准确率（>0.6 良好）
- `sep=2.1`: 平均类间/类内距离比（>2.0 良好）

### 3.4 保存到 checkpoint

这些指标会作为 `losses` dict 的一部分保存到 checkpoint 中：
```python
"losses": {
    "recon": 0.25,
    "uniform": -2.1,
    ...,
    "effective_rank": 58.3,
    "silhouette_score": 0.42,
    "nn_retrieval_acc": 0.67,
    "class_separation": 2.1,
}
```

---

## 四、实施步骤

| 步骤 | 时间 | 内容 |
|------|------|------|
| 1 | 30 min | 修改 `ddp_v10_temporal_trainer.py`，新增 `_compute_embedding_metrics` |
| 2 | 15 min | 修改 `train_ddp_v10.py`，更新 epoch 总结打印 |
| 3 | 20 min | 冒烟测试：运行 3 个 step，确认指标计算无 crash |
| 4 | 10 min | git commit + push |

---

## 五、预期效果

训练日志中每 5 个 epoch 会显示类似：

```
[Epoch 5] recon=0.251 uniform=-1.85 temporal=0.042 ... 
[Epoch 5] Metrics: eff_rank=52.1/64, silhouette=0.38, nn_acc=0.61, sep=1.8

[Epoch 10] recon=0.198 uniform=-2.34 temporal=0.051 ...
[Epoch 10] Metrics: eff_rank=57.4/64, silhouette=0.45, nn_acc=0.69, sep=2.3
```

通过这些指标可以实时监控：
- **embedding 是否坍缩**（eff_rank 是否下降）
- **语义信息是否被编码**（silhouette 是否上升）
- **局部结构是否良好**（nn_acc 是否上升）
- **类别判别性是否增强**（sep 是否上升）

---

## 六、待确认事项

1. **计算频率**: 每 5 个 epoch 计算一次是否合适？（太频繁增加开销，太少无法及时发现问题）
2. **WorldCover 标签**: 使用 patch-level 众数类别（当前 dataset 已有 `_get_worldcover_label`）还是 pixel-level 标签？
   - patch-level: 计算快，但粒度粗
   - pixel-level: 需要额外从 target_images 中解析 worldcover one-hot 回类别索引，计算慢但更精确
3. **是否需要 temporal 相关指标**: `temporal_consistency`（相邻月份 cosine sim）和 `temporal_discriminability`（变化/不变 distance 差异）是否也需要添加？

**请确认以上事项，我将立即开始实施。**
