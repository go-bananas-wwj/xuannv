# 下游评估系统 v2 — 设计方案

> 目标：修复 Round 1-3 评估 bug，建立标准化的、可复用的评估流水线。

---

## 1. 核心设计原则

1. **Embedding 预计算**：训练完成后，一次性提取所有 patch × 12 个月的 embedding，保存为文件。后续所有下游评估直接读取预计算文件，不再重复跑模型。
2. **时序充分利用**：每个 patch 保留 12 个月的 embedding，下游任务可灵活选择月份/季度/年度平均。
3. **评估结果集中管理**：一轮实验的所有评估结果保存在一个目录下，结构清晰。
4. **可视化第一**：变化检测评估必须输出 before/after embedding 的可视化对比，叠加 shapefile 标注，肉眼可验证。

---

## 2. 目录结构

```
/workspace/outputs/{experiment_name}/
├── checkpoints/                          # 训练 checkpoint
│   ├── epoch_best_xxx.pt
│   └── epoch_xxx.pt
├── train.log
├── evaluation/                           # ★ 所有评估结果
│   ├── embeddings/                       # 预计算 embedding
│   │   ├── patch_embeddings.npy          # [424, 12, D] 全局 mean
│   │   ├── patch_embedding_maps.npy      # [424, 12, D, H, W] 空间
│   │   └── metadata.json                 # patch_ids, months, 实验信息
│   ├── downstream/                       # 下游分类任务
│   │   ├── knn/                          # KNN 结果
│   │   │   ├── worldcover_report.json
│   │   │   ├── dynamicworld_report.json
│   │   │   └── jrcwater_report.json
│   │   └── mlp/                          # MLP 结果
│   │       ├── worldcover_report.json
│   │       ├── dynamicworld_report.json
│   │       └── jrcwater_report.json
│   ├── change_detection/                 # 变化检测
│   │   ├── auc_results.json              # 4 时期 AUC 汇总
│   │   ├── cd_head_trained.pt            # 训练好的 CD Head
│   │   ├── linear_discriminator.pkl      # 线性判别器
│   │   ├── cosine_baseline.json          # 余弦距离 baseline
│   │   └── predictions/                  # 逐 patch 预测
│   │       ├── period_apr_jun/
│   │       ├── period_jun_aug/
│   │       ├── period_aug_sep/
│   │       └── period_sep_oct/
│   └── visualizations/                   # 可视化
│       ├── cd_before_after/              # 变化检测前后 embedding
│       │   ├── period_apr_jun/
│       │   │   ├── patch_000001_before.png
│       │   │   ├── patch_000001_after.png
│       │   │   ├── patch_000001_change_prob.png
│       │   │   └── ...
│       │   └── ...
│       ├── embedding_tsne/               # t-SNE 聚类
│       └── downstream_heatmaps/          # 下游分类热力图
└── summary_report.json                   # 总汇总报告
```

---

## 3. Embedding 提取方案

### 3.1 输入数据

对每个 `(patch_id, month)` 对：
- 构造一个时间窗口：`[month-1, month+1]`（3个月窗口）
- 加载该 patch 在该窗口内的所有可用帧
- 输入给 AEFModel（skip_decoder=True）
- 输出 `embedding_map` [1, D, H, W] 和 `embedding` [1, D]

### 3.2 输出格式

保存为一个 `.npz` 文件：

```python
np.savez(
    "evaluation/embeddings/patch_embeddings.npz",
    global_mean=global_mean,      # [424, 12, D] float32
    spatial_maps=spatial_maps,    # [424, 12, D, H, W] float32
    patch_ids=patch_ids,          # [424] str
    months=months,                # [12] int
)
```

数据量估算（D=64, H=W=8）：
- global_mean: 424×12×64 × 4B = 1.3 MB
- spatial_maps: 424×12×64×8×8 × 4B = 83 MB
- 总计：~85 MB，完全可以接受

### 3.3 提取脚本

```python
# scripts/eval/extract_embeddings_v2.py
# 用法: python extract_embeddings_v2.py \
#           --config configs/xxx.yaml \
#           --checkpoint /path/to/epoch_best.pt \
#           --output-dir /workspace/outputs/xxx/evaluation/embeddings
```

---

## 4. 下游任务评估

### 4.1 月度策略

所有下游任务统一使用 **6 月 embedding** 作为静态标签的输入：
- 理由：6 月是哈尔滨夏季，云量最少，植被最丰富，WorldCover/DynamicWorld 标注最准确
- 变化检测使用各自时期的月份

### 4.2 KNN 评估

```python
# 输入: spatial_maps [424, 12, D, H, W]
# 使用 6 月: spatial_maps[:, 5]  # June = index 5
# 加载 WorldCover/DynamicWorld/JRC Water 标签（resize 到 H×W）
# Patch-stratified 80/20 分割
# torch.cdist + topk 在 NPU 上计算
```

### 4.3 MLP 评估

```python
# 冻结 backbone，训练 PixelMLPHead
# 输入: spatial_maps [B, D, H, W]
# 输出: [B, num_classes, H, W]
# 训练 100 epochs，lr=1e-3，AdamW
# 同样 patch-stratified 分割
```

---

## 5. 变化检测评估

### 5.1 4 个时期

| 时期 | before 月份 | after 月份 | 标注文件 | 图斑数 |
|------|------------|-----------|---------|--------|
| Apr→Jun | April (4) | June (6) | june.shp | 38 |
| Jun→Aug | June (6) | August (8) | aug.shp | 18 |
| Aug→Sep | August (8) | September (9) | September.shp | 25 |
| Sep→Oct | September (9) | October (10) | October.shp | 24 |

### 5.2 三种评估方法

#### 方法 1: Cosine Distance Baseline
```python
emb_before = global_mean[patch_idx, before_month-1]  # [D]
emb_after = global_mean[patch_idx, after_month-1]    # [D]
cos_sim = F.cosine_similarity(emb_before, emb_after)
change_score = 1.0 - cos_sim  # 越小越相似，越大越变化
```
保留作为 baseline，但不作为主要指标。

#### 方法 2: Linear Discriminator（主要指标）
```python
# 训练数据：所有标注 patch 的 (emb_before, emb_after) → label
X = np.concatenate([emb_before, emb_after], axis=1)  # [N, 2D]
y = labels  # 0=无变化, 1=变化

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
```

#### 方法 3: CD Head
```python
# 训练一个轻量级 ChangeDetectionHead
# 输入: concat(embedding_map_before, embedding_map_after)  # [B, 2D, H, W]
# 输出: [B, 1, H, W] 变化概率
# 标签: 将 shapefile 栅格化为 [H, W] 的二值 mask
# 训练 50 epochs，BCE loss
```

### 5.3 AUC 计算

每个时期单独计算 AUC，然后取 4 个时期的加权平均（按图斑数加权）。

```python
overall_auc = sum(auc_i * n_i for auc_i, n_i in zip(aucs, counts)) / sum(counts)
```

---

## 6. 可视化方案

### 6.1 变化检测 Before/After Embedding 可视化

对每个标注 patch，每个时期：

```python
# 1. 取 before 和 after 的 embedding_map [D, H, W]
emb_before = spatial_maps[patch_idx, before_month-1]  # [64, 8, 8]
emb_after = spatial_maps[patch_idx, after_month-1]    # [64, 8, 8]

# 2. PCA 降维到 3 通道 → RGB
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
emb_rgb_before = pca.fit_transform(emb_before.reshape(D, -1).T)  # [64, 3]
emb_rgb_before = emb_rgb_before.reshape(H, W, 3)
emb_rgb_before = (emb_rgb_before - emb_rgb_before.min()) / (emb_rgb_before.max() - emb_rgb_before.min())

# 3. 叠加 shapefile 标注边界
# 4. 显示变化概率热力图（linear discriminator 或 CD Head 输出）
# 5. 保存为 PNG
```

输出示例：
- `patch_000001_before.png`: PCA-RGB 的 before embedding，叠加红色变化边界
- `patch_000001_after.png`: PCA-RGB 的 after embedding，叠加红色变化边界
- `patch_000001_change_prob.png`: 变化概率热力图（红色=高变化概率），叠加绿色=实际变化区域

### 6.2 Embedding t-SNE 可视化

```python
# 对所有 patch 的 6 月 embedding 做 t-SNE
# 按 WorldCover 类别着色
# 观察同类是否聚类
```

---

## 7. 脚本列表

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `extract_embeddings_v2.py` | 提取所有 patch × month embedding | config, checkpoint | `patch_embeddings.npz` |
| `evaluate_knn_v2.py` | KNN 下游评估 | `patch_embeddings.npz` | `downstream/knn/*.json` |
| `evaluate_mlp_v2.py` | MLP 下游评估 | `patch_embeddings.npz` | `downstream/mlp/*.json` |
| `evaluate_cd_v2.py` | 变化检测 AUC + CD Head | `patch_embeddings.npz`, shapefiles | `change_detection/*.json` |
| `visualize_cd.py` | 变化检测可视化 | `patch_embeddings.npz`, predictions | `visualizations/cd_before_after/*.png` |
| `run_full_evaluation.py` | 一键运行全部 | config, checkpoint | 完整 `evaluation/` 目录 |

---

## 8. 执行顺序

```
Step 1: 训练完成 → checkpoint 保存
Step 2: python extract_embeddings_v2.py --config xxx --checkpoint xxx
Step 3: python run_full_evaluation.py --embedding-file xxx --output-dir xxx/evaluation
        └─ 内部调用:
           ├── evaluate_knn_v2.py
           ├── evaluate_mlp_v2.py
           ├── evaluate_cd_v2.py
           └── visualize_cd.py
Step 4: 查看 summary_report.json
```

---

## 9. Round 1-3 重跑计划

对每个已完成的 Round：
1. 找到最佳 checkpoint
2. 运行 `extract_embeddings_v2.py`
3. 运行 `run_full_evaluation.py`
4. 更新汇总报告

预计每个 Round 评估耗时：
- Embedding 提取：~30 分钟（424×12 = 5088 次 forward）
- KNN：~5 分钟
- MLP：~10 分钟
- CD AUC：~5 分钟
- CD Head 训练：~15 分钟
- 可视化：~10 分钟
- **总计：~75 分钟 / Round**

3 个 Round 共约 3.75 小时（可并行，7 张卡同时跑）。

---

## 10. Round 4 评估计划

8 个实验，7 张卡可用：
- 已完成的 7 个实验可以立即开始评估
- emb128 完成后立即评估
- 评估可以并行（每个实验独立，互不干扰）

---

## 待确认事项

1. **Embedding 提取窗口**：是否用 `[month-1, month+1]` 的 3 个月窗口？还是用单个月？
2. **MLP hidden_dim**：默认 256？
3. **CD Head 类型**：`ChangeDetectionHead`（已有）还是 `PixelMLPHead`？
4. **可视化数量**：105 个图斑 × 4 时期 = 420 张图，是否全部输出？还是抽样？
