# 下游评估系统 v2 — 详细执行计划

## 一、需求确认

用户最终确认的需求：

| 项目 | 确认内容 |
|------|---------|
| Embedding 提取 | **每个月单独提取**（非季度/非3月窗口），每个 patch 的 12 个月各 forward 一次 |
| MLP hidden_dim | **256** |
| CD Head | 使用现有 `ChangeDetectionHead` |
| 可视化 | **尽量多**，全部 420 组图（105 图斑 × 4 时期）都输出 |
| NPU 使用 | **0-6（7张卡）**，NPU 7 留给同事 |
| emb128 训练 | **已停止**，不再继续 |
| Round 1-3 | **需要重跑评估** |
| Round 4 | 已完成的 **7 个实验** 立即评估 |

---

## 二、技术方案

### 2.1 Embedding 提取

```
对每个 patch_id in 424 patches:
    对每个 month in [1, 2, ..., 12]:
        构造时间窗口: [month, month]  # 单个月
        加载该 patch 该月的所有可用帧
        model.forward(skip_decoder=True)
        保存:
            - global_mean: [D] 空间 mean
            - spatial_map: [D, H, W]
```

输出格式：`patch_embeddings.npz`
```python
{
    "global_mean":  [424, 12, D],       # float32, ~1.3 MB
    "spatial_maps": [424, 12, D, H, W], # float32, ~83 MB
    "patch_ids":    ["patch_000001", ...],
    "months":       [1, 2, ..., 12],
}
```

### 2.2 下游评估

#### KNN 评估
- 使用 **6 月** spatial_maps 作为输入
- Patch-stratified 80/20 分割
- torch.cdist + topk 在 NPU 上计算
- 评估 WorldCover(11类)、DynamicWorld(9类)、JRC Water(2类+)

#### MLP 评估
- PixelMLPHead(in_dim=D, hidden_dim=256, num_classes=N)
- 冻结 backbone，只训练 head
- 100 epochs, lr=1e-3, AdamW
- Patch-stratified 分割

### 2.3 变化检测评估

#### 时期定义

| 时期 | before 月份 | after 月份 | 标注文件 | 图斑数 |
|------|------------|-----------|---------|--------|
| Apr→Jun | 4 | 6 | june.shp | 38 |
| Jun→Aug | 6 | 8 | aug.shp | 18 |
| Aug→Sep | 8 | 9 | September.shp | 25 |
| Sep→Oct | 9 | 10 | October.shp | 24 |

#### 三种方法

1. **Cosine Distance** (Baseline)
   ```python
   score = 1.0 - F.cosine_similarity(emb_before, emb_after)
   ```

2. **Linear Discriminator** (主要指标)
   ```python
   X = concat(emb_before, emb_after)  # [N, 2D]
   clf = LogisticRegression(max_iter=1000, class_weight='balanced')
   auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
   ```

3. **CD Head** (空间级)
   ```python
   head = ChangeDetectionHead(in_dim=D, hidden_dim=128)
   input = concat(emb_map_before, emb_map_after)  # [B, 2D, H, W]
   output = head(input)  # [B, 1, H, W]
   label = rasterized_shapefile_mask  # [B, H, W]
   loss = BCEWithLogitsLoss
   ```

**最终 AUC** = 4 时期加权平均（按图斑数加权）

### 2.4 可视化方案

每个标注图斑的每个时期输出 **4 张图**：

1. **before_embedding_pca.png**: PCA降维到3通道的 before embedding，叠加红色变化边界
2. **after_embedding_pca.png**: 同上，after embedding
3. **change_probability.png**: 变化概率热力图（jet colormap），叠加绿色实际变化区域
4. **satellite_overlay.png**: S2 RGB 卫星图 + 半透明概率热力图 + 红色实际边界

总计：105 图斑 × 4 时期 × 4 张图 = **1680 张 PNG**

---

## 三、目录结构

```
/workspace/outputs/{experiment_name}/
├── checkpoints/
│   └── epoch_best_xxx.pt
├── evaluation/
│   ├── embeddings/
│   │   └── patch_embeddings.npz          # 预计算 embedding
│   ├── downstream/
│   │   ├── knn_worldcover.json
│   │   ├── knn_dynamicworld.json
│   │   ├── knn_jrcwater.json
│   │   ├── mlp_worldcover.json
│   │   ├── mlp_dynamicworld.json
│   │   └── mlp_jrcwater.json
│   ├── change_detection/
│   │   ├── summary.json                  # 加权平均 AUC
│   │   ├── period_apr_jun.json
│   │   ├── period_jun_aug.json
│   │   ├── period_aug_sep.json
│   │   ├── period_sep_oct.json
│   │   ├── cosine_baseline.json
│   │   ├── linear_discriminator.pkl
│   │   └── cd_head_trained.pt
│   └── visualizations/
│       ├── cd_before_after/
│       │   ├── period_apr_jun/
│       │   │   ├── patch_xxx_before_embedding_pca.png
│       │   │   ├── patch_xxx_after_embedding_pca.png
│       │   │   ├── patch_xxx_change_probability.png
│       │   │   └── patch_xxx_satellite_overlay.png
│       │   └── ...
│       ├── embedding_tsne/
│       │   ├── worldcover_tsne.png
│       │   └── dynamicworld_tsne.png
│       └── class_heatmaps/
│           ├── worldcover_heatmap_sample.png
│           └── dynamicworld_heatmap_sample.png
└── summary_report.json
```

---

## 四、脚本清单

| # | 脚本路径 | 功能 | 输入 | 输出 |
|---|---------|------|------|------|
| 1 | `scripts/eval/extract_embeddings_v2.py` | 提取所有 patch × 12月 embedding | config, checkpoint | `embeddings/patch_embeddings.npz` |
| 2 | `scripts/eval/evaluate_knn_v2.py` | KNN 三任务评估 | `patch_embeddings.npz` | `downstream/knn_*.json` |
| 3 | `scripts/eval/evaluate_mlp_v2.py` | MLP 三任务评估 | `patch_embeddings.npz` | `downstream/mlp_*.json` |
| 4 | `scripts/eval/evaluate_cd_v2.py` | 变化检测 AUC（cosine + LR） | `patch_embeddings.npz`, shapefiles | `change_detection/*.json`, `*.pkl` |
| 5 | `scripts/eval/train_cd_head.py` | CD Head 训练 | `patch_embeddings.npz`, shapefiles | `change_detection/cd_head_trained.pt` |
| 6 | `scripts/eval/visualize_cd.py` | 变化检测可视化 | `patch_embeddings.npz`, predictions, S2数据 | `visualizations/cd_before_after/*.png` |
| 7 | `scripts/eval/visualize_embedding_space.py` | t-SNE/PCA 可视化 | `patch_embeddings.npz`, labels | `visualizations/embedding_tsne/*.png` |
| 8 | `scripts/eval/run_full_evaluation.py` | **一键运行全部评估** | config, checkpoint | 完整 `evaluation/` 目录 |

---

## 五、执行计划

### Phase 1: 脚本开发（NPU 0，单卡）

**目标**：在 1 个已完成的 Round 4 实验上跑通全流程

| 步骤 | 任务 | 预计耗时 |
|------|------|---------|
| 1.1 | 写 `extract_embeddings_v2.py`（单月提取） | 30 min |
| 1.2 | 写 `evaluate_knn_v2.py` | 20 min |
| 1.3 | 写 `evaluate_mlp_v2.py` | 20 min |
| 1.4 | 写 `evaluate_cd_v2.py`（cosine + LR） | 30 min |
| 1.5 | 写 `train_cd_head.py` | 20 min |
| 1.6 | 写 `visualize_cd.py` | 30 min |
| 1.7 | 写 `visualize_embedding_space.py` | 15 min |
| 1.8 | 写 `run_full_evaluation.py`（调度脚本） | 15 min |
| 1.9 | 在 `round4_full_vicreg_baseline` 上测试全流程 | 90 min |

**Phase 1 总计：约 4.5 小时**

### Phase 2: Round 4 批量评估（NPU 0-6，7卡并行）

7 个已完成实验，7 张卡，1 对 1 并行：

| NPU | 实验 | 状态 |
|-----|------|------|
| 0 | round4_full_vicreg_baseline | ✅ 完成 |
| 1 | round4_full_high_var | ✅ 完成 |
| 2 | round4_full_high_temporal | ✅ 完成 |
| 3 | round4_full_low_recon | ✅ 完成 |
| 4 | round4_full_high_consist | ✅ 完成 |
| 5 | round4_full_high_kappa | ✅ 完成 |
| 6 | round4_full_low_decoder | ✅ 完成 |

每个实验评估耗时：~80 分钟
7 卡并行：总耗时 ≈ **80 分钟**

### Phase 3: Round 1-3 重跑评估（NPU 0-6，7卡并行）

Round 1: 8 个实验（但只需评估有意义的几个）
Round 2: 1 个实验
Round 3: 1 个实验

共约 10 个实验，分 2 批（7 + 3）：
- 第 1 批 7 个：~80 分钟
- 第 2 批 3 个：~80 分钟

**Phase 3 总计：约 2.5 小时**

### Phase 4: 汇总分析

1. 收集所有实验的 `summary_report.json`
2. 生成跨 Round 对比表格
3. 找出最佳配置

**Phase 4 总计：约 30 分钟**

---

## 六、时间线

```
Day 1 (今天):
  ├─ Phase 1: 脚本开发 + 单实验测试 (4.5h)
  └─ Phase 2: Round 4 批量评估 (1.5h)

Day 2:
  ├─ Phase 3: Round 1-3 重跑评估 (2.5h)
  └─ Phase 4: 汇总分析 (0.5h)

总计: ~9 小时实际工作，约 1.5 天完成
```

---

## 七、成功标准

| 指标 | 目标 |
|------|------|
| Bare AUC (Linear Discriminator) | > 0.55 |
| WorldCover KNN Acc | > 30% |
| WorldCover MLP Acc | > 35% |
| Dynamic World KNN Acc | > 40% |
| Dynamic World MLP Acc | > 45% |
| JRC Water KNN Acc | > 15% |
| JRC Water MLP Acc | > 20% |
| CD Head AUC | > 0.50 |

---

## 八、风险与应对

| 风险 | 应对 |
|------|------|
| 420 组可视化 PNG 占用空间大 | 每张 ~500KB，总计 ~200MB，可接受 |
| 7 卡并行评估时内存不足 | 每个评估脚本内存占用 < 2GB，7×2=14GB，足够 |
| shapefile 栅格化精度问题 | 用 rasterio.features.rasterize，8×8 分辨率 |
| Round 1 checkpoint 已删除 | 先确认 checkpoint 是否存在 |
| 现有 ChangeDetectionHead 接口不匹配 | 检查接口，必要时做适配层 |
