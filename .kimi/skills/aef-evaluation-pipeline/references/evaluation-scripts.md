# 评估脚本详细参数说明

## extract_v12_embeddings_all_months.py（推荐）

使用 DataLoader 提取 embedding，避免索引错位问题。

```bash
python scripts/eval/extract_v12_embeddings_all_months.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0 \
  --batch-size 16 \
  --num-workers 4 \
  --output-dir /workspace/outputs/{exp}/eval/embeddings_all_months
```

**参数**:
- `--batch-size`: 16 为安全值（24 OOM）
- `--num-workers`: 4 即可，更多 workers 收益递减

**输出**（per-patch npz）:
- `{patch_id}.npz`:
  - `embedding_map`: [N_months, D, H, W] — L2 normalized
  - `pre_norm_map`: [N_months, D, H, W] — 原始幅度
  - `year_months`: [N_months, 2] — int, e.g. [[2025, 4], ...]
  - `valid_starts`: [N_months] — float64 ms
  - `valid_ends`: [N_months] — float64 ms

**优势**: 无需 reshape，per-patch 文件天然避免排序假设错误。

---

## extract_embeddings_v2.py（备选，有注意事项）

手动 batch 提取 embedding。

```bash
python scripts/eval/extract_embeddings_v2.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --output-dir /workspace/outputs/{exp}/evaluation/embeddings \
  --device npu:0 \
  --batch-size 8 \
  --save-every 500
```

**参数**:
- `--save-every`: 每 N 个样本保存断点（默认 500）
- `--batch-size`: 16 为安全值（OOM 时降至 8）

**输出**:
- `patch_embeddings.npz`:
  - `global_mean`: [424, 12, D]
  - `spatial_maps`: [424, 12, D, H, W]
  - `patch_ids`: [424]
  - `months`: [12]

**⚠️ 关键警告**:
```python
# 以下 reshape 假设 monthly_samples 是 patch-major 排序
spatial_maps.reshape(num_patches, 12, D, H, W)
```

如果 `_build_monthly_samples` 逻辑改变（如过滤某些月份），reshape 会失败。**建议使用 `extract_v12_embeddings_all_months.py`。**

---

## validate_v12_bare.py（Bare AUC）

无 CD Head，像素级 cosine distance 评估变化检测能力。

```bash
python scripts/eval/validate_v12_bare.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0 \
  --use-pre-norm  # 可选：评估 pre-norm 空间
```

**关键设计**:
- 使用 `patch_month_to_idx` 映射避免索引错位
- Period: 2025 年月份对（4→6, 6→8, 8→9, 9→10）
- 像素级 cosine distance: `dist = 1 - cos(emb_before, emb_after)`
- 输出: `bare_auc.json`（全局 + 分 period）

**注意**: Bare AUC 容易低估模型能力（通常 0.50-0.65），不代表模型真实水平。

---

## train_cd_head_v12.py（CD Head 训练）

冻结 backbone，5-fold CV 训练 ChangeDetectionHeadV3。

```bash
python scripts/eval/train_cd_head_v12.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0 \
  --folds 5 \
  --epochs 200 \
  --lr 5e-4 \
  --hidden-dim 64 \
  --use-pre-norm  # 可选
```

**关键设计**:
- `extract_embedding_for_month` 正确 API（避免索引错位）
- ChangeDetectionHeadV3: 2 residual blocks + ECA 注意力
- Focal Loss + Dice Loss（处理类别不平衡）
- 数据增强: 随机水平/垂直翻转
- Early Stopping (patience=30)

**输出**:
- `cd_head_v12_best.pt`: 最佳 fold 的 CD Head checkpoint

**参数调参建议**:
| 参数 | 默认值 | 调参范围 | 影响 |
|------|:------:|:--------:|------|
| `--hidden-dim` | 64 | 64, 128, 256 | 容量，但对 mean AUC 提升有限 |
| `--epochs` | 200 | 100-200 | Early stopping 通常 40-60 epoch 触发 |
| `--lr` | 5e-4 | 1e-3 ~ 1e-4 | 过高不稳定，过低收敛慢 |

---

## evaluate_knn_v2.py

KNN 三任务评估（基线参考）。

```bash
python scripts/eval/evaluate_knn_v2.py \
  --embedding-file /path/to/patch_embeddings.npz \
  --output-dir /path/to/downstream \
  --device npu:0 \
  --k 5 \
  --month 6
```

**关键实现**:
- `knn_predict_npu`: 使用 `scipy.stats.mode` 在 CPU 上批量计算众数
- Patch-stratified 80/20 split
- 训练集 subsample 到 100K 像素

**局限性**: KNN 只是无参数基线，**不能反映 embedding 的真实判别力**。

---

## evaluate_mlp_v2.py

MLP 下游头训练（主要下游评估指标）。

```bash
python scripts/eval/evaluate_mlp_v2.py \
  --embedding-file /path/to/patch_embeddings.npz \
  --output-dir /path/to/downstream_mlp \
  --device npu:0 \
  --epochs 50 \
  --month 6 \
  --hidden-dim 256 \
  --dropout 0.3
```

**关键实现**:
- `PixelMLPHead(in_dim=D, hidden_dim=256, num_classes=N, dropout=0.3)`
- AdamW lr=1e-3, CosineAnnealingLR
- batch_size=1024
- 标签映射 + `label < num_classes` 过滤

**调参建议**:
- `--epochs`: 50 足够（200 epoch 几乎无提升）
- `--hidden-dim`: 256 足够（512 几乎无提升）
- 如果 Acc 停滞，问题在 embedding 质量，不在 MLP

---

## fewshot_change_detection.py

Few-Shot 变化检测（轻量 2-layer CD Head）。

```bash
python scripts/eval/fewshot_change_detection.py \
  --config configs/{exp}.yaml \
  --checkpoint /workspace/outputs/{exp}/epoch_best.pt \
  --device npu:0 \
  --k-shots 1,5,10,20 \
  --n-splits 5 \
  --use-pre-norm  # 可选
```

**注意**: 已修复 `MONTH_WINDOWS_2025` 为真正 2025 年值（原先是 2024 年）。

---

## visualize_cd_predictions.py

变化检测可视化（生成 pred/GT/overlay 三图）。

```bash
python scripts/eval/visualize_cd_predictions.py \
  --config configs/{exp}.yaml \
  --backbone /workspace/outputs/{exp}/epoch_best.pt \
  --cd-head /workspace/outputs/{exp}/cd_head_v12_best.pt \
  --device npu:0 \
  --max-patches 20
```

**输出**:
- `{patch_id}_{period}.png`: 三面板图（预测/GT/叠加）

---

## launch 脚本使用

### 7 卡并行提取
```bash
bash scripts/eval/launch_all_round4_eval.sh
```
自动分配 NPU 0-6，每个实验一个 tmux session。

### 下游评估批量启动
```bash
bash scripts/eval/launch_downstream_v2.sh
```
串行跑 KNN → MLP → CD，每个实验一个 tmux session。

### 只重新跑 MLP
```bash
bash scripts/eval/launch_mlp_v2.sh
```
用于修复 bug 后重新跑 MLP。
