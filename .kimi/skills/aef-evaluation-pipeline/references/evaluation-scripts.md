# 评估脚本详细参数说明

## extract_embeddings_v2.py

提取所有 patch × 12 个月的 embedding。

```bash
python scripts/eval/extract_embeddings_v2.py \
  --config configs/round4_8gpu/{exp}.yaml \
  --checkpoint /workspace/outputs/xuannv_round2/{exp}/epoch_best_epoch20.pt \
  --output-dir /workspace/outputs/xuannv_round2/{exp}/evaluation/embeddings \
  --device npu:0 \
  --batch-size 16 \
  --save-every 500
```

**参数**:
- `--save-every`: 每 N 个样本保存断点（默认 500）
- `--batch-size`: 16 为安全值（24 OOM）

**输出**:
- `patch_embeddings.npz`: global_mean[424,12,D], spatial_maps[424,12,D,H,W]
- `metadata.json`: 形状、checkpoint 路径、耗时

**断点续传**:
- 崩溃时保存 `patch_embeddings_partial.npz`
- 重启时自动加载并跳过已处理样本

## evaluate_knn_v2.py

KNN 三任务评估。

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

## evaluate_mlp_v2.py

MLP 下游头训练。

```bash
python scripts/eval/evaluate_mlp_v2.py \
  --embedding-file /path/to/patch_embeddings.npz \
  --output-dir /path/to/downstream \
  --device npu:0 \
  --epochs 50 \
  --month 6
```

**关键实现**:
- `PixelMLPHead(in_dim=64, hidden_dim=256, num_classes=N)`
- AdamW lr=1e-3, CosineAnnealingLR
- batch_size=1024
- 标签映射 + `label < num_classes` 过滤

## evaluate_cd_v2.py

变化检测 AUC。

```bash
python scripts/eval/evaluate_cd_v2.py \
  --embedding-file /path/to/patch_embeddings.npz \
  --output-dir /path/to/change_detection
```

**评估方法**:
- 4 个时期: Apr→Jun(38), Jun→Aug(18), Aug→Sep(25), Sep→Oct(24)
- Cosine AUC: `1 - cosine_similarity(before, after)`
- LR AUC: `LogisticRegression(class_weight='balanced')` on concat(before, after)
- 加权平均: Σ(AUC_i × n_i) / 105

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
