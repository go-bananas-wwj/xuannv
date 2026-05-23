# Round 4 评估计划（Grill-Me 版本）

## 当前状态
- 7个主要实验已完成（Epoch 20/20）
- emb128 仍在训练中（Epoch ~9/20，预计1小时后完成）

---

## 评估目标

| 优先级 | 目标 | 成功标准 |
|--------|------|---------|
| P0 | 找出 AUC > 0.55 的实验 | >0.55 及格，>0.7 良好 |
| P0 | 找出 KNN WorldCover > 30% 的实验 | >30% 及格 |
| P1 | 比较8个实验的 embedding 质量 | std_mean、active_dims、separation |
| P2 | 确定最佳超参数组合 | 为 Round 5 提供方向 |

---

## Phase 1: Checkpoint 确认（5分钟）

**做什么**：确认每个实验保存了有效的 best checkpoint

**具体命令**：
```bash
for exp in round4_full_{vicreg_baseline,high_var,high_temporal,low_recon,high_consist,high_kappa,low_decoder}; do
    ls -lh /workspace/outputs/xuannv_round2/$exp/epoch_best_*.pt
    ls -lh /workspace/outputs/xuannv_round2/$exp/epoch_20.pt
done
```

**预期输出**：每个实验应有 1-2 个 .pt 文件（~168MB）

**如果失败**：检查 train.log 最后几行，确认是否因 OOM 崩溃

---

## Phase 2: AUC 验证 — 变化检测能力（每实验 ~3分钟，共 ~25分钟）

**做什么**：对每个实验运行 validate_v12_bare.py，计算 Bare AUC

**为什么先做 AUC**：AUC 是核心指标，如果 AUC 仍然 ~0.5，说明即使数据量扩大到 424 patches，自监督框架仍然失败

**具体命令**（以 baseline 为例）：
```bash
cd /workspace/xuannv
source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv
export ASCEND_RT_VISIBLE_DEVICES=0

CKPT=$(ls -t /workspace/outputs/xuannv_round2/round4_full_vicreg_baseline/epoch_best_*.pt | head -1)

python scripts/eval/validate_v12_bare.py \
    --config configs/round4_8gpu/round4_full_vicreg_baseline.yaml \
    --checkpoint $CKPT \
    --device npu:0 \
    > /workspace/outputs/xuannv_round2/round4_full_vicreg_baseline/auc_eval.log 2>&1
```

**预期结果**：
- AUC > 0.55：说明数据量扩大产生了质变
- AUC ~0.5：说明问题不在数据量，需要改变框架
- Changed mean dist > Unchanged mean dist：separation 为正

**输出文件**：每个实验目录下的 `auc_eval.log`

---

## Phase 3: Embedding 提取（每实验 ~10分钟，共 ~70分钟）

**做什么**：加载每个实验的 checkpoint，提取所有 424 patches × 12 months = 5088 个样本的 embedding

**为什么需要提取**：KNN 评估需要预计算的 embedding，且 extract_embeddings_for_knn.py 会生成下游任务所需的特征向量

**具体问题**：`extract_embeddings_for_knn.py` 硬编码了 `/workspace/outputs/xuannv_round1/{exp}` 路径

**解决方案**：为每个实验创建 symlink
```bash
for exp in round4_full_{vicreg_baseline,high_var,high_temporal,low_recon,high_consist,high_kappa,low_decoder}; do
    ln -sf /workspace/outputs/xuannv_round2/$exp /workspace/outputs/xuannv_round1/$exp
done
```

**具体命令**：
```bash
cd /workspace/xuannv
export ASCEND_RT_VISIBLE_DEVICES=0

for exp in round4_full_{vicreg_baseline,high_var,high_temporal,low_recon,high_consist,high_kappa,low_decoder}; do
    echo "Extracting: $exp"
    python scripts/eval/extract_embeddings_for_knn.py \
        --experiment $exp \
        --device npu:0 \
        > /workspace/outputs/xuannv_round2/$exp/extract.log 2>&1
done
```

**预期输出**：每个实验目录下生成 `embeddings/` 文件夹，包含 .npy 文件

**如果失败**：
- OOM：降低 batch size 或分批提取
- 路径错误：检查 symlink 是否正确

---

## Phase 4: KNN 下游评估（每实验 ~2分钟，共 ~15分钟）

**做什么**：对提取的 embedding 运行 KNN 分类，评估 WorldCover、JRC Water、Dynamic World

**具体命令**：
```bash
for exp in round4_full_{vicreg_baseline,high_var,high_temporal,low_recon,high_consist,high_kappa,low_decoder}; do
    echo "KNN: $exp"
    python scripts/eval/run_knn_npu_fast.py \
        --experiment $exp \
        --device npu:0 \
        > /workspace/outputs/xuannv_round2/$exp/knn_eval.log 2>&1
done
```

**预期结果**：每个实验生成 `downstream_knn/results.json`

**成功标准**：
- WorldCover k20 accuracy > 30%
- Dynamic World k20 accuracy > 35%
- JRC Water k20 accuracy > 10%（基线很低，稍有提升就算成功）

---

## Phase 5: 可视化对比（~5分钟）

**做什么**：生成8个实验的对比图表

**具体命令**：
```bash
python scripts/eval/generate_comparison_viz.py \
    --output-dir /workspace/outputs/xuannv_round2
```

**预期输出**：
- `comparison_viz/knn_results_comparison.png`
- `comparison_viz/master_overview_all.png`
- 每个实验目录下的 `viz/all_periods_summary.png`

---

## Phase 6: 汇总报告（~5分钟）

**做什么**：读取所有 AUC 和 KNN 结果，生成汇总 JSON

**报告格式**：
```json
{
  "round4_results": {
    "round4_full_vicreg_baseline": {
      "auc": 0.55,
      "worldcover_k20": 0.32,
      "jrc_water_k20": 0.08,
      "dynamic_world_k20": 0.36,
      "std_mean": 1.05,
      "active_dims": "64/128"
    },
    ...
  },
  "best_experiment": "round4_full_xxx",
  "conclusion": "..."
}
```

---

## 关键假设和风险

1. **假设**：extract_embeddings_for_knn.py 能正确处理 424 patches 的数据集
   - **风险**：脚本可能硬编码了 100 patches 的期望，需要验证

2. **假设**：NPU 0 在评估时不会被 emb128 训练抢占
   - **风险**：emb128 在 NPU 6 上训练，评估用 NPU 0，无冲突

3. **假设**：AUC 验证脚本支持 424 patches 的 grid
   - **风险**：grid geojson 有 424 个 feature，标注 105 个，应该正常

---

## 时间估算

| Phase | 时间 | 是否可以并行 |
|-------|------|-------------|
| Checkpoint 确认 | 5分钟 | 否 |
| AUC 验证 | 25分钟 | 否（串行用1张NPU）|
| Embedding 提取 | 70分钟 | 否（串行用1张NPU）|
| KNN 评估 | 15分钟 | 否（串行用1张NPU）|
| 可视化 | 5分钟 | 否 |
| 汇总报告 | 5分钟 | 否 |
| **总计** | **约2小时** | |

**加速方案**：如果有第2张空闲NPU，可以并行跑2个实验的提取

---

## 决策点（Grill-Me）

1. **是否等 emb128 完成再一起评估？**
   - 选项A：等 emb128（+1小时），8个一起评估
   - 选项B：先评估7个，emb128 完成后单独补评

2. **AUC 如果仍然 ~0.5，下一步做什么？**
   - 选项A：放弃自监督，直接用变化检测标注训练 CD Head
   - 选项B：尝试不同的自监督任务（如掩码预测、跨源重建）
   - 选项C：检查数据质量，确认输入数据没有问题

3. **评估顺序**
   - 选项A：按我写的顺序（AUC → KNN → Viz）
   - 选项B：先快速过一遍所有AUC，确认有改善再做完整KNN

---

请逐条审阅，有任何不同意或需要修改的地方直接告诉我。
