# 评估报告模板

用于生成完整下游评估报告。

## 报告结构

```markdown
# Round N 完整下游评估报告

> 生成时间: YYYY-MM-DD
> 实验配置: X 个实验，Y 卡 DDP，训练 Z epochs
> 评估样本: 424 patches × 12 个月 = 5088 个 embedding

---

## 一、实验列表

| 实验名称 | 核心差异 | Embedding Dim | Active Dims | Recon Loss |
|---------|---------|:-------------:|:-----------:|:----------:|

## 二、Embedding 质量指标

| 实验 | Active / Total | Std Mean | Std Max | 质量评级 |
|------|:--------------:|:--------:|:-------:|:--------:|

---

## 三、KNN 评估结果（基线参考）

| 实验 | WC Acc | WC mIoU | JW Acc | JW mIoU | DW Acc | DW mIoU |
|------|:------:|:-------:|:------:|:-------:|:------:|:-------:|

## 四、MLP 评估结果（主要指标）

| 实验 | WC Acc | WC mIoU | JW Acc | JW mIoU | DW Acc | DW mIoU |
|------|:------:|:-------:|:------:|:-------:|:------:|:-------:|

**MLP vs KNN 对比**:
- 列出 MLP 相比 KNN 的提升百分比
- 如果 MLP 提升 < 10%，说明 embedding 信息有限

## 五、变化检测评估

### 5.1 Bare AUC（Cosine Distance）

| 实验 | 全局 | June | Aug | Sep | Oct |
|------|:----:|:----:|:---:|:---:|:---:|

### 5.2 CD Head AUC（ChangeDetectionHeadV3）

| 实验 | Mean | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Best |
|------|:----:|:------:|:------:|:------:|:------:|:------:|:----:|

**Bare AUC vs CD Head AUC 对比**:
- 如果 CD Head >> Bare（> +0.15），说明 embedding 包含丰富变化信息，cosine distance 是差的变化度量
- 如果 CD Head ≈ Bare（< +0.05），说明模型确实缺乏时间敏感性

### 5.3 Few-Shot CD（K=20）

| 实验 | Global AUC | Patch AUC | 相比 CD Head |
|------|:----------:|:---------:|:------------:|

## 六、综合排名

排序依据（权重）:
1. CD Head Mean AUC (40%)
2. MLP WorldCover mIoU (25%)
3. MLP JRC Water mIoU (15%)
4. MLP Dynamic World mIoU (10%)
5. Active Dims / Total (10%)

## 七、关键发现

1. **索引正确性验证**: 是否使用了 `extract_embedding_for_month` 或 `patch_month_to_idx`？
2. **时间窗口验证**: 是否使用 2025 年月份窗口？
3. **MLP 容量瓶颈**: 增加 epoch/hidden_dim 是否有提升？
4. **Embedding 坍缩**: active_dims 是否 > 30%？
5. **CD Head vs Bare 差距**: 差距越大，说明 embedding 越丰富

## 八、下一步实验建议

1. **如果 active < 15**: 优先增大 embedding_dim（64→128）或改进时间聚合
2. **如果 CD Head > 0.80 但 Bare < 0.60**: 模型成功，cosine distance 是瓶颈
3. **如果 MLP 提升 < 10% vs KNN**: embedding 信息量不足
4. **如果 Few-Shot K=20 < 0.60**: 标注数据太少或 embedding 太弱

```

## 生成脚本示例

```python
import json
from pathlib import Path

exps = [
    ('baseline', 'round4_full_vicreg_baseline'),
    # ...
]

for name, exp in exps:
    # Embedding 质量
    # 从训练日志提取 active_dims, std_mean, recon_loss
    
    # KNN
    knn = json.load(Path(f'{exp}/evaluation/downstream/knn_summary.json').open())
    
    # MLP
    mlp = json.load(Path(f'{exp}/evaluation/downstream_mlp/mlp_summary.json').open())
    
    # CD Head
    cd = json.load(Path(f'{exp}/cd_head_v12_best.json').open())
    
    # Bare AUC
    bare = json.load(Path(f'{exp}/bare_auc.json').open())
```

## 成功标准（2026-05-20 更新）

### 变化检测

| 方法 | 及格 | 良好 | 优秀 | 说明 |
|------|:----:|:----:|:----:|:----:|
| Bare AUC | >0.55 | >0.60 | >0.65 | 容易低估，仅供参考 |
| CD Head AUC | >0.70 | >0.80 | >0.85 | 真实能力 |
| Few-Shot K=20 | >0.55 | >0.60 | >0.65 | 轻量 head |

### 下游分类（MLP 为主要指标）

| 任务 | 及格 | 良好 | 优秀 | KNN 参考 |
|------|:----:|:----:|:----:|:--------:|
| WorldCover Acc | >45% | >50% | >55% | ~35-40% |
| WorldCover mIoU | >0.18 | >0.22 | >0.25 | ~0.15-0.17 |
| JRC Water Acc | >65% | >70% | >75% | ~65-68% |
| JRC Water mIoU | >0.35 | >0.40 | >0.45 | ~0.38-0.42 |
| Dynamic World Acc | >50% | >55% | >60% | ~45-50% |
| Dynamic World mIoU | >0.12 | >0.15 | >0.18 | ~0.12-0.15 |

### Embedding 质量

| 指标 | 及格 | 良好 | 优秀 |
|------|:----:|:----:|:----:|
| Active dims / Total | >30% | >50% | >70% |
| Std mean | >0.12 | >0.15 | >0.20 |
| Reconstruction Loss | <0.15 | <0.10 | <0.05 |
