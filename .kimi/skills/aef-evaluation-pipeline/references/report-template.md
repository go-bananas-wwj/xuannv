# 评估报告模板

用于生成 Round N 完整下游评估报告。

## 报告结构

```markdown
# Round N 完整下游评估报告

> 生成时间: YYYY-MM-DD
> 实验配置: X 个实验，Y 卡 DDP，训练 Z epochs
> 评估样本: 424 patches × 12 个月 = 5088 个 embedding

---

## 一、实验列表

| 实验名称 | 核心差异 |
|---------|---------|

## 二、KNN 评估结果

| 实验 | WC Acc | WC mIoU | JW Acc | JW mIoU | DW Acc | DW mIoU |

## 三、MLP 评估结果

| 实验 | WC Acc | WC mIoU | JW Acc | JW mIoU | DW Acc | DW mIoU |

## 四、变化检测 AUC

| 实验 | LR AUC | 对比 baseline |

## 五、综合排名

按下游分类综合 + AUC 排序

## 六、关键发现

1. 时序敏感性 trade-off
2. 一致性正则的双刃剑
3. 重建权重的平衡
4. VMF kappa 的影响

## 七、下一步实验建议

1. 混合方案
2. 降低 high_consist 权重
3. gap-aware 时序
4. 继续 Round N+1
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
    # KNN
    knn = json.load(Path(f'{exp}/evaluation/downstream/knn_summary.json').open())
    # MLP
    mlp = json.load(Path(f'{exp}/evaluation/downstream/mlp_summary.json').open())
    # CD
    cd_log = Path(f'{exp}/evaluation/cd_v2.log').read_text()
    # parse AUC
```

## 成功标准

| 指标 | 及格 | 良好 | 优秀 |
|------|------|------|------|
| CD AUC (LR) | >0.55 | >0.65 | >0.75 |
| WC KNN Acc | >30% | >35% | >40% |
| DW KNN Acc | >40% | >45% | >50% |
| JW KNN Acc | >15% | >20% | >25% |
