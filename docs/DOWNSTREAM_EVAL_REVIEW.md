# Few-Shot 下游评估方案全面审查报告

> 审查时间: 2026-05-16
> 审查范围: 8个v2实验的backbone训练 + 3个下游任务的few-shot评估
> 审查人: Kimi Code CLI

---

## 一、已发现的Bug及修复状态

### 🔴 严重Bug (已修复)

| # | 问题 | 影响 | 修复方案 | 状态 |
|---|------|------|----------|------|
| 1 | **embedding_map与mask尺寸不匹配** | embedding_map=[128,64,64], mask=128×128 → 训练/评估时张量维度错误直接crash | 所有mask统一下采样到64×64（最近邻保持标签） | ✅ 已修复 |
| 2 | **变化检测正负样本极端不平衡** | 正负比 884:1，模型倾向于全预测0，AUC/IoU虚高 | BCE Loss添加`pos_weight=total_neg/total_pos` | ✅ 已修复 |
| 3 | **训练epoch过多导致过拟合** | K=5时50个epoch严重过拟合小样本 | 减少到30 epoch + CosineAnnealingLR | ✅ 已修复 |
| 4 | **土地利用类别不平衡** | Bare(1.8%)、Wetland(3.6%)样本极少 | CrossEntropy添加`class_weights`逆频率加权 | ✅ 已修复 |

### 🟡 中等问题 (已修复)

| # | 问题 | 修复方案 | 状态 |
|---|------|----------|------|
| 5 | JRC Water 43×43上采样到128×128后又需下采样到64×64 | 直接上采样到64×64 | ✅ 已修复 |
| 6 | WorldCover 128×128需要下采样到64×64 | 映射类别后下采样到64×64 | ✅ 已修复 |

---

## 二、数据状态汇总

### 2.1 变化检测 (Change Detection)

```
数据来源: 4个shapefile (june/aug/september/october)
时间窗口:
  • june:      2024-04 → 2024-06  (47 patches有变化)
  • aug:       2024-06 → 2024-08  (21 patches有变化)
  • september: 2024-08 → 2024-09  (25 patches有变化)
  • october:   2024-09 → 2024-10  (28 patches有变化)

合并后统计:
  • 总窗口组合: 1696 (424 patches × 4 periods)
  • 有变化: 117 / 1696 (6.9%)
  • 总像素: 6,946,816
  • 变化像素: 7,849 (0.113%)
  • 正负比: 884:1 ⚠️ 极度不平衡
```

### 2.2 水体检测 (Water Detection)

```
数据来源: JRC Water (static.tif, 30m分辨率)
标注编码: ≥2 = 水体, 0/1 = 非水体

统计:
  • 总像素: 1,736,704 (424 patches × 64×64)
  • 水体像素: 441,478 (25.4%)
  • 正负比: 2.9:1 ✓ 相对平衡
```

### 2.3 土地利用分割 (Land Use Segmentation)

```
数据来源: WorldCover (static.tif, 10m分辨率)
类别映射 (7类):
  0: Tree cover      17.3%
  1: Grassland        8.7%
  2: Cropland        25.3%
  3: Built-up        18.1%
  4: Bare/sparse      1.8% ⚠️ 极少
  5: Water           25.4%
  6: Wetland          3.6% ⚠️ 较少
```

---

## 三、Few-Shot评估协议设计

### 3.1 核心参数

| 参数 | 设置 | 说明 |
|------|------|------|
| K-shot | 5, 10, 20, 50 | 训练patch数量 |
| N-splits | 5 | 每个K随机划分5次取平均 |
| Head架构 | 2-layer Conv | 参考AEF论文 supervised model |
| 优化器 | AdamW | lr=1e-3, weight_decay=1e-4 |
| 训练epoch | 30 | CosineAnnealingLR |
| 损失函数 | BCE+Dice (CD/Water) / CE (LU) | 带类别权重 |

### 3.2 评估指标

| 任务 | 指标 |
|------|------|
| 变化检测 | Global AUC, Patch AUC, IoU, Precision, Recall, F1 |
| 水体检测 | Global AUC, IoU, Precision, Recall, F1 |
| 土地利用 | Pixel Acc, Balanced Acc, F1-macro, mIoU, per-class IoU |

---

## 四、仍存在的风险与优化建议

### 4.1 🟠 时间窗口对齐风险 (中等)

**问题**: shapefile中的变化图斑时间是否对应2024年？

```
现状: extract_embeddings.py使用2024年时间窗口
      fewshot_change_detection.py原代码使用2025年时间窗口
      
风险: 如果实际变化发生在2025年，而embedding提取的是2024年数据，
      则before/after embedding可能包含相同的地物状态，导致变化信号弱
      
验证: 需要检查shapefile的采集时间，确认是否与2024年数据对齐
```

**建议**: 训练完成后，先验证一个patch的before/after embedding差异是否明显

### 4.2 🟠 无验证集/早停 (中等)

**问题**: 当前方案训练固定30 epoch，没有验证集选择最佳模型

**风险**: K=5时30 epoch仍可能过拟合，不同split间方差大

**建议**:
```python
# 方案A: 减少epoch到20
# 方案B: 从K-shot训练集中留出1个patch作为验证集
# 方案C: 使用Linear Probe作为对照（AEF论文标准做法）
```

### 4.3 🟡 K-shot划分未保证类别平衡 (低)

**问题**: 随机划分可能导致训练集中全是负样本（变化检测）

**建议**: 对变化检测使用分层抽样，确保训练集中有一定比例的正样本

### 4.4 🟡 缺少Linear Probe对照 (低)

**问题**: AEF论文下游评估的核心是Linear Probe，而我们使用2-layer Conv

**建议**: 未来补充Linear Probe (sklearn LogisticRegression) 作为对照实验

---

## 五、文件清单

### 已创建/修改的文件

```
scripts/downstream/
  ├── generate_change_masks.py      ✅ 4个月份段mask生成 (64×64)
  ├── prepare_labels.py             ✅ 水体/土地利用标注 (64×64)
  ├── extract_embeddings.py         ✅ Embedding提取 (5个月份)
  ├── fewshot_eval_unified.py       ✅ 统一Few-Shot评估 (3任务)
  ├── run_full_pipeline_v2.py       ✅ Master Pipeline V2
  └── evaluate_all.py               ✅ 全面评估 (备用)

src/downstream/heads.py             ✅ 轻量下游Heads定义

data/
  ├── change_masks/                 ✅ 4个period
  │   ├── june/      (47 patches有变化)
  │   ├── aug/       (21 patches有变化)
  │   ├── september/ (25 patches有变化)
  │   └── october/   (28 patches有变化)
  └── labels/
      ├── water/     ✅ 424 patches
      ├── building/  ✅ 424 patches (已弃用)
      └── landuse/   ✅ 424 patches
```

---

## 六、执行计划

### 阶段1: Backbone训练 (进行中)
- 8个实验并行训练，各1 NPU
- 目标: 10 epochs
- 预计时间: 2-3小时

### 阶段2: Embedding提取 (自动触发)
- 每个实验提取100 patches × 5个月份 = 500个embedding
- 输出: `data/embeddings/{exp_name}/{pid}_{month}.npy`

### 阶段3: Few-Shot评估 (自动触发)
- 每个实验运行3个任务 × 4个K-shot × 5 splits = 60次评估
- 预计时间: 30-60分钟/实验

### 阶段4: 结果汇总
- 输出: `/workspace/outputs/{exp_name}_10ep/fewshot_results.json`

---

## 七、关键监控指标

### Backbone训练监控
```bash
# 实时查看
for s in $(tmux list-sessions -F '#{session_name}' | grep v2_); do
  echo "=== $s ==="
  tmux capture-pane -t $s -p | grep -E "Epoch|vicreg|raw_unif|cov" | tail -5
done
```

### 关键阈值
| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| vicreg | < 5.0 | > 10 表示cov爆炸 |
| raw_unif | -4.0 ~ -1.0 | > -0.5 表示坍缩 |
| cov | < 1.0 | > 5.0 严重失控 |

### Few-Shot评估结果预期
```
变化检测 (K=20):
  • Global AUC: 0.60-0.80 (基线)
  • IoU: 0.05-0.20 (由于极端不平衡)

水体检测 (K=20):
  • Global AUC: 0.85-0.95
  • IoU: 0.50-0.70

土地利用 (K=20):
  • Balanced Acc: 0.40-0.60
  • mIoU: 0.25-0.45
```

---

## 八、后续优化方向

1. **Linear Probe对照**: 补充sklearn LogisticRegression评估，与AEF论文对齐
2. **分层抽样**: 变化检测K-shot划分时保证正负样本比例
3. **验证集早停**: 从训练集留1个patch作为验证集选最佳epoch
4. **多尺度评估**: 同时在64×64和128×128（上采样embedding）上评估
5. **混淆矩阵**: 输出per-class混淆矩阵，分析错误模式
6. **跨实验对比**: 自动绘制8个实验的K-shot曲线对比图

---

*最后更新: 2026-05-16*
