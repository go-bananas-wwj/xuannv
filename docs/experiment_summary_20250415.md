# AEF_qwen 双实验总结报告

> 生成时间：2025-04-15  
> 实验范围：HR-only Small Model (GPUs 3,4) + 主 Finetune (GPUs 5,6,7)  
> 核心目标：修复 temporal embedding 坍缩，提升变化检测 AUC

---

## 一、实验概览

本次共进行了两个并行实验：

| 实验 | 模型规模 | GPU | 数据特点 | 训练轮数 |
|---|---|---|---|---|
| **Experiment A — 主 Finetune** | 57.5M 参数 | 5,6,7 | `s2`+`s1`+`landsat` (2023–2024, 1 年间隔) | 20 epochs (已中断) |
| **Experiment B — HR-only Small** | 7.3M 参数 | 3,4 | `s2_hr`+`s1_hr` (2025 Apr–Oct, 6 个月间隔) | 99+ epochs (已中断) |

---

## 二、核心修复尝试

### 2.1 Anti-Diagonal InfoNCE (Global)
- **原理**：将同一 patch 不同时窗的 embedding 对 `(w1_i, w2_i)` 视为"负样本"，不同 patch 的交叉对 `(w1_i, w2_j, i≠j)` 视为"正样本"（相对更近）。
- **实现**：`F.cross_entropy(-logits, labels)`
- **意图**：防止模型通过"全局时间偏移"作弊。

### 2.2 Pixel-Level Anti-Diagonal InfoNCE
- **原理**：在空间 embedding map 上随机采样 16 个像素位置，对每个位置独立计算 Anti-Diagonal InfoNCE。
- **意图**：强制模型在每个局部空间位置都产生时间差异，从而自然形成"某些区域差异大、某些区域差异小"的空间模式。

### 2.3 数据采样增强
- **6 个月最小间隔**：在 `dataset.py` 中强制 `w1` 与 `w2` 的中心点间隔 ≥ 6 个月，确保训练分布与验证分布对齐。

### 2.4 权重激进上调
- `temporal_contrastive_weight: 5.0`
- `pixel_temporal_weight: 3.0`
- `pre_norm_uniform_weight: 5.0`
- `encoder_uniform_weight: 3.0`

---

## 三、HR-only Small Model 结果

### 3.1 训练指标（99 epochs）
- **Recon**: 从 7.54 降至 ~5.30，稳定收敛
- **Temporal**: 从 17.47 快速收敛至 ~0.03
- **PixTemp**: 从 6.12 收敛至 ~0.08
- **RawUnif**: 从 -1.90 改善至 -3.79

### 3.2 验证结果（`epoch_99.pt`）
| 指标 | 数值 |
|---|---|
| **AUC mean** | **0.495** |
| **AUC median** | **0.509** |
| **Mean embedding distance** | **0.495** |
| AUC > 0.55 | 9/19 |
| AUC > 0.60 | 5/19 |
| AUC > 0.70 | 2/19 |

### 3.3 可视化诊断
生成了 3 个代表 patch 的全域分割图和相邻月份变化强度图，关键发现：
1. **WorldCover 分类头已完全坍缩**：分割图几乎全为单一颜色（橙色），说明 7.3M 参数的小模型从头训练无法有效学习 11 类地物分类。
2. **变化强度缺乏空间定位能力**：青色标注框（真实变化区域）内的变化强度与周围背景没有显著差异。
3. **时间全局偏置**：Sep-Oct（9 月 vs 10 月）总是全图高亮，说明模型学到的是"时间越靠后差异越大"的全局偏置，而不是真正的地表变化。

### 3.4 结论
HR-only small 模型由于以下根本性限制，无法达到可用水平：
- **时间跨度太短**（仅 6 个月），真实建筑变化的影像特征不够显著
- **参数量太小**（7.3M），难以同时承担重建、分类和精细变化检测
- **缺少 LR 长时序信息**辅助，没有年际变化模式可学习

---

## 四、主 Finetune 结果

### 4.1 第一次验证（`epoch_19.pt`，修正前）
| 指标 | Finetune (新) | From Scratch (旧基线) |
|---|---|---|
| **AUC mean** | **0.488** | **0.504** |
| **Mean embedding distance** | **0.030** | **0.030** |
| AUC > 0.6 | 0/19 | 0/19 |

**Finetune 甚至不如旧的 From Scratch。**

### 4.2 根因分析：灾难性配置错误

在 `configs/qwen_v2_hr_finetune.yaml` 中，`filter_2025_monthly: true` 被错误保留：

- 数据加载代码 `_load_input_frames` 会过滤掉所有不在 **2025 年 4–10 月** 范围内的帧。
- 主模型的三个输入源（`s2`、`s1`、`landsat`）的文件名全部是 **2023–2024 年**（如 `20230103.tif`）。
- 因此训练时，这三个输入源全部被过滤成**零帧/空张量**。
- `s2_hr` / `s1_hr` 的合并逻辑依赖于在 `s2`/`s1` 目录中找到同名文件；既然这些文件被过滤掉了，HR 帧也**永远不会被合并进来**。

**结果**：模型实际上是在**空输入**上训练了 20 个 epoch，预训练权重 `best.pt` 的时间敏感表征被完全洗掉。

### 4.3 已应用的修复
- 将 `filter_2025_monthly: true → false`
- 删除被污染的检查点 `epoch_19.pt`
- 从 `best.pt` 重新启动训练（Epoch 1 已完成）
- **但训练在后续被用户要求全部停止**

---

## 五、Anti-Diagonal InfoNCE 的结构性缺陷

即使抛开数据过滤 bug，单独审视 Anti-Diagonal InfoNCE 本身：

- 它只约束了**相对排序**（对角线相似度必须最低），但没有约束**绝对距离**。
- 模型学会了"柔和坍缩（soft collapse）"策略：把所有 `w1` 和 `w2` 的 cosine similarity 都挤在 **0.94–0.96** 的极窄范围内，只要对角线略低一点点（如 0.940 vs 0.945），cross-entropy 就能收敛到 0.1–0.2。
- 这导致 L2 normalized embedding 的**方向几乎完全相同**，验证 distance 仅约 **0.03**。

**结论**：InfoNCE 必须配合 **Hinge Margin** 或 **L2 Distance Maximization** 一起使用，才能真正拉开 embedding。

---

## 六、关键文件修改记录

| 文件 | 修改内容 |
|---|---|
| `src/training/losses.py` | 新增 `temporal_info_nce_loss` (Anti-Diagonal) 和 `pixel_temporal_info_nce_loss` |
| `src/models/model.py` | `encode_dual_window` 返回 `pre_norm_map`；修复 `num_tgt` 动态读取；添加 `self.cfg` 保存 |
| `src/training/trainer.py` | 接入 Global + Pixel temporal loss；新增 `PixTemp` 日志输出 |
| `src/data/dataset.py` | 强制双窗口中心间隔 ≥ 6 个月；修复 `input_sources` 为 `None` 时的回退逻辑 |
| `src/config.py` | `TrainingConfig` 补全 `temporal_loss_type`、`pixel_temporal_weight`、`pixel_temporal_samples` 字段 |
| `configs/qwen_v2_hr_finetune.yaml` | `lr: 5e-5`、权重激进上调、`filter_2025_monthly: false`（已修正） |
| `configs/qwen_v2_hr_only_small.yaml` | 新增 2 输入 + 4 目标配置，权重上调 |
| `validate_hr_only.py` | 新建：针对 HR-only 的验证脚本（2025 Apr-May vs Sep-Oct） |
| `visualize_hr_changes.py` | 新建：生成小模型的分割图和相邻月份变化强度可视化图 |

---

## 七、遗留问题与后续建议

### 7.1 如果继续训练主 Finetune
由于 `filter_2025_monthly` 已修正，主模型现在可以正常读取 2023–2024 数据。但 Anti-Diagonal InfoNCE 的 soft collapse 问题仍未解决。建议：

```python
# 在 temporal_info_nce_loss 中加入 margin 约束
rank_loss = F.cross_entropy(-logits, labels)           # 防排序作弊
diag_sim = (flat_w1 * flat_w2).sum(-1)                 # [B]
margin_loss = F.relu(diag_sim - 0.0).mean() / temp     # 强制 ≤ 0.0
loss = rank_loss + margin_loss
```

这样模型无法再把 similarity 维持在 0.94，必须真正推到 **正交或相反** 的方向。

### 7.2 如果继续训练 HR-only Small
不建议投入更多时间。6 个月时间跨度 + 7.3M 参数的 ceiling 已肉眼可见。

### 7.3 快速验证策略
- 主模型 Epoch 20 后运行 `validate_comparison.py`
- 若 AUC > 0.60 且 distance > 0.15，说明修复方向正确
- 若仍低于 0.55，需重新设计 temporal loss（InfoNCE + Hinge 混合）

---

## 八、可视化文件位置

HR-only small 的相邻月份变化强度和分割图保存在：
```
/workspace/outputs/aef_qwen_v2_hr_only_small/visualizations/
```
包含：
- `patch_000146_change_intensity_annotated.png`
- `patch_000230_change_intensity_annotated.png`
- `patch_000235_change_intensity_annotated.png`
- `patch_*_segmentation.png`
- `patch_*_embedding_rgb.png`

---

**报告结束。**
