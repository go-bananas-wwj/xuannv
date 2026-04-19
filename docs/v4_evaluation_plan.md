# V4 Official 完整评估计划

> 生成时间：2026-04-15  
> 模型：AEF_qwen_v4_official (epoch_best_epoch231.pt)  
> 核心改进：uniformity = -3.04 (vs V2/V3 ~-0.55), 真实 WorldCover 分类标签, Student 扰动

---

## 一、已完成工作

### 1.1 权重选择
| Checkpoint | Total Loss | Uniformity | 选择理由 |
|-----------|-----------|-----------|---------|
| epoch_300 | -0.92 | -2.80 | 最终 epoch，uniformity 略差 |
| **epoch_231** | **-1.20** | **-3.04** | ✅ **选中** — 更低的 total loss + 更好的 embedding 分散度 |

### 1.2 Embedding 生成
- **输出目录**: `/workspace/outputs/aef_qwen_v4_official/monthly_embeddings_2025/`
- **规模**: 424 patches × 5 months = 2,120 个 `.npy` 文件
- **Shape**: `[128, H, W]` (128-dim, L2 normalized)
- **月份**: 2025-04, 2025-06, 2025-08, 2025-09, 2025-10

### 1.3 Level 1 — Backbone Bare AUC (已完成)
**方法**: 直接在 20 个高密度标注 patch 上，用 backbone 输出的 L2 normalized embedding 计算 cosine distance 作为变化分数。

| 指标 | V2 Baseline | V4 Official | Delta |
|------|------------|-------------|-------|
| AUC mean | ~0.49 | **0.503** | +0.013 |
| AUC > 0.6 | - | 2/19 | - |
| AUC > 0.7 | - | 0/19 | - |

**关键发现**: `changed` vs `unchanged` 区域的 cosine distance 几乎无差异（< 0.02）。
**结论**: Raw cosine distance 不是有效变化指标，**CD Head 必需**。

### 1.4 Level 3 — CD Head 训练与验证 (已完成)
**方法**: 冻结 V4 backbone，在月度 embedding 对上训练 ChangeDetectionHeadV3 + OHEM。

| 指标 | V2 Baseline | V4 Official | Delta |
|------|------------|-------------|-------|
| 单折 Val AUC | 0.908 (V3) | **0.908** | 持平 |
| **5-Fold Mean AUC** | **0.836** | **0.896** | **+0.060** 🚀 |
| 5-Fold Median AUC | - | **0.918** | - |
| Fold range | - | 0.854 ~ 0.933 | - |

**结论**: V4 embedding 的更高质量（uniformity -3.04）使 CD Head 学习效果显著提升。

---

## 二、待执行评估

### Level 2 — AlphaEarth 官方 Embedding 对比

#### 目标
回答核心问题：**V4 backbone bare 的 cosine distance 是否超越了 AlphaEarth 官方 embedding 的同类表现？**

#### 背景
- AlphaEarth 官方 embedding（64-dim vMF）在其论文中报告 **unsupervised CD BA = 71.3%**。
- 我们之前的对比实验（`alphaearth_auc_report.md`）显示：
  - AlphaEarth backbone bare AUC (87 patches, 2023 vs 2024): **0.658**
  - V2 backbone bare AUC (同期): **0.487**

#### 方法
复用 `scripts/eval/evaluate_alphaearth_cd_auc.py`，但将 V2 替换为 V4：

```python
# 在 evaluate_alphaearth_cd_auc.py 基础上新增 V4 分支
# 1. 加载 AlphaEarth 2023/2024 GeoTIFF (64-dim)
# 2. 加载 V4 月度 embedding (2025-04 vs 2025-10 近似全年)
# 3. 在同一 87 个 patch + 标注上分别计算 AUC
# 4. 对比: AlphaEarth vs V4 backbone bare cosine distance
```

#### 预期
- AlphaEarth 全年 embedding 质量很高（AUC ~0.66 bare）。
- V4 的月度 embedding 时间窗口更窄，可能更有优势（季节差异更小）。
- **目标**: V4 bare AUC > 0.60 即算有竞争力。

#### 输出
- `/workspace/outputs/aef_qwen_v4_official/eval/alphaearth_vs_v4_report.md`

---

### Level 3b — Full 69-Patch Benchmark (补充)

#### 目标
V2 有一个 **full 69 patches** 的 benchmark（非 5-fold），AUC = 0.8840。V4 也需要这个 benchmark 来做公平对比。

#### 方法
复用 `scripts/eval/benchmark_monthly_cd_head.py`，改为 V4 路径：
1. 加载 `monthly_cd_head_v3_ohem.pt` (已训练好)
2. 在所有 69 个有标注的 patch 上运行 head prediction
3. 计算 patch-level AUC mean/median/std

#### 预期
- V4 full benchmark AUC 应该 **> 0.90**（基于 5-fold mean = 0.896）。

---

### Level 4 — Few-Shot 分类评估

#### 目标
验证 V4 embedding 的 **语义表征质量** — 这是 V4 相比 V2/V3 最大的改进（使用了真实 WorldCover 分类标签）。

#### 背景
- AlphaEarth 论文核心卖点：在 **1-shot / 10-shot / max-trial** 下的 land cover 分类 BA。
- V2/V3 使用 dummy 分类标签（全 0），分类头完全无用。
- V4 使用真实 11 类 WorldCover 标签训练，分类头应该学到了有意义的语义信息。

#### 方法
设计 few-shot land cover classification benchmark：

**A. 数据准备**
1. 选取 20 个 patch 的 2025-08 embedding（夏季，云少，质量最好）。
2. 用 WorldCover 栅格作为逐像素标签（10m → 64×64 grid）。
3. 合并 11 类到 8 类（合并稀疏类）：
   - Tree cover, Shrubland, Grassland, Cropland, Built-up, Bare/sparse, Snow/ice, Water, Wetland, Mangroves, Moss → 合并后 8 类

**B. Few-shot 协议**
- 对每个类别，随机抽取 K 个像素作为训练样本（K = 1, 5, 10, 50, 100）。
- 用 **Linear Probe** 和 **kNN (k=1, 3)** 训练分类器。
- 在剩余像素上测试，报告 **Balanced Accuracy (BA)** 和 **Overall Accuracy (OA)**。
- 每个 K 值重复 5 次，取平均。

**C. 对比基线**
- V2 embedding（无真实分类训练）
- AlphaEarth embedding（若可获取）
- 随机猜测 baseline

#### 预期
- V4: BA > 60% (10-shot), > 70% (max-trial)
- V2: BA ~ 20-30%（随机水平，因为分类头没学到东西）

#### 输出
- `/workspace/outputs/aef_qwen_v4_official/eval/fewshot_classification_report.json`

---

## 三、评估优先级与时间估计

| Level | 任务 | 优先级 | 预计时间 | 状态 |
|-------|------|--------|---------|------|
| 1 | Backbone bare AUC (20 patches) | P0 | 10 min | ✅ 完成 |
| 2 | AlphaEarth 对比 (87 patches) | P1 | 30 min | ⏳ 待执行 |
| 3a | CD Head 5-fold crossval | P0 | 30 min | ✅ 完成 |
| 3b | CD Head full 69-patch benchmark | P1 | 10 min | ⏳ 待执行 |
| 4 | Few-shot classification | P2 | 1-2 hr | ⏳ 待执行 |

---

## 四、关键脚本清单

| 脚本 | 用途 | 修改点 |
|------|------|--------|
| `scripts/eval/validate_v4_level1_bare.py` | Level 1 | ✅ 已完成 |
| `scripts/train/train_v4_monthly_cd_head.py` | Level 3 CD Head 训练 | ✅ 已完成 |
| `scripts/train/crossval_v4_monthly_cd_head.py` | Level 3 5-fold | ✅ 已完成 |
| `scripts/eval/benchmark_monthly_cd_head.py` | Level 3b full benchmark | 改 EMBEDDING_DIR + HEAD_PATH → V4 |
| `scripts/eval/evaluate_alphaearth_cd_auc.py` | Level 2 AlphaEarth 对比 | 新增 V4 分支 |
| `scripts/eval/fewshot_v4_classification.py` | Level 4 Few-shot | 新建 |

---

## 五、SOTA 参照系

### 遥感变化检测 AUC 基准
| 数据集难度 | SOTA AUC | 我们的场景 |
|-----------|---------|-----------|
| 简单（同质） | >98% | 不适用 |
| 中等 | ~94-97% | 部分适用 |
| 困难城市异构 | ~92% | **最接近** |

**我们的 AUC = 0.896 (5-fold)** 处于中等偏上水平，考虑到：
- 输入是**异构多源时序数据**（S2 + S1 + Landsat），而非简单的双时相对比
- 时间窗口跨越**1.5 年**（2023Q3-Q4 vs 2024Q3-2025Q4），包含显著季节噪声
- 标注是**真实建筑变化**，变化率极低（~1.8% positive pixels）

### AlphaEarth 官方指标
| 协议 | AlphaEarth BA | 我们对标方式 |
|------|--------------|-------------|
| Unsupervised CD | 71.3% | Level 2 (AUC 转 BA) |
| Supervised 10-shot LC | ~78% | Level 4 (few-shot) |

---

## 六、下一步行动

1. **[立即]** 运行 Level 3b: `benchmark_monthly_cd_head.py` 适配 V4 → 获取 full 69-patch AUC
2. **[立即]** 运行 Level 2: 修改 `evaluate_alphaearth_cd_auc.py` 加入 V4 分支 → AlphaEarth vs V4 对比
3. **[今日]** 运行 Level 4: 新建 few-shot classification 脚本 → 验证语义表征质量
4. **[今日]** 汇总所有结果到 `v4_final_report.md`
