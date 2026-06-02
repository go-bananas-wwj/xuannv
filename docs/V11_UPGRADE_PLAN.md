# V11 全面升级方案：从数据扩展到架构重构

> 基于 2026-05-10 深度研究综合报告  
> 当前状态：V10 Difference Module E041 训练中，recon=0.32, raw_unif=-3.22, bare AUC 历史 ~0.495

---

## 一、核心问题诊断

### 1.1 根本瓶颈：数据量不足

| 维度 | AEF 论文 | 当前 V10 | 差距 |
|------|---------|---------|------|
| 训练序列 | 840 万 | 424 patches | **1:20,000** |
| 地理覆盖 | 全球 1.1% 陆地 | 哈尔滨单城市 | 单一区域 |
| 有效 batch | 256 | 32 | 8x |
| 训练步数 | 100,000 | ~4,000 | **1:25** |

**结论**：424 patches 无法驱动 57M 参数的自监督模型学习通用时序表征。

### 1.2 架构偏离：我们加了太多 AEF 没有的东西

AEF 原始损失：`Recon(1.0) + BatchUniformity(0.05) + Consistency(0.02) + TextCLIP(0.001)`

我们的 V10 损失：Recon + Uniformity(1.0) + Var(0.25) + Decorr(0.05) + Orth(0.01) + Temporal(0.02→0.08) + PixelChange(0.05) + ChangeConsist(0.05) + Consistency(0.05) + ...

**问题**：
- 损失数量过多 → 梯度冲突
- Uniformity weight 是 AEF 的 20 倍
- Consistency 未作为核心机制
- 没有 Text Contrastive（但 AEF weight 只有 0.001，影响有限）

### 1.3 静态-动态目标比例失衡

| 目标类型 | AEF | 我们 |
|---------|-----|------|
| 动态 | 7 个 (78%) | 3 个 (43%) |
| 静态/准静态 | 2 个 (22%) | 4 个 (57%) |
| 静态目标权重 | NLCD=0.5 | 全部为 1.0 |

**静态目标过多可能让模型倾向于学习时间无关的表征。**

### 1.4 嵌入质量诊断

| 指标 | V9 E80 | 健康阈值 | 评价 |
|------|--------|---------|------|
| raw_unif | -3.2 | [-4, -1] | ✅ 几何健康 |
| recon | 0.27 | < 0.3 | ✅ 重建可接受 |
| **bare AUC** | **0.495** | **> 0.7** | ❌ **时间盲** |
| cd_mean changed | 0.0152 | — | ❌ 变化信号≈0 |
| cd_mean unchanged | 0.0154 | — | ❌ 无区分度 |

**核心矛盾**：嵌入不坍缩，但语义上完全时间盲。

---

## 二、关键发现：可扩展的数据资源

### 2.1 现有未充分利用的数据

| 数据集 | Patches | S2 Frames | 完整度 | 备注 |
|--------|---------|-----------|--------|------|
| 哈尔滨云筛选 (当前) | 425 | ~9,300 | ✅ 完整 | 每月保留 2 帧 |
| 哈尔滨原始日度 | 424 | ~29,700 | ✅ 完整 | 未云筛选，182 帧/patch |
| 雅江 | 1,708 | ~22,200 | ✅ 完整 | **用户确认效果不佳，不使用** |
| 雅江扩展 | 2,884 | ~36,700 | ⚠️ 不完整 | 不使用 |

**关键发现**：哈尔滨原始数据有 182 帧/patch（云筛选前），当前只用了 22 帧/patch。**帧数可扩展 8 倍**。

### 2.2 用户确认的数据获取能力

- ✅ **Google Earth Engine 账号**：weijiewu0306@gmail.com
- ✅ **terragon 下载工具**：https://github.com/drnhhl/terragon （之前使用过）
- ✅ **愿意增加帧数**：接受重新云筛选，保留每月 3-4 帧

### 2.3 黑龙江省数据扩展方案

**目标**：从哈尔滨单一城市扩展到黑龙江省多个城市。

**候选城市**（按优先级）：
1. 齐齐哈尔（西部，干旱平原，农业区）
2. 大庆（石油城市，盐碱地）
3. 牡丹江（东南部，山地森林）
4. 佳木斯（东北部，三江平原，农业区）
5. 绥化（中部，农业区）

**每个城市目标**：100-200 个 patches
- 时间范围：2023-2025 年（与哈尔滨一致）
- 分辨率：10m Sentinel-2 L1C
- 数据源：S2, S1, Landsat, DEM, WorldCover, Dynamic World, JRC Water

**扩展后总规模估算**：
| 阶段 | Patches | 帧数/patch | 总帧数 | 地理多样性 |
|------|---------|-----------|--------|-----------|
| 当前 | 425 | 22 | ~9,300 | 哈尔滨单城市 |
| 阶段 1：增加哈尔滨帧数 | 425 | 40 | ~17,000 | 哈尔滨，时间更丰富 |
| 阶段 2：+2 个城市 | 825 | 40 | ~33,000 | 城市+农业/工业区 |
| 阶段 3：+4 个城市 | 1,225 | 40 | ~49,000 | 全省主要地貌覆盖 |

**硬件可行性评估**（8 × NPU 910B4）：
- 1,225 patches × 40 帧 × batch=32 ≈ 每 epoch 765 步
- 200 epochs = 153,000 步
- 每步 ~5-10 秒（8卡 DDP）
- 总训练时间 ~7-15 天 ✅ **可行**
- 缓存大小估算：~50-80GB（8 NPU 每张卡 HBM 64GB）✅ **可行**

---

## 三、升级方案（三选一）

### 方案 A：数据扩展优先 + 架构优化（⭐ 推荐）

**核心假设**：数据量不足是根本瓶颈。用户有 GEE + terragon 下载能力，应优先扩展黑龙江省数据。

#### Phase 1：哈尔滨数据优化（3-5 天）

1. **重新云筛选哈尔滨原始数据**
   - 当前：每月保留最 clear 的 2 帧 → 22 帧/patch
   - 优化：每月保留最 clear 的 **3-4 帧** → **40-50 帧/patch**
   - 总帧数从 ~9,300 增加到 ~17,000-21,000
   - 时间维度多样性翻倍，模型能看到更多季节内变化

2. **数据验证**
   - 检查新增帧的时空对齐
   - 确保 s1/landsat 对应关系正确

#### Phase 2：黑龙江省数据扩展（1-2 周）

3. **使用 terragon + GEE 下载新城市数据**
   - 目标城市：齐齐哈尔、大庆（优先，地理差异大）
   - 每个城市：100-200 patches
   - 时间范围：2023-2025（与哈尔滨一致）
   - 数据源：S2, S1, Landsat（DEM/WorldCover 可用全球产品裁剪）

4. **云筛选预处理**
   - 对新城市数据执行统一云筛选（每月 3-4 帧）
   - 生成各城市独立的 patch 目录

5. **合并数据集**
   - 统一命名：patch_000000 ~ patch_001224（哈尔滨 425 + 新城市 ~800）
   - 生成合并 manifest
   - 重新计算全局统计量
   - 生成新 cache

6. **静态目标准备**
   - DEM：使用 Copernicus DEM GLO-30 裁剪到新城市区域
   - WorldCover：使用 ESA WorldCover 2021 裁剪
   - Dynamic World：使用 Google Dynamic World 裁剪
   - JRC Water：使用 JRC Global Surface Water 裁剪

#### Phase 2：架构优化（1 周）

4. **实施源特定重建权重**
   ```yaml
   source_recon_weights:
     s2: 1.0
     s1: 1.0
     landsat: 1.0
     dem: 0.3          # 静态目标降低
     worldcover: 0.5   # 分类目标更容易
     dynamic_world: 0.5
     jrc_water: 0.5
   ```

5. **强化教师-学生一致性**
   - 将 consistency_weight 从 0.05 提升到 **0.2**
   - 作为核心训练机制（类似 AEF）
   - 学生扰动策略：drop frames 50% / drop sources 30%

6. **简化损失函数**
   - 移除：orthogonality_loss, decorrelation_loss（被 uniformity 覆盖）
   - 保留：reconstruction + uniformity + consistency + temporal
   - 权重调整 closer to AEF 比例

7. **降低静态目标时间编码影响**
   - ConditionInjector 中对静态目标的 gate × 0.1
   - 或完全绕过 relative_time 注入静态解码路径

#### Phase 3：架构优化（1 周，可与 Phase 2 并行）

8. **实施源特定重建权重**
   ```yaml
   source_recon_weights:
     s2: 1.0
     s1: 1.0
     landsat: 1.0
     dem: 0.3          # 静态目标降低
     worldcover: 0.5   # 分类目标更容易
     dynamic_world: 0.5
     jrc_water: 0.5
   ```

9. **强化教师-学生一致性**
   - 将 consistency_weight 从 0.05 提升到 **0.2**
   - 作为核心训练机制（类似 AEF）
   - 学生扰动策略：drop frames 50% / drop sources 30%

10. **简化损失函数**
    - 移除：orthogonality_loss, decorrelation_loss（被 uniformity 覆盖）
    - 保留：reconstruction + uniformity + consistency + temporal
    - 权重调整 closer to AEF 比例

11. **降低静态目标时间编码影响**
    - ConditionInjector 中对静态目标的 gate × 0.1
    - 或完全绕过 relative_time 注入静态解码路径

#### Phase 4：训练策略（持续）

12. **课程学习**
    - Epoch 0-30: 重建主导（recon=1.0, temporal=0.0, consistency=0.1）
    - Epoch 30-80: 引入时序（recon=1.0, temporal=0.05→0.1, consistency=0.2）
    - Epoch 80+: 强化一致性（recon=0.8, temporal=0.1, consistency=0.3）

13. **监控指标**
    - 新增：RankMe / Stable Rank（检测维度坍缩）
    - 新增：Temporal Discriminability（直接测量时间敏感性）
    - 新增：recon_static vs recon_dynamic 分离监控

14. **验证频率**
    - 每 20 epoch 做一次 bare AUC 验证
    - 目标：E100 bare AUC > 0.6，E200 bare AUC > 0.7

#### 预期效果
- 数据量 3x → 从 425 到 1,200+ patches
- 地理多样性 → 哈尔滨（城市）+ 齐齐哈尔（农业平原）+ 大庆（工业区）
- 时间维度翻倍 → 从 22 帧/patch 到 40 帧/patch
- 静态权重降低 → 模型被迫学习时间敏感表征
- 一致性强化 → 类似 AEF 的核心机制

#### 风险
- GEE 下载时间（每个城市 100-200 patches × 多源 × 3 年 ≈ 数万张图像）
- 缓存重建时间（1,200 patches 可能需 30-50GB 缓存）
- 训练时间增加（1,200 / 425 ≈ 3x 每 epoch 步数）

---

### 方案 B：纯架构优化（不扩展数据）

**核心假设**：数据量虽少，但通过更精巧的架构和训练策略仍可达可用水平。

#### 核心改动

1. **放弃 V10 Difference Module**
   - V10 的 diff_encoder + change_gate 增加了复杂度但效果未验证
   - 回归更简洁的架构

2. **实施 AEF 原教旨损失**
   ```
   L = Recon(1.0) + BatchUniformity(0.05) + Consistency(0.2)
   ```
   - 移除所有 temporal loss、pixel change loss、change consistency
   - 在 L2-normed embedding 上计算 batch uniformity（非 pre-norm）

3. **降低 embedding_dim 到 64**
   - 匹配 AEF 论文
   - 小数据下 64D 比 128D 更容易均匀分布

4. **PEFT 微调路线**
   - 训练 backbone 时注入 LoRA 模块
   - 下游 CD Head 训练时同时微调 LoRA
   - 参考 PeftCD 论文（DINOv3+LoRA +5.8 IoU）

5. **端到端 ChangeFormer（备选）**
   - 如果 embedding 路线 E100 仍失败，直接实现 ChangeFormer
   - 输入：两期 S2，输出：变化概率图
   - 无需预训练 backbone

#### 预期效果
- 更忠实于 AEF 设计哲学
- PEFT 可能突破 frozen backbone 的上限
- 工作量较小，不需要数据适配

#### 风险
- 不解决根本的数据量问题
- 64D embedding 可能不足以表达复杂地表特征
- 一致性 loss 在小 batch 下效果有限

---

### 方案 C：激进重构（端到端变化检测）

**核心假设**：embedding 路线在当前数据和算力下不可行，直接做端到端变化检测。

#### 核心改动

1. **冻结当前 V8/V9 backbone**
   - 不再训练 backbone
   - 将当前 embedding 作为初始化特征

2. **实现 ChangeFormer 架构**
   - 分层 Transformer Encoder（类似 STP Block）
   - Feature Difference Module（显式编码双时相差异）
   - 轻量 MLP Decoder

3. **端到端训练**
   - 输入：Before/After 两期 S2 图像
   - 输出：Pixel-level 变化概率
   - 损失：BCE + Dice

4. **数据准备**
   - 从 105 个变化标注构建训练/验证集
   - 数据增强：翻转、旋转、颜色抖动

#### 预期效果
- 直接优化变化检测目标
- 绕过 embedding 质量瓶颈
- 文献中最成熟的 SOTA 路线（ChangeFormer F1=90.43%）

#### 风险
- 放弃通用表征学习，变成专用模型
- 无法利用 S1/Landsat 多源信息（ChangeFormer 通常只用 S2）
- 项目方向完全改变

---

## 四、推荐决策树（更新）

用户确认有 GEE + terragon 下载能力，**方案 A 是首选**。

```
是否有时间执行 GEE 数据下载（1-2 周）？
├── 是 → 方案 A（黑龙江省数据扩展 + 架构优化）⭐ 强烈推荐
│         理由：用户有 GEE 能力，地理多样性是根本瓶颈
│         时间：1-2 周数据准备 + 2-3 周训练
│         目标：1,200+ patches，3 个城市，40 帧/patch
│
└── 否 → 是否接受纯时间维度扩展（哈尔滨 425 patches → 40 帧/patch）？
          ├── 是 → 方案 A-lite（仅增加帧数 + 架构优化）
          │       理由：不下载新数据，仅重新云筛选 + 架构改进
          │       时间：3-5 天准备 + 2 周训练
          │
          └── 否 → 方案 B（纯架构优化，不扩展数据）
                    理由：最小工作量，但上限受限
                    时间：1 周
```

---

## 五、执行优先级（方案 A：黑龙江省扩展）

### 阶段 1：数据准备（1-2 周）

| 优先级 | 任务 | 预计时间 | 阻塞项 |
|--------|------|---------|--------|
| P0 | 重新云筛选哈尔滨数据（3-4 帧/月） | 1-2 天 | 无 |
| P0 | terragon + GEE 下载齐齐哈尔/大庆数据 | 3-5 天 | GEE 配额/网络 |
| P0 | 新城市数据云筛选 | 1-2 天 | 依赖下载完成 |
| P1 | DEM/WorldCover/DynamicWorld/JRC 裁剪 | 1 天 | 依赖城市边界 |
| P1 | 合并数据集 + 统一命名 + manifest | 1 天 | 依赖所有数据就绪 |
| P1 | 全局统计量重计算 | 1 天 | 依赖 manifest |
| P1 | 新 dataset cache 生成 | 2-3 天 | 依赖统计量 |

### 阶段 2：架构改进（3-5 天，可与阶段 1 并行）

| 优先级 | 任务 | 预计时间 | 阻塞项 |
|--------|------|---------|--------|
| P0 | 源特定重建权重实现 | 1 天 | 无 |
| P0 | 简化损失函数（移除 orth/decorr） | 1 天 | 无 |
| P1 | 强化一致性（weight 0.05 → 0.2） | 0.5 天 | 无 |
| P1 | 静态目标时间编码弱化 | 1 天 | 无 |
| P2 | RankMe/Temporal Discriminability 指标 | 2 天 | 无 |

### 阶段 3：训练与验证（2-3 周）

| 优先级 | 任务 | 预计时间 | 阻塞项 |
|--------|------|---------|--------|
| P0 | V11 训练启动 | 1 天 | 依赖阶段 1+2 |
| P0 | 每 20 epoch bare AUC 验证 | 持续 | 依赖训练 |
| P1 | 根据验证结果调参 | 持续 | 依赖验证结果 |

---

## 六、用户已确认的关键信息

1. ✅ **数据获取能力**：有 GEE 账号（weijiewu0306@gmail.com）+ terragon 工具
2. ✅ **地理扩展意愿**：希望扩充到黑龙江省，不使用雅江数据
3. ✅ **帧数扩展意愿**：接受哈尔滨数据从 22 帧/patch 增加到 40 帧/patch
4. ⏳ **待确认**：优先下载哪些城市？建议齐齐哈尔（农业平原）+ 大庆（工业/盐碱地）
5. ⏳ **待确认**：每个城市目标 patch 数量？建议 150-200 个
6. ⏳ **待确认**：时间预算是否接受 3-4 周？

## 七、下一步行动

等待用户确认方案 A 后，立即执行：
1. 编写 terragon + GEE 下载脚本（齐齐哈尔、大庆）
2. 重新云筛选哈尔滨数据（3-4 帧/月）
3. 同步实施架构改进（源特定权重、简化损失、强化一致性）
4. 生成合并数据集与 cache
5. 启动 V11 训练
