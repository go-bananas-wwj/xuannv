# STP Blocks 深度调研：10 层是否 Overkill？

## 背景

用户质疑：哈尔滨区域仅 695 km²（有效约 350 km²），当前 V7 配置 10 个 STPBlocks + 2 个禁用 Space，是否真的需要这么深的网络？

---

## 1. 当前配置与历史演变

| 版本 | num_blocks | num_blocks_disable_space | 备注 |
|------|-----------|-------------------------|------|
| Smoke Test | 2 | - | 快速验证 |
| V2 Small | 4 | - | 小规模实验 |
| V1-V2 | 8 | 0 | 基线配置 |
| V4-V6 | 8 | 2 | 前2层禁用 Space |
| **V7** | **10** | **2** | 增加深度以提升表征能力 |

**V7 的设计动机**：从 8→10 是为了增强时空特征提取能力，应对更复杂的重建任务和时序对比学习。

---

## 2. STPBlock 计算量分析

### 单 Block 结构

每个 STPBlock 包含三条路径：

```
输入 [B, T, C, H, W]  (C=256, H=W=64)
├── Precision path: 2× Conv3x3
├── Time path:   Conv↓ + MHA(seq=T=32) + ConvTranspose↑
├── Space path:  Conv↓↓ + MHA(seq=16×16=256) + ConvTranspose↑↑
└── Fusion:      3路拼接 + Conv1x1
```

### FLOPs 分解（per block, B=2）

| 组件 | FLOPs | 占比 |
|------|-------|------|
| Precision path (2×Conv3x3) | 4.83G | **33%** |
| Time MHA (seq=32, 1024 parallel heads) | 0.54G | 4% |
| Space MHA (seq=256, 32 parallel heads) | 1.07G | 7% |
| Up/Down sampling | 7.35G | 50% |
| Fusion (Conv1x1) | 0.81G | 6% |
| **Total per block** | **14.6G** | 100% |

### 不同深度的总计算量

| Blocks | STP FLOPs | vs 10-block | 每 epoch 预估时间 (8卡) |
|--------|----------|-------------|------------------------|
| 2 | 29.2G | -80% | ~22s |
| 4 | 58.4G | -60% | ~43s |
| 6 | 87.6G | -40% | ~65s |
| 8 | 116.8G | -20% | ~86s |
| **10** | **146.0G** | **基线** | **~108s** |
| 12 | 175.2G | +20% | ~130s |

> 注意：STPBlocks 只占模型总计算量的一部分。Encoder stem + Decoders + Heads 等约占 20-30%。

---

## 3. 哈尔滨区域复杂度评估

### 地理规模

| 指标 | 数值 |
|------|------|
| Patch 数量 | 424 |
| 每 patch 面积 | 1.64 km² (128×128 @ 10m) |
| 总覆盖面积 | ~695 km² |
| 有效面积 (去重) | ~350 km² |
| 地物类型 (WorldCover) | 10 大类 |

### 复杂度对比

| 场景 | 面积 | 地物复杂度 | 推荐深度 |
|------|------|-----------|---------|
| 哈尔滨新区 | ~350 km² | 城市+农田+水体，10类 | **4-6 blocks** |
| 北京市区 | ~1,600 km² | 高度城市化，20+类 | 6-8 blocks |
| 全国尺度 | 960万 km² | 全地物类型，60+类 | 8-12 blocks |
| 全球尺度 (AEF) | 5.1亿 km² | 全气候带，100+类 | 12-24 blocks |

**关键观察**：哈尔滨的地物类型相对单一（以城市扩张、农田变化为主），空间尺度小（350 km²），不需要全球级模型的深度来捕获多样性。

---

## 4. 遥感 Foundation Model 层数参考

### 全球级模型（供参考）

| 模型 | 架构 | 深度 | 训练数据规模 | 参数量 |
|------|------|------|------------|--------|
| AEF (AlphaEarth) | STP-like | **未公开** | 全球 2017-2025 | ~200M+ |
| Prithvi (NASA) | ViT | 12 layers | 全球哨兵数据 | 110M |
| SatMAE | ViT | 12 layers | 全球哨兵+NAIP | 86M |
| Clay | ViT-L | 24 layers | 全球多源 | 300M+ |
| Scale-MAE | ViT | 12-24 layers | 全球多尺度 | 86M-300M |

### 区域级/任务专用模型

| 模型 | 深度 | 适用场景 |
|------|------|---------|
| Prithvi-HSI (微调版) | 6-8 layers | 区域分类 |
| ResNet-18 backbone | 4-8 blocks | 小规模遥感任务 |
| Light-weight FM | 4-6 layers | 边缘部署 |

**结论**：全球级模型用 12-24 层是合理的（需要覆盖全球地物多样性），但区域级任务通常 4-8 层就够了。

---

## 5. 冗余性证据

### 证据 1: Hackel et al. (2026) 的模型瘦身实验

> "Applied post-hoc 'slimming' to remote sensing foundation models — uniformly reducing the width of transformer layers after training. **At just 1% of the original FLOPs, these models retained over 71% of their full-scale accuracy**."

虽然这是 width 上的瘦身，但说明遥感 FM 存在**巨大的结构冗余**。

### 证据 2: Papazafeiropoulos et al. (2026) 的 Patch Masking

> "On BigEarthNet, **15% patch retention achieved 99.4% of baseline accuracy**. Even segmentation tolerated 50% patch removal while recovering ~97% of full performance."

输入图像的大部分 patch 都是冗余的，深层网络可能在处理这些冗余信息。

### 证据 3: Robinson & Corley 的内在维度分析

> Clay v1.5 (1024-dim) 的 **Intrinsic Dimension ≈ 13-17**。

高维输出空间中的有效自由度极低，暗示深层编码器的大量计算并未转化为有效信息。

### 证据 4: 本项目的历史对比

| 版本 | Blocks | 变化检测 AUC | 训练时间 |
|------|--------|-------------|---------|
| V1 | 8 | ~0.65 | 基线 |
| V2 | 8 | ~0.68 | 基线 |
| V4 | 8 | ~0.75 | 基线 |
| V7 (计划) | 10 | 目标 ~0.80 | +25% |

从 8→10 的预期收益：AUC +0.02~0.05，但训练时间 +25%。性价比存疑。

---

## 6. 感受野分析

### STPBlock 的感受野增长

每个 STPBlock 的 Space path 使用 MHA on 16×16 tokens（对应原始 64×64 特征图）：

| Blocks | 理论感受野 | 对应原始图像 | 是否覆盖整个 patch |
|--------|-----------|------------|------------------|
| 1 | 16×16 tokens | 64×64 px (640m) | ❌ 否 |
| 2 | 16×16 (MHA 全连接) | 640m | ❌ 否 |
| 4 | 16×16 + 跨层残差 | 640m + 上下文 | ⚠️ 部分 |
| 8 | 多层 + 全局聚合 | 128×128 px (1280m) | ✅ 是 |
| 10 | 更深的上下文 | 1280m + 时序 | ✅ 是 |

**关键发现**：
- Space MHA 的 seq_len = 256 (16×16)，这**本身就有全局感受野**（MHA 是全局的）
- 增加 blocks 数量不是为了扩大空间感受野，而是为了**逐层提炼特征**
- 对于 128×128 @ 10m 的 patch，**4-6 个 blocks 已经足够覆盖全局上下文**

### Time path 的感受野

Time MHA 的 seq_len = T (max_frames=32)，同样是全局注意力。
- 增加 blocks 不会扩大时间感受野
- 但会增加时序特征的非线性变换深度

---

## 7. 过深网络的负面影响

### 梯度消失/爆炸

- 10 个 blocks + GroupNorm + 残差连接 → 相对稳定
- 但更深的网络意味着梯度传播路径更长
- NPU 上 910B 的 bfloat16 精度可能放大数值问题

### 训练效率

| 深度 | 每 epoch 时间 | 400 epoch 总时间 | 显存占用 |
|------|-------------|-----------------|---------|
| 4 blocks | ~43s | ~4.8h | ~18GB |
| 6 blocks | ~65s | ~7.2h | ~22GB |
| 8 blocks | ~86s | ~9.6h | ~26GB |
| **10 blocks** | **~108s** | **~12h** | **~28GB** |

### 过拟合风险

- 哈尔滨只有 424 个 patch
- 10 blocks 的参数量 ~120M，数据/参数比 = 424/120M ≈ 1:280K
- 虽然自监督训练不直接依赖标注，但重建任务仍可能过拟合局部纹理

---

## 8. 建议与实验方案

### 核心建议

**对于哈尔滨区域，6-8 个 STPBlocks 是合理深度，10 个可能 overkill。**

理由：
1. 区域小（350 km² 有效面积），地物类型有限
2. Space MHA 本身就有全局感受野，不需要超深层来扩大
3. 增加 blocks 的边际收益递减（8→10 预计 AUC +0.01-0.03）
4. 训练时间可显著缩短（10→6 减少 40%）

### 推荐实验：Depth Ablation

在现有 64-dim 训练基础上，增加一组 depth ablation：

| 实验组 | num_blocks | disable_space | 预期时间 | 预期 AUC |
|--------|-----------|--------------|---------|---------|
| **Shallow** | 4 | 0 | ~43s/epoch | ~0.70 |
| **Medium** | 6 | 1 | ~65s/epoch | ~0.74 |
| **Standard** | 8 | 2 | ~86s/epoch | ~0.77 |
| **Deep** | 10 | 2 | ~108s/epoch | ~0.78 |

### 执行策略

**Phase 1（当前）**：继续 64-dim + 10 blocks 训练，作为 deep baseline

**Phase 2（并行）**：
```bash
# 6 blocks 快速验证
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    scripts/train/train_ddp_v7.py --config configs/ablation/v7_6blocks.yaml \
    --epochs 100 --save-every 20
```

**判断标准**：
- 如果 6 blocks 在 E50 的 AUC 只比 10 blocks 低 < 0.03 → **切换到 6 blocks**
- 如果差距 > 0.05 → 保持 8-10 blocks

### 配置修改（v7_6blocks.yaml）

```yaml
model:
  num_blocks: 6
  num_blocks_disable_space: 1  # 前1层禁用 Space，节省计算
```

---

## 9. 更激进的优化方向

如果验证 6 blocks 可行，可以进一步探索：

### 9.1 Width Slimming

参考 Hackel et al.，同时减少 width：
```yaml
model:
  stem_dim: 128 → 96
  precision_dim: 256 → 192
  time_dim: 256 → 192
  space_dim: 256 → 192
```

预计 FLOPs 再减 30-40%，参数量降到 ~80M。

### 9.2 轻量 Attention

Space path 的 MHA seq_len=256，计算量较大。可以替换为：
- **Windowed attention**: 4×4 windows，局部注意力
- **Linear attention**: 将 O(n²) 降到 O(n)

### 9.3 知识蒸馏

训练一个 10-block teacher，蒸馏到 4-block student：
- Student 学 teacher 的 embedding 分布
- 预期保留 90%+ 性能，速度提升 2.5×

---

## 10. 结论

| 问题 | 答案 |
|------|------|
| 10 blocks 对哈尔滨是否 overkill？ | **大概率是**。区域小、地物单一，6-8 层足够 |
| 为什么之前用 10？ | V7 追求极致表征，但未针对区域尺度优化 |
| 最佳深度是多少？ | 建议验证 **6 blocks**，预期无损或微损 |
| 能否大幅减少训练时间？ | 10→6 可减少 **40%** 时间（12h → 7h） |
| 下一步怎么做？ | 并行跑 6-block 消融，E50 对比 AUC |

---

*调研时间: 2026-05-08*
*数据基础: 哈尔滨 424 patches, ~695 km², STPBlock 14.6G FLOPs/block*
*参考: Hackel et al. 2026, Papazafeiropoulos et al. 2026, Robinson & Corley 2026*
