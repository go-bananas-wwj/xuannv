# Round 1 实验总结与 Round 2 改进方案

## 一、Round 1 实验设计

### 1.1 实验列表

| 实验名 | 关键变量 | 最佳 recon | AUC (Bare) | WorldCover Acc | Dynamic World Acc |
|--------|---------|-----------|-----------|---------------|-------------------|
| aef_baseline | recon=[1,1,1,0.05], kappa=50 | 0.0657 | 0.4997 | 26.64% | 33.39% |
| aef_high_consist | consistency=0.05 | 0.0659 | 0.5011 | 26.64% | 32.84% |
| aef_no_static | DEM=0 | 0.0647 | 0.4994 | 26.70% | 32.82% |
| aef_skip_l2 | skip_l2=true | 0.0580 | 0.4994 | 26.49% | 33.07% |
| aef_diff_recon | S2=1.0, S1=0.5, Landsat=0.8 | 0.0522 | 0.5012 | 26.49% | 32.94% |
| aef_high_kappa | kappa=2000 | 0.0657 | 0.5039 | 26.82% | 32.08% |
| aef_cyclic_unif | cyclic_shift uniformity | 0.0657 | 0.4981 | 26.36% | 32.89% |
| aef_no_uniform | uniform=0 | 0.0650 | 0.5132 | 26.52% | 33.13% |

### 1.2 核心结论

**全部 8 个实验全部失败。**

- **变化检测 AUC ≈ 0.50（随机）**：模型没有任何时序敏感性，无法区分 before/after
- **下游 KNN 表征质量极差**：WorldCover 10类 26.6%（随机10%），JRC Water 2类 6.5%（随机50%）
- **8 实验差异 <1%**：不同 loss 权重、uniformity、kappa、skip_l2 等对结果没有实质影响
- **模型只学了空间重建**：作为自编码器（Way A: 输入=目标同月），模型被迫记住空间纹理，完全忽略了时间维度

---

## 二、根因分析

### 2.1 架构缺陷：Way A 自编码器

当前训练方式是 **Autoencoder Way A**：
- 输入：某 patch 某月的所有帧
- 目标：重建同一 patch **同一月**的帧
- 结果：模型只需要学 "怎么把输入帧的空间信息压缩再还原"
- 时间编码器（time_encoder、window_encoder）完全没有被利用

### 2.2 重建目标与变化检测目标冲突

- 训练信号：重建同月帧 → 模型学到 "保持空间一致性"
- 评估信号：检测跨月变化 → 需要 "识别时间差异"
- 冲突：模型被训练为**最小化时间差异**（因为它要重建同一月），而评估要求它**放大时间差异**

### 2.3 损失函数失效

- `batch_uniformity_weight=0.05`：L2 uniformity 在 sphere 上强制均匀分布，但模型根本没学到有意义的语义特征，uniformity 只是让 embedding 在球面上均匀分散噪声
- `consistency_weight=0.02`： teacher-student consistency 只加强空间鲁棒性，不涉及时序
- `skip_l2_norm_training`：虽然 skip L2 保留了幅度信息，但没有时序对比信号来利用它

### 2.4 数据问题

- 100 patches（而非 424 全量），数据量有限
- 2025-only 过滤导致时间跨度短（仅 7 个月），季节变化信号弱
- 人类标注的变化（construction）与季节变化（vegetation greening）信号不一致

---

## 三、Round 2 改进方案

### 3.1 核心架构切换：Way B — 跨时相重建

**目标**：输入 month A，重建 month B。

实现方式：
1. **输入窗口**：month A 的所有帧（如 2025-04）
2. **目标窗口**：month B 的所有帧（如 2025-06）
3. **编码器**：只编码输入窗口 → embedding
4. **解码器**：用 embedding + month B 的时间条件 → 重建 month B 的帧

这样模型被迫学习：
- "从 4 月的植被状态推断 6 月的植被状态"
- 必须编码跨时间的映射关系
- embedding 必须携带时序变化信息

### 3.2 损失函数升级

| 损失 | Round 1 | Round 2 |
|------|---------|---------|
| Reconstruction | 同月自重建 | **跨月重建**（输入 A → 目标 B） |
| Temporal Contrast | 无 | **双窗口 hinge loss**（emb_A vs emb_B） |
| Uniformity | batch_uniformity 0.05 | **保持或降低**（避免抑制时序方差） |
| Consistency | 空间一致性 0.02 | **时序一致性**（同 patch 不同月预测一致） |

### 3.3 关键实现修改

#### 3.3.1 Dataset 修改

`HarbinPatchDataset.__getitem__` 需要修改：
- 当前：月度采样，输入/目标都是同一个月
- 目标：采样一个 **(patch, month_A, month_B)** 三元组
- month_A 和 month_B 需要有最小时间 gap（如 ≥2 个月）
- 如果 month_B 没有数据，可以回退到同月（但概率要很低，如 <5%）

#### 3.3.2 Model Forward 修改

`AEFModel.forward()` 需要修改：
- 当前：`target_relative_time` 对目标只传一个标量
- 目标：需要区分输入窗口和目标窗口的 `valid_start_ms` / `valid_end_ms`
- Decoder 需要接收目标窗口的时间编码，而不是输入窗口的

#### 3.3.3 Loss 计算修改

`reconstruction_loss` 需要修改：
- 当前：输入帧和目标帧是同一个，L1 损失直接比较
- 目标：输入帧来自 month A，目标帧来自 month B，需要 mask 掉 month A/B 没有的数据

### 3.4 实验设计

基于用户选择的 **aef_high_kappa**（kappa=2000，50 epochs）：

```yaml
# configs/round2_cross_temporal.yaml
experiment:
  name: round2_cross_temporal
  output_dir: /workspace/outputs/xuannv_round2/round2_cross_temporal

training:
  epochs: 50
  # 跨时相重建权重
  reconstruction_weight: 1.0
  # 时序对比损失（新增）
  temporal_contrastive_weight: 0.1
  # 降低 uniformity，避免抑制时序方差
  batch_uniformity_weight: 0.01
  # 保持 consistency
  consistency_weight: 0.02

data:
  # 最小时间 gap（2个月）
  cross_temporal_min_gap_months: 2
  # 跨时相采样概率
  cross_temporal_prob: 0.9
  # 回退到同月重建的概率
  same_month_fallback_prob: 0.1

model:
  vmf_kappa: 2000.0
```

### 3.5 评估方案

沿用 Round 1 的标准化流水线：
1. 训练 50 epochs
2. 提取 embedding（skip_decoder=True）
3. KNN 下游评估（WorldCover / JRC Water / Dynamic World）
4. 变化检测 Bare AUC 验证
5. 生成对比可视化

**成功标准**：
- Bare AUC > 0.55（明显高于随机）
- WorldCover KNN Acc > 30%（目前 26.6%）
- Dynamic World KNN Acc > 40%（目前 33.4%）
- Embedding distance 在变化区域明显 > 非变化区域

---

## 四、标准化流水线

已创建 `scripts/eval/run_full_pipeline.py`，一键完成：

```bash
cd /workspace/xuannv
# 训练 + 评估 + 可视化
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python scripts/eval/run_full_pipeline.py \
    --config configs/round2_cross_temporal.yaml --gpus 4

# 仅评估（跳过训练）
python scripts/eval/run_full_pipeline.py \
    --config configs/round2_cross_temporal.yaml --skip-train
```

---

## 五、下一步行动

1. [ ] 修改 `HarbinPatchDataset` 支持跨时相采样
2. [ ] 修改 `AEFModel.forward` 区分输入/目标时间窗口
3. [ ] 修改 `reconstruction_loss` 处理跨月目标
4. [ ] 创建 `configs/round2_cross_temporal.yaml`
5. [ ] 启动 50 epoch 训练（4-8 卡 DDP）
6. [ ] 运行标准化流水线评估
