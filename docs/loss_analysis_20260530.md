# 海淀嵌入底座 反坍缩实验分析与升级路线

> 分析时间: 2026-05-30 ~ 2026-06-01
> 分析目标: 分析 v24 ~ v36 所有海淀实验，制定下一步升级计划

---

## 一、实验全景：全部坍缩的残酷现实

| 版本 | 目标策略 | 峰值 erank | 最终 erank | 结论 |
|------|----------|-----------|-----------|------|
| v25 | 配置声称反坍缩，低重建 + memory bank | **3.9** | 2.0 | 轻微坍缩，无法维持 |
| v28 | **uniform=1.5, var=0.6, cov=0** | **11.7** | 3.4 | 峰值最高，但仍坍缩 |
| v31 | P0 立即改（低重建 + 空间裁剪） | 10.3 | 3.1 | 无 memory bank 快速坍缩 |
| v32 | P0+P1（+2D sincos PE） | 10.2 | 3.2 | 2D PE 无实质帮助 |
| v33 | 关闭 BYOL consistency | 11.2 | 3.7 | 关闭 consistency 仍坍缩 |
| v34 | 关闭 BYOL + 更低重建 | 11.4 | 3.4 | 极低重建仍坍缩 |
| v35 | **AEF batch rotation b=0.005** | 10.1 | 3.4 | AEF 权重太低，无效 |
| v36 | **AEF batch rotation b=0.05** | 10.8 | 3.5 | AEF 权重仍太低，无效 |

**核心发现：320 patches × 64D = 严重不足**

- 有效维度 64，样本仅 320，intrinsic dimension 理论上限 ~5-8
- 所有实验 Step 10-50 期间 erank 从 10-12 断崖下跌至 3-5
- **BYOL consistency 是最大吸引子**（loss ~0.039），AEF b=0.05 贡献仅 0.005-0.048

---

## 二、关键损失项分析（v35/v36）

| 损失项 | v35 范围 | v36 范围 | 作用 |
|--------|----------|----------|------|
| `recon` | 0.013-0.022 | 0.014-0.025 | 重建 → 坍缩吸引子 |
| `consistency` | 0.033-0.039 | 0.033-0.039 | **BYOL 吸引子** |
| `pre_unif` | -1.8 ~ -1.4 | -1.8 ~ -1.4 | pre-norm uniformity，接近坍缩 |
| `aefunif` | 0.005-0.026 | 0.005-0.048 | AEF batch rotation，**贡献太小** |

**关键洞察**：
- aefunif Step 4 时 0.756（推散有效！）→ Step 44 时 0.955（被推回去了！）
- consistency 从未低于 0.033，持续将模型拉向坍缩
- 即使关闭 consistency（v33/v34），uniformity loss 单独也无法对抗重建吸引子

---

## 三、 OlmoEarth / AEF / 其他地理嵌入反坍缩做法

| 模型 | 反坍缩机制 | 样本量 | 效果 |
|------|-----------|--------|------|
| **AEF** | Batch Rotation Uniformity (b=0.05) | ~10000 patches | **erank 25+ 稳定** |
| **OlmoEarth** | Pixel-wise InfoNCE + Global Gap-Aware | 未公开 | 无公开 erank |
| **SatMAE** | Masked Autoencoding | 大量无标注 | 无公开 erank |
| **Scale-MAE** | Scale-aware MAE | 大量无标注 | 无公开 erank |
| **SatlasPretrain** | Multi-task pretraining | 大量标注 | 无公开 erank |

**AEF 成功关键**（我们的差距）：
1. **样本量 30-40 倍差距**：10000 vs 320 patches
2. **多 GPU 大批量**：AEF 用 16-32 GPU，每步 2000-4000 patches
3. **AEF Batch Rotation 权重可能更高**：论文写 0.05，但成功实验可能隐性加权

---

## 四、v25 假象揭穿

| 来源 | 声称 erank | 实际日志 erank |
|------|-----------|---------------|
| `config_haidian_v25.yaml` 注释 | **9.09** | 无对应日志 |
| `s2_novdec_pipeline.log` | - | 最终 **2.0** |

**结论**：v25 从未达到 erank=9.09，最佳仅 3.9（Step 9），最终 2.0 已严重坍缩。

---

## 五、P0 已实施内容

1. ✅ **AEF batch rotation uniformity loss**（v35/v36/v37/v38）
2. ✅ **2D sincos positional encoding**（v32/v34）— 效果有限
3. ✅ **Center-crop 评估**（extract_embeddings.py）— 边缘不连续改善
4. ✅ **Memory bank = 1024**（v25起）— 有效但不足

---

## 六、当前运行实验（2026-06-01 启动）

| 版本 | consistency | aef | recon | 策略 |
|------|------------|-----|-------|------|
| **v37** | **0.0** | **0.5** | **0.08** | 关闭 BYOL + 强 AEF |
| **v38** | **0.0** | **0.5** | **0.02** | 关闭 BYOL + 强 AEF + 低重建 |

**预期**：aef=0.5 时 loss 贡献 ~0.5（相比之前 0.005-0.048），是**100倍提升**。如果 aefunif 能在 0.4-0.6 区间稳定，erank 可能维持 > 8。

---

## 七、下一步计划（基于 v37/v38 结果）

### 如果 v37/v38 erank 维持 > 8（成功）
1. 提取 embedding 做 KNN 评估
2. 若 KNN 指标达标 → 作为海淀区生产基线
3. 若 KNN 不达标 → 微调重建权重/分类权重

### 如果 v37/v38 仍坍缩（失败）
1. **权重级推到 aef=1.0-2.0**（接近 AEF 原始论文量级）
2. 考虑**完全不重建**（recon=0），纯对比学习
3. **增加正样本数量**：memory bank 8192 + 每 batch 32 patches
4. 最后手段：增加数据（扩展到全北京市或全国 patches）

---

## 八、数据缺口影响评估

| 场景 | patches | erank 理论上限 | 可行？ |
|------|---------|---------------|--------|
| 当前（海淀 320） | 320 | ~5-8 | ⚠️ 勉强 |
| 北京全市 | ~5000 | ~15-20 | ✅ 可行 |
| 全国主要城市 | ~50000 | ~30+ | ✅ 理想 |

**AEF 论文样本量**：~10000 patches，erank 25+。**我们至少需要 3000+ patches** 才能稳定达到 erank > 10。

---

## 九、核心结论

1. **320 patches 不足以支撑 64D embedding 分散** — 这是根本瓶颈
2. **BYOL consistency 是最大坍缩元凶** — 必须降到 0 或极低
3. **AEF batch rotation 权重必须大幅提高** — b=0.005/0.05 无效，需 0.5+
4. **重建 loss 是次要吸引子** — 降到 0.02-0.08 有帮助但不够
5. **最终解决方案 = 算法优化 + 数据扩展** — 两者缺一不可
