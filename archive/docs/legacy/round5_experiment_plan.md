# Round 5 实验计划 — 融合 V4 成功机制 + V13 Spatial VICReg

## 一、调研结论

### 1.1 V4 Official 成功密码

从 V4 官方训练日志（Epoch 286-300）提取关键指标：

| 机制 | 配置权重 | 实际值 | 作用 |
|------|---------|--------|------|
| batch_uniformity | 1.0 | **-2.78 ~ -3.03** | 在 pre_norm 空间强制 batch 内分散 |
| variance (VICReg) | 0.25 | 0.53 | 维度方差约束 |
| **decorrelation** | **0.05** | **31-32** | Barlow Twins 去相关，**核心反坍缩机制** |
| **orthogonality** | **0.01** | **0.0039** | bottleneck Conv1x1 权重正交约束 |
| **consistency** | **0.05** | **0.04-0.05** | EMA teacher + student 扰动，**正常工作** |
| classification | 0.03 | 0.39-0.49 | WorldCover 语义监督 |
| reconstruction | 1.0 | 0.125-0.133 | **核心目标，不坍缩** |
| 训练 epoch | — | 300 | 充分训练 |

**V7 对比**（Epoch 75-84）：
- uniform = **-0.51**（仅为 V4 的 1/5）
- consist = **0.00002**（几乎为 0，consistency **失效**）
- cls = 1.28-1.77（过大）

**核心结论**：V4 能抵抗 reconstruction 坍缩，靠的是 **decorr + orth + consistency + cls** 四个互补机制 + 300 epoch 训练。

### 1.2 V13 Round 4 教训

| 实验 | 配置 | active_dims | 结论 |
|------|------|-------------|------|
| ExpM | var=0.3, cov=0.1, **recon=0** | **107** | Spatial VICReg 本身有效 |
| Staged recon=0.1 | var=0.3, cov=0.1, recon 渐进 0→0.1 | **108** | recon 加入后短暂有效 |
| ExpL | var=0.3, cov=0.1, **recon=0.5** | **81** | **recon>0 导致坍缩** |
| ExpQ | var=1.0, cov=**0** | **44** | **covariance 必要** |
| ExpR | var=0.3, cov=0.1, recon 渐进 0→0.2 | **2** | **渐进 recon 失败** |

**核心结论**：Spatial VICReg 单独无法抵抗 reconstruction 的坍缩力，需要 **额外的反坍缩机制**。

### 1.3 文献支撑

- **VICReg 论文**：λ(inv)=25, μ(var)=25, ν(cov)=1 → var:cov = **25:1**
- **AEF 官方**：D=64, κ=8000, recon 为核心目标
- **V4 官方**：batch_uniformity 在 S⁶³ 上计算（GMP 后），配合 decorr + orth

---

## 二、核心假设

> **Spatial VICReg（空间级，替代 GMP 后的 batch_uniformity）+ V4 的 decorr + orth + consistency + cls + recon warm-up = 不坍缩的重建学习**

V13 的失败不是因为 Spatial VICReg 不好，而是因为 **去掉了 V4 中真正起作用的反坍缩机制**。

---

## 三、实验设计（只做预期有用的）

### 3.1 实验 T — 完整组合（最高优先级）

**目标**：验证 "V4 全部反坍缩机制 + Spatial VICReg" 是否能在 reconstruction > 0 时不坍缩。

**配置**：
- **Base**: V13 Spatial VICReg（var=0.3, cov=0.1）
- **Add decorrelation**: weight=0.05（Barlow Twins）
- **Add orthogonality**: weight=0.01（bottleneck 权重正交）
- **Add consistency**: weight=0.05 + EMA teacher (m=0.996) + student perturbation
- **Add classification**: weight=0.03（WorldCover 众数监督）
- **Reconstruction**: warm-up 0→1.0（前 10 epoch linear ramp）
- **Epoch**: 50（验证是否能在 50 epoch 内稳定）

**预期**：
- 前 10 epoch：recon 从 0 逐渐增加，active_dims 保持 >100
- 10-30 epoch：recon 下降至 <0.3，active_dims 稳定在 80-100
- 30-50 epoch：recon 继续下降至 <0.2，active_dims >70
- 不坍缩（active_dims >50）

### 3.2 实验 U — 轻量版（验证 decorr+orth 是否足够）

**目标**：验证 consistency + cls 是否必要，还是 decorr + orth 已足够。

**配置**：
- **Base**: V13 Spatial VICReg（var=0.3, cov=0.1）
- **Add decorrelation**: weight=0.05
- **Add orthogonality**: weight=0.01
- **去掉**: consistency, classification（简化配置）
- **Reconstruction**: warm-up 0→1.0（前 10 epoch）
- **Epoch**: 50

**预期**：
- 如果 ExpU 也成功 → decorr + orth 是核心，consistency 不是必要
- 如果 ExpU 坍缩但 ExpT 成功 → consistency/cls 是必要的

---

## 四、不做（预期坍缩）的实验

| 实验 | 原因 |
|------|------|
| 只用 Spatial VICReg + recon | V13 已验证：ExpL 坍缩到 81 |
| 只用 Spatial VICReg + recon + consistency | 缺少 decorr+orth，可能仍不足 |
| 渐进 recon 0→0.2 | ExpR 已验证：坍缩到 2 |
| 高 weight VICReg (var=1.0, cov=0.5) | ExpP 已验证：active=71 |
| 更多数据 (400 patches) | ExpS 已验证：active=63 |

---

## 五、执行计划

1. **修改 `ddp_v12_trainer.py`**：
   - 添加 EMA teacher + `update_teacher()`
   - 添加 student perturbation（丢帧/丢源/截断）
   - 添加 decorrelation loss（在 gathered_pre_norm 上）
   - 添加 orthogonality loss（在 bottleneck Conv1x1 权重上）
   - 添加 consistency loss（teacher.detach() vs student）
   - 添加 classification loss（WorldCover 众数）
   - 添加 reconstruction warm-up
   - 保留 Spatial VICReg（在 pre_norm_map.reshape(-1, D) 上）

2. **创建两个 config**（ExpT 和 ExpU）

3. **并行训练**（NPU 0 和 NPU 1 各跑一个）

4. **监控指标**：每 5 epoch 记录 active_dims, recon, var, cov, decorr, orth, consist, cls

---

## 六、成功标准

| 指标 | 成功标准 | 失败标准 |
|------|---------|---------|
| active_dims | 50 epoch 后 >70 | <50 |
| recon | 50 epoch 后 <0.3 | >0.5 且持续上升 |
| var_reg | 接近 0 | >0.5 |
| 趋势 | recon 持续下降 | recon 停滞或上升 |
