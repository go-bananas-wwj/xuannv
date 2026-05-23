# V12 实施差距分析与修复计划

> 生成时间: 2026-05-13
> 背景: 用户指出当前训练中 (1) Active dims 展示/驱动不足 (2) 实验间差异分析缺失 (3) 前期讨论计划未完整实施

---

## 一、前期计划实施状态

### 已实施的 ✅

| 计划来源 | 内容 | 状态 |
|---------|------|------|
| plan_v12_vicreg_prenorm.md | pre-norm VICReg (variance_regularizer) | ✅ 已实施 |
| plan_v12_vicreg_prenorm.md | covariance_loss | ✅ 已实施 |
| plan_v12_vicreg_prenorm.md | Memory Bank (K=512) | ✅ 已实施 |
| plan_v12_vicreg_prenorm.md | per-dimension std 诊断 (std_mean, active_dims) | ✅ 已实施 |
| plan_v12_per_dim_uniform.md | spatial VICReg (use_spatial_vicreg) | ✅ 已实施 |
| plan_v12_per_dim_uniform.md | inter-sample variance | ✅ 已实施（刚修复config bug） |

### 未实施的 ❌

| 计划来源 | 内容 | 当前状态 | 影响 |
|---------|------|---------|------|
| plan_v12_per_dim_uniform.md | **active_dim_loss** (死亡维度惩罚) | `src/training/losses.py` 中不存在该函数 | active_dims 只能观测，不能主动驱动 |
| plan_v12_vicreg_prenorm.md | **L2 辅助 batch_uniformity** (weight=0.05) | 配置中 `batch_uniformity_weight: 0.0` | L2 空间完全无监督，推理时可能坍缩 |
| EXPERIMENT_PLAN_v13_uniformity.md | **5组V13对照实验** | 配置存在但未运行 | 未验证 spatial uniformity / 大权重 / min_std修正 的独立效果 |

---

## 二、用户反馈的具体问题

### 问题1: "Active dims 怎么没有在训练中显示出来"

**现状分析:**
- Step 日志中有 `spatial=[128/128:0.5790]`，其中 `128/128` 就是 active_dims
- Epoch 摘要中有 `active=128/128 std_mean=0.5790`
- **但缺少 `inter_active` 在 epoch 摘要中的显示**（只出现在 step 日志的 `inter=[125/128:0.1699]`）
- **更关键：没有 `active_dim_loss`** — active_dims 只是被动观测指标，不是损失驱动项

**根因:**
- `plan_v12_per_dim_uniform.md` 中设计了 `active_dim_loss`（惩罚 std<threshold 的死亡维度），但代码中从未实现

### 问题2: "不同派系之间的差异怎么也没有显示出来"

**现状分析:**
- "派系"指 ExpA/B/C/D 四个实验配置
- Monitor 脚本只有简单表格对比，缺少：
  1. **实时差异分析**：ExpB vs ExpA 的 inter_var 差了多少？ recon 变化趋势如何？
  2. **趋势图**：各指标随 epoch 的变化曲线
  3. **关键差异高亮**：哪个实验的 spatial 更好？哪个的 inter-sample 更优？
  4. **epoch 摘要中标识实验配置**：当前 epoch 摘要看不出是哪个实验

### 问题3: "你按照之前讨论的计划，有在实施吗？"

**诚实回答：**
- VICReg + pre-norm + memory bank + spatial VICReg + inter-variance 已实施
- **但 active_dim_loss、L2 辅助 uniformity、完整的 V13 对照实验未实施**
- 当前 4 实验（ExpA-D）是 V12 的 inter-variance 参数扫描，**不是** V13 计划中验证的 5 组反坍缩策略对照实验

---

## 三、修复计划（待用户确认）

### 阶段1: 补充缺失的反坍缩损失

#### 1.1 添加 `active_dim_loss` ⭐

```python
# src/training/losses.py
def active_dim_loss(embeddings: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """惩罚死亡维度 — 每个维度的 std 必须大于阈值.
    
    在 pre-norm 空间计算，threshold=0.02 是"有信息量"的最低标准。
    """
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-6)  # [D]
    return F.relu(threshold - std).mean()
```

**集成到 trainer:**
- 在 `gathered_pre`（或 `all_pre`）上计算
- weight 配置项：`active_dim_weight`（默认 0.0，backward compatible）
- 日志显示：`active_loss=X.XXXX`

#### 1.2 恢复 L2 辅助 batch_uniformity

```yaml
# 在4个实验配置中调整
training:
  batch_uniformity_weight: 0.05  # 0.0 → 0.05
```

**原因:**
- plan_v12_vicreg_prenorm.md 明确设计 L2 轻量监督作为保险
- 当前 weight=0，推理时 L2 空间无约束

### 阶段2: 改进日志与诊断

#### 2.1 Epoch 摘要增强

当前格式：
```
Epoch 001/50 | total=X.XXX recon=X.XXX ... active=128/128 std_mean=0.5790 lr=0.000001
```

建议格式：
```
[ExpA] Epoch 001/50 | recon=X.XXX consist=X.XXX cls=X.XXX
  VICReg: var=X.XXX cov=X.XXX | Spatial: active=128/128 std=0.5790
  Inter: active=125/128 std=0.1699 inter_var=0.1200
  Uniform: l2unif=X.XXX | Bank=512/512 | lr=0.000001 | time=XXs
```

**改动点:**
- 添加 `[ExpX]` 标识
- 分组显示：VICReg / Spatial / Inter / Uniform
- `inter_active` 和 `inter_std_mean` 加入 epoch 摘要

#### 2.2 Step 日志优化

- 保留当前格式（已足够详细）
- 添加 `active_loss` 显示（如果阶段1实施）

### 阶段3: 增强 Monitor 差异分析

#### 3.1 实验间差异对比

Monitor 报告增加：
```
## 实验间差异分析

| 对比 | Recon Δ | Spatial Active Δ | Inter Active Δ | Inter Var Δ |
|------|---------|-----------------|----------------|-------------|
| B vs A | +0.002 | 0/128 | +2/128 | +0.120 (B有inter_var) |
| C vs B | +0.005 | 0/128 | -1/128 | -0.015 (C weight更高但效果?) |
| D vs B | -0.015 | -5/128 | -5/128 | -0.025 (recon更高→diversity?) |
```

#### 3.2 趋势图生成

每 5 分钟保存 `4exp_trends.json`，供后续绘图：
```json
{
  "expA": {"epochs": [1,2,3], "recon": [0.40, 0.38, ...], "spatial_active": [128, 128, ...]},
  "expB": {...}
}
```

### 阶段4: 对齐 V13 对照实验（可选，需确认优先级）

如果用户认为当前 ExpA-D 不是正确的实验设计，可以切换为 V13 计划的 5 组对照实验：

| 实验 | 核心变量 | 目的 |
|------|---------|------|
| Exp1 | Spatial uniformity | 验证 GMP 是坍缩主因 |
| Exp2 | batch_uniformity_weight=0.5 | 验证权重不足 |
| Exp3 | vicreg_min_std=0.1 | 验证 min_std 不切实际 |
| Exp4 | 组合最优 | 验证组合效果 |
| Exp5 | pre-norm raw uniformity | 验证 L2 Jacobian 屏障 |

---

## 四、改动文件清单

| 文件 | 改动内容 |
|------|---------|
| `src/training/losses.py` | 新增 `active_dim_loss()` |
| `src/training/ddp_v12_trainer.py` | 集成 active_dim_loss；inter_active/inter_std 加入 loss_accum；改进 step 日志 |
| `scripts/train/train_ddp_v12.py` | Epoch 摘要格式增强（分组显示、Exp标识、inter_active） |
| `scripts/monitor_4exp.py` | 增加实验间差异对比、趋势数据保存 |
| `src/config.py` | 添加 `active_dim_weight` 字段 |
| `configs/v12_exp*.yaml` | 添加 `active_dim_weight`；可选调整 `batch_uniformity_weight` |

---

## 五、执行 Checklist

- [ ] 用户确认本计划
- [ ] 实施阶段1（active_dim_loss + L2辅助）
- [ ] 实施阶段2（日志增强）
- [ ] 实施阶段3（monitor增强）
- [ ] 终止当前训练，清理日志，重启
- [ ] 验证新日志格式正确
- [ ] 监控 5-10 epoch，确认指标健康
