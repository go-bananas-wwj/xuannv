# V12 实验总结与下一步计划

> 生成时间: 2026-05-14
> 实验状态: 4 实验全部进入平台期，已停止

---

## 一、实验结果总结

### 1.1 基本数据

| 实验 | 配置 | 轮数 | 最佳 recon | 最佳轮次 | Recon 降幅 |
|------|------|------|-----------|---------|-----------|
| **ExpA** | Baseline (inter=0.0) | 39 | **0.0448** | 39 | 87.0% |
| **ExpB** | InterVar=0.1 | 39 | 0.0449 | 33 | 85.5% |
| **ExpC** | InterVar=0.2 | 38 | 0.0450 | 36 | 86.4% |
| **ExpD** | Recon=0.1 + Inter=0.1 | 38 | **0.0445** | 33 | 85.3% |

### 1.2 关键发现

**✅ 已确认可靠：**
- **Spatial VICReg 零坍缩**：128/128 全维度活跃，39 轮无衰减
- **Recon 收敛极限 ~0.045**：所有实验收敛到同一水平，差异 <0.0005
- **平台期明确**：Epoch 12-17 后 recon 不再下降，最后 10 轮变化 <0.0023

**⚠️ 已确认问题：**
- **Inter-variance 持续衰减**：ExpB(-41%), ExpC(-36%), ExpD(-47%)
- **Inter=0.2 不如 0.1**：ExpC 终点 0.094 < ExpB 终点 0.103
- **Recon 权重增加加速衰减**：ExpD(recon=0.1) 衰减最严重
- **L2 uniformity 完全缺失**：配置 weight=0，推理时 L2 空间无约束

**❓ 未解答问题：**
- Inter-variance 是否提升了变化检测 AUC？
- L2 embedding 在推理时的质量如何？
- Inter-var 衰减能否通过调参缓解？

---

## 二、下一步计划选项

### 选项 A：先验证再决定（⭐ 推荐）

**逻辑**：在投入更多训练资源之前，先验证当前实验的实际价值。

**步骤：**
1. 用 **ExpA epoch 39** (recon=0.0448) 和 **ExpB epoch 39** (recon=0.0449) 跑 AUC 验证
2. 对比两者变化检测能力
3. 如果 ExpB AUC > ExpA AUC（哪怕 +0.01）：证明 inter-variance 有价值，继续优化
4. 如果 ExpB AUC ≈ ExpA AUC：inter-variance 在此配置下效果有限，转向其他方向

**耗时**：~30 分钟（2 个 checkpoint × 2 卡并行）
**风险**：无

---

### 选项 B：减缓 Inter-variance 衰减

**假设**：inter-var 衰减是因为 min_std=0.3 太弱，模型容易"满足"要求。

**实验设计（8 卡并行，每组 2 卡 × 30 epoch）：**

| 实验 | 配置变化 | 目的 |
|------|---------|------|
| **ExpE** | inter_min_std=0.5 (其他同 ExpB) | 增大门槛，强制更高区分度 |
| **ExpF** | inter_min_std=0.3 + weight=0.15 | 增大权重，增强驱动力 |
| **ExpG** | inter_min_std=0.5 + weight=0.15 | 组合增强 |
| **ExpH** | inter_min_std=0.3 + recon=0.03 | 降低 recon 权重，减少挤压 |

**验证指标**：
- Inter-var 衰减幅度是否 <30%？
- Recon 是否仍能收敛到 <0.05？

**耗时**：~9 小时（30 epoch × 18 分钟）

---

### 选项 C：引入 L2 辅助 + Active Dim Loss

**假设**：当前缺失两个计划中的关键组件：L2 空间监督和死亡维度惩罚。

**实验设计（8 卡并行）：**

| 实验 | 配置变化 | 目的 |
|------|---------|------|
| **ExpI** | batch_uniformity_weight=0.05 (其他同 ExpB) | L2 空间轻量监督 |
| **ExpJ** | active_dim_weight=0.3 (新损失) | 死亡维度惩罚 |
| **ExpK** | L2(0.05) + active_dim(0.3) + inter(0.1) | 组合验证 |
| **ExpL** | L2(0.05) + active_dim(0.3) + inter(0.0) | 无 inter 的 baseline |

**代码改动**：
- `src/training/losses.py`：新增 `active_dim_loss()`
- `src/training/ddp_v12_trainer.py`：集成两个新损失
- `src/config.py`：添加 `active_dim_weight`、`batch_uniformity_weight` 字段

**耗时**：~9 小时 + 代码修改 30 分钟

---

### 选项 D：V13 对照实验（验证其他反坍缩策略）

**来源**：`EXPERIMENT_PLAN_v13_uniformity.md`

**实验设计（8 卡并行）：**

| 实验 | 核心变量 | 目的 |
|------|---------|------|
| **V13-1** | Spatial uniformity (embedding_map 上计算) | 验证 GMP 是坍缩主因 |
| **V13-2** | batch_uniformity_weight=0.5 | 验证权重不足 |
| **V13-3** | vicreg_min_std=0.1 (L2 空间) | 验证 min_std 不切实际 |
| **V13-4** | 组合最优 (1+2+3) | 验证组合效果 |

**注意**：V13 实验关注的是 **L2 空间的反坍缩**，与 V12 的 pre-norm 策略不同。

**耗时**：~9 小时 + 配置创建 15 分钟

---

## 三、建议的优先级

| 优先级 | 选项 | 理由 |
|--------|------|------|
| **P0** | **选项 A** (AUC 验证) | 成本低（30分钟），高信息价值，决定后续方向 |
| **P1** | **选项 B** (减缓衰减) | 如果 A 证明 inter-var 有价值，这是最直接的问题 |
| **P2** | **选项 C** (L2 + active_dim) | 补充计划中缺失的组件，但改动较大 |
| **P3** | **选项 D** (V13 对照) | 探索不同策略空间，但当前 V12 策略已有基础 |

---

## 四、立即可以执行的任务

1. **AUC 验证**（如果你选 A）
   - 用 `scripts/eval/validate_v*.py` 或写新脚本
   - 对比 ExpA epoch39 vs ExpB epoch39
   - 输出：AUC 差异、embedding 分布对比

2. **保存最佳 checkpoint 清单**
   - ExpA: `/workspace/outputs/v12_expA_baseline/epoch_best_epoch39.pt`
   - ExpB: `/workspace/outputs/v12_expB_inter01/epoch_best_epoch39.pt`
   - ExpC: `/workspace/outputs/v12_expC_inter02/epoch_best_epoch36.pt`
   - ExpD: `/workspace/outputs/v12_expD_recon01/epoch_best_epoch33.pt`

---

## 五、需要你做决定的问题

1. **是否先跑 AUC 验证？**（推荐，30 分钟，无风险）
2. **如果验证通过，优先测试哪个方向？**
   - B（减缓 inter-var 衰减）
   - C（L2 + active_dim）
   - D（V13 对照）
3. **下一轮实验的 epoch 数**：继续 50 epoch 还是缩短到 30 epoch？（平台期在 12-17 epoch 开始，30 epoch 可能足够）
