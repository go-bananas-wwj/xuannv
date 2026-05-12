# V13 Uniformity 坍缩对照实验计划

## 背景

V13 训练 5 个 epoch 后出现严重 uniformity 坍缩：
- `l2unif`: 0.55 → 0.86（越接近 1 越坍缩）
- `active_dims`: 35 → 7（有效维度从 35 跌至 7）

已确认的坍缩原因：
1. **GMP（Global Mean Pool）**: 把 spatial embedding map [B,D,H,W] 压成 [B,D]，抹掉空间区分度，uniformity 从 0.606 恶化到 1.000
2. **Uniformity 权重太小**: `batch_uniformity_weight=0.05`，无法对抗 reconstruction（0.5）和 consistency（0.2）的坍缩拉力
3. **VICReg min_std 不切实际**: `min_std=1.0` 对 L2 归一化 embedding 是不可能达到的天花板（理论 std≈0.088），variance loss 几乎恒为 0.99，无实际约束

## 实验目标

通过 5 组对照实验（每组 2 卡 × 10 epoch），确定最有效的反坍缩策略。

## 实验设计

### 实验 0：Baseline（已有数据，不复跑）

- **配置**: `configs/xuannv_v12_clean.yaml`（当前配置）
- **已有数据**:
  | Epoch | l2unif | active_dims | recon |
  |-------|--------|-------------|-------|
  | 1 | 0.5488 | 35 | 0.3637 |
  | 2 | 0.5977 | 25 | 0.3573 |
  | 3 | 0.7023 | 17 | 0.3329 |
  | 4 | 0.8078 | 9 | 0.2955 |
  | 5 | ~0.86 | 7 | ~0.27 |

---

### 实验 1：Spatial Uniformity（去掉 GMP 影响）

**假设**: GMP 把 spatial 信息抹掉是 uniformity 恶化的主因。如果在 spatial level（embedding_map）计算 uniformity，可以 bypass GMP 的坍缩效应。

**修改**:
- 在 `ddp_v12_trainer.py` 中，把 uniformity loss 的输入从 `student_out.embedding` [B,D] 改为 `student_out.embedding_map` [B,D,H,W]
- `batch_uniformity_loss_l2` 已原生支持 4D 输入（会自动 reshape 为 [B×H×W, D] 并随机采样 512 个空间位置）
- 其他所有配置保持不变

**配置**:
```yaml
experiment:
  name: v13_exp1_spatial_unif
  output_dir: /workspace/outputs/v13_exp1_spatial_unif

# 其余同 baseline，仅 trainer 代码修改
training:
  batch_uniformity_weight: 0.05  # 不变
```

**评估指标**: l2unif, active_dims, recon, total

---

### 实验 2：增大 Uniformity 权重

**假设**: 当前 uniformity weight=0.05 太小，无法对抗 reconstruction（weight=0.5）的坍缩拉力。增大到 0.5 可以给 uniformity 足够的话语权。

**修改**: 仅改配置文件
```yaml
experiment:
  name: v13_exp2_high_weight
  output_dir: /workspace/outputs/v13_exp2_high_weight

training:
  batch_uniformity_weight: 0.5  # 0.05 → 0.5
```

**评估指标**: l2unif, active_dims, recon, total

---

### 实验 3：修正 VICReg min_std

**假设**: `min_std=1.0` 对 L2 归一化 embedding 是不可能的（理论 std≈0.088），导致 variance loss 恒为 ~0.99，无实际约束。将 min_std 降到 0.1 可以让 variance loss 真正起效。

**修改**:
- 在 `ddp_v12_trainer.py` 中，`variance_regularizer(all_pre.float(), min_std=1.0)` → `min_std=0.1`
- 其他配置不变

**配置**:
```yaml
experiment:
  name: v13_exp3_vicreg_fix
  output_dir: /workspace/outputs/v13_exp3_vicreg_fix

training:
  batch_uniformity_weight: 0.05  # 不变
```

**评估指标**: l2unif, active_dims, recon, var, total

---

### 实验 4：组合最优（Spatial + 大权重 + 修正 min_std）

**假设**: 单一修改可能不够，组合所有已知有效的修改可以获得最佳效果。

**修改**:
- 实验 1（Spatial uniformity）+ 实验 2（weight=0.5）+ 实验 3（min_std=0.1）

**配置**:
```yaml
experiment:
  name: v13_exp4_combined
  output_dir: /workspace/outputs/v13_exp4_combined

training:
  batch_uniformity_weight: 0.5
```

**评估指标**: l2unif, active_dims, recon, var, total

---

### 实验 5：Pre-norm Raw Uniformity（替代 L2 uniformity）

**假设**: L2 归一化本身有 Jacobian 梯度屏障 `(I-uu^T)/||x||`，当 embedding 坍缩时梯度趋于 0。在 pre-norm 欧氏空间计算 `raw_uniformity_loss` 可以绕过这个屏障。

**修改**:
- 在 `ddp_v12_trainer.py` 中，把 L2 uniformity 替换为 pre-norm raw uniformity
- `raw_uniformity_loss(gathered_pre.float())` 替代 `batch_uniformity_loss_l2(gathered_l2.float())`
- weight 相应调整

**配置**:
```yaml
experiment:
  name: v13_exp5_prenorm_raw
  output_dir: /workspace/outputs/v13_exp5_prenorm_raw

training:
  # 禁用 L2 uniformity，启用 pre-norm uniformity
  batch_uniformity_weight: 0.0
  pre_norm_uniform_weight: 0.3  # 新增
```

**评估指标**: raw_unif, active_dims, recon, total

---

## 资源规划

| 实验 | 卡数 | 预计时间 | NPU 分配 |
|------|------|----------|----------|
| Exp1 | 2 | ~3.5h | NPU 0,1 |
| Exp2 | 2 | ~3.5h | NPU 2,3 |
| Exp3 | 2 | ~3.5h | NPU 4,5 |
| Exp4 | 2 | ~3.5h | NPU 6,7 |
| Exp5 | 2 | ~3.5h | NPU 0,1（第二轮） |

**并行策略**: 第一轮同时跑 Exp1~Exp4（占满 8 卡），第二轮跑 Exp5（2 卡）。
**总预计时间**: ~7 小时。

---

## 需要创建的文件

### 1. 配置文件（5 个）
- `configs/v13_exp1_spatial_unif.yaml`
- `configs/v13_exp2_high_weight.yaml`
- `configs/v13_exp3_vicreg_fix.yaml`
- `configs/v13_exp4_combined.yaml`
- `configs/v13_exp5_prenorm_raw.yaml`

### 2. Trainer 修改
- `src/training/ddp_v12_trainer.py`: 增加实验变体支持
  - `use_spatial_uniformity`: bool（实验 1, 4）
  - `vicreg_min_std`: float（实验 3, 4）
  - `use_pre_norm_uniform`: bool（实验 5）
  - 通过配置文件读取，默认行为保持 backward compatible

### 3. 启动脚本
- `scripts/train/launch_experiments.sh`: 一键启动所有实验（tmux session）

---

## 评估方案

每个实验跑完 10 epoch 后，提取以下指标：

| 指标 | 含义 | 正常范围 | 失败信号 |
|------|------|----------|----------|
| `l2unif` / `raw_unif` | uniformity | < 0.6 | > 0.8 |
| `active_dims` | 有效维度数 | > 30 | < 15 |
| `recon` | 重建损失 | < 0.35 | > 0.5 |
| `var` | VICReg variance | < 0.5 | > 0.9（min_std  unreachable） |
| `total` | 总损失 | 稳定下降 | 震荡或上升 |

**决策规则**:
1. 哪个实验的 `l2unif` 在 epoch 10 最低？
2. 哪个实验的 `active_dims` 保持在 30 以上？
3. 重建损失是否仍然收敛？
4. 综合以上，选择最优策略应用到主训练

---

## 风险控制

1. **OOM 风险**: 实验 1 的 spatial uniformity 处理 4D tensor [B,D,H,W]，内存开销略大。若 OOM，将 `max_samples` 从 512 降到 256。
2. **训练发散风险**: 实验 2/4 的 weight=0.5 可能使 uniformity 过度主导。若 recon 不下降，降至 0.3。
3. **NaN/Inf**: 实验 5 的 pre-norm 空间计算可能遇到数值问题。已有 encode_frames fallback 修复，应无问题。

---

## 执行 checklist

- [ ] 用户确认实验计划
- [ ] 创建 5 个配置文件
- [ ] 修改 trainer 支持实验变体
- [ ] 启动第一轮实验（Exp1~4）
- [ ] 监控训练，确认无 OOM/NaN
- [ ] 启动第二轮实验（Exp5）
- [ ] 汇总 10 epoch 数据，生成对比表格
- [ ] 根据实验结果确定最终策略
