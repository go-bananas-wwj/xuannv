# 重建+蒸馏专用训练计划 — Recon+Distill v1

> 目标: **最大化重建质量 + AEF蒸馏对齐**，不关注 erank / AUC / 下游分类指标  
> 日期: 2026-06-06  
> 配置: `configs/config_recon_distill_v1.yaml`  
> 实验名: `exp_recon_distill_v1`

---

## 一、核心变更总结

### 1.1 评估策略改变

| 旧策略 | 新策略 |
|--------|--------|
| 每10 epoch 跑 kNN mIoU + 变化检测 AUC | 每10 epoch 跑 **重建质量评估（PSNR/SSIM）** + kNN mIoU（仅参考） |
| AUC < 0.7 视为不及格 | **AUC 不再作为评判标准** |
| erank < 10 视为坍缩失败 | **erank 不再关注** |

### 1.2 训练损失调整

```yaml
# 核心: 重建主导
reconstruction_weight: 1.0          # 从 0.05 → 1.0 (20倍提升)
source_recon_weights: [1.0, 1.0, 1.0, 0.02, 0.1, 0.1]

# AEF 蒸馏辅助
aef_spatial_distill_weight: 0.5     # 空间蒸馏
aef_global_distill_weight: 0.25    # 全局蒸馏

# 全部关闭的反坍缩损失
pre_norm_uniform_weight: 0.0
variance_weight: 0.0
covariance_weight: 0.0
decorrelation_weight: 0.0
erank_loss_weight: 0.0
coding_rate_weight: 0.0
orthogonality_weight: 0.0
temporal_gap_aware_weight: 0.0
consistency_weight: 0.0
classification_weight: 0.0
patch_id_loss_weight: 0.0
```

### 1.3 代码修改清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/train/train.py` | ① `run_periodic_eval.py` 调用添加 `--skip-cd`（跳过AUC）<br>② 新增 `evaluate_reconstruction.py` 调用，每 eval_every epoch 自动跑 PSNR/SSIM |
| `scripts/eval/evaluate_reconstruction.py` | **新建**。加载模型 → 采样 patch → 前向(skip_decoder=False) → 逐源计算 PSNR/SSIM → 输出 JSON |
| `configs/config_recon_distill_v1.yaml` | **新建**。重建权重=1.0 + AEF蒸馏 + 全部反坍缩关闭 |

---

## 二、训练启动命令

```bash
cd /workspace/xuannv
conda activate xuannv

# 检查 NPU 占用
npu-smi info

# 8卡训练（推荐）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 \
    scripts/train/train.py \
    --config configs/config_recon_distill_v1.yaml \
    --save-every 10

# 或 4卡快速验证
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
    scripts/train/train.py \
    --config configs/config_recon_distill_v1.yaml \
    --save-every 10 \
    --epochs 20
```

---

## 三、评估流程

### 3.1 自动评估（训练过程中）

每 `eval_every=10` epoch，训练脚本自动执行：

1. **kNN 语义分割**（保留，仅作参考，不用于 early stop）
2. **重建质量评估**（新增，核心指标）：
   ```
   [ReconEval] s2: PSNR=25.86±22.68 SSIM=0.1503±0.3473 (n=528)
   [ReconEval] s1: PSNR=15.93±6.99 SSIM=0.0305±0.1685 (n=100)
   [ReconEval] landsat: PSNR=48.97±27.50 SSIM=0.4579±0.4715 (n=456)
   [ReconEval] dem: PSNR=xx.xx±xx.xx SSIM=xxxx (n=xxx)
   [ReconEval] worldcover: PSNR=xx.xx±xx.xx SSIM=xxxx (n=xxx)
   ```

### 3.2 手动评估（任意 checkpoint）

```bash
python scripts/eval/evaluate_reconstruction.py \
    --config configs/config_recon_distill_v1.yaml \
    --checkpoint /workspace/xuannv/outputs/exp_recon_distill_v1/epoch_40.pt \
    --output /workspace/xuannv/outputs/exp_recon_distill_v1/recon_epoch_40.json \
    --device npu:0 \
    --num-samples 50 \
    --months 4,5,6,7,8,9,10
```

### 3.3 重建可视化（手动）

评估脚本目前只输出 PSNR/SSIM 数值。如需可视化对比图，需后续扩展 `visualize_reconstruction()` 函数。

---

## 四、预期效果与基准

### 4.1 历史基准（HRE 困难重建实验，epoch 49）

| 源 | PSNR | SSIM | 备注 |
|------|------|------|------|
| S2 | 25.86 | 0.1503 | 方差大，部分易重建 |
| S1 | 15.93 | 0.0305 | SAR 最困难 |
| Landsat | 48.97 | 0.4579 | 较易重建 |

### 4.2 本实验预期

- **S2 PSNR > 30**（重建权重从0.05提升到1.0，模型被迫聚焦像素恢复）
- **S1 PSNR > 20**（SAR困难，但权重提升应带来改善）
- **Landsat PSNR > 50**（本身已较高，继续提升）
- **AEF蒸馏损失收敛**（aef_sp < 0.3, aef_gl < 0.15）

### 4.3 不关注的指标

| 指标 | 当前值 | 态度 |
|------|--------|------|
| erank | 8.5 | 不关心 |
| l2unif | -2.78 | 不关心 |
| CD AUC (cosine) | 0.54 | 不测 |
| kNN mIoU | 0.295 | 仅参考 |

---

## 五、风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| 重建权重1.0导致loss数值过大，梯度爆炸 | 中 | 已设置 `grad_clip_norm=1.0`，监控训练log中的 `total` 值 |
| AEF蒸馏与重建目标冲突 | 低 | AEF蒸馏权重0.5/0.25低于重建1.0，重建主导；如冲突可降至0.2/0.1 |
| 分类目标（WC/DW）重建质量差 | 中 | 分类用CE loss，PSNR/SSIM对分类目标意义有限；可单独看CE loss |
| 训练过拟合到重建，embedding完全无判别力 | 低 | 保留kNN评估作参考，如mIoU暴跌至<0.15则回调 |
| 评估脚本NPU OOM | 低 | `evaluate_reconstruction.py` 单卡评估，batch_size=1，num-samples=30 |

---

## 六、迭代方向（根据训练结果）

### 场景A: 重建质量好 + 蒸馏收敛
→ **成功**。可固定此配置，延长训练至100-200 epoch。

### 场景B: 重建质量好但蒸馏不收敛
→ 提高 AEF 蒸馏权重至 0.8/0.4，或检查 AEF embedding 数据路径是否正确。

### 场景C: 重建质量差（PSNR无提升）
→ 检查 source_recon_weights 是否过均衡；尝试给 SAR 更高权重；或检查 decoder 容量（增大 decoder_hidden_mult）。

### 场景D: 训练不稳定（NaN/Inf）
→ 降低 reconstruction_weight 至 0.5；增大 warmup_epochs；检查是否有源数据异常。

---

## 七、关键监控命令

```bash
# 实时查看训练日志
tail -f /workspace/xuannv/outputs/exp_recon_distill_v1/train_*.log

# 查看重建评估结果
cat /workspace/xuannv/outputs/exp_recon_distill_v1/recon_epoch_10.json

# 查看 kNN 参考结果
cat /workspace/xuannv/outputs/exp_recon_distill_v1/eval_epoch_10.json

# NPU 占用监控
watch -n 2 npu-smi info
```

---

## 八、一句话总结

> **重建权重拉到 1.0，反坍缩全部关闭，AEF 蒸馏辅助对齐。每 10 epoch 自动跑 PSNR/SSIM 评估，AUC 不测。目标是让模型把像素重建做到极致。**
