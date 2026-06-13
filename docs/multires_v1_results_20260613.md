# 多分辨率 v1 训练结果摘要

> 备份时间：2026-06-13  
> 对应 tag：`multires_v1_baseline`  
> 权重备份：`outputs/exp_multires_v1_0612_backup/`

## 1. 训练配置

- 配置：`configs/config_multires_v1.yaml`
- 输入源：S2 / S1 / Landsat / 天仪 SAR（10m）
- 训练区域：海淀 320 patch
- 训练时段：2025-04 至 2025-10（配置默认月份）
- 反坍缩/对比损失：全部关闭
- 训练 epochs：80

## 2. 训练结果

| 检查点 | total loss | recon loss | 内部 kNN acc/mIoU |
|--------|-----------|-----------|------------------|
| epoch 40 (best) | — | — | 0.7520 / 0.3788 |
| epoch 80 | 1.585 | 1.476 | 0.6758 / 0.2569 |

- embedding 维度：D=64
- **erank = 8.88 / 64**（严重坍缩，正常应 >32）

## 3. 重建评估

| 源 | PSNR | SSIM |
|-----|------|------|
| S2 | 49.10 | 0.817 |
| Landsat | 45.49 | 0.737 |
| S1 | 14.63 | -0.028 |
| 天仪 SAR | 12.02 | 0.031 |

## 4. 海淀区 2026 标注下游评估

### 4.1 单时相 2026-04 embedding

| 类别 | Linear AUC | MLP AUC |
|------|-----------|---------|
| 施工工地 (gongdi) | 0.610 | 0.562 |
| 建筑用地 (jianzhudongdi) | 0.396 | 0.435 |
| 疑似违建 (weijian) | 0.370 | 0.529 |
| 农用地变化 (nongyongdi) | 0.799 | 0.541 |
| 建筑消失 (chaichu) | 0.557 | 0.433 |
| 施工道路 (daolubianhua) | 0.665 | 0.629 |

### 4.2 双时相 2025-12 + 2026-04 embedding

| 类别 | Linear AUC | MLP AUC |
|------|-----------|---------|
| 施工工地 (gongdi) | 0.661 | 0.675 |
| 建筑用地 (jianzhudongdi) | 0.532 | 0.505 |
| 疑似违建 (weijian) | 0.396 | 0.600 |
| 农用地变化 (nongyongdi) | 0.817 | 0.754 |
| 建筑消失 (chaichu) | 0.624 | 0.488 |
| 施工道路 (daolubianhua) | 0.680 | 0.647 |

### 4.3 关键结论

- MLP 在极端类别不平衡下基本失效（BAcc≈0.5），Linear 更可靠。
- 双时相比单时相对施工工地、建筑用地、建筑消失等有明显提升。
- 天花板受 embedding 坍缩限制，需恢复反坍缩/时序/蒸馏损失。

## 5. 关键文件路径

- 权重：`outputs/exp_multires_v1_0612_backup/epoch_80.pt`
- 单时相 2026-04 评估：`outputs/exp_multires_v1_0612_backup/haidian2026_eval_202604/metrics.json`
- 双时相评估：`outputs/exp_multires_v1_0612_backup/haidian2026_eval_bitemporal/metrics.json`
- 可视化：`outputs/exp_multires_v1_0612_backup/haidian2026_eval_bitemporal/visualizations/`
