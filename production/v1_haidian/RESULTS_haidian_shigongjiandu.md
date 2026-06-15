# Haidian 施工工地监测（shigongjiandu）优化结果

## 目标
- F1 ≥ 0.6
- AUC ≥ 0.8

## 最终模型
- **微调 checkpoint**: `production/v1_haidian/outputs/finetune_haidian_cd_phase2/best.pt`
- **部署目录**: `production/v1_haidian/model_finetuned_haidian_cd_phase2/`
- **基座**: 多分辨率 v1 生产模型（`model/epoch_80.pt`）
- **微调方式**: 在 63 个有标注 patch 上，用 `(2025-12, 2026-04)` 双时相对 + `ChangeDetectionHeadV2` + Focal/Dice 损失，端到端微调 backbone。

## 训练阶段
1. **Phase 1**（冻结 sensor encoder）:
   ```bash
   python -m torch.distributed.run --nproc_per_node=1 \
     scripts/finetune_embedding_haidian.py \
     --model-dir model --label-dir /workspace/xuannv/haidian_label/labeljson \
     --output-dir outputs/finetune_haidian_cd_e20_pw50 \
     --device npu:0 --epochs 20 --lr 1e-4 --batch-size 4 --accum-steps 2 \
     --freeze-sensor-encoder 1 --val-patches 10 --patience 10 --pos-weight 50
   ```
2. **Phase 2**（解冻 sensor encoder，从 Phase 1 best 继续）:
   ```bash
   python -m torch.distributed.run --nproc_per_node=1 \
     scripts/finetune_embedding_haidian.py \
     --model-dir model --label-dir /workspace/xuannv/haidian_label/labeljson \
     --resume-model outputs/finetune_haidian_cd_e20_pw50/best.pt \
     --output-dir outputs/finetune_haidian_cd_phase2 \
     --device npu:0 --epochs 30 --lr 5e-5 --batch-size 4 --accum-steps 2 \
     --freeze-sensor-encoder 0 --val-patches 10 --patience 15 --pos-weight 100
   ```

## 下游头评估（在 Phase 2 best embedding 上）
```bash
python scripts/compare_heads.py \
  --model-dir model \
  --checkpoint outputs/finetune_haidian_cd_phase2/best.pt \
  --label-dir /workspace/xuannv/haidian_label/labeljson \
  --output-dir outputs/eval_phase2best_all \
  --device npu:1 --mode bitemporal \
  --heads mlp_torch_v2,mlp_diff_upsample,unet,cdhead \
  --task shigongjiandu
```

## 结果汇总

| Head | F1 | AUC | IoU | Balanced Acc | 阈值来源 |
|------|----|-----|----|-------------|----------|
| cdhead | **0.684** | **0.931** | 0.519 | 0.826 | 0.5（默认） |
| unet | 0.630 | 0.909 | 0.460 | 0.846 | 验证集 F1 调优 |
| mlp_diff_upsample | 0.630 | 0.908 | 0.460 | 0.799 | 验证集 F1 调优 |
| mlp_torch_v2 | 0.613 | 0.890 | 0.442 | 0.788 | 0.5（默认） |

所有 head 均超过目标阈值（F1≥0.6，AUC≥0.8）。最佳组合为 **Phase 2 微调 embedding + CDHead**，达到 **F1=0.684，AUC=0.931**。

## 关键改进点
1. **嵌入模型微调**：将预训练 AEF backbone 用 Haidian 施工工地标注做有监督变化检测微调，显著提升 embedding 对变化区域的敏感性。
2. **下游头优化**：为 `PixelMLPHeadV3` 与 `UNetHead` 增加验证集 F1 阈值自动搜索；横向对比 4 种 head，选择最佳方案。
3. **双阶段训练策略**：先冻结 sensor encoder 稳定训练，再解冻进行端到端精调。

## 文件说明
- `scripts/finetune_embedding_haidian.py`: backbone 有监督微调入口。
- `scripts/compare_heads.py`: 下游头横向对比（支持 `--checkpoint` 加载微调模型）。
- `xuannv_v1/haidian_heads.py`: `PixelMLPHeadV3` / `UNetHead` 等下游头实现（含验证集阈值调优）。
- `xuannv_v1/haidian_tasks.py`: 任务训练/评估流程。
