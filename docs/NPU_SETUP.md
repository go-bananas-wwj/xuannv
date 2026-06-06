# NPU 环境适配说明

## 环境
- conda 环境: `xuannv`
- 基于 `yuncheng` 环境克隆
- Python 3.11.15, torch 2.1.0, torch_npu 2.1.0.post18
- 8× Ascend 910B4 NPU

## NPU 适配修改
1. 所有 `.cuda()` → `.npu()`
2. 所有 `torch.cuda` → `torch.npu`
3. 所有 `torch.autocast(device_type="cuda")` → `torch.autocast(device_type="npu")`
4. 所有 `torch.cuda.amp.GradScaler` → `torch_npu.npu.amp.GradScaler`
5. 所有 `backend="nccl"` → `backend="hccl"`
6. `src/utils/device.py` 全面适配 NPU
7. 所有训练脚本添加 `import torch_npu`

## 快速开始
```bash
conda activate xuannv
cd /workspace/xuannv

# 单卡测试
python scripts/train/train_single_gpu.py --config configs/qwen_v1_scenes.yaml --device npu:0

# DDP 多卡训练 (3卡示例)
torchrun --nproc_per_node=3 scripts/train/train_ddp.py --config configs/qwen_v1_scenes.yaml
```

## 数据路径
- 权重: `/workspace/outputs/aef_qwen_v5_mixed_scale/` (软链接)
- 原始数据: 待从 `raw_backup.tar.gz` 解压到 `/workspace/raw/harbin_scenes`
