#!/usr/bin/env python
"""V7 极简验证 DDP 训练入口.

用法:
    torchrun --nproc_per_node=8 \
        scripts/train/train_ddp_v7_minimal.py --config configs/qwen_v7_minimal.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AEF_qwen V7 Minimal DDP Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    args = parser.parse_args()

    # 确保项目根目录在 path 中
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 如果通过 torchrun 启动 (多个进程)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    import torch
    import torch_npu
    import torch.distributed as dist

    from src.config import load_config

    cfg = load_config(args.config)

    # 覆盖命令行参数
    if args.warmup_epochs is not None:
        cfg.training.warmup_epochs = args.warmup_epochs
    if args.save_every is not None:
        cfg.training.save_every = args.save_every

    # 设置随机种子
    torch.manual_seed(cfg.experiment.seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(cfg.experiment.seed)

    # DDP 训练
    from src.training.ddp_v7_minimal_trainer import train_worker
    train_worker(local_rank, cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
