#!/usr/bin/env python3
"""V7 Phase 1 DDP 训练入口."""
from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu  # 必须先导入以注册 HCCL 后端
import torch.distributed as dist

from src.config import load_config
from src.training.ddp_v7_trainer import train_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--save-every", type=int, default=20, help="每 N epoch 保存")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="warmup epoch 数")
    parser.add_argument("--wandb-project", type=str, default="xuannv-backbone", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb-offline", action="store_true", help="wandb 离线模式")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.save_every:
        cfg.training.save_every = args.save_every
    if args.warmup_epochs:
        cfg.training.warmup_epochs = args.warmup_epochs

    # 处理 resume 配置
    resume_from = args.resume
    if resume_from:
        print(f"[ddp] Resume from {resume_from}")
    else:
        print("[ddp] Training from scratch")

    # wandb 配置
    wandb_config = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": "offline" if args.wandb_offline else "online",
        "config_path": args.config,
        "resume": bool(resume_from),
    }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_worker(local_rank, cfg, resume_from=resume_from, wandb_config=wandb_config)


if __name__ == "__main__":
    import os
    main()
