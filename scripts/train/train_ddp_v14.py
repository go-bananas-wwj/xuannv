#!/usr/bin/env python3
"""DDP V14 训练入口 — 共享参数Teacher-Student + 多区域混合 + Coding Rate Loss.

用法:
    cd /workspace/xuannv
    torchrun --nproc_per_node=1 \
        scripts/train/train_ddp_v14.py --config configs/v14_shared_ts_multi.yaml \
        --save-every 10

V14 核心:
  - 共享参数 Teacher-Student（对齐AEF原文）
  - 多区域混合训练（哈尔滨+大庆+海淀）
  - Coding Rate Loss（MCR²）
  - Memory Bank 扩大有效batch
  - 有效秩监控
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import time

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.data.multi_region_dataset import MultiRegionPatchDataset
from src.training.ddp_v14_trainer import DDPv14Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--soft-restart", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=0)
    return parser.parse_args()


class FileLogger:
    def __init__(self, log_path: str):
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    def log(self, msg: str):
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.save_every is not None:
        cfg.training.save_every = args.save_every

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch_npu.npu.set_device(local_rank)

    if "RANK" in os.environ:
        dist.init_process_group(backend="hccl")
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
    else:
        world_size = 1
        global_rank = 0

    if global_rank == 0:
        print(f"[V14] Config: {args.config}")
        print(f"[V14] Epochs: {cfg.training.epochs}")
        print(f"[V14] World size: {world_size}")
        print(f"[V14] Local rank: {local_rank}")
        print(f"[V14] Multi-region manifest: {getattr(cfg.data, 'multi_region_manifest', 'N/A')}")

    # 设置随机种子
    seed = getattr(cfg.experiment, "seed", 42)
    random.seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    torch.manual_seed(seed + global_rank)

    # 创建数据集和DataLoader
    if getattr(cfg.data, 'multi_region_manifest', None):
        dataset = MultiRegionPatchDataset(cfg)
    else:
        from src.data.dataset import HarbinPatchDataset
        dataset = HarbinPatchDataset(cfg)

    dataloader = build_dataloader(dataset, cfg, is_train=True)

    if global_rank == 0:
        print(f"[V14] Dataset size: {len(dataset)} patches")

    # 创建训练器
    trainer = DDPv14Trainer(cfg, local_rank)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if global_rank == 0:
            print(f"[V14] Resumed from epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_start = time.time()
        metrics = trainer.train_epoch(epoch, dataloader)
        epoch_time = time.time() - epoch_start

        if global_rank == 0:
            print(f"[Epoch {epoch}] time={epoch_time:.1f}s | " + " | ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            ))

        # 保存检查点
        if (epoch + 1) % cfg.training.save_every == 0 or epoch == cfg.training.epochs - 1:
            if global_rank == 0:
                trainer.save_checkpoint(epoch)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
