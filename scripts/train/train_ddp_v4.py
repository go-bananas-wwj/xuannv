#!/usr/bin/env python3
"""DDP V4 训练入口 — GPU 6/7 双卡并行.

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
        scripts/train/train_ddp_v4.py --config configs/qwen_v4_cd_upgrade.yaml

支持:
  - EMA Teacher + DINO/iBOT 自蒸馏
  - 跨时相掩码重建
  - VICReg + KoLeo 反坍缩
  - 相邻月份双窗口采样
  - resume 断点续训
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# torchrun 已带 -u (unbuffered)，无需手动重定向 stdout

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.distributed as dist

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.ddp_v4_trainer import DDPv4Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--save-every", type=int, default=None, help="每隔多少 epoch 保存检查点")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


class FileLogger:
    """同时写入 stdout 和文件的 logger（仅 rank 0）."""
    def __init__(self, filepath: str, rank: int):
        self.rank = rank
        self.file = None
        if rank == 0:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(filepath, "a", buffering=1, encoding="utf-8")

    def write(self, msg: str):
        if self.rank == 0:
            sys.stdout.write(msg)
            sys.stdout.flush()
            if self.file:
                self.file.write(msg)
                self.file.flush()

    def print(self, msg: str):
        self.write(msg + "\n")


def main():
    args = parse_args()

    # DDP 初始化
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger = FileLogger("/workspace/outputs/aef_qwen_v4_cd_upgrade/train.log", global_rank)

    cfg = load_config(args.config)

    # 覆盖参数
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.save_every is not None:
        cfg.training.save_every = args.save_every

    # 固定随机种子（每个 rank 不同以避免完全同步）
    seed = getattr(cfg.experiment, "seed", 42) + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if global_rank == 0:
        logger.print("=" * 70)
        logger.print(f"  DDP V4 训练  [GPU 6,7]  —  {cfg.experiment.name}")
        logger.print(f"  World size: {world_size}  |  Rank: {global_rank}")
        logger.print("=" * 70)
        logger.print(f"  Config: {args.config}")
        logger.print(f"  Epochs: {cfg.training.epochs}")
        logger.print(f"  Batch per GPU: {cfg.data.batch_size}")
        logger.print(f"  Grad accum: {getattr(cfg.training, 'gradient_accumulation_steps', 4)}")
        logger.print(f"  Effective batch: {cfg.data.batch_size * world_size * getattr(cfg.training, 'gradient_accumulation_steps', 4)}")
        logger.print(f"  Window mode: {getattr(cfg.data, 'window_mode', 'random_split')}")
        logger.print("=" * 70)

    # DataLoader (distributed)
    dataloader = build_dataloader(
        cfg,
        training=True,
        distributed=True,
        world_size=world_size,
        rank=global_rank,
    )

    # Trainer
    trainer = DDPv4Trainer(cfg, local_rank=local_rank)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if global_rank == 0:
            logger.print(f"[train] Resumed from {args.resume}, starting at epoch {start_epoch + 1}")
        dist.barrier()

    total_epochs = cfg.training.epochs
    if args.resume and start_epoch > 0:
        if global_rank == 0:
            logger.print(f"[train] Will train from epoch {start_epoch + 1} to {total_epochs}")

    best_loss = float("inf")
    save_every = args.save_every if args.save_every is not None else getattr(cfg.training, "save_every", 20)

    for epoch in range(start_epoch, total_epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        losses = trainer.train_epoch(epoch, dataloader)

        if global_rank == 0:
            logger.print(
                f"Epoch {epoch + 1:03d}/{cfg.training.epochs} | "
                f"total={losses['total']:.4f} recon={losses['recon']:.4f} "
                f"ct_recon={losses['ct_recon']:.4f} dino={losses.get('dino', 0.0):.4f} "
                f"vicreg={losses['vicreg']:.4f} koleo={losses['koleo']:.4f} "
                f"temporal={losses['temporal']:.4f} lr={losses['lr']:.6f}"
            )

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, losses)

        if losses["total"] < best_loss:
            best_loss = losses["total"]
            # 保存 best 时传入实际 epoch 数字，避免 resume 时字符串问题
            trainer.save_checkpoint(f"best_epoch{epoch + 1}", losses)

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if global_rank == 0:
        logger.print("[train] Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
