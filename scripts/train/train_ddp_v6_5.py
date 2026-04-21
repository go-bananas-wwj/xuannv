#!/usr/bin/env python3
"""DDP V6.5 Gap-Aware Temporal 训练入口 — GPU 6/7 双卡并行.

核心改进 (vs V6):
  - Gap-aware temporal cosine loss: 根据时间 gap 设定 target
  - 降低 temporal 权重, 增强 uniformity/variance
  - 软重启 (保留 encoder, 重置 bottleneck/decoder/head)
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
import torch.distributed as dist

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.ddp_v6_5_gap_aware_trainer import DDPv6_5GapAwareTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--soft-restart", type=str, default=None, help="软重启: 从旧 checkpoint 加载 encoder，重置其余")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--save-every", type=int, default=None, help="每隔多少 epoch 保存检查点")
    parser.add_argument("--local-rank", type=int, default=0)
    return parser.parse_args()


class FileLogger:
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

    def Print(self, msg: str):
        self.write(msg + "\n")


def main():
    args = parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = load_config(args.config)
    log_path = Path(cfg.experiment.output_dir) / "train.log"
    logger = FileLogger(str(log_path), global_rank)

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.save_every is not None:
        cfg.training.save_every = args.save_every

    seed = getattr(cfg.experiment, "seed", 42) + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if global_rank == 0:
        logger.Print("=" * 70)
        logger.Print(f"  DDP V6.5 Gap-Aware 训练  [GPU 6,7]  —  {cfg.experiment.name}")
        logger.Print(f"  World size: {world_size}  |  Rank: {global_rank}")
        logger.Print("=" * 70)
        logger.Print(f"  Config: {args.config}")
        logger.Print(f"  Epochs: {cfg.training.epochs}")
        logger.Print(f"  Batch per GPU: {cfg.data.batch_size}")
        logger.Print(f"  Grad accum: {getattr(cfg.training, 'gradient_accumulation_steps', 12)}")
        logger.Print(f"  Effective batch: {cfg.data.batch_size * world_size * getattr(cfg.training, 'gradient_accumulation_steps', 12)}")
        logger.Print(f"  Window mode: {getattr(cfg.data, 'window_mode', 'random_split')}")
        logger.Print(f"  --- 损失权重 ---")
        logger.Print(f"  Recon: {getattr(cfg.training, 'reconstruction_weight', 1.0):.2f}")
        logger.Print(f"  Consistency: {getattr(cfg.training, 'consistency_weight', 0.0):.2f}")
        logger.Print(f"  Classification: {getattr(cfg.training, 'classification_weight', 0.0):.3f}")
        logger.Print(f"  Uniformity (global): {getattr(cfg.training, 'uniformity_weight', 0.0):.2f}")
        logger.Print(f"  Uniformity (spatial): {getattr(cfg.training, 'spatial_uniformity_weight', 0.0):.2f}")
        logger.Print(f"  Variance: {getattr(cfg.training, 'variance_weight', 0.0):.2f}")
        logger.Print(f"  Decorrelation: {getattr(cfg.training, 'decorrelation_weight', 0.0):.3f}")
        logger.Print(f"  Orthogonality: {getattr(cfg.training, 'orthogonality_weight', 0.0):.3f}")
        logger.Print(f"  Temporal magnitude: {getattr(cfg.training, 'temporal_magnitude_weight', 0.0):.2f}")
        logger.Print(f"  Gap-aware cosine: {getattr(cfg.training, 'temporal_cosine_pixel_weight', 0.0):.2f}")
        logger.Print(f"  Pixel InfoNCE: {getattr(cfg.training, 'pixel_temporal_info_nce_weight', 0.0):.2f}")
        logger.Print(f"  Kappa: {getattr(cfg.training, 'kappa_start', 50.0)} → {getattr(cfg.training, 'kappa_end', 500.0)}")
        logger.Print("=" * 70)

    dataloader = build_dataloader(
        cfg,
        training=True,
        distributed=True,
        world_size=world_size,
        rank=global_rank,
    )

    trainer = DDPv6_5GapAwareTrainer(cfg, local_rank=local_rank)

    start_epoch = 0
    if args.soft_restart:
        trainer.soft_restart(args.soft_restart)
        if global_rank == 0:
            logger.Print(f"[train] Soft restart from {args.soft_restart}")
        dist.barrier()
    elif args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if global_rank == 0:
            logger.Print(f"[train] Resumed from {args.resume}, starting at epoch {start_epoch + 1}")
        dist.barrier()

    total_epochs = cfg.training.epochs
    best_loss = float("inf")
    save_every = args.save_every if args.save_every is not None else getattr(cfg.training, "save_every", 20)

    for epoch in range(start_epoch, total_epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        losses = trainer.train_epoch(epoch, dataloader)

        if global_rank == 0:
            logger.Print(
                f"Epoch {epoch + 1:03d}/{cfg.training.epochs} | "
                f"total={losses['total']:.4f} recon={losses['recon']:.4f} "
                f"consist={losses['consist']:.4f} cls={losses['cls']:.4f} "
                f"uniform={losses['uniform']:.4f} spatial_u={losses['spatial_uniform']:.4f} "
                f"var={losses['var']:.4f} decorr={losses['decorr']:.4f} "
                f"orth={losses['orth']:.4f} temporal={losses['temporal']:.4f} "
                f"gap_aware={losses['gap_aware']:.4f} ptnce={losses['ptnce']:.4f} "
                f"lr={losses['lr']:.6f}"
            )

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, losses)

        if losses["total"] < best_loss:
            best_loss = losses["total"]
            trainer.save_checkpoint(f"best_epoch{epoch + 1}", losses)

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if global_rank == 0:
        logger.Print("[train] Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
