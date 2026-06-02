#!/usr/bin/env python3
"""DDP V11 训练入口 — 数据扩展 + 架构优化.

用法:
    cd /workspace/xuannv
    torchrun --nproc_per_node=8 \
        scripts/train/train_ddp_v11.py --config configs/xuannv_v11.yaml \
        --soft-restart /workspace/outputs/xuannv_backbone_v10_temporal/epoch_60.pt \
        --epochs 200

V11 改进:
  - 源特定重建权重
  - 移除 decorrelation + orthogonality
  - 强化一致性 (0.2)
  - 静态目标时间编码弱化
  - 课程学习训练
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

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.ddp_v11_trainer import DDPv11Trainer


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

    def Print(self, msg: str):
        self.write(msg + "\n")


def main():
    args = parse_args()

    # DDP 初始化
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.npu.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = load_config(args.config)
    log_path = Path(cfg.experiment.output_dir) / "train.log"
    logger = FileLogger(str(log_path), global_rank)

    # 覆盖参数
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.save_every is not None:
        cfg.training.save_every = args.save_every

    # 固定随机种子
    seed = getattr(cfg.experiment, "seed", 42) + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

    if global_rank == 0:
        logger.Print("=" * 70)
        logger.Print(f"  DDP V11 训练  [8×NPU]  —  {cfg.experiment.name}")
        logger.Print(f"  World size: {world_size}  |  Rank: {global_rank}")
        logger.Print("=" * 70)
        logger.Print(f"  Config: {args.config}")
        logger.Print(f"  Epochs: {cfg.training.epochs}")
        logger.Print(f"  Batch per GPU: {cfg.data.batch_size}")
        logger.Print(f"  Grad accum: {getattr(cfg.training, 'gradient_accumulation_steps', 8)}")
        logger.Print(f"  Effective batch: {cfg.data.batch_size * world_size * getattr(cfg.training, 'gradient_accumulation_steps', 8)}")
        logger.Print(f"  Recon weight: {getattr(cfg.training, 'reconstruction_weight', 1.0)}")
        logger.Print(f"  Consistency weight: {getattr(cfg.training, 'consistency_weight', 0.0)}")
        logger.Print(f"  Classification weight: {getattr(cfg.training, 'classification_weight', 0.0)}")
        logger.Print(f"  Uniformity weight: {getattr(cfg.training, 'uniformity_weight', 0.0)}")
        logger.Print(f"  Variance weight: {getattr(cfg.training, 'variance_weight', 0.0)}")
        src_weights = getattr(cfg.training, 'source_recon_weights', [1.0]*7)
        logger.Print(f"  Source recon weights: {src_weights}")
        logger.Print(f"  Temporal gap-aware weight: {getattr(cfg.training, 'temporal_gap_aware_weight', 0.0)} → {getattr(cfg.training, 'temporal_gap_aware_weight_end', 0.08)}")
        logger.Print(f"  Temporal warmup epochs: {getattr(cfg.training, 'temporal_gap_aware_warmup_epochs', 30)}")
        logger.Print(f"  Temporal ramp epochs: {getattr(cfg.training, 'temporal_gap_aware_weight_ramp_epochs', 50)}")
        logger.Print(f"  Pixel change weight: {getattr(cfg.training, 'pixel_change_supervision_weight', 0.0)}")
        logger.Print(f"  Change consistency weight: {getattr(cfg.training, 'change_consistency_weight', 0.0)}")
        logger.Print(f"  Window mode: {getattr(cfg.data, 'window_mode', 'random_split')}")
        logger.Print(f"  Kappa: {getattr(cfg.training, 'kappa_start', 50.0)} → {getattr(cfg.training, 'kappa_end', 500.0)}")
        logger.Print("=" * 70)

    # DataLoader
    dataloader = build_dataloader(
        cfg,
        training=True,
        distributed=True,
        world_size=world_size,
        rank=global_rank,
    )

    # Trainer
    trainer = DDPv11Trainer(cfg, local_rank=local_rank)

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
    best_recon = float("inf")
    save_every = args.save_every if args.save_every is not None else getattr(cfg.training, "save_every", 20)

    for epoch in range(start_epoch, total_epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        losses = trainer.train_epoch(epoch, dataloader)

        if global_rank == 0:
            temporal_str = f"temporal={losses['temporal']:.4f}"
            pixel_str = f" pc={losses.get('pixel_change', 0.0):.3f}" if losses.get('pixel_change', 0.0) > 0 else ""
            change_str = f" chg={losses.get('change_consist', 0.0):.3f}" if losses.get('change_consist', 0.0) > 0 else ""
            logger.Print(
                f"Epoch {epoch + 1:03d}/{cfg.training.epochs} | "
                f"total={losses['total']:.4f} recon={losses['recon']:.4f} "
                f"consist={losses['consist']:.4f} cls={losses['cls']:.4f} "
                f"uniform={losses['uniform']:.4f} "
                f"{temporal_str}{pixel_str}{change_str} lr={losses['lr']:.6f}"
            )

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, losses)

        # 基于 reconstruction loss 选 best
        if losses["recon"] < best_recon:
            best_recon = losses["recon"]
            trainer.save_checkpoint(f"best_epoch{epoch + 1}", losses)
            if global_rank == 0:
                logger.Print(f"  [Best] New best recon={best_recon:.4f} at epoch {epoch + 1}")

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if global_rank == 0:
        logger.Print("[train] Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
