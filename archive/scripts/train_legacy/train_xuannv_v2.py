#!/usr/bin/env python3
"""玄女V2 训练入口 — 严格对齐AEF论文 STP Block 架构.

用法:
    cd /workspace/xuannv
    torchrun --nproc_per_node=8 \
        scripts/train/train_xuannv_v2.py --config configs/xuannv_v2_baseline.yaml \
        --save-every 5

玄女V2 核心:
  - STP Block 三条独立路径(1024/512/128) + 跨尺度信息传递
  - 15 blocks, 严格对齐论文
  - 重建 S2/S1/Landsat/DEM
  - 64维 Embedding, VMF κ=8000
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
from src.training.ddp_xuannv_v2_trainer import XuannvV2Trainer


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
        logger.Print(f"  玄女V2 训练  [8×NPU]  —  {cfg.experiment.name}")
        logger.Print(f"  World size: {world_size}  |  Rank: {global_rank}")
        logger.Print("=" * 70)
        logger.Print(f"  Config: {args.config}")
        logger.Print(f"  Epochs: {cfg.training.epochs}")
        logger.Print(f"  Batch per GPU: {cfg.data.batch_size}")
        logger.Print(f"  Grad accum: {getattr(cfg.training, 'gradient_accumulation_steps', 2)}")
        logger.Print(f"  Effective batch: {cfg.data.batch_size * world_size * getattr(cfg.training, 'gradient_accumulation_steps', 2)}")
        logger.Print(f"  Recon weight: {getattr(cfg.training, 'reconstruction_weight', 1.0)}")
        logger.Print(f"  Consistency weight: {getattr(cfg.training, 'consistency_weight', 0.0)}")
        logger.Print(f"  BatchUniformity weight: {getattr(cfg.training, 'batch_uniformity_weight', 0.0)}")
        src_weights = getattr(cfg.training, 'source_recon_weights', [1.0]*3)
        logger.Print(f"  Source recon weights: {src_weights}")
        logger.Print(f"  Embedding dim: {getattr(cfg.model, 'embedding_dim', 128)}")
        logger.Print(f"  VMF kappa: {getattr(cfg.model, 'vmf_kappa', 2000.0)}")
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
    trainer = XuannvV2Trainer(cfg, local_rank=local_rank)

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
    best_score = float("-inf")
    save_best_balanced = getattr(cfg.training, "save_best_balanced", False)
    save_every = args.save_every if args.save_every is not None else getattr(cfg.training, "save_every", 20)
    epoch_start_time = time.time()
    bank_size = 0

    for epoch in range(start_epoch, total_epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        epoch_t0 = time.time()
        losses = trainer.train_epoch(epoch, dataloader)
        epoch_dt = time.time() - epoch_t0
        elapsed = time.time() - epoch_start_time
        remain_epochs = total_epochs - (epoch + 1)
        eta_sec = epoch_dt * remain_epochs if remain_epochs > 0 else 0
        eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"

        # 获取 Memory Bank 大小
        if hasattr(trainer, 'memory_bank'):
            bank_size = trainer.memory_bank.size

        if global_rank == 0:
            bank_size = int(losses.get('bank', 0))
            active_dims = int(losses.get('active_dims', 0))
            std_mean = losses.get('std_mean', 0.0)
            inter_var = losses.get('inter_var', 0.0)
            logger.Print(
                f"[{time.strftime('%H:%M:%S')}] Epoch {epoch + 1:03d}/{cfg.training.epochs} | "
                f"total={losses['total']:.4f} recon={losses['recon']:.4f} "
                f"consist={losses['consist']:.4f} var={losses['var']:.4f} "
                f"cov={losses['cov']:.4f} l2unif={losses['l2unif']:.4f} "
                f"inter_var={inter_var:.4f} "
                f"bank={bank_size}/512 active={active_dims}/128 std_mean={std_mean:.4f} "
                f"lr={losses['lr']:.6f} | "
                f"time={epoch_dt:.1f}s elapsed={int(elapsed//60)}m ETA={eta_str}"
            )

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, losses)

        # 选 best checkpoint
        if save_best_balanced:
            # 综合评分: std_mean 越高越好, recon 越低越好
            # recon 典型范围 0.08-0.30, std_mean 典型范围 0.40-1.00
            # 加权平衡两者贡献
            score = losses.get("std_mean", 0.0) - losses["recon"] * 0.5
            if score > best_score:
                best_score = score
                trainer.save_checkpoint(f"best_epoch{epoch + 1}", losses)
                if global_rank == 0:
                    logger.Print(f"  [Best] New best score={best_score:.4f} (std_mean={losses.get('std_mean', 0):.4f}, recon={losses['recon']:.4f}) at epoch {epoch + 1}")
        else:
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
