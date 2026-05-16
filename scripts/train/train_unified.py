#!/usr/bin/env python3
"""DDP Unified 训练入口 — AEF 对齐版本.

用法:
    cd /workspace/xuannv
    torchrun --nproc_per_node=1 \
        scripts/train/train_unified.py --config configs/aef_baseline.yaml

8 卡并行 (每卡一个独立实验):
    bash scripts/train/launch_aef_8gpu.sh
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
from src.training.ddp_unified_trainer import DDPUnifiedTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    # --save-every 已移除：现在只自动保留 recon 最低的 best checkpoint
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--wandb-group", type=str, default="aef_v1", help="Wandb group name")
    parser.add_argument("--wandb-off", action="store_true", help="禁用 wandb")
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
    # save_every 不再使用，只保留 best checkpoint

    # 固定随机种子
    seed = getattr(cfg.experiment, "seed", 42) + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

    if global_rank == 0:
        logger.Print("=" * 70)
        logger.Print(f"  DDP Unified 训练  [AEF 对齐]  —  {cfg.experiment.name}")
        logger.Print(f"  World size: {world_size}  |  Rank: {global_rank}")
        logger.Print("=" * 70)
        logger.Print(f"  Config: {args.config}")
        logger.Print(f"  Epochs: {cfg.training.epochs}")
        logger.Print(f"  Batch per GPU: {cfg.data.batch_size}")
        logger.Print(f"  Effective batch: {cfg.data.batch_size * world_size}")
        logger.Print(f"  Recon weight: {getattr(cfg.training, 'reconstruction_weight', 1.0)}")
        logger.Print(f"  Consistency weight: {getattr(cfg.training, 'consistency_weight', 0.02)}")
        logger.Print(f"  BatchUniformity weight: {getattr(cfg.training, 'batch_uniformity_weight', 0.05)}")
        logger.Print(f"  Source recon weights: {getattr(cfg.training, 'source_recon_weights', [1.0]*3)}")
        logger.Print(f"  Embedding dim: {getattr(cfg.model, 'embedding_dim', 64)}")
        logger.Print(f"  Skip L2 training: {getattr(cfg.model, 'skip_l2_norm_training', False)}")
        logger.Print(f"  VMF kappa: {getattr(cfg.model, 'vmf_kappa', 50.0)}")
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
    wandb_group = args.wandb_group if not args.wandb_off else None
    trainer = DDPUnifiedTrainer(cfg, local_rank=local_rank, wandb_group=wandb_group or "aef_v1")

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if global_rank == 0:
            logger.Print(f"[train] Resumed from {args.resume}, starting at epoch {start_epoch + 1}")
        dist.barrier()

    total_epochs = cfg.training.epochs
    epoch_start_time = time.time()
    
    # 只保留1个 best checkpoint（基于 recon loss）
    best_checkpoints: list[tuple[float, Path]] = []  # [(recon, path), ...]

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

        if global_rank == 0:
            logger.Print(
                f"[{time.strftime('%H:%M:%S')}] Epoch {epoch + 1:03d}/{cfg.training.epochs} | "
                f"total={losses['total']:.4f} recon={losses['recon']:.4f} "
                f"consist={losses['consist']:.4f} l2unif={losses['l2unif']:.4f} "
                f"lr={losses['lr']:.6f} | "
                f"time={epoch_dt:.1f}s elapsed={int(elapsed//60)}m ETA={eta_str}"
            )

        # 只保留最好的 best checkpoint（基于 recon loss）
        if global_rank == 0:
            recon_val = losses["recon"]
            # 当前 recon 比之前最好的还低才保存
            is_best = len(best_checkpoints) == 0 or recon_val < best_checkpoints[0][0]
            if is_best:
                # ★ 修复：ckpt_path 必须与 save_checkpoint 生成的文件名一致
                ckpt_path = trainer.output_dir / f"epoch_best_epoch{epoch + 1}.pt"
                trainer.save_checkpoint(f"best_epoch{epoch + 1}", losses)
                best_checkpoints.append((recon_val, ckpt_path))
                best_checkpoints.sort(key=lambda x: x[0])
                logger.Print(f"  [Best] New best recon={recon_val:.4f} at epoch {epoch + 1}")
                
                # 只保留1个最好的，删除旧的
                while len(best_checkpoints) > 1:
                    old_recon, old_path = best_checkpoints.pop(-1)
                    if old_path.exists():
                        old_path.unlink()
                        logger.Print(f"  [Best] Removed old best: recon={old_recon:.4f}")

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if global_rank == 0:
        logger.Print("[train] Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
