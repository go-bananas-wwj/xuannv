#!/usr/bin/env python3
"""单 GPU 训练脚本 — GPU6 专用.

支持:
  - EMA Teacher + DINO/iBOT 自蒸馏
  - 跨时相掩码重建
  - VICReg + KoLeo 反坍缩
  - 相邻月份双窗口采样

用法:
    cd /workspace/xuannv
    CUDA_VISIBLE_DEVICES=6 python scripts/train/train_single_gpu.py \
        --config configs/qwen_v2_temporal.yaml \
        --resume /workspace/outputs/aef_qwen_v2/epoch_499.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# 强制无缓冲输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace')

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.single_gpu_trainer import SingleGPUTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--device", type=str, default="cuda:6", help="训练设备")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--save-every", type=int, default=50, help="每隔多少 epoch 保存检查点")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 覆盖参数
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    # 固定随机种子
    seed = getattr(cfg.experiment, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print(f"  单 GPU 训练  [{args.device}]  —  {cfg.experiment.name}")
    print("=" * 70)
    print(f"  Config: {args.config}")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch size: {cfg.data.batch_size} (effective via grad accum)")
    print(f"  Window mode: {getattr(cfg.data, 'window_mode', 'random_split')}")
    print("=" * 70)

    # DataLoader
    dataloader = build_dataloader(
        cfg,
        training=True,
        distributed=False,
        world_size=1,
        rank=0,
    )

    # Trainer
    trainer = SingleGPUTrainer(cfg, device_str=args.device)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"[train] Resumed from {args.resume}, starting at epoch {start_epoch + 1}")

    total_epochs = cfg.training.epochs
    if args.resume and start_epoch > 0:
        total_epochs = start_epoch + cfg.training.epochs
        print(f"[train] Will train from epoch {start_epoch + 1} to {total_epochs}")

    best_loss = float("inf")
    for epoch in range(start_epoch, total_epochs):
        losses = trainer.train_epoch(epoch, dataloader)
        print(
            f"Epoch {epoch + 1:03d}/{total_epochs} | "
            f"total={losses['total']:.4f} recon={losses['recon']:.4f} "
            f"ct_recon={losses['ct_recon']:.4f} dino={losses['dino']:.4f} "
            f"vicreg={losses['vicreg']:.4f} koleo={losses['koleo']:.4f} "
            f"temporal={losses['temporal']:.4f} lr={losses['lr']:.6f}"
        )

        if (epoch + 1) % args.save_every == 0:
            trainer.save_checkpoint(epoch + 1, losses)

        # 保存最佳模型 (基于 total loss)
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            trainer.save_checkpoint("best", losses)

    print("[train] Training complete.")


if __name__ == "__main__":
    main()
