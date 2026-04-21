#!/usr/bin/env python3
"""V6 训练器完整验证 — 跑 2 个 step 的 train_epoch."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch.distributed as dist

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.ddp_v6_enhanced_temporal_trainer import DDPv6EnhancedTemporalTrainer


def main():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29507")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    
    torch.cuda.set_device(0)
    
    cfg = load_config("configs/qwen_v6_enhanced_temporal.yaml")
    cfg.training.epochs = 1
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.training.gradient_accumulation_steps = 1  # 简化测试
    
    dataloader = build_dataloader(cfg, training=True, distributed=False)
    trainer = DDPv6EnhancedTemporalTrainer(cfg, local_rank=0)
    
    print("=" * 60)
    print("V6 Trainer Full Test — 1 epoch, 2 steps")
    print("=" * 60)
    
    losses = trainer.train_epoch(0, dataloader)
    
    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✅ V6 trainer test passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
