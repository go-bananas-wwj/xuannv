#!/usr/bin/env python3
"""Unified Trainer 冒烟测试 — 只跑 1 个 step 确认无 crash."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.ddp_unified_trainer import DDPUnifiedTrainer


def main():
    cfg = load_config("configs/aef_baseline.yaml")
    cfg.training.epochs = 1
    cfg.training.max_steps_per_epoch = 1
    
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    
    print("Building dataloader...")
    dataloader = build_dataloader(cfg, training=True, distributed=False, world_size=1, rank=0)
    
    print("Building trainer...")
    trainer = DDPUnifiedTrainer(cfg, local_rank=0)
    
    print("Running 1 step...")
    losses = trainer.train_epoch(0, dataloader)
    print(f"SUCCESS! Losses: {losses}")


if __name__ == "__main__":
    main()
