#!/usr/bin/env python3
"""玄女V2 快速验证 — 仅跑 3 个 step 确认无 crash, shape 正确."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch.distributed as dist

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.ddp_xuannv_v2_trainer import XuannvV2Trainer


def main():
    # 单卡模拟 DDP
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29507")
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
    
    torch.npu.set_device(0)
    
    cfg = load_config("configs/xuannv_v2_baseline.yaml")
    cfg.training.epochs = 1
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    
    print("=" * 60)
    print("玄女V2 Quick Test — 3 steps")
    print("=" * 60)
    print(f"  Config: xuannv_v2_baseline.yaml")
    print(f"  precision_dim: {cfg.model.precision_dim}")
    print(f"  time_dim: {cfg.model.time_dim}")
    print(f"  space_dim: {cfg.model.space_dim}")
    print(f"  num_blocks: {cfg.model.num_blocks}")
    print(f"  embedding_dim: {cfg.model.embedding_dim}")
    print("=" * 60)
    
    dataloader = build_dataloader(cfg, training=True, distributed=False)
    trainer = XuannvV2Trainer(cfg, local_rank=0)
    
    for step, batch in enumerate(dataloader):
        if step >= 3:
            break
        
        batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # 测试完整训练 step (forward + loss + backward)
        losses = trainer.train_epoch(0, [batch])
        
        print(f"\nStep {step}: OK")
        print(f"  total_loss: {losses.get('total', 0):.4f}")
        print(f"  recon: {losses.get('recon', 0):.4f}")
        print(f"  consist: {losses.get('consist', 0):.4f}")
        print(f"  var: {losses.get('var', 0):.4f}")
        print(f"  l2unif: {losses.get('l2unif', 0):.4f}")
        print(f"  active_dims: {losses.get('active_dims', 0)}")
        
    print("\n✅ 玄女V2 quick test passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
