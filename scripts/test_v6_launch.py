#!/usr/bin/env python3
"""V6 训练快速验证 — 仅跑 5 个 step 确认无 crash."""
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
    # 单卡模拟 DDP
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29506")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    
    torch.cuda.set_device(0)
    
    cfg = load_config("configs/qwen_v6_enhanced_temporal.yaml")
    cfg.training.epochs = 1
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    
    dataloader = build_dataloader(cfg, training=True, distributed=False)
    trainer = DDPv6EnhancedTemporalTrainer(cfg, local_rank=0)
    
    print("=" * 60)
    print("V6 Quick Test — 5 steps")
    print("=" * 60)
    
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        
        batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # 只测试 forward + loss computation
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            student_out = trainer.model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
                target_loss_type=batch.get("target_loss_type"),
                target_source_idx=batch.get("target_source_idx"),
            )
            
            emb_w1, emb_w2, pre_w1, pre_w2 = trainer.model.module.encode_dual_window(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_w1=batch["valid_start_w1"],
                valid_end_w1=batch["valid_end_w1"],
                valid_start_w2=batch["valid_start_w2"],
                valid_end_w2=batch["valid_end_w2"],
            )
        
        print(f"Step {step}: student_out shapes OK")
        print(f"  pre_norm_embedding: {student_out.pre_norm_embedding.shape}")
        print(f"  pre_norm_map: {student_out.pre_norm_map.shape if student_out.pre_norm_map is not None else None}")
        print(f"  pre_w1: {pre_w1.shape}, pre_w2: {pre_w2.shape}")
        
        # Test spatial uniformity
        from src.training.ddp_v6_enhanced_temporal_trainer import _gather_spatial_embeddings
        spatial_emb = _gather_spatial_embeddings(student_out.pre_norm_map.float(), 256)
        print(f"  spatial_emb: {spatial_emb.shape}")
        
        # Test new losses
        from src.training.losses import (
            temporal_cosine_pixel_loss,
            pixel_temporal_info_nce_loss,
        )
        tc = temporal_cosine_pixel_loss(pre_w1.float(), pre_w2.float(), temperature=0.05)
        ptnce = pixel_temporal_info_nce_loss(pre_w1.float(), pre_w2.float(), temperature=0.1, num_samples=16)
        print(f"  tc_pixel_loss: {tc.item():.4f}")
        print(f"  pixel_infoNCE: {ptnce.item():.4f}")
        
    print("\n✅ V6 quick test passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
