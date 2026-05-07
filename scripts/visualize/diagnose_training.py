#!/usr/bin/env python3
"""快速诊断 — 检查一个 batch 的前向传播和损失计算."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
from src.config import load_config
from src.data.builder import build_dataloader
from src.training.single_gpu_trainer import SingleGPUTrainer

cfg = load_config("configs/qwen_v4_cd_upgrade.yaml")
cfg.training.epochs = 1  # 不实际训练

dataloader = build_dataloader(cfg, training=True, distributed=False, world_size=1, rank=0)
trainer = SingleGPUTrainer(cfg, device_str="npu:0")

batch = next(iter(dataloader))
print(f"Batch loaded: patch={batch['patch_id']}")
print(f"  source_frames: {batch['source_frames'].shape}")
print(f"  target_images: {batch['target_images'].shape}")
print(f"  spatial_mask: {batch.get('spatial_mask')}")

# 前向传播
print("\n--- Forward pass ---")
student_out = trainer.model(
    batch["source_frames"].npu(0),
    batch["source_timestamps_ms"].npu(0),
    batch["source_frame_mask"].npu(0),
    batch["source_input_mask"].npu(0),
    batch["source_type_ids"].npu(0),
    batch["valid_start_ms"].npu(0),
    batch["valid_end_ms"].npu(0),
    batch["target_relative_time"].npu(0),
    batch["target_metadata"].npu(0),
    batch["target_loss_type"].npu(0),
    batch["target_source_idx"].npu(0),
)
print(f"  embedding: {student_out.embedding.shape}")
print(f"  embedding min/max: {student_out.embedding.min().item():.3f} / {student_out.embedding.max().item():.3f}")
print(f"  embedding mean/std: {student_out.embedding.mean().item():.3f} / {student_out.embedding.std().item():.3f}")
print(f"  embedding has NaN: {torch.isnan(student_out.embedding).any().item()}")
print(f"  pre_norm_embedding: {student_out.pre_norm_embedding.shape}")
print(f"  pre_norm_embedding has NaN: {torch.isnan(student_out.pre_norm_embedding).any().item()}")

# 重建
print("\n--- Reconstruction ---")
for t_idx, recon in enumerate(student_out.reconstructions):
    if recon is not None:
        print(f"  target {t_idx}: {recon.shape} min={recon.min().item():.3f} max={recon.max().item():.3f} NaN={torch.isnan(recon).any().item()}")
    else:
        print(f"  target {t_idx}: None")

# Teacher
print("\n--- Teacher forward ---")
with torch.no_grad():
    teacher_out = trainer.teacher(
        batch["source_frames"].npu(0),
        batch["source_timestamps_ms"].npu(0),
        batch["source_frame_mask"].npu(0),
        batch["source_input_mask"].npu(0),
        batch["source_type_ids"].npu(0),
        batch["valid_start_ms"].npu(0),
        batch["valid_end_ms"].npu(0),
        batch["target_relative_time"].npu(0),
        batch["target_metadata"].npu(0),
        batch["target_loss_type"].npu(0),
        batch["target_source_idx"].npu(0),
    )
print(f"  teacher embedding NaN: {torch.isnan(teacher_out.embedding).any().item()}")

# DINO
print("\n--- DINO ---")
student_dino = trainer.dino_head(student_out.embedding)
teacher_dino = trainer.dino_head(teacher_out.embedding)
print(f"  student_dino: {student_dino.shape} min={student_dino.min().item():.3f} max={student_dino.max().item():.3f}")
print(f"  teacher_dino: {teacher_dino.shape} min={teacher_dino.min().item():.3f} max={teacher_dino.max().item():.3f}")

# VICReg
print("\n--- VICReg ---")
student_pre = student_out.pre_norm_embedding
teacher_pre = teacher_out.pre_norm_embedding
print(f"  student_pre: {student_pre.shape} min={student_pre.min().item():.3f} max={student_pre.max().item():.3f}")
print(f"  teacher_pre: {teacher_pre.shape} min={teacher_pre.min().item():.3f} max={teacher_pre.max().item():.3f}")

# 检查 batch 中是否有 NaN
print("\n--- Batch data NaN check ---")
print(f"  source_frames NaN: {torch.isnan(batch['source_frames']).any().item()}")
print(f"  target_images NaN: {torch.isnan(batch['target_images']).any().item()}")
print(f"  spatial_mask NaN: {torch.isnan(batch.get('spatial_mask', torch.tensor(0.0))).any().item()}")

print("\n--- DONE ---")
