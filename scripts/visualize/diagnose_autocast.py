#!/usr/bin/env python3
"""诊断混合精度 (float16) 是否导致 NaN."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
from src.config import load_config
from src.data.builder import build_dataloader
from src.training.single_gpu_trainer import SingleGPUTrainer

cfg = load_config("configs/qwen_v4_cd_upgrade.yaml")
dataloader = build_dataloader(cfg, training=True, distributed=False, world_size=1, rank=0)
trainer = SingleGPUTrainer(cfg, device_str="npu:0")

for i, batch in enumerate(dataloader):
    batch = {k: v.npu(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
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
        
        recon = trainer._compute_recon_loss(student_out.reconstructions, batch)
        
        with torch.no_grad():
            teacher_out = trainer.teacher(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
            )
        
        s_logits = trainer.dino_head(student_out.embedding)
        t_logits = trainer.dino_head(teacher_out.embedding)
        dino = trainer.dino_loss(s_logits, t_logits)
        
        z_s = student_out.pre_norm_embedding
        z_t = teacher_out.pre_norm_embedding
        if trainer.expander is not None:
            z_s = trainer.expander(z_s)
            z_t = trainer.expander(z_t)
        
        from src.training.single_gpu_trainer import vicreg_loss, koleo_loss
        vicreg = vicreg_loss(z_s, z_t)
        koleo = koleo_loss(torch.cat([z_s, z_t], dim=0))
        
        ct_recon = torch.tensor(0.0, device="npu:0")
        if getattr(cfg.training, "ct_reconstruction_weight", 0.0) > 0 and "spatial_mask" in batch:
            ct_recon = trainer._cross_temporal_masked_recon(batch, student_out)
        
        total = (
            cfg.training.reconstruction_weight * recon
            + cfg.training.ct_reconstruction_weight * ct_recon
            + cfg.training.dino_weight * dino
            + cfg.training.vicreg_weight * vicreg
            + cfg.training.koleo_weight * koleo
        )
    
    has_nan = torch.isnan(total).item()
    print(f"Step {i}: total={total.item():.4f} recon={recon.item():.4f} dino={dino.item():.4f} vicreg={vicreg.item():.4f} koleo={koleo.item():.4f} ct_recon={ct_recon.item():.4f} NaN={has_nan}")
    
    if has_nan:
        print("  FOUND NaN! Breaking.")
        break
    
    if i >= 20:
        print("  No NaN in first 21 steps.")
        break

print("\nDONE")
