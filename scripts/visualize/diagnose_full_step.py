#!/usr/bin/env python3
"""诊断完整训练 step (含 backward + step)."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
from src.config import load_config
from src.data.builder import build_dataloader
from src.training.single_gpu_trainer import SingleGPUTrainer

cfg = load_config("configs/qwen_v4_cd_upgrade.yaml")
dataloader = build_dataloader(cfg, training=True, distributed=False, world_size=1, rank=0)
trainer = SingleGPUTrainer(cfg, device_str="cuda:0")

accum_steps = cfg.training.gradient_accumulation_steps

for i, batch in enumerate(dataloader):
    batch = {k: v.cuda(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
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
        
        ct_recon = torch.tensor(0.0, device="cuda:0")
        if getattr(cfg.training, "ct_reconstruction_weight", 0.0) > 0 and "spatial_mask" in batch:
            ct_recon = trainer._cross_temporal_masked_recon(batch, student_out)
        
        total = (
            cfg.training.reconstruction_weight * recon
            + cfg.training.ct_reconstruction_weight * ct_recon
            + cfg.training.dino_weight * dino
            + cfg.training.vicreg_weight * vicreg
            + cfg.training.koleo_weight * koleo
        ) / accum_steps
    
    total.backward()
    
    if (i + 1) % accum_steps == 0:
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.update_teacher()
    
    # 检查参数是否被 NaN 污染
    model_has_nan = any(torch.isnan(p).any().item() for p in trainer.model.parameters())
    teacher_has_nan = any(torch.isnan(p).any().item() for p in trainer.teacher.parameters())
    
    print(f"Step {i}: total={total.item()*accum_steps:.4f} NaN_params={model_has_nan} NaN_teacher={teacher_has_nan}")
    
    if model_has_nan:
        print("  MODEL PARAMS NaN! Breaking.")
        break
    
    if i >= 23:
        print("  No NaN in first 24 steps.")
        break

print("\nDONE")
