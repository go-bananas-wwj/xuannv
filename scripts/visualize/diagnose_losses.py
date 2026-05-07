#!/usr/bin/env python3
"""诊断哪个损失导致 NaN."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.data.builder import build_dataloader
from src.training.single_gpu_trainer import SingleGPUTrainer, vicreg_loss, koleo_loss, DINOLoss

cfg = load_config("configs/qwen_v4_cd_upgrade.yaml")
dataloader = build_dataloader(cfg, training=True, distributed=False, world_size=1, rank=0)
trainer = SingleGPUTrainer(cfg, device_str="npu:0")

batch = next(iter(dataloader))
batch = {k: v.npu(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print("=== Step 1: Forward ===")
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
print(f"  embedding NaN: {torch.isnan(student_out.embedding).any().item()}")
print(f"  reconstructions NaN: {torch.isnan(student_out.reconstructions).any().item()}")

print("\n=== Step 2: Recon Loss ===")
from src.training.loops import compute_recon_loss
recon = compute_recon_loss(
    student_out.reconstructions, batch["target_images"], batch["target_mask"],
    batch.get("target_loss_type"), cfg.data.num_classes,
)
print(f"  recon: {recon.item():.4f} NaN: {torch.isnan(recon).item()}")

print("\n=== Step 3: Teacher Forward ===")
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

print("\n=== Step 4: DINO ===")
s_logits = trainer.dino_head(student_out.embedding)
t_logits = trainer.dino_head(teacher_out.embedding)
dino = trainer.dino_loss(s_logits, t_logits)
print(f"  dino: {dino.item():.4f} NaN: {torch.isnan(dino).item()}")

print("\n=== Step 5: VICReg ===")
z_s = student_out.pre_norm_embedding
z_t = teacher_out.pre_norm_embedding
if trainer.expander is not None:
    z_s = trainer.expander(z_s)
    z_t = trainer.expander(z_t)
vicreg = vicreg_loss(z_s, z_t)
print(f"  vicreg: {vicreg.item():.4f} NaN: {torch.isnan(vicreg).item()}")

print("\n=== Step 6: KoLeo ===")
koleo = koleo_loss(torch.cat([z_s, z_t], dim=0))
print(f"  koleo: {koleo.item():.4f} NaN: {torch.isnan(koleo).item()}")

print("\n=== Step 7: CT Recon ===")
ct_recon = torch.tensor(0.0, device="npu:0")
if getattr(cfg.training, "ct_reconstruction_weight", 0.0) > 0 and "spatial_mask" in batch:
    spatial_mask = batch["spatial_mask"]
    if spatial_mask.dim() == 2:
        spatial_mask = spatial_mask.unsqueeze(0)
    pred = student_out.reconstructions
    tgt = batch["target_images"]
    mask = batch["target_mask"]
    B, T, C, H, W = pred.shape
    sm = F.interpolate(spatial_mask.unsqueeze(1), size=(H, W), mode="nearest")
    sm = sm.unsqueeze(1)
    pixel_valid = (~torch.isnan(tgt)).float()
    tgt_mask = mask[:, :, None, None, None].float() * pixel_valid
    diff = torch.abs(pred - tgt) * tgt_mask * sm
    denom = torch.clamp((tgt_mask * sm).sum(), min=1.0)
    ct_recon = diff.sum() / denom
    print(f"  sm sum: {sm.sum().item():.1f}")
    print(f"  denom: {denom.item():.4f}")
print(f"  ct_recon: {ct_recon.item():.4f} NaN: {torch.isnan(ct_recon).item()}")

print("\n=== Step 8: Total ===")
total = (
    cfg.training.reconstruction_weight * recon
    + cfg.training.ct_reconstruction_weight * ct_recon
    + cfg.training.dino_weight * dino
    + cfg.training.vicreg_weight * vicreg
    + cfg.training.koleo_weight * koleo
)
print(f"  total: {total.item():.4f} NaN: {torch.isnan(total).item()}")

print("\n=== Step 9: Backward ===")
total.backward()
print("  backward OK")
for name, p in trainer.model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"  NaN grad in: {name}")
        break
else:
    print("  No NaN grad found")

print("\n=== DONE ===")
