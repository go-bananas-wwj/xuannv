#!/usr/bin/env python3
"""Profile a single training step to find bottlenecks."""
import sys, os, time
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch.nn.functional as F

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

from src.config import load_config
from src.data.builder import build_dataloader
from src.models.model import AEFModel
from src.training.trainer import DDPv13Trainer, _build_student_view
from src.training.losses import (
    reconstruction_loss, batch_uniformity_loss_l2, raw_uniformity_loss,
    variance_regularizer, covariance_loss, erank_maximization_loss,
    inter_patch_infonce_loss, decorrelation_loss, classification_loss,
    temporal_contrastive_loss, consistency_loss_spatial,
)

def profile_step():
    cfg = load_config("configs/config_dual_teacher_v1.yaml")
    device = torch.device("npu:0") if HAS_NPU else torch.device("cpu")
    
    dataloader = build_dataloader(cfg, training=True, distributed=False)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    model = AEFModel(cfg).to(device)
    teacher = AEFModel(cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    t = cfg.training
    timings = {}
    
    def tic(label):
        timings[label] = time.perf_counter()
    
    def toc(label):
        dt = time.perf_counter() - timings[label]
        timings[label] = dt
        return dt
    
    # Warmup: 2 steps
    for _ in range(2):
        model.zero_grad()
        out = model(
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
        loss = out.reconstructions.sum() * 0.0
        loss.backward()
        if HAS_NPU:
            torch.npu.synchronize()
    
    # Profile step
    model.train()
    timings = {}
    
    # Teacher forward
    tic("teacher_fwd")
    with torch.no_grad():
        teacher_out = teacher(**{k: batch[k] for k in [
            "source_frames", "source_timestamps_ms", "source_frame_mask",
            "source_input_mask", "source_type_ids", "valid_start_ms", "valid_end_ms",
            "target_relative_time", "target_metadata"
        ] if k in batch})
    if HAS_NPU:
        torch.npu.synchronize()
    teacher_dt = toc("teacher_fwd")
    
    # Student view build
    tic("student_view")
    student_frames, student_frame_mask, student_input_mask, _ = _build_student_view(
        batch["source_frames"], batch["source_timestamps_ms"],
        batch["source_frame_mask"], batch["source_input_mask"],
        drop_rate=0.5, source_drop_rate=0.3, front_drop_prob=0.15, back_drop_prob=0.15,
    )
    if HAS_NPU:
        torch.npu.synchronize()
    view_dt = toc("student_view")
    
    # Student forward
    tic("student_fwd")
    student_out = model(
        source_frames=student_frames, source_timestamps_ms=batch["source_timestamps_ms"],
        source_frame_mask=student_frame_mask, source_input_mask=student_input_mask,
        source_type_ids=batch["source_type_ids"], valid_start_ms=batch["valid_start_ms"],
        valid_end_ms=batch["valid_end_ms"], target_relative_time=batch["target_relative_time"],
        target_metadata=batch["target_metadata"],
        target_loss_type=batch.get("target_loss_type"),
        target_source_idx=batch.get("target_source_idx"),
    )
    if HAS_NPU:
        torch.npu.synchronize()
    student_dt = toc("student_fwd")
    
    # Reconstruction loss
    tic("recon")
    from src.training.loops import compute_recon_loss
    recon = compute_recon_loss(
        student_out.reconstructions, batch["target_images"], batch["target_mask"],
        batch.get("target_loss_type"), cfg.data.num_classes,
    )
    if HAS_NPU:
        torch.npu.synchronize()
    recon_dt = toc("recon")
    
    # Gather pre_norm
    tic("gather")
    pre_norm = student_out.pre_norm_embedding
    gathered_pre = pre_norm  # single card, skip gather
    if HAS_NPU:
        torch.npu.synchronize()
    gather_dt = toc("gather")
    
    # VICReg variance + covariance
    tic("vicreg")
    var = variance_regularizer(gathered_pre.float(), min_std=1.0)
    cov = covariance_loss(gathered_pre.float())
    if HAS_NPU:
        torch.npu.synchronize()
    vicreg_dt = toc("vicreg")
    
    # Uniformity
    tic("uniform")
    l2_uniform = raw_uniformity_loss(gathered_pre.float())
    if HAS_NPU:
        torch.npu.synchronize()
    uniform_dt = toc("uniform")
    
    # erank (SVD)
    tic("erank")
    erank_val = erank_maximization_loss(gathered_pre.float())
    if HAS_NPU:
        torch.npu.synchronize()
    erank_dt = toc("erank")
    
    # Diagnosis SVD
    tic("diag_svd")
    _z = gathered_pre.float()
    _z = _z - _z.mean(dim=0)
    try:
        _svs = torch.linalg.svdvals(_z.T @ _z)
    except Exception:
        pass
    if HAS_NPU:
        torch.npu.synchronize()
    diag_svd_dt = toc("diag_svd")
    
    # AEF distill
    tic("aef_distill")
    aef_spatial_val = torch.tensor(0.0, device=device)
    aef_global_val = torch.tensor(0.0, device=device)
    aef_spatial_emb = batch.get("aef_spatial_emb")
    if aef_spatial_emb is not None:
        aef_spatial_emb = aef_spatial_emb.to(device).float()
        student_64 = student_out.pre_norm_map.float()
        if aef_spatial_emb.shape[2:] != student_64.shape[2:]:
            aef_spatial_emb = F.adaptive_avg_pool2d(aef_spatial_emb, student_64.shape[2:])
        aef_spatial_val = (1.0 - F.cosine_similarity(student_64, aef_spatial_emb, dim=1, eps=1e-6)).mean()
        aef_global_emb = batch.get("aef_global_emb")
        if aef_global_emb is not None:
            aef_global_emb = aef_global_emb.to(device).float()
        else:
            aef_global_emb = aef_spatial_emb.mean(dim=(2, 3))
        student_global = student_64.mean(dim=(2, 3))
        aef_global_val = (1.0 - F.cosine_similarity(student_global, aef_global_emb, dim=1, eps=1e-6)).mean()
    if HAS_NPU:
        torch.npu.synchronize()
    aef_dt = toc("aef_distill")
    
    # OlmoEarth distill
    tic("olmo_distill")
    olmo_spatial_val = torch.tensor(0.0, device=device)
    olmo_global_val = torch.tensor(0.0, device=device)
    teacher_tok = batch.get("teacher_spatial_tokens")
    if teacher_tok is not None and student_out.distill_map is not None:
        teacher_tok = teacher_tok.to(device)
        teacher_raw = teacher_tok.permute(0, 3, 1, 2).float()
        student_map = student_out.distill_map.float()
        if teacher_raw.shape[2:] != student_map.shape[2:]:
            teacher_sp = F.adaptive_avg_pool2d(teacher_raw, student_map.shape[2:])
        else:
            teacher_sp = teacher_raw
        t_cent = teacher_sp - teacher_sp.mean(dim=(2, 3), keepdim=True)
        s_cent = student_map - student_map.mean(dim=(2, 3), keepdim=True)
        olmo_spatial_val = (1.0 - F.cosine_similarity(s_cent, t_cent, dim=1, eps=1e-6)).mean()
        teacher_global_ref = batch.get("teacher_global_emb")
        if teacher_global_ref is not None:
            teacher_global_ref = teacher_global_ref.to(device).float()
        else:
            teacher_global_ref = teacher_raw.mean(dim=(2, 3))
        olmo_global_val = (1.0 - F.cosine_similarity(student_out.distill_global.float(), teacher_global_ref, dim=1, eps=1e-6)).mean()
    if HAS_NPU:
        torch.npu.synchronize()
    olmo_dt = toc("olmo_distill")
    
    # Classification
    tic("cls")
    cls = classification_loss(student_out.logits.float(), batch["label"])
    if HAS_NPU:
        torch.npu.synchronize()
    cls_dt = toc("cls")
    
    # Total + backward
    tic("backward")
    total = recon + 0.03 * cls + 0.2 * var + 0.05 * cov + 1.0 * l2_uniform + 0.3 * erank_val + 0.75 * aef_spatial_val + 0.3 * aef_global_val + 0.3 * olmo_spatial_val + 0.15 * olmo_global_val
    total.backward()
    if HAS_NPU:
        torch.npu.synchronize()
    backward_dt = toc("backward")
    
    # Optimizer step
    tic("optim")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if HAS_NPU:
        torch.npu.synchronize()
    optim_dt = toc("optim")
    
    print("\n" + "="*60)
    print("  STEP PROFILING RESULTS")
    print("="*60)
    total_step = teacher_dt + view_dt + student_dt + recon_dt + gather_dt + vicreg_dt + uniform_dt + erank_dt + diag_svd_dt + aef_dt + olmo_dt + cls_dt + backward_dt + optim_dt
    for name, dt in [
        ("teacher_fwd", teacher_dt), ("student_view", view_dt), ("student_fwd", student_dt),
        ("recon_loss", recon_dt), ("gather", gather_dt), ("vicreg", vicreg_dt),
        ("uniformity", uniform_dt), ("erank_loss", erank_dt), ("diag_svd", diag_svd_dt),
        ("aef_distill", aef_dt), ("olmo_distill", olmo_dt), ("cls_loss", cls_dt),
        ("backward", backward_dt), ("optimizer", optim_dt),
    ]:
        pct = dt / total_step * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:20s} {dt:7.3f}s ({pct:5.1f}%) {bar}")
    print(f"  {'TOTAL':20s} {total_step:7.3f}s")
    print("="*60)

if __name__ == "__main__":
    profile_step()
