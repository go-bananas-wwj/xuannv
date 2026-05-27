#!/usr/bin/env python3
"""调试 V13 NaN/Inf 问题."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.training.losses import (
    reconstruction_loss, consistency_loss_spatial,
    variance_regularizer, covariance_loss, batch_uniformity_loss_l2
)

print("="*60)
print("V13 NaN Debug")
print("="*60)

cfg = load_config("configs/xuannv_v12_clean.yaml")
cfg.data.preload = False

device = "npu:0"
model = AEFModel(cfg).to(device)
model.train()

ds = HarbinPatchDataset(cfg)
print(f"Dataset: {len(ds)} samples")

# 取几个样本
for i in range(3):
    item = ds[i]
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in item.items()}
    
    print(f"\n--- Sample {i} ---")
    print(f"Patch: {item['patch_id']}, YearMonth: {item['year_month']}")
    
    try:
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
        
        # 检查各张量是否有 NaN/Inf
        def check_nan(name, tensor):
            if tensor is None or not isinstance(tensor, torch.Tensor):
                return
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan or has_inf:
                print(f"  {name}: NaN={has_nan}, Inf={has_inf}, shape={tensor.shape}")
        
        check_nan("embedding_map", out.embedding_map)
        check_nan("embedding", out.embedding)
        check_nan("pre_norm_embedding", out.pre_norm_embedding)
        check_nan("reconstructions", out.reconstructions)
        
        # 计算各个损失
        recon = reconstruction_loss(out.reconstructions, batch["target_images"], batch["target_mask"])
        print(f"  recon: {recon.item():.4f}")
        
        consist = consistency_loss_spatial(out.embedding_map.detach(), out.embedding_map)
        print(f"  consist: {consist.item():.4f}")
        
        pre_norm = out.pre_norm_embedding
        var = variance_regularizer(pre_norm.float(), min_std=1.0)
        print(f"  var: {var.item():.4f}")
        
        cov = covariance_loss(pre_norm.float())
        print(f"  cov: {cov.item():.4f}")
        
        l2_unif = batch_uniformity_loss_l2(out.embedding.float())
        print(f"  l2_unif: {l2_unif.item():.4f}")
        
        total = 0.5 * recon + 0.1 * consist + 0.3 * var + 0.1 * cov + 0.01 * l2_unif
        print(f"  total: {total.item():.4f}")
        
        # 反向传播
        total.backward()
        
        # 检查梯度
        has_nan_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print(f"  NaN/Inf grad in {name}")
                has_nan_grad = True
                break
        if not has_nan_grad:
            print(f"  Gradients OK")
        
        model.zero_grad()
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("  Debug complete")
print("="*60)
