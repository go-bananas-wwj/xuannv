#!/usr/bin/env python3
"""玄女V2 backward 测试 — 验证模型能正确计算梯度."""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
from src.config import load_config
from src.models.model import AEFModel


def test_backward():
    cfg = load_config("configs/xuannv_v2_baseline.yaml")
    device = torch.device("npu:0")
    
    print("=" * 60)
    print("玄女V2 Backward Test")
    print("=" * 60)
    
    model = AEFModel(cfg).to(device)
    model.train()
    
    B, S, T, C, H, W = 1, 3, 4, 6, 128, 128
    num_tgt = cfg.data.num_target_sources
    meta_dim = cfg.data.metadata_dim
    
    source_frames = torch.randn(B, S, T, C, H, W, device=device)
    source_timestamps_ms = torch.randint(0, 1000000000, (B, S, T), device=device).float()
    source_frame_mask = torch.ones(B, S, T, dtype=torch.bool, device=device)
    source_input_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    source_type_ids = torch.zeros(B, S, dtype=torch.long, device=device)
    valid_start_ms = torch.tensor([0], device=device)
    valid_end_ms = torch.tensor([500000000], device=device)
    target_relative_time = torch.zeros(B, num_tgt, device=device)
    target_metadata = torch.zeros(B, num_tgt, meta_dim, device=device)
    
    print("Forward...")
    out = model(
        source_frames=source_frames,
        source_timestamps_ms=source_timestamps_ms,
        source_frame_mask=source_frame_mask,
        source_input_mask=source_input_mask,
        source_type_ids=source_type_ids,
        valid_start_ms=valid_start_ms,
        valid_end_ms=valid_end_ms,
        target_relative_time=target_relative_time,
        target_metadata=target_metadata,
    )
    
    print(f"  embedding_map: {out.embedding_map.shape}")
    print(f"  reconstructions: {out.reconstructions.shape}")
    
    # 构造一个简单 loss
    loss = out.embedding_map.mean() + out.reconstructions.mean()
    
    print("Backward...")
    loss.backward()
    
    # 检查梯度
    has_grad = False
    max_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            has_grad = True
            max_grad = max(max_grad, p.grad.abs().max().item())
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"  ❌ NaN/Inf grad in {name}")
                return False
    
    if not has_grad:
        print("  ❌ No gradients found!")
        return False
    
    print(f"  Max grad: {max_grad:.6f}")
    print("  ✅ Backward OK")
    
    # Test optimizer step
    print("Optimizer step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    print("  ✅ Optimizer step OK")
    
    print("\n" + "=" * 60)
    print("✅ 玄女V2 backward test passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_backward()
    sys.exit(0 if success else 1)
