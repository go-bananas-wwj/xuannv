#!/usr/bin/env python3
"""玄女V2 模型架构快速验证 — 用 dummy 数据测试 forward + shape, 不加载真实数据集."""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
from src.config import load_config
from src.models.model import AEFModel


def test_model_shapes():
    """测试玄女V2模型各层输出shape是否正确."""
    
    cfg = load_config("configs/xuannv_v2_baseline.yaml")
    device = torch.device("cpu")  # CPU测试即可，不占用NPU
    
    print("=" * 60)
    print("玄女V2 Model Shape Test")
    print("=" * 60)
    print(f"  precision_dim: {cfg.model.precision_dim}")
    print(f"  time_dim: {cfg.model.time_dim}")
    print(f"  space_dim: {cfg.model.space_dim}")
    print(f"  num_blocks: {cfg.model.num_blocks}")
    print(f"  embedding_dim: {cfg.model.embedding_dim}")
    print(f"  image_size: {cfg.data.image_size}")
    print("=" * 60)
    
    model = AEFModel(cfg).to(device)
    model.eval()
    
    B, S, T, C, H, W = 2, 3, 8, 6, 128, 128
    num_tgt = cfg.data.num_target_sources
    meta_dim = cfg.data.metadata_dim
    
    # Dummy inputs
    source_frames = torch.randn(B, S, T, C, H, W, device=device)
    source_timestamps_ms = torch.randint(0, 1000000000, (B, S, T), device=device).float()
    source_frame_mask = torch.ones(B, S, T, dtype=torch.bool, device=device)
    source_input_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    source_type_ids = torch.zeros(B, S, dtype=torch.long, device=device)
    valid_start_ms = torch.tensor([0, 100000], device=device)
    valid_end_ms = torch.tensor([500000000, 600000000], device=device)
    target_relative_time = torch.zeros(B, num_tgt, device=device)
    target_metadata = torch.zeros(B, num_tgt, meta_dim, device=device)
    
    print("\n1. Testing full forward...")
    with torch.no_grad():
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
    
    print(f"   embedding_map:      {out.embedding_map.shape}     (expect [B, {cfg.model.embedding_dim}, 64, 64])")
    print(f"   embedding:          {out.embedding.shape}              (expect [B, {cfg.model.embedding_dim}])")
    print(f"   pre_norm_embedding: {out.pre_norm_embedding.shape}     (expect [B, {cfg.model.embedding_dim}])")
    print(f"   pre_norm_map:       {out.pre_norm_map.shape if out.pre_norm_map is not None else None}  (expect [B, {cfg.model.embedding_dim}, 64, 64])")
    print(f"   reconstructions:    {out.reconstructions.shape}  (expect [B, {num_tgt}, max_ch, 128, 128])")
    
    # Verify shapes
    assert out.embedding_map.shape == (B, cfg.model.embedding_dim, 64, 64), "embedding_map shape mismatch"
    assert out.embedding.shape == (B, cfg.model.embedding_dim), "embedding shape mismatch"
    assert out.pre_norm_map is not None and out.pre_norm_map.shape == (B, cfg.model.embedding_dim, 64, 64), "pre_norm_map shape mismatch"
    assert out.reconstructions.shape[0] == B and out.reconstructions.shape[1] == num_tgt
    assert out.reconstructions.shape[3:] == (128, 128), "reconstruction should be 128x128"
    print("   ✅ Full forward OK")
    
    print("\n2. Testing encode_dual_window...")
    with torch.no_grad():
        emb_w1, emb_w2, pre_w1, pre_w2 = model.encode_dual_window(
            source_frames=source_frames,
            source_timestamps_ms=source_timestamps_ms,
            source_frame_mask=source_frame_mask,
            source_input_mask=source_input_mask,
            source_type_ids=source_type_ids,
            valid_start_w1=valid_start_ms,
            valid_end_w1=valid_end_ms,
            valid_start_w2=valid_start_ms,
            valid_end_w2=valid_end_ms,
        )
    
    print(f"   emb_w1: {emb_w1.shape}  (expect [B, {cfg.model.embedding_dim}, 64, 64])")
    print(f"   emb_w2: {emb_w2.shape}  (expect [B, {cfg.model.embedding_dim}, 64, 64])")
    print(f"   pre_w1: {pre_w1.shape}  (expect [B, {cfg.model.embedding_dim}, 64, 64])")
    print(f"   pre_w2: {pre_w2.shape}  (expect [B, {cfg.model.embedding_dim}, 64, 64])")
    
    assert emb_w1.shape == (B, cfg.model.embedding_dim, 64, 64)
    assert emb_w2.shape == (B, cfg.model.embedding_dim, 64, 64)
    print("   ✅ Dual window OK")
    
    print("\n3. Testing STPEncoder internal shapes...")
    # 直接测试 STPEncoder
    from src.models.blocks import STPEncoder
    stp = STPEncoder(
        precision_dim=128,
        time_dim=512,
        space_dim=1024,
        num_blocks=2,  # 只用2个block测试，节省内存
        num_heads=8,
        use_checkpoint=False,
    ).to(device)
    
    test_x = torch.randn(B, T, 128, 64, 64, device=device)
    test_ts = torch.randint(0, 1000000000, (B, T), device=device).float()
    test_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        stp_out = stp(test_x, test_ts, test_mask)
    
    print(f"   STPEncoder input:  {test_x.shape}")
    print(f"   STPEncoder output: {stp_out.shape}  (expect [B, T, 128, 64, 64])")
    assert stp_out.shape == test_x.shape, "STPEncoder output shape mismatch"
    assert not torch.isnan(stp_out).any(), "STPEncoder output contains NaN"
    assert not torch.isinf(stp_out).any(), "STPEncoder output contains Inf"
    print("   ✅ STPEncoder OK")
    
    print("\n" + "=" * 60)
    print("✅ 玄女V2 model shape test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model_shapes()
