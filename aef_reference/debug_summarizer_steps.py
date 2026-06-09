"""Test summarizer step by step y-corr"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.aef_module import TemporalSummarizer, TimePooling
from src.aef.architecture.encoder import STPEncoder

def y_corr(x):
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        if x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
            x = x.permute(2, 0, 1)
    elif x.dim() == 3:
        if x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
            x = x.permute(2, 0, 1)
    x = x.detach().cpu().float()
    C, H, W = x.shape
    if H < 2:
        return 0.0
    row_means = x.mean(dim=2, keepdim=True)
    row_stds = x.std(dim=2, keepdim=True).clamp(min=1e-8)
    xn = (x - row_means) / row_stds
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            row1 = xn[c, h]
            row2 = xn[c, h + 1]
            corr = (row1 * row2).sum() / (row1.norm() * row2.norm() + 1e-8)
            corrs.append(corr.item())
    return np.mean(corrs)

# Create encoder + summarizer
encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
summarizer = TemporalSummarizer(feature_dim=64, embed_dim=64, num_heads=8)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)
valid_periods = torch.tensor([[0.0, 1.0]])

with torch.no_grad():
    feats = encoder(x, ts)
    print(f"encoder output y-corr: {y_corr(feats):.4f}")
    
    # Step 1: spatial_smooth
    B, T, H, W, C = feats.shape
    feats_2d = feats.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    feats_2d = summarizer.spatial_smooth(feats_2d)
    feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B, T, H, W, C)
    print(f"after spatial_smooth y-corr: {y_corr(feats_smooth):.4f}")
    
    # Step 2: build query
    q = summarizer.summarizer_q(valid_periods)
    print(f"query shape: {q.shape}")
    
    # Step 3: time_pool (manual)
    time_pool = summarizer.time_pool
    z = time_pool(feats_smooth, q, mask=None)
    print(f"after time_pool y-corr: {y_corr(z):.4f}")
    
    # Step 4: proj_64
    mu = summarizer.proj_64(z)
    print(f"after proj_64 y-corr: {y_corr(mu):.4f}")
    
    # Full summarizer
    mu_full = summarizer(feats, ts, valid_periods)
    print(f"full summarizer y-corr: {y_corr(mu_full):.4f}")
    
    # Test time_pool in detail
    print("\n=== TimePool detail ===")
    # What if feats_smooth has no y-corr?
    feats_random = torch.randn_like(feats_smooth)
    z_random = time_pool(feats_random, q, mask=None)
    print(f"random input -> time_pool y-corr: {y_corr(z_random):.4f}")
    
    # What if feats_smooth has strong y-corr?
    feats_striped = feats_smooth.clone()
    for h in range(H):
        feats_striped[:, :, h, :, :] += h * 0.1
    z_striped = time_pool(feats_striped, q, mask=None)
    print(f"striped input -> time_pool y-corr: {y_corr(z_striped):.4f}")
