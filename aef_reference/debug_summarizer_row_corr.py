"""Test summarizer steps with row_mean_corr"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.aef_module import TemporalSummarizer
from einops import rearrange

def row_mean_corr(x):
    x = x.detach().cpu().float()
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        if x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
            x = x.permute(2, 0, 1)
    C, H, W = x.shape
    rm = x.mean(dim=2)
    rm_mean = rm.mean(dim=1, keepdim=True)
    rm_std = rm.std(dim=1, keepdim=True).clamp(min=1e-8)
    rm_norm = (rm - rm_mean) / rm_std
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            v1 = rm_norm[c, h].item()
            v2 = rm_norm[c, h + 1].item()
            corrs.append(v1 * v2)
    return np.mean(corrs)

torch.manual_seed(42)

encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
summarizer = TemporalSummarizer(feature_dim=64, embed_dim=64, num_heads=8)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)
valid_periods = torch.tensor([[0.0, 1.0]])

with torch.no_grad():
    feats = encoder(x, ts)
    print(f"Encoder output row_mean_corr: {row_mean_corr(feats):.4f}")
    
    # Summarizer step by step
    B, T, H, W, C = feats.shape
    feats_2d = feats.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    print(f"Before spatial_smooth: {row_mean_corr(feats_2d):.4f}")
    
    feats_2d = summarizer.spatial_smooth(feats_2d)
    print(f"After spatial_smooth: {row_mean_corr(feats_2d):.4f}")
    
    feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B, T, H, W, C)
    print(f"After reshape: {row_mean_corr(feats_smooth):.4f}")
    
    q = summarizer.summarizer_q(valid_periods)
    z = summarizer.time_pool(feats_smooth, q, mask=None)
    print(f"After time_pool: {row_mean_corr(z):.4f}")
    
    mu = summarizer.proj_64(z)
    print(f"After proj_64: {row_mean_corr(mu):.4f}")
    
    # Full summarizer
    mu_full = summarizer(feats, ts, valid_periods)
    print(f"Full summarizer: {row_mean_corr(mu_full):.4f}")
    
    # Test with random input to summarizer
    rand_feats = torch.randn_like(feats)
    mu_rand = summarizer(rand_feats, ts, valid_periods)
    print(f"\nRandom input -> summarizer: {row_mean_corr(mu_rand):.4f}")
    
    # Test time_pool with random input
    z_rand = summarizer.time_pool(rand_feats, q, mask=None)
    print(f"Random input -> time_pool: {row_mean_corr(z_rand):.4f}")
    
    # Test proj_64 with random input
    mu_proj = summarizer.proj_64(z_rand)
    print(f"Random input -> proj_64: {row_mean_corr(mu_proj):.4f}")
