"""Test with 1x1 conv instead of 3x3"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.aef_module import TemporalSummarizer
from einops import rearrange

def y_corr(x):
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
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

torch.manual_seed(42)

encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
summarizer = TemporalSummarizer(feature_dim=64, embed_dim=64, num_heads=8)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)
valid_periods = torch.tensor([[0.0, 1.0]])

with torch.no_grad():
    B, T, H, W, C = x.shape
    x_proj = encoder.input_projection(x)
    
    space_features = encoder.space_projection(x_proj)
    space_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(space_features, 'b t h w c -> (b t) c h w'),
        (H // 8, W // 8)
    )
    space_features = rearrange(space_features, '(b t) c h w -> b t h w c', b=B, t=T)
    
    time_features = encoder.time_projection(x_proj)
    time_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(time_features, 'b t h w c -> (b t) c h w'),
        (H // 4, W // 4)
    )
    time_features = rearrange(time_features, '(b t) c h w -> b t h w c', b=B, t=T)
    
    precision_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(x_proj, 'b t h w c -> (b t) c h w'),
        (H, W)
    )
    precision_features = rearrange(precision_features, '(b t) c h w -> b t h w c', b=B, t=T)
    
    for i, block in enumerate(encoder.blocks):
        space_features, time_features, precision_features = block(
            space_features, time_features, precision_features, ts
        )
    
    space_global = space_features.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    time_global = time_features.mean(dim=(2, 3))
    time_ctx = encoder.time_to_precision(time_global)
    time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    final_features = precision_features + space_broadcast + time_broadcast
    
    # Replace spatial_fusion with 1x1 conv
    conv1x1 = torch.nn.Conv2d(64, 64, kernel_size=1)
    final_2d = rearrange(final_features, 'b t h w c -> (b t) c h w')
    final_2d = conv1x1(final_2d)
    final_features = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
    
    feats_1x1 = encoder.norm(final_features)
    print(f"Encoder (1x1 conv) y-corr: {y_corr(feats_1x1):.4f}")
    
    # Summarizer with 1x1 spatial_smooth
    B, T, H, W, C = feats_1x1.shape
    feats_2d = feats_1x1.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    conv1x1_2 = torch.nn.Conv2d(64, 64, kernel_size=1)
    feats_2d = conv1x1_2(feats_2d)
    feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B, T, H, W, C)
    
    q = summarizer.summarizer_q(valid_periods)
    z = summarizer.time_pool(feats_smooth, q, mask=None)
    print(f"After time_pool (1x1) y-corr: {y_corr(z):.4f}")
    
    mu = summarizer.proj_64(z)
    print(f"After proj_64 (1x1) y-corr: {y_corr(mu):.4f}")
    
    # Check PCA
    emb = mu[0].cpu().numpy()
    emb_flat = emb.reshape(-1, 64)
    mean = emb_flat.mean(axis=0)
    centered = emb_flat - mean
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total_var = eigvals.sum()
    print(f"PCA top-3 cumulative: {eigvals[:3].sum() / total_var:.4f}")
