"""Compare step-by-step y-corr between two methods"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
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

x = torch.randn(1, 1, 128, 128, 32)
ts = torch.tensor([[0.0]])

with torch.no_grad():
    B, T, H, W, C = x.shape
    
    # Method A: exact copy from debug_block_accumulation_fixed.py
    x_proj = encoder.input_projection(x)
    
    space_features_a = encoder.space_projection(x_proj)
    space_features_a = torch.nn.functional.adaptive_avg_pool2d(
        space_features_a.reshape(B*T, -1, H, W),
        (H // 8, W // 8)
    )
    space_features_a = space_features_a.reshape(B, T, H//8, W//8, 512)
    
    time_features_a = encoder.time_projection(x_proj)
    time_features_a = torch.nn.functional.adaptive_avg_pool2d(
        time_features_a.reshape(B*T, -1, H, W),
        (H // 4, W // 4)
    )
    time_features_a = time_features_a.reshape(B, T, H//4, W//4, 256)
    
    precision_features_a = torch.nn.functional.adaptive_avg_pool2d(
        x_proj.reshape(B*T, -1, H, W),
        (H, W)
    )
    precision_features_a = precision_features_a.reshape(B, T, H, W, 64)
    
    for i, block in enumerate(encoder.blocks):
        space_features_a, time_features_a, precision_features_a = block(
            space_features_a, time_features_a, precision_features_a, ts
        )
    
    space_global_a = space_features_a.mean(dim=(2, 3))
    space_ctx_a = encoder.space_to_precision(space_global_a)
    space_broadcast_a = space_ctx_a.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    time_global_a = time_features_a.mean(dim=(2, 3))
    time_ctx_a = encoder.time_to_precision(time_global_a)
    time_broadcast_a = time_ctx_a.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    final_features_a = precision_features_a + space_broadcast_a + time_broadcast_a
    
    final_2d_a = final_features_a.reshape(B*T, -1, H, W)
    final_2d_a = encoder.spatial_fusion(final_2d_a)
    final_features_a = final_2d_a.reshape(B, T, H, W, 64)
    
    out_a = encoder.norm(final_features_a)
    
    # Method B: exact copy from debug_same_seed.py
    x_proj_b = encoder.input_projection(x)
    
    space_features_b = encoder.space_projection(x_proj_b)
    space_features_b = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(space_features_b, 'b t h w c -> (b t) c h w'),
        (H // 8, W // 8)
    )
    space_features_b = rearrange(space_features_b, '(b t) c h w -> b t h w c', b=B, t=T)
    
    time_features_b = encoder.time_projection(x_proj_b)
    time_features_b = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(time_features_b, 'b t h w c -> (b t) c h w'),
        (H // 4, W // 4)
    )
    time_features_b = rearrange(time_features_b, '(b t) c h w -> b t h w c', b=B, t=T)
    
    precision_features_b = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(x_proj_b, 'b t h w c -> (b t) c h w'),
        (H, W)
    )
    precision_features_b = rearrange(precision_features_b, '(b t) c h w -> b t h w c', b=B, t=T)
    
    for i, block in enumerate(encoder.blocks):
        space_features_b, time_features_b, precision_features_b = block(
            space_features_b, time_features_b, precision_features_b, ts
        )
    
    space_global_b = space_features_b.mean(dim=(2, 3))
    space_ctx_b = encoder.space_to_precision(space_global_b)
    space_broadcast_b = space_ctx_b.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    time_global_b = time_features_b.mean(dim=(2, 3))
    time_ctx_b = encoder.time_to_precision(time_global_b)
    time_broadcast_b = time_ctx_b.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    final_features_b = precision_features_b + space_broadcast_b + time_broadcast_b
    
    final_2d_b = rearrange(final_features_b, 'b t h w c -> (b t) c h w')
    final_2d_b = encoder.spatial_fusion(final_2d_b)
    final_features_b = rearrange(final_2d_b, '(b t) c h w -> b t h w c', b=B, t=T)
    
    out_b = encoder.norm(final_features_b)
    
    print(f"Method A (reshape): y-corr={y_corr(out_a):.4f}")
    print(f"Method B (rearrange): y-corr={y_corr(out_b):.4f}")
    print(f"Max diff A vs B: {(out_a - out_b).abs().max().item():.6f}")
    
    # Direct
    out_direct = encoder(x, ts)
    print(f"Direct: y-corr={y_corr(out_direct):.4f}")
    print(f"Max diff A vs Direct: {(out_a - out_direct).abs().max().item():.6f}")
    print(f"Max diff B vs Direct: {(out_b - out_direct).abs().max().item():.6f}")
