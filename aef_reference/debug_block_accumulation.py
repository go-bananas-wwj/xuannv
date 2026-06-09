"""Test how y-corr accumulates across STPBlocks"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder

def y_corr(x):
    if x.dim() == 5:
        x = x[0, 0]  # (C, H, W)
    elif x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    C, H, W = x.shape
    if H < 2:
        return 0.0
    row_means = x.mean(dim=(1, 2), keepdim=True)
    row_stds = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    xn = (x - row_means) / row_stds
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            row1 = xn[c, h]
            row2 = xn[c, h + 1]
            corr = (row1 * row2).sum() / (row1.norm() * row2.norm() + 1e-8)
            corrs.append(corr.item())
    return np.mean(corrs)

# Create encoder
encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)

# Random input
x = torch.randn(1, 1, 128, 128, 32)
ts = torch.tensor([[0.0]])

with torch.no_grad():
    # Manually run encoder step by step
    B, T, H, W, C = x.shape
    x_proj = encoder.input_projection(x)
    
    space_features = encoder.space_projection(x_proj)
    space_features = torch.nn.functional.adaptive_avg_pool2d(
        space_features.reshape(B*T, -1, H, W),
        (H // 8, W // 8)
    )
    space_features = space_features.reshape(B, T, H//8, W//8, 512)
    
    time_features = encoder.time_projection(x_proj)
    time_features = torch.nn.functional.adaptive_avg_pool2d(
        time_features.reshape(B*T, -1, H, W),
        (H // 4, W // 4)
    )
    time_features = time_features.reshape(B, T, H//4, W//4, 256)
    
    precision_features = torch.nn.functional.adaptive_avg_pool2d(
        x_proj.reshape(B*T, -1, H, W),
        (H, W)
    )
    precision_features = precision_features.reshape(B, T, H, W, 64)
    
    print(f"After init:")
    print(f"  space_features y-corr:     {y_corr(space_features):.4f}")
    print(f"  time_features y-corr:      {y_corr(time_features):.4f}")
    print(f"  precision_features y-corr: {y_corr(precision_features):.4f}")
    
    for i, block in enumerate(encoder.blocks):
        space_features, time_features, precision_features = block(
            space_features, time_features, precision_features, ts
        )
        print(f"After block {i+1}:")
        print(f"  space_features y-corr:     {y_corr(space_features):.4f}")
        print(f"  time_features y-corr:      {y_corr(time_features):.4f}")
        print(f"  precision_features y-corr: {y_corr(precision_features):.4f}")
    
    # Final fusion
    space_global = space_features.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    time_global = time_features.mean(dim=(2, 3))
    time_ctx = encoder.time_to_precision(time_global)
    time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    final_features = precision_features + space_broadcast + time_broadcast
    print(f"After fusion (before spatial_fusion):")
    print(f"  final_features y-corr:     {y_corr(final_features):.4f}")
    
    final_2d = final_features.reshape(B*T, -1, H, W)
    final_2d = encoder.spatial_fusion(final_2d)
    final_features = final_2d.reshape(B, T, H, W, 64)
    print(f"After spatial_fusion:")
    print(f"  final_features y-corr:     {y_corr(final_features):.4f}")
    
    final_features = encoder.norm(final_features)
    print(f"After norm:")
    print(f"  final_features y-corr:     {y_corr(final_features):.4f}")
