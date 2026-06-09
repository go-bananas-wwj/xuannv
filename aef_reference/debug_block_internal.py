"""Test internal y-corr within STPBlock"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.STPBlock import STPBlock

def y_corr(x):
    if x.dim() == 5:
        x = x[0, 0]  # (C, H, W)
    elif x.dim() == 4:
        x = x[0]
    elif x.dim() == 3:
        x = x[0]  # (C,) or (H, W)
        if x.dim() == 1:
            return 0.0
    x = x.detach().cpu().float()
    if x.dim() == 2:
        # (H, W) single channel
        C, H, W = 1, x.shape[0], x.shape[1]
        x = x.unsqueeze(0)
    else:
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

block = STPBlock(space_dim=512, time_dim=256, precision_dim=64)

# Simulate inputs after projection/downsample
space_x = torch.randn(1, 1, 16, 16, 512)  # H//8 = 16
time_x = torch.randn(1, 1, 32, 32, 256)    # H//4 = 32
precision_x = torch.randn(1, 1, 128, 128, 64)  # H = 128
ts = torch.tensor([[0.0]])

with torch.no_grad():
    # Step by step
    space_out = block.space_op(space_x)
    time_out = block.time_op(time_x, ts)
    precision_out = block.precision_op(precision_x)
    
    print(f"space_op output y-corr:     {y_corr(space_out):.4f}")
    print(f"time_op output y-corr:      {y_corr(time_out):.4f}")
    print(f"precision_op output y-corr: {y_corr(precision_out):.4f}")
    
    # Global means
    space_global = space_out.mean(dim=(2, 3))    # (B, T, space_dim)
    time_global = time_out.mean(dim=(2, 3))      # (B, T, time_dim)
    precision_global = precision_out.mean(dim=(2, 3))  # (B, T, precision_dim)
    
    print(f"\nspace_global shape: {space_global.shape}")
    print(f"time_global shape: {time_global.shape}")
    print(f"precision_global shape: {precision_global.shape}")
    
    # Projections
    time_to_space = block.time_to_space_proj(time_global).unsqueeze(2).unsqueeze(3)
    precision_to_space = block.precision_to_space_proj(precision_global).unsqueeze(2).unsqueeze(3)
    space_to_time = block.space_to_time_proj(space_global).unsqueeze(2).unsqueeze(3)
    precision_to_time = block.precision_to_time_proj(precision_global).unsqueeze(2).unsqueeze(3)
    space_to_precision = block.space_to_precision_proj(space_global).unsqueeze(2).unsqueeze(3)
    time_to_precision = block.time_to_precision_proj(time_global).unsqueeze(2).unsqueeze(3)
    
    print(f"\ntime_to_space y-corr (broadcast): {y_corr(time_to_space):.4f}")
    print(f"precision_to_space y-corr (broadcast): {y_corr(precision_to_space):.4f}")
    
    # Exchange
    space_exchange = space_out + time_to_space.expand(1, 1, 16, 16, 512) + precision_to_space.expand(1, 1, 16, 16, 512)
    time_exchange = time_out + space_to_time.expand(1, 1, 32, 32, 256) + precision_to_time.expand(1, 1, 32, 32, 256)
    precision_exchange = precision_out + space_to_precision.expand(1, 1, 128, 128, 64) + time_to_precision.expand(1, 1, 128, 128, 64)
    
    print(f"\nspace_exchange y-corr:     {y_corr(space_exchange):.4f}")
    print(f"time_exchange y-corr:      {y_corr(time_exchange):.4f}")
    print(f"precision_exchange y-corr: {y_corr(precision_exchange):.4f}")
    
    # Check if adding broadcast values changes y-corr
    # space_out + 0 vs space_out + time_to_space
    space_add_zero = space_out + 0
    space_add_time = space_out + time_to_space.expand(1, 1, 16, 16, 512)
    print(f"\nspace_out + 0 y-corr:      {y_corr(space_add_zero):.4f}")
    print(f"space_out + time_to_space y-corr: {y_corr(space_add_time):.4f}")
    
    # What if time_to_space has the same value repeated?
    print(f"\ntime_to_space value std across space: {time_to_space.std(dim=(2,3)).item():.6f}")
    print(f"precision_to_space value std across space: {precision_to_space.std(dim=(2,3)).item():.6f}")
