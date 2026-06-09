"""Test time_pool with centered input"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.aef_module import TimePooling

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

time_pool = TimePooling(dim=64, num_heads=8)

B, T, H, W, C = 1, 4, 128, 128, 64

# Use actual encoder output
from src.aef.architecture.encoder import STPEncoder
from einops import rearrange

torch.manual_seed(42)
encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)

with torch.no_grad():
    B, T, H, W, C_in = x.shape
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
    
    final_2d = rearrange(final_features, 'b t h w c -> (b t) c h w')
    final_2d = encoder.spatial_fusion(final_2d)
    final_features = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
    
    feats = encoder.norm(final_features)

print(f"Encoder output row_mean_corr: {row_mean_corr(feats):.4f}")

q = torch.randn(B, 64)

# Test 1: Normal time_pool
with torch.no_grad():
    z1 = time_pool(feats, q, mask=None)
print(f"Normal time_pool output: {row_mean_corr(z1):.4f}")

# Test 2: Center each spatial position across time
feats_centered = feats - feats.mean(dim=1, keepdim=True)
with torch.no_grad():
    z2 = time_pool(feats_centered, q, mask=None)
print(f"Time-centered input -> time_pool: {row_mean_corr(z2):.4f}")

# Test 3: Center each row across space and time
feats_row_centered = feats.clone()
for h in range(H):
    feats_row_centered[:, :, h, :, :] -= feats[:, :, h, :, :].mean()
with torch.no_grad():
    z3 = time_pool(feats_row_centered, q, mask=None)
print(f"Row-centered input -> time_pool: {row_mean_corr(z3):.4f}")

# Test 4: What if we shuffle rows before time_pool?
feats_shuffled = feats.clone()
perm = torch.randperm(H)
feats_shuffled = feats_shuffled[:, :, perm, :, :]
with torch.no_grad():
    z4 = time_pool(feats_shuffled, q, mask=None)
print(f"Shuffled rows -> time_pool: {row_mean_corr(z4):.4f}")

# Test 5: Check if the issue is in q_proj
# Use zero query
q_zero = torch.zeros(B, 64)
with torch.no_grad():
    z5 = time_pool(feats, q_zero, mask=None)
print(f"Zero query -> time_pool: {row_mean_corr(z5):.4f}")
