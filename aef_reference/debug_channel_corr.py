"""Check channel correlation of encoder output"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
from einops import rearrange

torch.manual_seed(42)

encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)

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
    
    final_2d = rearrange(final_features, 'b t h w c -> (b t) c h w')
    final_2d = encoder.spatial_fusion(final_2d)
    final_features = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
    
    feats = encoder.norm(final_features)

# Check channel correlation
ff = feats[0, 0].cpu().numpy()  # (128, 128, 64)
ff_flat = ff.reshape(-1, 64)
corr_matrix = np.corrcoef(ff_flat.T)
off_diag = corr_matrix[~np.eye(64, dtype=bool)]
print(f"Channel correlation: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, max={off_diag.max():.4f}")

# Check per-row channel correlation
print("\nPer-row channel correlation (first 5 rows):")
for h in range(5):
    row = ff[h, :, :].reshape(-1, 64)
    corr = np.corrcoef(row.T)
    off = corr[~np.eye(64, dtype=bool)]
    print(f"  row {h}: mean={off.mean():.4f}, max={off.max():.4f}")

# Compare with random
rand_feats = torch.randn_like(feats)
rf = rand_feats[0, 0].cpu().numpy()
rf_flat = rf.reshape(-1, 64)
rcorr = np.corrcoef(rf_flat.T)
roff = rcorr[~np.eye(64, dtype=bool)]
print(f"\nRandom channel correlation: mean={roff.mean():.4f}, std={roff.std():.4f}, max={roff.max():.4f}")

# Check if channel correlation causes time_pool stripes
from src.aef.architecture.aef_module import TimePooling

time_pool = TimePooling(dim=64, num_heads=8)
q = torch.randn(1, 64)

with torch.no_grad():
    z_real = time_pool(feats, q, mask=None)
    z_rand = time_pool(rand_feats, q, mask=None)

# Row mean corr
def row_mean_corr(x):
    x = x.detach().cpu().float()
    if x.dim() == 4:
        x = x[0]
        x = x.permute(2, 0, 1)
    C, H, W = x.shape
    rm = x.mean(dim=2)
    rm_mean = rm.mean(dim=1, keepdim=True)
    rm_std = rm.std(dim=1, keepdim=True).clamp(min=1e-8)
    rm_norm = (rm - rm_mean) / rm_std
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            corrs.append(rm_norm[c, h].item() * rm_norm[c, h+1].item())
    return np.mean(corrs)

print(f"\nReal feats -> time_pool: {row_mean_corr(z_real):.4f}")
print(f"Random feats -> time_pool: {row_mean_corr(z_rand):.4f}")

# What if we decorrelate channels of real feats?
# Project to random basis
rand_proj = torch.randn(64, 64)
rand_proj = rand_proj / rand_proj.norm(dim=0, keepdim=True)
feats_decorr = feats @ rand_proj.unsqueeze(0).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    z_decorr = time_pool(feats_decorr, q, mask=None)
print(f"Decorrelated real feats -> time_pool: {row_mean_corr(z_decorr):.4f}")
