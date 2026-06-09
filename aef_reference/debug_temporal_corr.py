"""Check temporal correlation of encoder output"""
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

# Check temporal correlation
ff = feats[0].cpu().numpy()  # (T, H, W, C)
print(f"feats shape: {ff.shape}")

# For each spatial position, check temporal correlation
# Sample some positions
np.random.seed(42)
sample_positions = [(np.random.randint(H), np.random.randint(W)) for _ in range(10)]

print("\nTemporal correlation at sample positions:")
for h, w in sample_positions:
    ts_data = ff[:, h, w, :]  # (T, C)
    corr_matrix = np.corrcoef(ts_data.T)
    off_diag = corr_matrix[~np.eye(64, dtype=bool)]
    print(f"  ({h},{w}): mean={off_diag.mean():.4f}, max={off_diag.max():.4f}")

# Check if temporal frames are highly correlated
temporal_means = ff.mean(axis=(1,2,3))  # (T,)
print(f"\nTemporal means: {temporal_means}")
print(f"Temporal std: {temporal_means.std():.6f}")

# For random comparison
rand_feats = torch.randn_like(feats)
rf = rand_feats[0].cpu().numpy()
rand_temporal_means = rf.mean(axis=(1,2,3))
print(f"\nRandom temporal means: {rand_temporal_means}")
print(f"Random temporal std: {rand_temporal_means.std():.6f}")

# Check per-frame spatial mean
print("\nPer-frame spatial mean (first 5 frames):")
for t in range(min(5, T)):
    print(f"  frame {t}: mean={ff[t].mean():.6f}, std={ff[t].std():.6f}")
    
# Check row means per frame
print("\nRow mean correlation across frames:")
for t1 in range(T):
    for t2 in range(t1+1, T):
        rm1 = ff[t1].mean(axis=(1,2))  # (H,)
        rm2 = ff[t2].mean(axis=(1,2))  # (H,)
        corr = np.corrcoef(rm1, rm2)[0, 1]
        print(f"  frame {t1} vs {t2}: row_mean_corr={corr:.4f}")
