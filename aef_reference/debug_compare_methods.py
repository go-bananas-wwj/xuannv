"""Compare encoder(x, ts) vs manual steps"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
from src.aef.architecture.encoder import STPEncoder
from einops import rearrange

encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
encoder.eval()

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)

with torch.no_grad():
    # Method 1: direct forward
    out1 = encoder(x, ts)
    
    # Method 2: manual steps
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
    
    out2 = encoder.norm(final_features)
    
    print(f"out1 shape: {out1.shape}")
    print(f"out2 shape: {out2.shape}")
    print(f"max diff: {(out1 - out2).abs().max().item():.6f}")
    print(f"mean diff: {(out1 - out2).abs().mean().item():.6f}")
