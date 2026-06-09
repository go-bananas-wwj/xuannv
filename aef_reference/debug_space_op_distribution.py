"""Analyze space_op output distribution across channels"""
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
        return np.array([]), 0.0
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
    return np.array(corrs), np.mean(corrs)

torch.manual_seed(42)

encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)

x = torch.randn(1, 1, 128, 128, 32)
ts = torch.tensor([[0.0]])

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
    
    print("After init:")
    _, s = y_corr(space_features)
    print(f"  space y-corr mean: {s:.4f}")
    _, t = y_corr(time_features)
    print(f"  time y-corr mean: {t:.4f}")
    _, p = y_corr(precision_features)
    print(f"  precision y-corr mean: {p:.4f}")
    
    for i, block in enumerate(encoder.blocks):
        space_features, time_features, precision_features = block(
            space_features, time_features, precision_features, ts
        )
        corrs_s, mean_s = y_corr(space_features)
        corrs_t, mean_t = y_corr(time_features)
        corrs_p, mean_p = y_corr(precision_features)
        print(f"After block {i+1}:")
        print(f"  space y-corr: mean={mean_s:.4f}, std={corrs_s.std():.4f}, min={corrs_s.min():.4f}, max={corrs_s.max():.4f}")
        print(f"  time y-corr: mean={mean_t:.4f}, std={corrs_t.std():.4f}, min={corrs_t.min():.4f}, max={corrs_t.max():.4f}")
        print(f"  precision y-corr: mean={mean_p:.4f}, std={corrs_p.std():.4f}, min={corrs_p.min():.4f}, max={corrs_p.max():.4f}")
    
    # Final fusion
    space_global = space_features.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    time_global = time_features.mean(dim=(2, 3))
    time_ctx = encoder.time_to_precision(time_global)
    time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    
    final_features = precision_features + space_broadcast + time_broadcast
    corrs_f, mean_f = y_corr(final_features)
    print(f"After fusion: y-corr mean={mean_f:.4f}, std={corrs_f.std():.4f}, min={corrs_f.min():.4f}, max={corrs_f.max():.4f}")
    
    # Check correlation between channels
    ff = final_features[0, 0].cpu().numpy()  # (128, 128, 64)
    ff_flat = ff.reshape(-1, 64)
    corr_matrix = np.corrcoef(ff_flat.T)
    off_diag = corr_matrix[~np.eye(64, dtype=bool)]
    print(f"Channel correlation: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, max={off_diag.max():.4f}")
