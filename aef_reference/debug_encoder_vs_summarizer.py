"""Compare encoder output y-corr under same conditions as summarizer test"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder

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

# Use aef_module defaults (faster)
encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)

with torch.no_grad():
    feats = encoder(x, ts)
    print(f"encoder output shape: {feats.shape}")
    print(f"encoder output y-corr: {y_corr(feats):.4f}")
    
    # Also test with T=1
    x1 = torch.randn(1, 1, 128, 128, 32)
    ts1 = torch.rand(1, 1)
    feats1 = encoder(x1, ts1)
    print(f"encoder output (T=1) y-corr: {y_corr(feats1):.4f}")
    
    # Test with different seeds
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        encoder2 = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
        x2 = torch.randn(1, 4, 128, 128, 32)
        ts2 = torch.rand(1, 4)
        feats2 = encoder2(x2, ts2)
        print(f"  seed={seed}: encoder output y-corr: {y_corr(feats2):.4f}")
