"""Debug why adding broadcast increases y-corr - detailed channel analysis"""
import torch
import numpy as np

def y_corr_detail(x):
    if x.dim() == 5:
        x = x[0, 0]
    elif x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    C, H, W = x.shape
    if H < 2:
        return np.array([]), 0.0
    row_means = x.mean(dim=2, keepdim=True)  # (C, H, 1)
    row_stds = x.std(dim=2, keepdim=True).clamp(min=1e-8)  # (C, H, 1)
    xn = (x - row_means) / row_stds
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            row1 = xn[c, h]
            row2 = xn[c, h + 1]
            corr = (row1 * row2).sum() / (row1.norm() * row2.norm() + 1e-8)
            corrs.append(corr.item())
    return np.array(corrs), np.mean(corrs)

import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")
from src.aef.architecture.STPBlock import STPBlock

block = STPBlock(space_dim=512, time_dim=256, precision_dim=64)
space_x = torch.randn(1, 1, 16, 16, 512)
time_x = torch.randn(1, 1, 32, 32, 256)
ts = torch.tensor([[0.0]])

with torch.no_grad():
    space_out = block.space_op(space_x)
    time_out = block.time_op(time_x, ts)
    
    time_global = time_out.mean(dim=(2, 3))
    time_to_space = block.time_to_space_proj(time_global).unsqueeze(2).unsqueeze(3)
    
    space_exchange = space_out + time_to_space.expand(1, 1, 16, 16, 512)
    
    corrs_so, mean_so = y_corr_detail(space_out)
    corrs_se, mean_se = y_corr_detail(space_exchange)
    
    print(f"space_out: mean={mean_so:.4f}, std={corrs_so.std():.4f}, min={corrs_so.min():.4f}, max={corrs_so.max():.4f}")
    print(f"space_exchange: mean={mean_se:.4f}, std={corrs_se.std():.4f}, min={corrs_se.min():.4f}, max={corrs_se.max():.4f}")
    
    # Find channels where y-corr changed most
    diff = corrs_se - corrs_so
    print(f"\nDiff: mean={diff.mean():.4f}, std={diff.std():.4f}")
    print(f"Top 10 increased channels: {np.argsort(diff)[-10:]}")
    print(f"Top 10 diff values: {np.sort(diff)[-10:]}")
    
    # Check if these channels have large time_to_space values
    k = time_to_space[0, 0, 0, 0].cpu().numpy()
    print(f"\ntime_to_space abs mean: {np.abs(k).mean():.4f}")
    print(f"time_to_space abs max: {np.abs(k).max():.4f}")
    
    top_channels = np.argsort(diff)[-10:]
    print(f"\ntime_to_space abs values for top channels: {np.abs(k[top_channels])}")
    
    # What about space_out std for these channels?
    so_std = space_out[0, 0].std(dim=(1,2)).cpu().numpy()
    print(f"space_out std for top channels: {so_std[top_channels]}")
    
    # Check: does time_to_space correlate with space_out's row mean differences?
    so_rowmeans = space_out[0, 0].mean(dim=2).cpu().numpy()  # (512, 16)
    so_rowdiff = np.diff(so_rowmeans, axis=1)  # (512, 15)
    so_rowdiff_mean = so_rowdiff.mean(axis=1)  # (512,)
    
    print(f"\nCorrelation between time_to_space and so_rowdiff_mean: {np.corrcoef(k, so_rowdiff_mean)[0,1]:.4f}")
