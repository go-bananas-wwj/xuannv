"""Debug why adding broadcast increases y-corr"""
import torch
import numpy as np

def y_corr(x):
    if x.dim() == 5:
        x = x[0, 0]
    elif x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    C, H, W = x.shape
    if H < 2:
        return 0.0
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
    return np.mean(corrs)

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
    
    print(f"space_out y-corr:      {y_corr(space_out):.4f}")
    print(f"space_exchange y-corr: {y_corr(space_exchange):.4f}")
    
    # Manual check for first channel
    c = 0
    s = space_out[0, 0, c]  # (16, 16)
    k = time_to_space[0, 0, 0, 0, c].item()
    se = s + k
    
    s_rm = s.mean(dim=1)  # row means
    se_rm = se.mean(dim=1)
    
    s_std = s.std(dim=1).clamp(min=1e-8)
    se_std = se.std(dim=1).clamp(min=1e-8)
    
    s_norm = (s - s_rm.unsqueeze(1)) / s_std.unsqueeze(1)
    se_norm = (se - se_rm.unsqueeze(1)) / se_std.unsqueeze(1)
    
    for h in range(3):
        s_corr = (s_norm[h] * s_norm[h+1]).sum() / (s_norm[h].norm() * s_norm[h+1].norm() + 1e-8)
        se_corr = (se_norm[h] * se_norm[h+1]).sum() / (se_norm[h].norm() * se_norm[h+1].norm() + 1e-8)
        print(f"  ch={c}, rows {h},{h+1}: space_out={s_corr:.4f}, space_exchange={se_corr:.4f}")
    
    # Now check: is it because time_to_space changes the std?
    print(f"\nspace_out row stds (first 4 rows, ch=0): {s_std[:4].numpy()}")
    print(f"space_exchange row stds (first 4 rows, ch=0): {se_std[:4].numpy()}")
    
    # What about precision_to_space?
    precision_x = torch.randn(1, 1, 128, 128, 64)
    precision_out = block.precision_op(precision_x)
    precision_global = precision_out.mean(dim=(2, 3))
    precision_to_space = block.precision_to_space_proj(precision_global).unsqueeze(2).unsqueeze(3)
    
    space_exchange2 = space_out + time_to_space.expand(1, 1, 16, 16, 512) + precision_to_space.expand(1, 1, 16, 16, 512)
    print(f"\nspace_exchange (with precision) y-corr: {y_corr(space_exchange2):.4f}")
    
    # What if we ONLY add precision_to_space?
    space_exchange3 = space_out + precision_to_space.expand(1, 1, 16, 16, 512)
    print(f"space_out + precision_only y-corr: {y_corr(space_exchange3):.4f}")
