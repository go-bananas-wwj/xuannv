"""Verify broadcast does not affect per-channel y-corr"""
import torch
import numpy as np

def y_corr_single_channel(x):
    """x: (H, W)"""
    x = x.detach().cpu().float()
    H, W = x.shape
    if H < 2:
        return 0.0
    rm = x.mean(dim=1, keepdim=True)  # (H, 1)
    rs = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    xn = (x - rm) / rs
    corrs = []
    for h in range(H - 1):
        corr = (xn[h] * xn[h+1]).sum() / (xn[h].norm() * xn[h+1].norm() + 1e-8)
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
    
    # Compare per-channel y-corr for a few channels
    for c in [0, 16, 120, 158, 255]:
        so_c = space_out[0, 0, :, :, c]  # (16, 16)
        se_c = space_exchange[0, 0, :, :, c]  # (16, 16)
        y_so = y_corr_single_channel(so_c)
        y_se = y_corr_single_channel(se_c)
        diff = y_se - y_so
        k_c = time_to_space[0, 0, 0, 0, c].item()
        print(f"ch={c}: space_out={y_so:.4f}, space_exchange={y_se:.4f}, diff={diff:.6f}, k={k_c:.4f}")
    
    # Now test with synthetic: add a constant to a channel
    print("\n=== Synthetic test ===")
    test_tensor = torch.randn(16, 16)
    test_const = 5.0
    y1 = y_corr_single_channel(test_tensor)
    y2 = y_corr_single_channel(test_tensor + test_const)
    print(f"Random: y-corr={y1:.4f}, after +{test_const}: y-corr={y2:.4f}")
    
    # Test with structured tensor
    structured = torch.randn(16, 16)
    for h in range(16):
        structured[h] += h * 0.1
    y3 = y_corr_single_channel(structured)
    y4 = y_corr_single_channel(structured + test_const)
    print(f"Structured: y-corr={y3:.4f}, after +{test_const}: y-corr={y4:.4f}")
