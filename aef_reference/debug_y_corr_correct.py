"""Test corrected y-corr computation"""
import torch
import numpy as np

def y_corr_old(x):
    if x.dim() == 5:
        x = x[0, 0]
    elif x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
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

def y_corr_correct(x):
    if x.dim() == 5:
        x = x[0, 0]
    elif x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    C, H, W = x.shape
    if H < 2:
        return 0.0
    # Row-wise mean and std
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

# Test with synthetic data
print("=== Synthetic tests ===")

# 1. Pure horizontal stripes
stripes = torch.zeros(10, 128, 128)
for h in range(128):
    stripes[:, h, :] = h / 128.0
print(f"Horizontal stripes - old: {y_corr_old(stripes):.4f}, correct: {y_corr_correct(stripes):.4f}")

# 2. Pure vertical stripes  
vert = torch.zeros(10, 128, 128)
for w in range(128):
    vert[:, :, w] = w / 128.0
print(f"Vertical stripes - old: {y_corr_old(vert):.4f}, correct: {y_corr_correct(vert):.4f}")

# 3. Random noise
noise = torch.randn(10, 128, 128)
print(f"Random noise - old: {y_corr_old(noise):.4f}, correct: {y_corr_correct(noise):.4f}")

# 4. Random + small horizontal bias
biased = torch.randn(10, 128, 128)
for h in range(128):
    biased[:, h, :] += h * 0.01
print(f"Random + h-bias - old: {y_corr_old(biased):.4f}, correct: {y_corr_correct(biased):.4f}")

# 5. Constant + channel-wise offset
const = torch.ones(10, 128, 128)
for c in range(10):
    const[c] *= c
print(f"Constant (channel offset) - old: {y_corr_old(const):.4f}, correct: {y_corr_correct(const):.4f}")

# Test with real model output
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")
from src.aef.architecture.STPBlock import STPBlock

block = STPBlock(space_dim=512, time_dim=256, precision_dim=64)
space_x = torch.randn(1, 1, 16, 16, 512)
time_x = torch.randn(1, 1, 32, 32, 256)
precision_x = torch.randn(1, 1, 128, 128, 64)
ts = torch.tensor([[0.0]])

with torch.no_grad():
    space_out = block.space_op(space_x)
    time_out = block.time_op(time_x, ts)
    precision_out = block.precision_op(precision_x)
    
    print(f"\n=== STPBlock random init ===")
    print(f"space_op - old: {y_corr_old(space_out):.4f}, correct: {y_corr_correct(space_out):.4f}")
    print(f"time_op - old: {y_corr_old(time_out):.4f}, correct: {y_corr_correct(time_out):.4f}")
    print(f"precision_op - old: {y_corr_old(precision_out):.4f}, correct: {y_corr_correct(precision_out):.4f}")
    
    # Exchange
    space_global = space_out.mean(dim=(2, 3))
    time_global = time_out.mean(dim=(2, 3))
    precision_global = precision_out.mean(dim=(2, 3))
    
    time_to_space = block.time_to_space_proj(time_global).unsqueeze(2).unsqueeze(3)
    space_exchange = space_out + time_to_space.expand(1, 1, 16, 16, 512)
    
    print(f"space_exchange - old: {y_corr_old(space_exchange):.4f}, correct: {y_corr_correct(space_exchange):.4f}")
