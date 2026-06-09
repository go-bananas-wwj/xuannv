"""Understand why time_pool amplifies row_mean_corr"""
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

# Create time_pool
time_pool = TimePooling(dim=64, num_heads=8)

# Test 1: controlled input with specific row_mean_corr
print("=== Test 1: Controlled row_mean_corr ===")
B, T, H, W, C = 1, 4, 128, 128, 64

# Create input with row_mean_corr = 0.05
feats = torch.randn(B, T, H, W, C)
for h in range(H):
    feats[:, :, h, :, :] += (h / H) * 0.1  # Add small horizontal gradient

print(f"Input row_mean_corr: {row_mean_corr(feats):.4f}")

q = torch.randn(B, C)
with torch.no_grad():
    z = time_pool(feats, q, mask=None)
print(f"Output row_mean_corr: {row_mean_corr(z):.4f}")

# Test 2: pure horizontal stripes
print("\n=== Test 2: Pure stripes ===")
feats2 = torch.zeros(B, T, H, W, C)
for h in range(H):
    feats2[:, :, h, :, :] = (h // 32) * 0.2

print(f"Input row_mean_corr: {row_mean_corr(feats2):.4f}")
with torch.no_grad():
    z2 = time_pool(feats2, q, mask=None)
print(f"Output row_mean_corr: {row_mean_corr(z2):.4f}")

# Test 3: random input
print("\n=== Test 3: Random input ===")
feats3 = torch.randn(B, T, H, W, C)
print(f"Input row_mean_corr: {row_mean_corr(feats3):.4f}")
with torch.no_grad():
    z3 = time_pool(feats3, q, mask=None)
print(f"Output row_mean_corr: {row_mean_corr(z3):.4f}")

# Test 4: Check if amplification depends on T
print("\n=== Test 4: Effect of T ===")
for T_test in [1, 2, 4, 8, 16]:
    feats_t = torch.randn(B, T_test, H, W, C)
    for h in range(H):
        feats_t[:, :, h, :, :] += (h / H) * 0.1
    with torch.no_grad():
        z_t = time_pool(feats_t, q[:, :C], mask=None)
    print(f"  T={T_test}: input={row_mean_corr(feats_t):.4f}, output={row_mean_corr(z_t):.4f}")

# Test 5: What if we use mean pooling instead of attention?
print("\n=== Test 5: Mean pooling vs attention ===")
feats5 = torch.randn(B, T, H, W, C)
for h in range(H):
    feats5[:, :, h, :, :] += (h / H) * 0.1

# Mean pooling over time
mean_pooled = feats5.mean(dim=1)  # (B, H, W, C)
print(f"Mean pooled row_mean_corr: {row_mean_corr(mean_pooled):.4f}")

with torch.no_grad():
    z5 = time_pool(feats5, q, mask=None)
print(f"Attention pooled row_mean_corr: {row_mean_corr(z5):.4f}")
