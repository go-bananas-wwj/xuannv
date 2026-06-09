"""深入检查 SpaceOperator 的 attention 矩阵结构"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from einops import rearrange
from src.aef.architecture.stp_operators import SpaceOperator

torch.manual_seed(42)

def y_corr(arr):
    B, T, H, W, D = arr.shape
    corrs = []
    for bi in range(B):
        for t in range(T):
            for c in range(D):
                for yi in range(H - 1):
                    c1 = np.corrcoef(arr[bi, t, yi, :, c], arr[bi, t, yi+1, :, c])[0, 1]
                    if not np.isnan(c1):
                        corrs.append(c1)
    return np.mean(corrs)

B, T, H, W, C = 1, 1, 16, 16, 64
x = torch.randn(B, T, H, W, C).npu()

op = SpaceOperator(dim=C, num_heads=8).npu()
op.eval()

with torch.no_grad():
    # Manual forward to inspect attention
    x_flat = rearrange(x, 'b t h w c -> (b t) (h w) c')
    residual = x_flat
    x_norm = op.norm1(x_flat)
    
    qkv = op.qkv(x_norm)
    qkv = rearrange(qkv, 'bt hw (three heads d) -> three bt heads hw d', 
                   three=3, heads=op.num_heads, d=op.head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    attn = (q @ k.transpose(-2, -1)) * (op.head_dim ** -0.5)
    attn_probs = torch.softmax(attn, dim=-1)
    
    # Check attention pattern
    attn_np = attn_probs.cpu().numpy()  # (BT, heads, HW, HW)
    print(f"Attention shape: {attn_np.shape}")
    
    # For each head, check if attention has y-structure
    # Compute correlation between attention patterns of adjacent rows
    for hi in range(op.num_heads):
        a = attn_np[0, hi]  # (HW, HW)
        # Reshape to (H, W, H, W)
        a_4d = a.reshape(H, W, H, W)
        
        # Check: does token (i,j) attend more to tokens in same row or column?
        y_corrs = []
        for yi in range(H - 1):
            for xi in range(W):
                idx1 = yi * W + xi
                idx2 = (yi + 1) * W + xi
                # Correlation between attention patterns of these two tokens
                c1 = np.corrcoef(a[idx1], a[idx2])[0, 1]
                if not np.isnan(c1):
                    y_corrs.append(c1)
        
        print(f"Head {hi}: mean y-attn-corr = {np.mean(y_corrs):.4f}")
    
    # Check if attention is uniform
    print(f"\nAttention entropy: {-np.mean(attn_np * np.log(attn_np + 1e-10)):.4f}")
    print(f"Max attention weight: {attn_np.max():.4f}")
    print(f"Min attention weight: {attn_np.min():.4f}")
    print(f"Std attention weight: {attn_np.std():.4f}")
    
    # Forward pass
    out = op(x)
    print(f"\nInput y_corr: {y_corr(x.cpu().numpy()):.4f}")
    print(f"Output y_corr: {y_corr(out.cpu().numpy()):.4f}")
