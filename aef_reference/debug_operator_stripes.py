"""Test which operator introduces stripes"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.stp_operators import SpaceOperator, TimeOperator, PrecisionOperator

def y_corr(x):
    """Compute mean Pearson correlation between adjacent rows"""
    if x.dim() == 4:
        x = x[0]
    elif x.dim() == 5:
        x = x[0, 0]
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

# Create operators
space_op = SpaceOperator(1024)
time_op = TimeOperator(512)
precision_op = PrecisionOperator(128)

# Random input
space_x = torch.randn(1, 1, 128, 128, 1024)
time_x = torch.randn(1, 1, 128, 128, 512)
precision_x = torch.randn(1, 1, 128, 128, 128)
timestamps = torch.tensor([[0.0]])

with torch.no_grad():
    space_out = space_op(space_x)
    time_out = time_op(time_x, timestamps)
    precision_out = precision_op(precision_x)
    
print(f"space_op y-corr:  {y_corr(space_out):.4f}")
print(f"time_op y-corr:   {y_corr(time_out):.4f}")
print(f"precision_op y-corr: {y_corr(precision_out):.4f}")

# Test time_op components
print("\n--- TimeOperator structure ---")
for name, module in time_op.named_modules():
    if len(list(module.children())) == 0 and not isinstance(module, (torch.nn.Identity,)):
        print(f"  {name}: {module.__class__.__name__}")

# Check time operator attention
if hasattr(time_op, 'attn'):
    qkv = time_op.attn.qkv.weight.data
    print(f"\nTimeOperator qkv weight shape: {qkv.shape}")
    print(f"  qkv weight std: {qkv.std():.4f}")
    print(f"  qkv weight mean abs: {qkv.abs().mean():.4f}")
    
    # Test attention on input with row structure
    time_x_structured = time_x.clone()
    time_x_structured[:, :, 60:68, :, :] += 2.0
    time_out_structured = time_op(time_x_structured, timestamps)
    print(f"\n  time_op with row structure input y-corr: {y_corr(time_out_structured):.4f}")
    print(f"  time_op random input y-corr: {y_corr(time_out):.4f}")
