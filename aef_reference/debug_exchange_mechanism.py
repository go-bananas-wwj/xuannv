"""检查全局交换的哪个具体部分导致条带"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from src.aef.architecture.STPBlock import STPBlock

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

B, T = 1, 1
space = torch.randn(B, T, 16, 16, 64).npu()
time = torch.randn(B, T, 16, 16, 32).npu()
prec = torch.randn(B, T, 16, 16, 16).npu()
ts = torch.randn(B, T).npu()

block = STPBlock(space_dim=64, time_dim=32, precision_dim=16).npu()
block.eval()

# Run ops without exchange
with torch.no_grad():
    space_out = block.space_op(space)
    time_out = block.time_op(time, ts)
    prec_out = block.precision_op(prec)

print(f"After ops (no exchange):")
print(f"  space: {y_corr(space_out.cpu().numpy()):.4f}")
print(f"  time:  {y_corr(time_out.cpu().numpy()):.4f}")
print(f"  prec:  {y_corr(prec_out.cpu().numpy()):.4f}")

# Test individual exchange components
space_H, space_W = space_out.shape[2:4]
time_H, time_W = time_out.shape[2:4]
prec_H, prec_W = prec_out.shape[2:4]

space_global = space_out.mean(dim=(2, 3))
time_global = time_out.mean(dim=(2, 3))
prec_global = prec_out.mean(dim=(2, 3))

# Test: space + time_to_space only
time_to_space = block.time_to_space(time_global).unsqueeze(2).unsqueeze(3).expand(B, T, space_H, space_W, block.space_dim)
space_plus_time = space_out + time_to_space
print(f"\nspace + time_to_space: {y_corr(space_plus_time.cpu().numpy()):.4f}")

# Test: space + prec_to_space only
prec_to_space = block.precision_to_space(prec_global).unsqueeze(2).unsqueeze(3).expand(B, T, space_H, space_W, block.space_dim)
space_plus_prec = space_out + prec_to_space
print(f"space + prec_to_space: {y_corr(space_plus_prec.cpu().numpy()):.4f}")

# Test: space + time_to_space + prec_to_space
space_full = space_out + time_to_space + prec_to_space
print(f"space + both: {y_corr(space_full.cpu().numpy()):.4f}")

# Check if the constant itself has y-corr
print(f"\ntime_to_space alone: {y_corr(time_to_space.cpu().numpy()):.4f}")
print(f"prec_to_space alone: {y_corr(prec_to_space.cpu().numpy()):.4f}")

# Check: does adding a constant change y-corr?
const = torch.ones_like(space_out) * 0.5
space_plus_const = space_out + const
print(f"space + const(0.5): {y_corr(space_plus_const.cpu().numpy()):.4f}")
