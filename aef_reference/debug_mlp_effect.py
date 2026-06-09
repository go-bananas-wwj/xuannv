"""测试 MLP 对条带的贡献"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from src.aef.architecture.encoder import STPEncoder

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

B, T, H, W, C = 2, 4, 128, 128, 20
x = torch.randn(B, T, H, W, C).npu()
ts = torch.randn(B, T).npu()

# Full model
encoder = STPEncoder(input_channels=C, d_s=1024, d_t=512, d_p=128, num_blocks=15).npu()
encoder.eval()
with torch.no_grad():
    out = encoder(x, ts)
print(f"Full (d_s=1024, blocks=15): y_corr={y_corr(out.cpu().numpy()):.4f}")

# Disable MLP in all SpaceOperators
for b in encoder.blocks:
    for layer in b.space_op.mlp:
        if hasattr(layer, 'weight'):
            layer.weight.data.zero_()
            if layer.bias is not None:
                layer.bias.data.zero_()

with torch.no_grad():
    out = encoder(x, ts)
print(f"No Space MLP: y_corr={y_corr(out.cpu().numpy()):.4f}")

# Also disable MLP in TimeOperators
for b in encoder.blocks:
    for layer in b.time_op.mlp:
        if hasattr(layer, 'weight'):
            layer.weight.data.zero_()
            if layer.bias is not None:
                layer.bias.data.zero_()

with torch.no_grad():
    out = encoder(x, ts)
print(f"No Space/Time MLP: y_corr={y_corr(out.cpu().numpy()):.4f}")

# Also disable PrecisionOperator conv
for b in encoder.blocks:
    b.precision_op.conv1.weight.data.zero_()
    b.precision_op.conv2.weight.data.zero_()

with torch.no_grad():
    out = encoder(x, ts)
print(f"No block ops at all: y_corr={y_corr(out.cpu().numpy()):.4f}")
