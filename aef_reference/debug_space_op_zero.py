"""测试：将 SpaceOperator 的 qkv 权重设为 0，看是否还有条带"""
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

encoder = STPEncoder(input_channels=C, d_s=64, d_t=32, d_p=16, num_blocks=2).npu()
encoder.eval()

# Zero out qkv weights in all SpaceOperators
for block in encoder.blocks:
    block.space_op.qkv.weight.data.zero_()
    if block.space_op.qkv.bias is not None:
        block.space_op.qkv.bias.data.zero_()
    block.space_op.proj.weight.data.zero_()
    if block.space_op.proj.bias is not None:
        block.space_op.proj.bias.data.zero_()

with torch.no_grad():
    out = encoder(x, ts)

print(f"With zero qkv: encoder_output y_corr = {y_corr(out.cpu().numpy()):.4f}")

# Now test with normal weights but zero projections in STPBlock exchange
encoder2 = STPEncoder(input_channels=C, d_s=64, d_t=32, d_p=16, num_blocks=2).npu()
encoder2.eval()
for block in encoder2.blocks:
    for proj in [block.space_to_precision, block.time_to_precision,
                 block.precision_to_space, block.precision_to_time,
                 block.space_to_time, block.time_to_space]:
        proj.weight.data.zero_()
        if proj.bias is not None:
            proj.bias.data.zero_()

with torch.no_grad():
    out2 = encoder2(x, ts)

print(f"With zero exchange proj: encoder_output y_corr = {y_corr(out2.cpu().numpy()):.4f}")
