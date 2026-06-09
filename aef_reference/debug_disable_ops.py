"""正确禁用所有 ops，定位条带来源"""
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

# Test 1: Full model
encoder = STPEncoder(input_channels=C, d_s=1024, d_t=512, d_p=128, num_blocks=15).npu()
encoder.eval()
with torch.no_grad():
    out = encoder(x, ts)
print(f"1. Full model: y_corr={y_corr(out.cpu().numpy()):.4f}")

# Test 2: Disable ALL self-attention (qkv=0) but keep MLPs
def disable_qkv(e):
    for b in e.blocks:
        b.space_op.qkv.weight.data.zero_()
        if b.space_op.qkv.bias is not None:
            b.space_op.qkv.bias.data.zero_()
        b.time_op.qkv.weight.data.zero_()
        if b.time_op.qkv.bias is not None:
            b.time_op.qkv.bias.data.zero_()

test_encoder = STPEncoder(input_channels=C, d_s=1024, d_t=512, d_p=128, num_blocks=15).npu()
test_encoder.eval()
disable_qkv(test_encoder)
with torch.no_grad():
    out = test_encoder(x, ts)
print(f"2. No self-attention: y_corr={y_corr(out.cpu().numpy()):.4f}")

# Test 3: Disable ALL ops (qkv=0, mlp=0, conv=0)
def disable_all_ops(e):
    for b in e.blocks:
        b.space_op.qkv.weight.data.zero_()
        for layer in b.space_op.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.data.zero_()
        b.time_op.qkv.weight.data.zero_()
        for layer in b.time_op.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.data.zero_()
        b.precision_op.conv1.weight.data.zero_()
        b.precision_op.conv2.weight.data.zero_()

test_encoder2 = STPEncoder(input_channels=C, d_s=1024, d_t=512, d_p=128, num_blocks=15).npu()
test_encoder2.eval()
disable_all_ops(test_encoder2)
with torch.no_grad():
    out = test_encoder2(x, ts)
print(f"3. No block ops at all: y_corr={y_corr(out.cpu().numpy()):.4f}")

# Test 4: Disable global exchange in STPBlock
class STPBlockNoExchange(torch.nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.space_op = original_block.space_op
        self.time_op = original_block.time_op
        self.precision_op = original_block.precision_op
    
    def forward(self, space_x, time_x, precision_x, timestamps):
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps)
        precision_out = self.precision_op(precision_x)
        return space_out, time_out, precision_out

test_encoder3 = STPEncoder(input_channels=C, d_s=1024, d_t=512, d_p=128, num_blocks=15).npu()
test_encoder3.eval()
for i in range(len(test_encoder3.blocks)):
    test_encoder3.blocks[i] = STPBlockNoExchange(test_encoder3.blocks[i])
with torch.no_grad():
    out = test_encoder3(x, ts)
print(f"4. No global exchange: y_corr={y_corr(out.cpu().numpy()):.4f}")
