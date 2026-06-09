"""验证：添加位置编码后随机初始化是否还有条带"""
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

with torch.no_grad():
    out = encoder(x, ts)

print(f"With pos enc: encoder_output y_corr = {y_corr(out.cpu().numpy()):.4f}")

# Compare: without pos enc (by zeroing pos embed)
for block in encoder.blocks:
    # Replace forward to skip pos enc
    original_forward = block.space_op.forward
    def no_pos_forward(x):
        B, T, H, W, C = x.shape
        from einops import rearrange
        x_flat = rearrange(x, 'b t h w c -> (b t) (h w) c')
        residual = x_flat
        x_norm = block.space_op.norm1(x_flat)
        qkv = block.space_op.qkv(x_norm)
        qkv = rearrange(qkv, 'bt hw (three heads d) -> three bt heads hw d',
                       three=3, heads=block.space_op.num_heads, d=block.space_op.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (block.space_op.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        x_attn = attn @ v
        x_attn = rearrange(x_attn, 'bt heads hw d -> bt hw (heads d)')
        x_flat = residual + block.space_op.proj(x_attn)
        x_flat = x_flat + block.space_op.mlp(block.space_op.norm2(x_flat))
        return rearrange(x_flat, '(b t) (h w) c -> b t h w c', b=B, t=T, h=H, w=W)
    block.space_op.forward = no_pos_forward

with torch.no_grad():
    out2 = encoder(x, ts)

print(f"Without pos enc: encoder_output y_corr = {y_corr(out2.cpu().numpy()):.4f}")
