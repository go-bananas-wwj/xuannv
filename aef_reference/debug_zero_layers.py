"""逐层禁用，定位条带来源"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from einops import rearrange
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

def test_encoder(modify_fn, label):
    encoder = STPEncoder(input_channels=C, d_s=64, d_t=32, d_p=16, num_blocks=2).npu()
    encoder.eval()
    modify_fn(encoder)
    with torch.no_grad():
        out = encoder(x, ts)
    print(f"{label}: y_corr = {y_corr(out.cpu().numpy()):.4f}")

# Baseline
test_encoder(lambda e: None, "Baseline")

# Zero space_op qkv only
def zero_space_qkv(e):
    for b in e.blocks:
        b.space_op.qkv.weight.data.zero_()
        if b.space_op.qkv.bias is not None:
            b.space_op.qkv.bias.data.zero_()
test_encoder(zero_space_qkv, "Zero space qkv")

# Zero space_op proj only
def zero_space_proj(e):
    for b in e.blocks:
        b.space_op.proj.weight.data.zero_()
        if b.space_op.proj.bias is not None:
            b.space_op.proj.bias.data.zero_()
test_encoder(zero_space_proj, "Zero space proj")

# Zero space_op mlp
def zero_space_mlp(e):
    for b in e.blocks:
        for layer in b.space_op.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
test_encoder(zero_space_mlp, "Zero space mlp")

# Zero time_op qkv
def zero_time_qkv(e):
    for b in e.blocks:
        b.time_op.qkv.weight.data.zero_()
        if b.time_op.qkv.bias is not None:
            b.time_op.qkv.bias.data.zero_()
test_encoder(zero_time_qkv, "Zero time qkv")

# Zero precision_op conv
def zero_prec_conv(e):
    for b in e.blocks:
        b.precision_op.conv1.weight.data.zero_()
        b.precision_op.conv2.weight.data.zero_()
test_encoder(zero_prec_conv, "Zero prec conv")

# Zero all qkv and convs in blocks
def zero_all_ops(e):
    for b in e.blocks:
        b.space_op.qkv.weight.data.zero_()
        b.space_op.proj.weight.data.zero_()
        for layer in b.space_op.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.data.zero_()
        b.time_op.qkv.weight.data.zero_()
        b.precision_op.conv1.weight.data.zero_()
        b.precision_op.conv2.weight.data.zero_()
test_encoder(zero_all_ops, "Zero ALL block ops")

# Zero spatial_fusion
def zero_spatial_fusion(e):
    for layer in e.spatial_fusion:
        if hasattr(layer, 'weight'):
            layer.weight.data.zero_()
            if layer.bias is not None:
                layer.bias.data.zero_()
test_encoder(zero_spatial_fusion, "Zero spatial_fusion")

# Identity blocks (skip all block ops)
def identity_blocks(e):
    for b in e.blocks:
        b.space_op.qkv.weight.data.zero_()
        b.space_op.proj.weight.data.zero_()
        for layer in b.space_op.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.data.zero_()
        b.time_op.qkv.weight.data.zero_()
        b.precision_op.conv1.weight.data.zero_()
        b.precision_op.conv2.weight.data.zero_()
    e.spatial_fusion[0].weight.data.zero_()
    e.spatial_fusion[2].weight.data.zero_()
test_encoder(identity_blocks, "Identity blocks + fusion")
