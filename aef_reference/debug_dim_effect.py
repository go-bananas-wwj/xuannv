"""测试不同维度设置对条带的影响"""
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

for d_s, d_t, d_p, num_blocks in [
    (64, 32, 16, 2),
    (128, 64, 32, 2),
    (256, 128, 64, 4),
    (512, 256, 128, 8),
    (1024, 512, 128, 15),
]:
    encoder = STPEncoder(input_channels=C, d_s=d_s, d_t=d_t, d_p=d_p, num_blocks=num_blocks).npu()
    encoder.eval()
    with torch.no_grad():
        out = encoder(x, ts)
    print(f"d_s={d_s:4d}, d_t={d_t:3d}, d_p={d_p:3d}, blocks={num_blocks:2d}: y_corr={y_corr(out.cpu().numpy()):.4f}")
