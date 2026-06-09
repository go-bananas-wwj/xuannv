"""检查完整 encoder 中 STP block 的输入/输出"""
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

encoder = STPEncoder(input_channels=C, d_s=64, d_t=32, d_p=16, num_blocks=1).npu()
encoder.eval()

with torch.no_grad():
    x_proj = encoder.input_projection(x)
    print(f"x_proj: {y_corr(x_proj.cpu().numpy()):.4f}")
    
    space = encoder.space_projection(x_proj)
    space = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(space, 'b t h w c -> (b t) c h w'), (H//8, W//8)
    )
    space = rearrange(space, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"space (before block): {y_corr(space.cpu().numpy()):.4f}")
    
    time = encoder.time_projection(x_proj)
    time = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(time, 'b t h w c -> (b t) c h w'), (H//4, W//4)
    )
    time = rearrange(time, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"time (before block): {y_corr(time.cpu().numpy()):.4f}")
    
    prec = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(x_proj, 'b t h w c -> (b t) c h w'), (H, W)
    )
    prec = rearrange(prec, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"prec (before block): {y_corr(prec.cpu().numpy()):.4f}")
    
    # Manually run STP block 0
    block = encoder.blocks[0]
    space_out = block.space_op(space)
    time_out = block.time_op(time, ts)
    prec_out = block.precision_op(prec)
    print(f"\nAfter individual ops (no exchange):")
    print(f"  space: {y_corr(space_out.cpu().numpy()):.4f}")
    print(f"  time:  {y_corr(time_out.cpu().numpy()):.4f}")
    print(f"  prec:  {y_corr(prec_out.cpu().numpy()):.4f}")
    
    # Now do global exchange manually
    B_t, T_t = space_out.shape[:2]
    space_H, space_W = space_out.shape[2:4]
    time_H, time_W = time_out.shape[2:4]
    prec_H, prec_W = prec_out.shape[2:4]
    
    space_global = space_out.mean(dim=(2, 3))
    time_global = time_out.mean(dim=(2, 3))
    prec_global = prec_out.mean(dim=(2, 3))
    
    print(f"\nGlobal means:")
    print(f"  space_global std: {space_global.std().item():.4f}")
    print(f"  time_global std:  {time_global.std().item():.4f}")
    print(f"  prec_global std:  {prec_global.std().item():.4f}")
    
    time_to_space = block.time_to_space_proj(time_global).unsqueeze(2).unsqueeze(3).expand(B_t, T_t, space_H, space_W, block.space_dim)
    prec_to_space = block.precision_to_space_proj(prec_global).unsqueeze(2).unsqueeze(3).expand(B_t, T_t, space_H, space_W, block.space_dim)
    
    space_ex = space_out + time_to_space + prec_to_space
    print(f"\nAfter global exchange:")
    print(f"  space: {y_corr(space_ex.cpu().numpy()):.4f}")
