"""逐步检查 encoder 每一层的 y-corr"""
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
    """Compute mean y-correlation."""
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

# Random input
B, T, H, W, C = 2, 4, 128, 128, 20
x = torch.randn(B, T, H, W, C).npu()
ts = torch.randn(B, T).npu()

encoder = STPEncoder(input_channels=C, d_s=64, d_t=32, d_p=16, num_blocks=2).npu()
encoder.eval()

with torch.no_grad():
    # Step 1: input_projection
    x_proj = encoder.input_projection(x)
    print(f"1. input_projection: y_corr={y_corr(x_proj.cpu().numpy()):.4f}")
    
    # Step 2: space_projection + pool
    space = encoder.space_projection(x_proj)
    space = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(space, 'b t h w c -> (b t) c h w'),
        (H // 8, W // 8)
    )
    space = rearrange(space, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"2. space_projection+pool: y_corr={y_corr(space.cpu().numpy()):.4f}")
    
    # Step 3: time_projection + pool
    time = encoder.time_projection(x_proj)
    time = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(time, 'b t h w c -> (b t) c h w'),
        (H // 4, W // 4)
    )
    time = rearrange(time, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"3. time_projection+pool: y_corr={y_corr(time.cpu().numpy()):.4f}")
    
    # Step 4: precision (pool at full res)
    prec = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(x_proj, 'b t h w c -> (b t) c h w'),
        (H, W)
    )
    prec = rearrange(prec, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"4. precision_pool: y_corr={y_corr(prec.cpu().numpy()):.4f}")
    
    # Step 5: After STP blocks
    for i, block in enumerate(encoder.blocks):
        space, time, prec = block(space, time, prec, ts)
        print(f"5. STP block {i}: space_y={y_corr(space.cpu().numpy()):.4f}, time_y={y_corr(time.cpu().numpy()):.4f}, prec_y={y_corr(prec.cpu().numpy()):.4f}")
    
    # Step 6: Final combination
    space_global = space.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, encoder.precision_dim)
    
    time_global = time.mean(dim=(2, 3))
    time_ctx = encoder.time_to_precision(time_global)
    time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, encoder.precision_dim)
    
    final = prec + space_broadcast + time_broadcast
    print(f"6. after combine: y_corr={y_corr(final.cpu().numpy()):.4f}")
    
    # Step 7: spatial_fusion
    final_2d = rearrange(final, 'b t h w c -> (b t) c h w')
    final_2d = encoder.spatial_fusion(final_2d)
    final = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
    print(f"7. after spatial_fusion: y_corr={y_corr(final.cpu().numpy()):.4f}")
    
    # Step 8: norm
    final = encoder.norm(final)
    print(f"8. after norm: y_corr={y_corr(final.cpu().numpy()):.4f}")
