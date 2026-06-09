"""Debug encoder layer by layer"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.encoder import STPEncoder

def adjacent_row_cos_sim(t):
    """t: (H, W, C)"""
    sims = []
    for h in range(t.shape[0] - 1):
        row_h = t[h, :, :].reshape(-1)
        row_hp1 = t[h+1, :, :].reshape(-1)
        sim = torch.cosine_similarity(row_h, row_hp1, dim=0)
        sims.append(sim.item())
    return np.mean(sims)

# Create a simple encoder
encoder = STPEncoder(input_channels=19, d_s=1024, d_t=512, d_p=128, num_blocks=15)
device = "npu:0"
encoder = encoder.to(device)
encoder.eval()

B, T, H, W, C = 1, 4, 128, 128, 19
torch.manual_seed(42)
x = torch.randn(B, T, H, W, C, device=device)
timestamps = torch.rand(B, T, device=device)

with torch.no_grad():
    # Step by step through encoder
    x_proj = encoder.input_projection(x)
    print(f"x_proj adjacent_row_cos_sim: {adjacent_row_cos_sim(x_proj[0, 0]):.4f}")
    
    space_features = encoder.space_projection(x_proj)
    space_features = torch.nn.functional.adaptive_avg_pool2d(
        space_features.permute(0, 1, 4, 2, 3).reshape(B*T, -1, H, W),
        (H // 8, W // 8)
    )
    space_features = space_features.view(B, T, -1, H//8, W//8).permute(0, 1, 3, 4, 2)
    print(f"space_features (after init) adjacent_row_cos_sim: {adjacent_row_cos_sim(space_features[0, 0]):.4f}")
    
    time_features = encoder.time_projection(x_proj)
    time_features = torch.nn.functional.adaptive_avg_pool2d(
        time_features.permute(0, 1, 4, 2, 3).reshape(B*T, -1, H, W),
        (H // 4, W // 4)
    )
    time_features = time_features.view(B, T, -1, H//4, W//4).permute(0, 1, 3, 4, 2)
    print(f"time_features (after init) adjacent_row_cos_sim: {adjacent_row_cos_sim(time_features[0, 0]):.4f}")
    
    precision_features = torch.nn.functional.adaptive_avg_pool2d(
        x_proj.permute(0, 1, 4, 2, 3).reshape(B*T, -1, H, W),
        (H, W)
    )
    precision_features = precision_features.view(B, T, -1, H, W).permute(0, 1, 3, 4, 2)
    print(f"precision_features (after init) adjacent_row_cos_sim: {adjacent_row_cos_sim(precision_features[0, 0]):.4f}")
    
    # Run through blocks one by one
    for i, block in enumerate(encoder.blocks):
        space_features, time_features, precision_features = block(
            space_features, time_features, precision_features, timestamps
        )
        if i % 3 == 0 or i == 14:
            print(f"Block {i}: precision adjacent_row_cos_sim={adjacent_row_cos_sim(precision_features[0, 0]):.4f}, "
                  f"space adjacent_row_cos_sim={adjacent_row_cos_sim(space_features[0, 0]):.4f}")
    
    # Final combination
    space_global = space_features.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, encoder.precision_dim)
    
    time_global = time_features.mean(dim=(2, 3))
    time_ctx = encoder.time_to_precision(time_global)
    time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, encoder.precision_dim)
    
    final_features = precision_features + space_broadcast + time_broadcast
    print(f"final_features (before fusion) adjacent_row_cos_sim: {adjacent_row_cos_sim(final_features[0, 0]):.4f}")
    
    final_2d = final_features.permute(0, 1, 4, 2, 3).reshape(B*T, -1, H, W)
    final_2d = encoder.spatial_fusion(final_2d)
    final_features = final_2d.view(B, T, -1, H, W).permute(0, 1, 3, 4, 2)
    print(f"final_features (after fusion) adjacent_row_cos_sim: {adjacent_row_cos_sim(final_features[0, 0]):.4f}")
