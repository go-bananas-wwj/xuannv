"""Find source of grid artifacts"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch_npu
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.aef_module import TimePooling
from einops import rearrange

def show_grid(name, x, save_path):
    x = x.detach().cpu().float()
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        x = x.permute(2, 0, 1)
    C, H, W = x.shape
    
    # PCA to 3 channels
    x_flat = x.reshape(C, -1).T
    mean = x_flat.mean(axis=0)
    centered = x_flat - mean
    if C > 3:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        proj = centered @ eigvecs[:, :3]
    else:
        proj = centered
    for i in range(3):
        c = proj[:, i]
        proj[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
    rgb = proj.reshape(H, W, 3)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title(name)
    ax.axis('off')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

torch.manual_seed(42)
encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
time_pool = TimePooling(dim=64, num_heads=8)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)

with torch.no_grad():
    B, T, H, W, C = x.shape
    x_proj = encoder.input_projection(x)
    show_grid("x_proj", x_proj, "/workspace/xuannv/aef_reference/debug_grid_x_proj.png")
    
    space_features = encoder.space_projection(x_proj)
    space_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(space_features, 'b t h w c -> (b t) c h w'), (H // 8, W // 8))
    space_features = rearrange(space_features, '(b t) c h w -> b t h w c', b=B, t=T)
    show_grid("space_features init", space_features, "/workspace/xuannv/aef_reference/debug_grid_space_init.png")
    
    time_features = encoder.time_projection(x_proj)
    time_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(time_features, 'b t h w c -> (b t) c h w'), (H // 4, W // 4))
    time_features = rearrange(time_features, '(b t) c h w -> b t h w c', b=B, t=T)
    show_grid("time_features init", time_features, "/workspace/xuannv/aef_reference/debug_grid_time_init.png")
    
    precision_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(x_proj, 'b t h w c -> (b t) c h w'), (H, W))
    precision_features = rearrange(precision_features, '(b t) c h w -> b t h w c', b=B, t=T)
    show_grid("precision_features init", precision_features, "/workspace/xuannv/aef_reference/debug_grid_prec_init.png")
    
    for i, block in enumerate(encoder.blocks):
        space_features, time_features, precision_features = block(
            space_features, time_features, precision_features, ts)
        if i == 0:
            show_grid("space_features after block1", space_features, "/workspace/xuannv/aef_reference/debug_grid_space_b1.png")
            show_grid("precision_features after block1", precision_features, "/workspace/xuannv/aef_reference/debug_grid_prec_b1.png")
    
    show_grid("space_features after all blocks", space_features, "/workspace/xuannv/aef_reference/debug_grid_space_final.png")
    show_grid("precision_features after all blocks", precision_features, "/workspace/xuannv/aef_reference/debug_grid_prec_final.png")
    
    space_global = space_features.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    show_grid("space_broadcast", space_broadcast, "/workspace/xuannv/aef_reference/debug_grid_space_broadcast.png")
    
    final_features = precision_features + space_broadcast
    show_grid("final_features (no spatial_fusion)", final_features, "/workspace/xuannv/aef_reference/debug_grid_final_nosf.png")
    
    final_2d = rearrange(final_features, 'b t h w c -> (b t) c h w')
    final_2d = encoder.spatial_fusion(final_2d)
    final_features = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
    show_grid("final_features (with spatial_fusion)", final_features, "/workspace/xuannv/aef_reference/debug_grid_final_sf.png")
    
    feats = encoder.norm(final_features)
    show_grid("encoder output", feats, "/workspace/xuannv/aef_reference/debug_grid_encoder_out.png")
    
    q = torch.randn(B, 64)
    z = time_pool(feats, q, mask=None)
    show_grid("time_pool output", z, "/workspace/xuannv/aef_reference/debug_grid_timepool.png")
    
    z_centered = z - z.mean(dim=2, keepdim=True)
    show_grid("time_pool + row center", z_centered, "/workspace/xuannv/aef_reference/debug_grid_timepool_rc.png")
