"""Test row centering without normalization"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.aef_module import TimePooling
from einops import rearrange

def rmc(x):
    x = x.detach().cpu().float()
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 3:
        x = x.permute(2, 0, 1)
    C, H, W = x.shape
    rm = x.mean(dim=2)
    rm_mean = rm.mean(dim=1, keepdim=True)
    rm_std = rm.std(dim=1, keepdim=True).clamp(min=1e-8)
    rm_norm = (rm - rm_mean) / rm_std
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            corrs.append(rm_norm[c, h].item() * rm_norm[c, h+1].item())
    return np.mean(corrs)

torch.manual_seed(42)
encoder = STPEncoder(input_channels=32, d_s=512, d_t=256, d_p=64, num_blocks=6)
time_pool = TimePooling(dim=64, num_heads=8)

x = torch.randn(1, 4, 128, 128, 32)
ts = torch.rand(1, 4)

with torch.no_grad():
    B, T, H, W, C = x.shape
    x_proj = encoder.input_projection(x)
    space_features = encoder.space_projection(x_proj)
    space_features = torch.nn.functional.adaptive_avg_pool2d(rearrange(space_features, 'b t h w c -> (b t) c h w'), (H // 8, W // 8))
    space_features = rearrange(space_features, '(b t) c h w -> b t h w c', b=B, t=T)
    time_features = encoder.time_projection(x_proj)
    time_features = torch.nn.functional.adaptive_avg_pool2d(rearrange(time_features, 'b t h w c -> (b t) c h w'), (H // 4, W // 4))
    time_features = rearrange(time_features, '(b t) c h w -> b t h w c', b=B, t=T)
    precision_features = torch.nn.functional.adaptive_avg_pool2d(rearrange(x_proj, 'b t h w c -> (b t) c h w'), (H, W))
    precision_features = rearrange(precision_features, '(b t) c h w -> b t h w c', b=B, t=T)
    for block in encoder.blocks:
        space_features, time_features, precision_features = block(space_features, time_features, precision_features, ts)
    space_global = space_features.mean(dim=(2, 3))
    space_ctx = encoder.space_to_precision(space_global)
    space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    time_global = time_features.mean(dim=(2, 3))
    time_ctx = encoder.time_to_precision(time_global)
    time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 64)
    final_features = precision_features + space_broadcast + time_broadcast
    final_2d = rearrange(final_features, 'b t h w c -> (b t) c h w')
    final_2d = encoder.spatial_fusion(final_2d)
    final_features = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
    feats = encoder.norm(final_features)

q = torch.randn(B, 64)

with torch.no_grad():
    z = time_pool(feats, q, mask=None)
    print(f"Normal: {rmc(z):.4f}")
    
    # Row center only
    z_rc = z - z.mean(dim=2, keepdim=True)
    print(f"Row center: {rmc(z_rc):.4f}")
    
    # Row center + small noise
    z_rcn = z_rc + torch.randn_like(z_rc) * 0.01
    print(f"Row center + noise: {rmc(z_rcn):.4f}")
    
    # Also check if row center affects PCA RGB
    emb = z[0].cpu().numpy()
    emb_flat = emb.reshape(-1, 64)
    mean = emb_flat.mean(axis=0)
    centered = emb_flat - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    proj = emb_flat @ eigvecs[:, :3]
    for i in range(3):
        c = proj[:, i]
        proj[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
    rgb = proj.reshape(128, 128, 3)
    
    emb_rc = z_rc[0].cpu().numpy()
    emb_rc_flat = emb_rc.reshape(-1, 64)
    mean_rc = emb_rc_flat.mean(axis=0)
    centered_rc = emb_rc_flat - mean_rc
    cov_rc = np.cov(centered_rc.T)
    eigvals_rc, eigvecs_rc = np.linalg.eigh(cov_rc)
    idx_rc = np.argsort(eigvals_rc)[::-1]
    eigvecs_rc = eigvecs_rc[:, idx_rc]
    proj_rc = emb_rc_flat @ eigvecs_rc[:, :3]
    for i in range(3):
        c = proj_rc[:, i]
        proj_rc[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
    rgb_rc = proj_rc.reshape(128, 128, 3)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Normal")
    axes[0].axis('off')
    axes[1].imshow(rgb_rc)
    axes[1].set_title("Row Center")
    axes[1].axis('off')
    plt.savefig("/workspace/xuannv/aef_reference/debug_row_center_viz.png", dpi=150)
    print("Saved visualization")
