"""Check every step of time_pool - simplified"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.aef_module import TimePooling
from einops import rearrange

def rmc(x):
    """row mean corr for (B,H,W,C) or (B,T,H,W,C)"""
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
    B2, T2, H2, W2, C2 = feats.shape
    BHW = B2 * H2 * W2
    
    x_tp = feats.view(BHW, T2, C2)
    print(f"input: {rmc(feats):.4f}")
    
    kv = time_pool.kv(x_tp)
    print(f"after kv: {rmc(kv.view(B2, T2, H2, W2, 128)):.4f}")
    
    kv_rs = kv.view(BHW, T2, 2, 8, 8)
    K = kv_rs[:, :, 0].permute(0, 2, 1, 3)  # (BHW, 8, T, 8)
    V = kv_rs[:, :, 1].permute(0, 2, 1, 3)
    
    K_spatial = K.permute(1, 2, 3, 0).reshape(8, T2, 8, H2, W2).permute(0, 1, 4, 3, 2)
    print(f"after K: {rmc(K_spatial):.4f}")
    
    V_spatial = V.permute(1, 2, 3, 0).reshape(8, T2, 8, H2, W2).permute(0, 1, 4, 3, 2)
    print(f"after V: {rmc(V_spatial):.4f}")
    
    qh = time_pool.q_proj(q).view(B2, 8, 8)
    qh = qh.unsqueeze(1).expand(B2, H2*W2, 8, 8).reshape(BHW, 8, 1, 8)
    
    logits = (qh * K).sum(-1) / (8 ** 0.5)
    logits = logits.squeeze(2)  # (BHW, 8, T)
    logits_spatial = logits.permute(1, 2, 0).reshape(8, T2, H2, W2).unsqueeze(0)
    print(f"after logits: {rmc(logits_spatial):.4f}")
    
    attn = torch.softmax(logits, dim=-1)
    attn_spatial = attn.permute(1, 2, 0).reshape(8, T2, H2, W2).unsqueeze(0)
    print(f"after attn: {rmc(attn_spatial):.4f}")
    
    z = torch.einsum('bht,bhtd->bhd', attn, V)  # (BHW, 8, 8)
    z_spatial = z.permute(1, 2, 0).reshape(8, 8, H2, W2).unsqueeze(0)
    print(f"after einsum: {rmc(z_spatial):.4f}")
    
    z_flat = z.reshape(BHW, 64)
    z_out = time_pool.out(z_flat)
    print(f"after out: {rmc(z_out.view(B2, H2, W2, 64)):.4f}")
