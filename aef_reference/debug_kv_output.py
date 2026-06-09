"""Check kv output structure"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.aef_module import TimePooling
from einops import rearrange

def row_mean_corr(x):
    x = x.detach().cpu().float()
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        if x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
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
    space_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(space_features, 'b t h w c -> (b t) c h w'),
        (H // 8, W // 8)
    )
    space_features = rearrange(space_features, '(b t) c h w -> b t h w c', b=B, t=T)
    
    time_features = encoder.time_projection(x_proj)
    time_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(time_features, 'b t h w c -> (b t) c h w'),
        (H // 4, W // 4)
    )
    time_features = rearrange(time_features, '(b t) c h w -> b t h w c', b=B, t=T)
    
    precision_features = torch.nn.functional.adaptive_avg_pool2d(
        rearrange(x_proj, 'b t h w c -> (b t) c h w'),
        (H, W)
    )
    precision_features = rearrange(precision_features, '(b t) c h w -> b t h w c', b=B, t=T)
    
    for i, block in enumerate(encoder.blocks):
        space_features, time_features, precision_features = block(
            space_features, time_features, precision_features, ts
        )
    
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

print(f"feats row_mean_corr: {row_mean_corr(feats):.4f}")

q = torch.randn(B, 64)

with torch.no_grad():
    B2, T2, H2, W2, C2 = feats.shape
    BHW = B2 * H2 * W2
    
    x_tp = feats.view(BHW, T2, C2)
    print(f"x_tp shape: {x_tp.shape}")
    print(f"x_tp row_mean_corr: {row_mean_corr(x_tp.view(B2, T2, H2, W2, C2)):.4f}")
    
    kv = time_pool.kv(x_tp)
    print(f"kv shape: {kv.shape}")
    
    kv_reshaped = kv.view(BHW, T2, 2, time_pool.num_heads, time_pool.head_dim)
    K = kv_reshaped[:, :, 0]
    V = kv_reshaped[:, :, 1]
    
    K_spatial = K.view(B2, H2, W2, T2, time_pool.num_heads, time_pool.head_dim)
    V_spatial = V.view(B2, H2, W2, T2, time_pool.num_heads, time_pool.head_dim)
    
    # Check row_mean_corr of K and V for each head
    for head in range(time_pool.num_heads):
        K_head = K_spatial[0, :, :, :, head, :].reshape(H2, W2, T2 * time_pool.head_dim)
        K_head = K_head.permute(2, 0, 1)
        print(f"K head {head} row_mean_corr: {row_mean_corr(K_head.unsqueeze(0).unsqueeze(0)):.4f}")
        
        V_head = V_spatial[0, :, :, :, head, :].reshape(H2, W2, T2 * time_pool.head_dim)
        V_head = V_head.permute(2, 0, 1)
        print(f"V head {head} row_mean_corr: {row_mean_corr(V_head.unsqueeze(0).unsqueeze(0)):.4f}")
        
    # Check qh
    qh = time_pool.q_proj(q).view(B2, time_pool.num_heads, time_pool.head_dim)
    print(f"\nqh shape: {qh.shape}")
    
    # Check logits
    K_perm = K.permute(0, 2, 1, 3)  # (BHW, heads, T, d)
    qh_expanded = qh.unsqueeze(1).expand(B2, H2 * W2, time_pool.num_heads, time_pool.head_dim).reshape(BHW, time_pool.num_heads, 1, time_pool.head_dim)
    logits = (qh_expanded * K_perm).sum(-1) / (time_pool.head_dim ** 0.5)
    logits = logits.squeeze(2)  # (BHW, heads, T)
    
    logits_spatial = logits.view(B2, H2, W2, time_pool.num_heads, T2)
    for head in range(time_pool.num_heads):
        log_head = logits_spatial[0, :, :, head, :].reshape(H2, W2, T2)
        log_head = log_head.permute(2, 0, 1)
        print(f"logits head {head} row_mean_corr: {row_mean_corr(log_head.unsqueeze(0).unsqueeze(0)):.4f}")
    
    # Check attn
    attn = torch.softmax(logits, dim=-1)
    attn_spatial = attn.view(B2, H2, W2, time_pool.num_heads, T2)
    for head in range(time_pool.num_heads):
        att_head = attn_spatial[0, :, :, head, :].reshape(H2, W2, T2)
        att_head = att_head.permute(2, 0, 1)
        print(f"attn head {head} row_mean_corr: {row_mean_corr(att_head.unsqueeze(0).unsqueeze(0)):.4f}")
