"""Debug V spatial correlation"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

def spatial_row_mean_corr(t, H, W):
    """t: (BHW, ...) or flattened spatial tensor"""
    # Reshape to (B, H, W, ...)
    if t.shape[0] == H * W or t.shape[0] % (H * W) == 0:
        B = t.shape[0] // (H * W)
        rest = t.shape[1:]
        t_reshaped = t.view(B, H, W, *rest)
        # Compute row means: mean over W
        rm = t_reshaped.mean(dim=2)  # (B, H, ...)
        # Compute correlation between adjacent rows
        rm_np = rm.cpu().numpy() if hasattr(rm, 'cpu') else rm
        B_val = rm_np.shape[0]
        H_val = rm_np.shape[1]
        total_corr = []
        for b in range(B_val):
            # Flatten all dims except H for this batch
            rm_b = rm_np[b]  # (H, ...)
            # Reshape to (H, -1)
            rm_flat = rm_b.reshape(H_val, -1)
            corrs = []
            for c in range(rm_flat.shape[1]):
                col = rm_flat[:, c]
                if np.std(col) < 1e-8:
                    continue
                corr = np.corrcoef(col[:-1], col[1:])[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
            if corrs:
                total_corr.append(np.mean(corrs))
        return float(np.mean(total_corr)) if total_corr else 0.0
    return 0.0

input_sources = {"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4}
decode_sources = {"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4, "dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1}
model = AlphaEarthFoundations(
    model_size="small",
    input_sources=input_sources,
    decode_sources=decode_sources,
    per_source_latent=32,
    enable_text_align=False,
)

device = "npu:0"
model = model.to(device)
model.eval()

B, T, H, W = 1, 4, 128, 128
torch.manual_seed(42)
source_data = {
    "s1": torch.randn(B, T, H, W, 2, device=device),
    "s2": torch.randn(B, T, H, W, 6, device=device),
    "tianyi_sar": torch.randn(B, T, H, W, 1, device=device),
    "landsat": torch.randn(B, T, H, W, 6, device=device),
    "planet": torch.randn(B, T, H, W, 4, device=device),
}
timestamps = {k: torch.rand(B, T, device=device) for k in source_data.keys()}
valid_periods = [(0.0, 1.0)]

with torch.no_grad():
    x = model._stack_inputs(source_data)
    first_src = next(iter(model.input_sources.keys()))
    ts = timestamps[first_src]
    feats = model.encoder(x, ts)
    
    B2, T2, H2, W2, C2 = feats.shape
    feats_2d = feats.view(B2 * T2, H2, W2, C2).permute(0, 3, 1, 2).contiguous()
    feats_2d = model.summarizer.spatial_smooth(feats_2d)
    feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B2, T2, H2, W2, C2)
    
    vp = torch.tensor(valid_periods, device=device)
    q = model.summarizer.summarizer_q(vp)
    
    print("=== Random init model ===")
    print(f"feats_smooth spatial_row_mean_corr: {spatial_row_mean_corr(feats_smooth[0], H2, W2):.4f}")
    
    # Manually run TimePooling components
    B3, T3, H3, W3, C3 = feats_smooth.shape
    BHW = B3 * H3 * W3
    x_in = feats_smooth.view(BHW, T3, C3)
    kv = model.summarizer.time_pool.kv(x_in).view(BHW, T3, 2, 8, C3 // 8)
    K, V = kv[:, :, 0], kv[:, :, 1]
    K = K.permute(0, 2, 1, 3)  # (BHW, heads, T, d)
    V = V.permute(0, 2, 1, 3)  # (BHW, heads, T, d)
    
    print(f"K spatial_row_mean_corr: {spatial_row_mean_corr(K, H3, W3):.4f}")
    print(f"V spatial_row_mean_corr: {spatial_row_mean_corr(V, H3, W3):.4f}")
    
    qh = model.summarizer.time_pool.q_proj(q).view(B3, 8, C3 // 8)
    qh = qh.unsqueeze(1).expand(B3, H3 * W3, 8, C3 // 8).reshape(BHW, 8, 1, C3 // 8)
    
    logits = (qh * K).sum(-1) / ((C3 // 8) ** 0.5)
    logits = logits.squeeze(2)  # (BHW, heads, T)
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
    
    print(f"attn spatial_row_mean_corr: {spatial_row_mean_corr(attn, H3, W3):.4f}")
    
    z2 = torch.einsum('bht,bhtd->bhd', attn, V)  # (BHW, heads, d)
    print(f"z2 spatial_row_mean_corr: {spatial_row_mean_corr(z2, H3, W3):.4f}")
    
    # Check if z2 is actually constant across rows
    z2_reshaped = z2.view(B3, H3, W3, 8, C3 // 8)
    for h in [0, 1, 2, 63, 64, 65, 126, 127]:
        row_mean = z2_reshaped[0, h, :, :, :].mean()
        row_std = z2_reshaped[0, h, :, :, :].std()
        print(f"  z2 row {h}: mean={row_mean:.6f}, std={row_std:.6f}")
