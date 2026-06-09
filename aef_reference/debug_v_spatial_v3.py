"""Debug V spatial correlation - simple version"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

def spatial_row_mean_corr_simple(t, H, W):
    """t: (BHW, ...) or (B, ..., H, W, C)"""
    if t.shape[0] == H * W or t.shape[0] % (H * W) == 0:
        B = t.shape[0] // (H * W)
        rest = t.shape[1:]
        t_r = t.view(B, H, W, *rest)
    else:
        # Try to find H and W in the shape
        return 0.0
    
    # Row mean over W dimension
    rm = t_r.mean(dim=2)  # (B, H, ...)
    rm_np = rm.detach().cpu().numpy()
    
    # Flatten to (B, H, -1)
    B_val, H_val = rm_np.shape[0], rm_np.shape[1]
    rm_flat = rm_np.reshape(B_val, H_val, -1)
    
    corrs = []
    for b in range(B_val):
        for c in range(rm_flat.shape[2]):
            col = rm_flat[b, :, c]
            if np.std(col) < 1e-8:
                continue
            corr = np.corrcoef(col[:-1], col[1:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
    return float(np.mean(corrs)) if corrs else 0.0

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
    print(f"feats_smooth spatial_row_mean_corr: {spatial_row_mean_corr_simple(feats_smooth.view(B2*T2*H2*W2, C2), H2, W2):.4f}")
    
    # Manually run TimePooling components
    B3, T3, H3, W3, C3 = feats_smooth.shape
    BHW = B3 * H3 * W3
    x_in = feats_smooth.view(BHW, T3, C3)
    kv = model.summarizer.time_pool.kv(x_in).view(BHW, T3, 2, 8, C3 // 8)
    K, V = kv[:, :, 0], kv[:, :, 1]
    K = K.permute(0, 2, 1, 3)  # (BHW, heads, T, d)
    V = V.permute(0, 2, 1, 3)  # (BHW, heads, T, d)
    
    print(f"K spatial_row_mean_corr: {spatial_row_mean_corr_simple(K, H3, W3):.4f}")
    print(f"V spatial_row_mean_corr: {spatial_row_mean_corr_simple(V, H3, W3):.4f}")
    
    qh = model.summarizer.time_pool.q_proj(q).view(B3, 8, C3 // 8)
    qh = qh.unsqueeze(1).expand(B3, H3 * W3, 8, C3 // 8).reshape(BHW, 8, 1, C3 // 8)
    
    logits = (qh * K).sum(-1) / ((C3 // 8) ** 0.5)
    logits = logits.squeeze(2)  # (BHW, heads, T)
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
    
    print(f"attn spatial_row_mean_corr: {spatial_row_mean_corr_simple(attn, H3, W3):.4f}")
    
    z2 = torch.einsum('bht,bhtd->bhd', attn, V)  # (BHW, heads, d)
    print(f"z2 spatial_row_mean_corr: {spatial_row_mean_corr_simple(z2, H3, W3):.4f}")
    
    # Check if self.kv weight has row structure
    W_kv = model.summarizer.time_pool.kv.weight  # (2*C, C)
    W_np = W_kv.detach().cpu().numpy()
    corrs = []
    for i in range(W_np.shape[0] - 1):
        corr = np.corrcoef(W_np[i], W_np[i+1])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    print(f"\nW_kv adjacent row corr: {np.mean(corrs):.4f}")
    
    # Check x_in correlation
    print(f"x_in spatial_row_mean_corr: {spatial_row_mean_corr_simple(x_in, H3, W3):.4f}")
