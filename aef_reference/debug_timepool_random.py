"""Debug TimePooling with random init model"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

def row_mean_corr(t):
    if t.ndim == 3:
        rm = t.mean(dim=1)
    else:
        rm = t.mean(dim=1)
    if rm.shape[0] < 2:
        return 0.0
    rm_np = rm.cpu().numpy() if hasattr(rm, 'cpu') else rm
    if rm_np.ndim == 2:
        corrs = []
        for c in range(rm_np.shape[1]):
            col = rm_np[:, c]
            if np.std(col) < 1e-8:
                continue
            corr = np.corrcoef(col[:-1], col[1:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
        return float(np.mean(corrs)) if corrs else 0.0
    else:
        if np.std(rm_np) < 1e-8:
            return 0.0
        return float(np.corrcoef(rm_np[:-1], rm_np[1:])[0, 1])

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
    print(f"feats_smooth row_mean_corr: {row_mean_corr(feats_smooth[0, 0]):.4f}")
    
    z = model.summarizer.time_pool(feats_smooth, q, mask=None)
    print(f"z (TimePooling output) row_mean_corr: {row_mean_corr(z[0]):.4f}")
    
    # Break down TimePooling
    B, T, H, W, C = feats_smooth.shape
    BHW = B * H * W
    x_in = feats_smooth.view(BHW, T, C)
    kv = model.summarizer.time_pool.kv(x_in).view(BHW, T, 2, 8, C // 8)
    K, V = kv[:, :, 0], kv[:, :, 1]
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    
    qh = model.summarizer.time_pool.q_proj(q).view(B, 8, C // 8)
    qh = qh.unsqueeze(1).expand(B, H * W, 8, C // 8).reshape(BHW, 8, 1, C // 8)
    
    logits = (qh * K).sum(-1) / ((C // 8) ** 0.5)
    logits = logits.squeeze(2)
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
    
    print(f"K row_mean_corr: {row_mean_corr(K[0, 0]):.4f}")
    print(f"V row_mean_corr: {row_mean_corr(V[0, 0]):.4f}")
    print(f"attn (first head) row_mean_corr: {row_mean_corr(attn[:, 0, :]):.4f}")
    
    # Check if attn is uniform across (h,w)
    attn_reshaped = attn.view(B, H, W, 8, T)
    print(f"attn std across (h,w) for first head, first time: {attn_reshaped[0, :, :, 0, 0].std():.6f}")
    print(f"attn mean across (h,w) for first head, first time: {attn_reshaped[0, :, :, 0, 0].mean():.6f}")
    
    # Check z composition
    z2 = torch.einsum('bht,bhtd->bhd', attn, V)
    print(f"z2 (before out) row_mean_corr: {row_mean_corr(z2.view(B, H, W, C)[0]):.4f}")
    
    z3 = model.summarizer.time_pool.out(z2).view(B, H, W, C)
    print(f"z3 (after out) row_mean_corr: {row_mean_corr(z3[0]):.4f}")
