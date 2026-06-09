"""Debug z2 row structure"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

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
    
    # Run TimePooling
    B3, T3, H3, W3, C3 = feats_smooth.shape
    BHW = B3 * H3 * W3
    x_in = feats_smooth.view(BHW, T3, C3)
    kv = model.summarizer.time_pool.kv(x_in).view(BHW, T3, 2, 8, C3 // 8)
    K, V = kv[:, :, 0], kv[:, :, 1]
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    
    qh = model.summarizer.time_pool.q_proj(q).view(B3, 8, C3 // 8)
    qh = qh.unsqueeze(1).expand(B3, H3 * W3, 8, C3 // 8).reshape(BHW, 8, 1, C3 // 8)
    
    logits = (qh * K).sum(-1) / ((C3 // 8) ** 0.5)
    logits = logits.squeeze(2)
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
    
    z2 = torch.einsum('bht,bhtd->bhd', attn, V)
    z2_r = z2.view(B3, H3, W3, 8, C3 // 8)
    
    # Compare adjacent rows
    print("=== z2 adjacent row comparison ===")
    for h in [0, 1, 63, 64, 126]:
        row_h = z2_r[0, h, :, :, :]
        row_hp1 = z2_r[0, h+1, :, :, :]
        diff = (row_h - row_hp1).abs().mean()
        sim = torch.cosine_similarity(row_h.reshape(-1), row_hp1.reshape(-1), dim=0)
        print(f"  rows {h}-{h+1}: abs_diff={diff:.6f}, cosine_sim={sim:.4f}")
    
    # Check if z2 is actually constant within each row
    print("\n=== z2 within-row variation ===")
    for h in [0, 64, 127]:
        row = z2_r[0, h, :, :, :]
        within_row_std = row.std(dim=0).mean()  # std across W, averaged over channels
        print(f"  row {h}: within-row std={within_row_std:.6f}")
    
    # Check V within-row variation
    V_r = V.reshape(B3, H3, W3, 8, T3, C3 // 8)
    print("\n=== V within-row variation ===")
    for h in [0, 64, 127]:
        row = V_r[0, h, :, :, :, :]
        within_row_std = row.std(dim=0).mean()
        print(f"  row {h}: within-row std={within_row_std:.6f}")
    
    # Check if V is smooth in H direction
    print("\n=== V smoothness in H direction ===")
    V_per_h = V.reshape(B3, H3, W3, -1)
    for h in [0, 1, 2, 63, 64, 65]:
        print(f"  V row {h}: mean={V_per_h[0, h, :, :].mean():.6f}, std={V_per_h[0, h, :, :].std():.6f}")
    
    # Check attention within-row variation
    attn_r = attn.reshape(B3, H3, W3, 8, T3)
    print("\n=== attn within-row variation ===")
    for h in [0, 64, 127]:
        row = attn_r[0, h, :, :, :]
        within_row_std = row.std(dim=0).mean()
        print(f"  row {h}: within-row std={within_row_std:.6f}")
    
    # Check if the issue is that attn is nearly constant across (h,w)
    print("\n=== attn global stats ===")
    print(f"  attn mean: {attn.mean():.6f}")
    print(f"  attn std: {attn.std():.6f}")
    print(f"  attn min: {attn.min():.6f}")
    print(f"  attn max: {attn.max():.6f}")
    
    # Key insight: if attn is constant, z2 = sum(attn_t * V_t)
    # The correlation comes from V itself
    # Let's check V's row mean correlation directly
    V_rm = V.reshape(B3, H3, W3, 8, T3, C3 // 8).mean(dim=2)  # (B, H, 8, T, d)
    print("\n=== V row mean correlation ===")
    V_rm_np = V_rm.detach().cpu().numpy()
    B_val, H_val = V_rm_np.shape[0], V_rm_np.shape[1]
    V_rm_flat = V_rm_np.reshape(B_val, H_val, -1)
    corrs = []
    for b in range(B_val):
        for c in range(V_rm_flat.shape[2]):
            col = V_rm_flat[b, :, c]
            if np.std(col) < 1e-8:
                continue
            corr = np.corrcoef(col[:-1], col[1:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
    print(f"  V row mean corr: {np.mean(corrs):.4f}")
