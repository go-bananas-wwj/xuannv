"""Debug feats_smooth row structure"""
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
    
    print("=== feats_smooth row structure ===")
    
    # Check adjacent row similarity for each time frame
    for t in range(T2):
        f_t = feats_smooth[0, t, :, :, :]  # (H, W, C)
        sims = []
        for h in range(H2 - 1):
            row_h = f_t[h, :, :].reshape(-1)
            row_hp1 = f_t[h+1, :, :].reshape(-1)
            sim = torch.cosine_similarity(row_h, row_hp1, dim=0)
            sims.append(sim.item())
        print(f"  Time {t}: adjacent row cos_sim mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")
    
    # Check within-row std for each time frame
    for t in range(T2):
        f_t = feats_smooth[0, t, :, :, :]
        within_row_stds = []
        for h in range(H2):
            row_std = f_t[h, :, :].std()
            within_row_stds.append(row_std.item())
        print(f"  Time {t}: within-row std mean={np.mean(within_row_stds):.4f}, std={np.std(within_row_stds):.4f}")
    
    # Now check x_in (same data, different view)
    x_in = feats_smooth.view(B2 * H2 * W2, T2, C2)
    
    # Check if x_in has the same row structure
    print("\n=== x_in row structure ===")
    x_in_r = x_in.reshape(B2, H2, W2, T2, C2)
    for t in range(T2):
        x_t = x_in_r[0, :, :, t, :]  # (H, W, C)
        sims = []
        for h in range(H2 - 1):
            row_h = x_t[h, :, :].reshape(-1)
            row_hp1 = x_t[h+1, :, :].reshape(-1)
            sim = torch.cosine_similarity(row_h, row_hp1, dim=0)
            sims.append(sim.item())
        print(f"  Time {t}: adjacent row cos_sim mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")
    
    # Now check V
    kv = model.summarizer.time_pool.kv(x_in).view(B2 * H2 * W2, T2, 2, 8, C2 // 8)
    K, V = kv[:, :, 0], kv[:, :, 1]
    V = V.permute(0, 2, 1, 3)  # (BHW, heads, T, d)
    
    print("\n=== V row structure ===")
    V_r = V.reshape(B2, H2, W2, 8, T2, C2 // 8)
    for h_idx in [0, 64]:
        # Compare all w positions for this row
        row_h = V_r[0, h_idx, :, :, :, :]  # (W, 8, T, d)
        row_hp1 = V_r[0, h_idx + 1, :, :, :, :]
        sim = torch.cosine_similarity(row_h.reshape(-1), row_hp1.reshape(-1), dim=0)
        print(f"  Rows {h_idx}-{h_idx+1}: cos_sim={sim:.4f}")
    
    # Check V within-row std
    V_flat = V.reshape(B2, H2, W2, -1)
    for h in [0, 64, 127]:
        within_row_std = V_flat[0, h, :, :].std(dim=0).mean()
        print(f"  Row {h}: within-row std={within_row_std:.4f}")
    
    # Check if V is actually constant across W
    print("\n=== V across-W variation ===")
    for h in [0, 64]:
        row = V_flat[0, h, :, :]
        # Compute std for each w position across all channels
        w_stds = row.std(dim=1)
        print(f"  Row {h}: per-w std mean={w_stds.mean():.4f}, std={w_stds.std():.4f}")
