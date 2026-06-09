"""Debug correlation chain in summarizer"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

def row_mean_corr(t):
    """t: (H, W, C) or (H, W)"""
    if t.ndim == 3:
        rm = t.mean(dim=1)  # (H, C)
    else:
        rm = t.mean(dim=1)  # (H,)
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

checkpoint_path = "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000200_seed42.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["model_state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)

device = "npu:0"
model = model.to(device)
model.eval()

B, T, H, W = 1, 4, 128, 128
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
    z = model.summarizer.time_pool(feats_smooth, q, mask=None)
    
    print("=== Correlation chain ===")
    print(f"feats_smooth row_mean_corr: {row_mean_corr(feats_smooth[0, 0]):.4f}")
    print(f"z (before center) row_mean_corr: {row_mean_corr(z[0]):.4f}")
    
    z_centered = z - z.mean(dim=2, keepdim=True)
    print(f"z (after center) row_mean_corr: {row_mean_corr(z_centered[0]):.4f}")
    
    mu = model.summarizer.proj_64(z)
    print(f"mu (from z, no center) row_mean_corr: {row_mean_corr(mu[0]):.4f}")
    
    mu_centered = model.summarizer.proj_64(z_centered)
    print(f"mu (from z_centered) row_mean_corr: {row_mean_corr(mu_centered[0]):.4f}")
    
    # Also check what the actual forward returns
    out = model(source_data, timestamps, valid_periods)
    mu_s = out['student_embeddings'][0]
    print(f"\nActual student embedding row_mean_corr: {row_mean_corr(mu_s):.4f}")
    
    # Check row means std
    for name, t in [("z", z[0]), ("z_centered", z_centered[0]), ("mu", mu[0]), ("mu_centered", mu_centered[0]), ("mu_s", mu_s)]:
        rm = t.mean(dim=1)  # (H, C)
        print(f"{name} row means: mean={rm.mean():.4f}, std={rm.std():.4f}")
