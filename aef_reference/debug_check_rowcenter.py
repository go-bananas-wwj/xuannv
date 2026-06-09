"""Check if row centering is effective in trained model"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

def rmc(x):
    x = x.detach().cpu().float()
    if x.dim() == 4:
        x = x[0]
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
    out = model(source_data, timestamps, valid_periods)
    
    print(f"teacher_embeddings row_mean_corr: {rmc(out['embeddings']):.4f}")
    print(f"student_embeddings row_mean_corr: {rmc(out['student_embeddings']):.4f}")
    
    # Check if row centering is in the code
    import inspect
    source = inspect.getsource(model.summarizer.forward)
    if "z = z - z.mean(dim=2, keepdim=True)" in source:
        print("\nRow centering IS present in code")
    else:
        print("\nRow centering IS NOT present in code")
    
    # Manually run summarizer to verify
    x = model._stack_inputs(source_data)
    first_src = next(iter(model.input_sources.keys()))
    ts = timestamps[first_src]
    feats = model.encoder(x, ts)
    
    # Before summarizer
    print(f"\nencoder output row_mean_corr: {rmc(feats):.4f}")
    
    # Inside summarizer step by step
    B2, T2, H2, W2, C2 = feats.shape
    feats_2d = feats.view(B2 * T2, H2, W2, C2).permute(0, 3, 1, 2).contiguous()
    feats_2d = model.summarizer.spatial_smooth(feats_2d)
    feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B2, T2, H2, W2, C2)
    
    q = model.summarizer.summarizer_q(torch.tensor(valid_periods, device=device))
    z = model.summarizer.time_pool(feats_smooth, q, mask=None)
    print(f"after time_pool: {rmc(z):.4f}")
    
    # Check if row centering is applied
    z_centered = z - z.mean(dim=2, keepdim=True)
    print(f"after manual row center: {rmc(z_centered):.4f}")
    
    # Actual summarizer output
    mu = model.summarizer(feats, ts, torch.tensor(valid_periods, device=device))
    print(f"actual summarizer output: {rmc(mu):.4f}")
