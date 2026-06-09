"""Debug row means of student embedding"""
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
    mu_t = out['embeddings'][0]  # (128, 128, 64)
    mu_s = out['student_embeddings'][0]
    
    print("Teacher row means (first 5 channels):")
    for i in range(5):
        rm = mu_t[:, :, i].mean(dim=1)  # (128,)
        print(f"  ch{i}: mean={rm.mean():.6f}, std={rm.std():.6f}, min={rm.min():.6f}, max={rm.max():.6f}")
    
    print("\nStudent row means (first 5 channels):")
    for i in range(5):
        rm = mu_s[:, :, i].mean(dim=1)
        print(f"  ch{i}: mean={rm.mean():.6f}, std={rm.std():.6f}, min={rm.min():.6f}, max={rm.max():.6f}")
    
    # Check z inside summarizer
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
    
    print("\nz (before row center) row means (first 5 channels):")
    for i in range(5):
        rm = z[0, :, :, i].mean(dim=1)
        print(f"  ch{i}: mean={rm.mean():.6f}, std={rm.std():.6f}")
    
    z_centered = z - z.mean(dim=2, keepdim=True)
    print("\nz (after row center) row means (first 5 channels):")
    for i in range(5):
        rm = z_centered[0, :, :, i].mean(dim=1)
        print(f"  ch{i}: mean={rm.mean():.6f}, std={rm.std():.6f}")
    
    mu_manual = model.summarizer.proj_64(z_centered)
    print("\nmu_manual row means (first 5 channels):")
    for i in range(5):
        rm = mu_manual[0, :, :, i].mean(dim=1)
        print(f"  ch{i}: mean={rm.mean():.6f}, std={rm.std():.6f}")
