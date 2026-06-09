"""Test corrected y-corr that detects row-mean patterns"""
import torch
import numpy as np

def row_mean_corr(x):
    """Correlate row means across channels"""
    x = x.detach().cpu().float()
    if x.dim() == 4:
        # (B, H, W, C)
        x = x[0]  # (H, W, C)
        x = x.permute(2, 0, 1)  # (C, H, W)
    elif x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    C, H, W = x.shape
    rm = x.mean(dim=2)  # (C, H)
    rm_mean = rm.mean(dim=1, keepdim=True)
    rm_std = rm.std(dim=1, keepdim=True).clamp(min=1e-8)
    rm_norm = (rm - rm_mean) / rm_std
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            v1 = rm_norm[c, h].item()
            v2 = rm_norm[c, h + 1].item()
            corrs.append(v1 * v2)
    return np.mean(corrs)

# Test with synthetic stripe pattern
stripes = torch.zeros(1, 1, 128, 128, 64)
for h in range(128):
    stripes[0, 0, h, :, :] = (h // 32) * 0.25

print("Synthetic stripes:")
print(f"  row_mean_corr: {row_mean_corr(stripes):.4f}")

noise = torch.randn(1, 1, 128, 128, 64)
print(f"Random noise: {row_mean_corr(noise):.4f}")

# Now test with actual model
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")
from src.aef.architecture.aef_module import AlphaEarthFoundations
import torch_npu

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
    embeddings = out['embeddings']
    
    print(f"\nStep 200 checkpoint:")
    print(f"  row_mean_corr: {row_mean_corr(embeddings):.4f}")
    
    # Compare with random init
    model_rand = AlphaEarthFoundations(
        model_size="small",
        input_sources=input_sources,
        decode_sources=decode_sources,
        per_source_latent=32,
        enable_text_align=False,
    ).to(device)
    with torch.no_grad():
        out_rand = model_rand(source_data, timestamps, valid_periods)
        emb_rand = out_rand['embeddings']
    
    print(f"Random init: {row_mean_corr(emb_rand):.4f}")
    
    # Also check encoder output
    x = model._stack_inputs(source_data)
    first_src = next(iter(model.input_sources.keys()))
    ts = timestamps[first_src]
    feats = model.encoder(x, ts)
    print(f"Encoder output (step 200): {row_mean_corr(feats):.4f}")
    
    # Random init encoder
    feats_rand = model_rand.encoder(x, ts)
    print(f"Encoder output (random): {row_mean_corr(feats_rand):.4f}")
