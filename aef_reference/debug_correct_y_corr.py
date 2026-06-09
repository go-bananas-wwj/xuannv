"""Test corrected y-corr that detects row-mean patterns"""
import torch
import numpy as np

def y_corr_old(x):
    """Old: per-row std then correlate"""
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    x = x.detach().cpu().float()
    C, H, W = x.shape
    row_means = x.mean(dim=2, keepdim=True)
    row_stds = x.std(dim=2, keepdim=True).clamp(min=1e-8)
    xn = (x - row_means) / row_stds
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            row1 = xn[c, h]
            row2 = xn[c, h + 1]
            corr = (row1 * row2).sum() / (row1.norm() * row2.norm() + 1e-8)
            corrs.append(corr.item())
    return np.mean(corrs)

def row_mean_corr(x):
    """New: correlate row means"""
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    x = x.detach().cpu().float()
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
            # Single value correlation = v1*v2 (since norms are 1)
            corrs.append(v1 * v2)
    return np.mean(corrs)

def global_row_corr(x):
    """Even simpler: just check if all rows in all channels are correlated"""
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    x = x.detach().cpu().float()
    C, H, W = x.shape
    # Compute row means for all channels
    rm = x.mean(dim=2)  # (C, H)
    # Flatten all row means into one vector
    rm_flat = rm.reshape(-1)
    # Compute autocorrelation with lag 1 (adjacent rows, same channel)
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            v1 = rm[c, h].item()
            v2 = rm[c, h + 1].item()
            corrs.append(v1 * v2)
    return np.mean(corrs)

# Test with synthetic stripe pattern
stripes = torch.zeros(1, 1, 128, 128, 64)
for h in range(128):
    stripes[0, 0, h, :, :] = (h // 32) * 0.25  # 4 horizontal bands

print("Synthetic stripes:")
print(f"  y_corr_old: {y_corr_old(stripes):.4f}")
print(f"  row_mean_corr: {row_mean_corr(stripes):.4f}")

# Test with constant within row
constant_rows = torch.zeros(1, 1, 128, 128, 64)
for h in range(128):
    constant_rows[0, 0, h, :, :] = h / 128.0

print("\nConstant within rows:")
print(f"  y_corr_old: {y_corr_old(constant_rows):.4f}")
print(f"  row_mean_corr: {row_mean_corr(constant_rows):.4f}")

# Test with random
noise = torch.randn(1, 1, 128, 128, 64)
print("\nRandom noise:")
print(f"  y_corr_old: {y_corr_old(noise):.4f}")
print(f"  row_mean_corr: {row_mean_corr(noise):.4f}")

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
    print(f"  y_corr_old: {y_corr_old(embeddings):.4f}")
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
    
    print(f"\nRandom init:")
    print(f"  y_corr_old: {y_corr_old(emb_rand):.4f}")
    print(f"  row_mean_corr: {row_mean_corr(emb_rand):.4f}")
