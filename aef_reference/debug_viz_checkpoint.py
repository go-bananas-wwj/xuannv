"""Generate PCA RGB visualization from step 200 checkpoint"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

# Build model
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

# Random input
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
    embeddings = out['embeddings'][0].cpu().numpy()  # (128, 128, 64)

# PCA to 3 channels
emb_flat = embeddings.reshape(-1, 64)
mean = emb_flat.mean(axis=0)
centered = emb_flat - mean
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# Project to 3D
proj = emb_flat @ eigvecs[:, :3]  # (16384, 3)

# Normalize each channel to [0, 1]
for i in range(3):
    c = proj[:, i]
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        proj[:, i] = (c - c_min) / (c_max - c_min)
    else:
        proj[:, i] = 0.5

rgb = proj.reshape(128, 128, 3)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb)
axes[0].set_title("Step 200 PCA RGB (1x1 conv fix)")
axes[0].axis('off')

# Also show AEF official style: compare with random init model
model_rand = AlphaEarthFoundations(
    model_size="small",
    input_sources=input_sources,
    decode_sources=decode_sources,
    per_source_latent=32,
    enable_text_align=False,
).to(device)
with torch.no_grad():
    out_rand = model_rand(source_data, timestamps, valid_periods)
    emb_rand = out_rand['embeddings'][0].cpu().numpy()

emb_flat_rand = emb_rand.reshape(-1, 64)
mean_rand = emb_flat_rand.mean(axis=0)
centered_rand = emb_flat_rand - mean_rand
cov_rand = np.cov(centered_rand.T)
eigvals_rand, eigvecs_rand = np.linalg.eigh(cov_rand)
idx_rand = np.argsort(eigvals_rand)[::-1]
eigvecs_rand = eigvecs_rand[:, idx_rand]
proj_rand = emb_flat_rand @ eigvecs_rand[:, :3]
for i in range(3):
    c = proj_rand[:, i]
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        proj_rand[:, i] = (c - c_min) / (c_max - c_min)
    else:
        proj_rand[:, i] = 0.5
rgb_rand = proj_rand.reshape(128, 128, 3)

axes[1].imshow(rgb_rand)
axes[1].set_title("Random Init PCA RGB")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("/workspace/xuannv/aef_reference/debug_viz_step200.png", dpi=150)
print("Saved to /workspace/xuannv/aef_reference/debug_viz_step200.png")
