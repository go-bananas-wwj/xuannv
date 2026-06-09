"""Analyze PCA structure of student embeddings"""
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
    emb = out['student_embeddings'][0].cpu().numpy()  # (128, 128, 64)

# PCA
emb_flat = emb.reshape(-1, 64)
mean = emb_flat.mean(axis=0)
centered = emb_flat - mean
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# Project to top 3
proj = centered @ eigvecs[:, :3]

# Check row mean of each PC
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(3):
    pc = proj[:, i].reshape(128, 128)
    
    # Show PC image
    axes[0, i].imshow(pc, cmap='viridis')
    axes[0, i].set_title(f'PC {i+1} (var={eigvals[idx[i]]/eigvals.sum():.4f})')
    axes[0, i].axis('off')
    
    # Show row mean
    row_means = pc.mean(axis=1)
    axes[1, i].plot(row_means)
    axes[1, i].set_title(f'PC {i+1} row means')
    axes[1, i].set_xlabel('Row')
    axes[1, i].set_ylabel('Mean')
    axes[1, i].grid(True)

plt.tight_layout()
plt.savefig("/workspace/xuannv/aef_reference/debug_pca_structure.png", dpi=150)
print("Saved debug_pca_structure.png")

# Check if row means have periodic structure
print("\nRow mean stats:")
for i in range(3):
    pc = proj[:, i].reshape(128, 128)
    row_means = pc.mean(axis=1)
    print(f"PC {i+1}: min={row_means.min():.4f}, max={row_means.max():.4f}, std={row_means.std():.4f}")
    
    # Check autocorrelation of row means
    corr = np.correlate(row_means - row_means.mean(), row_means - row_means.mean(), mode='full')
    corr = corr[len(corr)//2:]
    corr = corr / corr[0]
    print(f"  Autocorr (lag 1-5): {corr[1:6]}")

# Also check column means
print("\nColumn mean stats:")
for i in range(3):
    pc = proj[:, i].reshape(128, 128)
    col_means = pc.mean(axis=0)
    print(f"PC {i+1}: min={col_means.min():.4f}, max={col_means.max():.4f}, std={col_means.std():.4f}")
