"""Test full model output y-corr (FIXED)"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
from src.aef.architecture.aef_module import AlphaEarthFoundations

def y_corr(x):
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        if x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
            x = x.permute(2, 0, 1)
    elif x.dim() == 3:
        if x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
            x = x.permute(2, 0, 1)
    x = x.detach().cpu().float()
    C, H, W = x.shape
    if H < 2:
        return 0.0
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

# Create model
model = AlphaEarthFoundations(
    model_size="small",
    input_sources={"s1": 2, "s2": 6, "landsat": 6},
    decode_sources={"s2": 6},
    per_source_latent=32,
)

# Random input
B, T, H, W = 1, 4, 128, 128
source_data = {
    "s1": torch.randn(B, T, H, W, 2),
    "s2": torch.randn(B, T, H, W, 6),
    "landsat": torch.randn(B, T, H, W, 6),
}
timestamps = {"s1": torch.rand(B, T), "s2": torch.rand(B, T), "landsat": torch.rand(B, T)}
valid_periods = [(0.0, 1.0)]

with torch.no_grad():
    out = model(source_data, timestamps, valid_periods)
    
    embeddings = out['embeddings']  # (B, H, W, 64)
    print(f"embeddings shape: {embeddings.shape}")
    print(f"embeddings y-corr: {y_corr(embeddings):.4f}")
    
    # Check each channel's spatial std
    emb = embeddings[0].cpu().numpy()  # (128, 128, 64)
    print(f"embeddings per-channel spatial std: mean={emb.std(axis=(0,1)).mean():.4f}")
    
    # Check PCA
    emb_flat = emb.reshape(-1, 64)
    mean = emb_flat.mean(axis=0)
    centered = emb_flat - mean
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total_var = eigvals.sum()
    print(f"PCA explained variance ratios: {eigvals[:5] / total_var}")
    print(f"PCA top-3 cumulative: {eigvals[:3].sum() / total_var:.4f}")
