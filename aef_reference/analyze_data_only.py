"""Analyze input sources and AEF embeddings without NPU."""
from __future__ import annotations
import os, sys
sys.path.insert(0, '/workspace/xuannv/aef_reference')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn

dataset = HaidianAEFDataset(
    data_root='/workspace/xuannv/data_raw/haidian/scenes',
    planet_root='/workspace/xuannv/data_raw/beijing/planetscene',
    stats_dir='/workspace/xuannv/statistics/haidian',
    cache_dir='/workspace/xuannv/aef_reference/src/aef/.cache',
    source_names=['s1','s2','tianyi_sar','landsat','planet'],
    required_sources=['s2'], split='train', train_ratio=0.9, seed=42,
    max_frames=16, start_date='20251201', end_date='20260430',
    aef_embedding_root='/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches',
)

patch_indices = [0, 10, 25, 50, 75, 100, 150, 200]
source_names = ['s1', 's2', 'tianyi_sar', 'landsat', 'planet']

print("="*70)
print("MULTI-PATCH INPUT SOURCE & AEF ANALYSIS")
print("="*70)

all_aef_zeros = []
all_aef_min = []
all_aef_y_corr = []
input_y_corr = {src: [] for src in source_names}

for idx in patch_indices:
    sample = dataset[idx]
    batch = collate_fn([sample])
    
    # AEF analysis
    aef_emb = batch['aef_embedding'][0].numpy()  # D, H, W
    aef_emb = np.transpose(aef_emb, (1, 2, 0))  # H, W, D
    aef_mag = np.linalg.norm(aef_emb, axis=-1)
    zeros = (aef_mag < 0.01).sum()
    all_aef_zeros.append(zeros)
    all_aef_min.append(aef_mag.min())
    
    H, W, D = aef_emb.shape
    max_corr = 0
    for ch in range(min(16, D)):
        for y in range(H):
            row = aef_emb[y, :, ch]
            if row.std() > 1e-6:
                for yy in range(y+1, H):
                    corr = np.corrcoef(row, aef_emb[yy, :, ch])[0,1]
                    if not np.isnan(corr):
                        max_corr = max(max_corr, abs(corr))
    all_aef_y_corr.append(max_corr)
    
    # Input source analysis
    source_data = batch['source_data']
    for src in source_names:
        data = source_data[src][0].numpy()  # T, H, W, C
        T, H3, W3, C = data.shape
        mid_t = min(T // 2, T - 1)
        frame = data[mid_t]
        max_corr3 = 0
        for ch in range(min(3, C)):
            for y in range(H3):
                row = frame[y, :, ch]
                if row.std() > 1e-6:
                    for yy in range(y+1, H3):
                        corr = np.corrcoef(row, frame[yy, :, ch])[0,1]
                        if not np.isnan(corr):
                            max_corr3 = max(max_corr3, abs(corr))
        input_y_corr[src].append(max_corr3)

print(f"\nAEF Embedding (8 patches):")
print(f"  Zero locations:     {all_aef_zeros}")
print(f"  Min magnitude:      {[f'{x:.4f}' for x in all_aef_min]}")
print(f"  Max y-correlation:  {[f'{x:.4f}' for x in all_aef_y_corr]}")
print(f"  Mean y-correlation: {np.mean(all_aef_y_corr):.4f}")

print(f"\nInput Source Y-Correlation (middle frame, max over channels):")
for src in source_names:
    vals = input_y_corr[src]
    print(f"  {src:12s}: {vals}  mean={np.mean(vals):.4f}")

# Visualization for first patch
sample = dataset[0]
batch = collate_fn([sample])
source_data = batch['source_data']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, src in enumerate(source_names):
    data = source_data[src][0].numpy()
    T, H, W, C = data.shape
    mid_t = min(T // 2, T - 1)
    frame = data[mid_t]
    
    rgb = np.zeros((H, W, 3))
    for c in range(min(3, C)):
        ch = frame[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax - vmin > 1e-6:
            rgb[:, :, c] = (ch - vmin) / (vmax - vmin)
    
    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f'{src} (T={T}, mid frame)')
    axes[0, i].axis('off')
    
    # Y-gradient plot
    if C > 0:
        y_grad = np.abs(frame[1:, :, 0] - frame[:-1, :, 0]).mean(axis=1)
        axes[1, i].plot(y_grad, range(len(y_grad)))
        axes[1, i].invert_yaxis()
        axes[1, i].set_title(f'{src} y-grad mean={np.mean(y_grad):.3f}')
        axes[1, i].set_xlabel('abs grad')

plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/multi_patch_inputs.png', dpi=150)
plt.close()
print("\nSaved multi_patch_inputs.png")

# AEF zero location map
aef_emb = batch['aef_embedding'][0].numpy()
aef_emb = np.transpose(aef_emb, (1, 2, 0))
aef_mag = np.linalg.norm(aef_emb, axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].imshow(aef_mag, cmap='viridis')
axes[0].set_title(f'AEF Magnitude\nmin={aef_mag.min():.4f}, max={aef_mag.max():.4f}')
plt.colorbar(im0, ax=axes[0])

axes[1].imshow(aef_mag < 0.01, cmap='hot')
axes[1].set_title(f'Zero Locations (red=zero)\ncount={(aef_mag < 0.01).sum()}')

# PCA RGB
flat = aef_emb.reshape(-1, aef_emb.shape[-1])
mean = flat.mean(axis=0)
centered = flat - mean
cov = (centered.T @ centered) / (flat.shape[0] - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
proj = flat @ eigvecs[:, :3]
p2, p98 = np.percentile(proj, [2, 98])
proj = np.clip((proj - p2) / (p98 - p2 + 1e-8), 0, 1)
rgb = proj.reshape(aef_emb.shape[0], aef_emb.shape[1], 3)
axes[2].imshow(rgb)
axes[2].set_title('AEF PCA RGB (patch_0)')

plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/aef_zero_map.png', dpi=150)
plt.close()
print("Saved aef_zero_map.png")
