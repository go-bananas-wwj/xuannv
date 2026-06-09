"""Multi-patch analysis: input sources, AEF embeddings, and random init model output."""
from __future__ import annotations
import os, sys
os.environ.setdefault('ASCEND_LAUNCH_BLOCKING', '1')
sys.path.insert(0, '/workspace/xuannv/aef_reference')

import numpy as np, torch, torch_npu, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import rearrange
from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn

device = torch.device('npu:0')

# Load dataset
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

# Random init model
model = AlphaEarthFoundations(
    model_size='small',
    input_sources={'s1':2,'s2':6,'tianyi_sar':1,'landsat':6,'planet':4},
    decode_sources={'s1':2,'s2':6,'tianyi_sar':1,'landsat':6,'planet':4,'dem':1,'worldcover':11,'dynamic_world':9,'jrc_water':1},
    per_source_latent=32, enable_text_align=False,
).to(device)
model.eval()

# Select 8 diverse patches
patch_indices = [0, 10, 25, 50, 75, 100, 150, 200]

source_names = ['s1', 's2', 'tianyi_sar', 'landsat', 'planet']

results = {
    'aef_zero_count': [],
    'aef_min_mag': [],
    'aef_max_y_corr': [],
    'random_init_max_y_corr': [],
    'input_y_corr': {src: [] for src in source_names},
}

for idx in patch_indices:
    sample = dataset[idx]
    batch = collate_fn([sample])
    source_data = {k:v.to(device) for k,v in batch['source_data'].items()}
    timestamps = {k:v.to(device) for k,v in batch['timestamps'].items()}
    valid_periods = batch['valid_periods'].to(device)
    
    # AEF embedding analysis
    aef_emb = batch['aef_embedding'][0].numpy()  # D, H, W
    aef_emb = np.transpose(aef_emb, (1, 2, 0))  # H, W, D
    aef_mag = np.linalg.norm(aef_emb, axis=-1)
    results['aef_zero_count'].append((aef_mag < 0.01).sum())
    results['aef_min_mag'].append(aef_mag.min())
    
    # AEF y-correlation
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
    results['aef_max_y_corr'].append(max_corr)
    
    # Random init model y-correlation
    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)
        student_emb = out['teacher_embeddings'][0].cpu().numpy()
    H2, W2, D2 = student_emb.shape
    max_corr2 = 0
    for ch in range(min(16, D2)):
        for y in range(H2):
            row = student_emb[y, :, ch]
            if row.std() > 1e-6:
                for yy in range(y+1, H2):
                    corr = np.corrcoef(row, student_emb[yy, :, ch])[0,1]
                    if not np.isnan(corr):
                        max_corr2 = max(max_corr2, abs(corr))
    results['random_init_max_y_corr'].append(max_corr2)
    
    # Input source y-correlation
    for src in source_names:
        data = source_data[src][0].cpu().numpy()  # T, H, W, C
        T, H3, W3, C = data.shape
        # Take middle frame
        mid_t = T // 2
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
        results['input_y_corr'][src].append(max_corr3)

# Print summary
print("="*70)
print("MULTI-PATCH ANALYSIS SUMMARY")
print("="*70)
print(f"\nAEF Embedding Stats:")
print(f"  Zero-magnitude locations: {results['aef_zero_count']}")
print(f"  Min magnitude: {[f'{x:.4f}' for x in results['aef_min_mag']]}")
print(f"  Max y-correlation: {[f'{x:.4f}' for x in results['aef_max_y_corr']]}")

print(f"\nRandom Init Model Stats:")
print(f"  Max y-correlation: {[f'{x:.4f}' for x in results['random_init_max_y_corr']]}")

print(f"\nInput Source Y-Correlation (middle frame):")
for src in source_names:
    vals = results['input_y_corr'][src]
    print(f"  {src:12s}: {[f'{x:.4f}' for x in vals]}  mean={np.mean(vals):.4f}")

# Visualization: Input sources for first patch
sample = dataset[0]
batch = collate_fn([sample])
source_data = {k:v.to(device) for k,v in batch['source_data'].items()}

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, src in enumerate(source_names):
    data = source_data[src][0].cpu().numpy()  # T, H, W, C
    T, H, W, C = data.shape
    mid_t = T // 2
    frame = data[mid_t]
    
    # RGB visualization
    rgb = np.zeros((H, W, 3))
    for c in range(min(3, C)):
        ch = frame[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax - vmin > 1e-6:
            rgb[:, :, c] = (ch - vmin) / (vmax - vmin)
    
    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f'{src} (mid frame)')
    axes[0, i].axis('off')
    
    # Y-gradient visualization
    y_grad = np.abs(frame[1:, :, 0] - frame[:-1, :, 0]).mean(axis=1)
    axes[1, i].plot(y_grad, range(len(y_grad)))
    axes[1, i].invert_yaxis()
    axes[1, i].set_title(f'{src} y-gradient')
    axes[1, i].set_xlabel('mean abs grad')

plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/multi_patch_analysis.png', dpi=150)
plt.close()
print("\nSaved multi_patch_analysis.png")

# AEF zero location visualization
aef_emb = batch['aef_embedding'][0].numpy()
aef_emb = np.transpose(aef_emb, (1, 2, 0))
aef_mag = np.linalg.norm(aef_emb, axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(aef_mag, cmap='viridis')
axes[0].set_title(f'AEF Magnitude\nmin={aef_mag.min():.4f}')
axes[1].imshow(aef_mag < 0.01, cmap='hot')
axes[1].set_title(f'Zero Locations\ncount={(aef_mag < 0.01).sum()}')

# PCA RGB of AEF
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
axes[2].set_title('AEF PCA RGB')

plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/aef_analysis_patch0.png', dpi=150)
plt.close()
print("Saved aef_analysis_patch0.png")
