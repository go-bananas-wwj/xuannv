"""Generate visualization for control experiment (spatial distillation, step 200)."""
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

# Dataset
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

sample = dataset[0]
batch = collate_fn([sample])
source_data = {k:v.to(device) for k,v in batch['source_data'].items()}
timestamps = {k:v.to(device) for k,v in batch['timestamps'].items()}
valid_periods = batch['valid_periods'].to(device)

# Load trained model
model = AlphaEarthFoundations(
    model_size='small',
    input_sources={'s1':2,'s2':6,'tianyi_sar':1,'landsat':6,'planet':4},
    decode_sources={'s1':2,'s2':6,'tianyi_sar':1,'landsat':6,'planet':4,'dem':1,'worldcover':11,'dynamic_world':9,'jrc_water':1},
    per_source_latent=32, enable_text_align=False,
).to(device)

checkpoint_path = '/workspace/xuannv/aef_reference/outputs/aef_distill_control_exp/step_000200_seed42.pt'
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    has_module = any(k.startswith('module.') for k in state_dict.keys())
    if has_module:
        state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint from {checkpoint_path}')
else:
    print(f'Checkpoint not found: {checkpoint_path}')
    sys.exit(1)

model.eval()
with torch.no_grad():
    out = model(source_data, timestamps, valid_periods)
    student_emb = out['teacher_embeddings'][0].cpu().numpy()  # H, W, 64
    aef_emb = batch['aef_embedding'].numpy()[0]  # D, H, W
    aef_emb = np.transpose(aef_emb, (1, 2, 0))  # H, W, D

H, W, D = student_emb.shape

# === Stripe detection ===
def analyze_stripes(emb):
    max_corr = 0
    for ch in range(min(16, D)):
        for y in range(H):
            row = emb[y, :, ch]
            if row.std() > 1e-6:
                for yy in range(y+1, H):
                    corr = np.corrcoef(row, emb[yy, :, ch])[0,1]
                    if not np.isnan(corr):
                        max_corr = max(max_corr, abs(corr))
    # PCA
    flat = emb.reshape(-1, D)
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = (centered.T @ centered) / (flat.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    pca0_ratio = eigvals[0] / (eigvals.sum() + 1e-8)
    return max_corr, pca0_ratio

stu_max, stu_pca = analyze_stripes(student_emb)
aef_max, aef_pca = analyze_stripes(aef_emb)

print(f'Student: max_y_corr={stu_max:.4f}, PCA0_ratio={stu_pca:.4f}')
print(f'AEF:     max_y_corr={aef_max:.4f}, PCA0_ratio={aef_pca:.4f}')

# === PCA RGB ===
aef_flat = aef_emb.reshape(-1, D)
mean = aef_flat.mean(axis=0)
aef_centered = aef_flat - mean
cov = (aef_centered.T @ aef_centered) / (aef_flat.shape[0] - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

def to_pca_rgb(emb):
    flat = emb.reshape(-1, D) - mean
    proj = flat @ eigvecs[:, :3]
    p2, p98 = np.percentile(proj, [2, 98])
    proj = np.clip((proj - p2) / (p98 - p2 + 1e-8), 0, 1)
    return proj.reshape(H, W, 3)

stu_rgb = to_pca_rgb(student_emb)
aef_rgb = to_pca_rgb(aef_emb)
diff = np.abs(stu_rgb - aef_rgb).mean(axis=-1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(stu_rgb)
axes[0].set_title(f'Student PCA RGB\nmax_y_corr={stu_max:.2f}')
axes[1].imshow(aef_rgb)
axes[1].set_title(f'AEF PCA RGB\nmax_y_corr={aef_max:.2f}')
im = axes[2].imshow(diff, cmap='hot')
axes[2].set_title(f'|Student-AEF|\nmean={diff.mean():.3f}')
plt.colorbar(im, ax=axes[2], fraction=0.046)
axes[3].axis('off')
axes[3].text(0.1, 0.7, f'Student metrics:\nmax_y_corr={stu_max:.4f}\nPCA0_ratio={stu_pca:.4f}', fontsize=11, family='monospace')
axes[3].text(0.1, 0.3, f'AEF metrics:\nmax_y_corr={aef_max:.4f}\nPCA0_ratio={aef_pca:.4f}', fontsize=11, family='monospace')

plt.suptitle('Control Exp: Spatial distill weight=50.0, step=200', fontsize=14)
plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/viz_control_pca.png', dpi=150)
plt.close()
print('Saved viz_control_pca.png')

# === Individual channels ===
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i in range(16):
    ax = axes[i // 4, i % 4]
    ch = student_emb[:, :, i]
    vmin, vmax = ch.min(), ch.max()
    ch_norm = (ch - vmin) / (vmax - vmin + 1e-8)
    ax.imshow(ch_norm, cmap='viridis')
    ax.set_title(f'ch{i}', fontsize=10)
    ax.axis('off')
plt.suptitle('Control Exp - Student Individual Channels (no PCA)', fontsize=14)
plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/viz_control_channels.png', dpi=150)
plt.close()
print('Saved viz_control_channels.png')
