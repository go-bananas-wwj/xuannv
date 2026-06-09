"""Compare 4 ablation experiments at step 50."""
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

model = AlphaEarthFoundations(
    model_size='small',
    input_sources={'s1':2,'s2':6,'tianyi_sar':1,'landsat':6,'planet':4},
    decode_sources={'s1':2,'s2':6,'tianyi_sar':1,'landsat':6,'planet':4,'dem':1,'worldcover':11,'dynamic_world':9,'jrc_water':1},
    per_source_latent=32, enable_text_align=False,
).to(device)

# AEF PCA basis
aef_emb = batch['aef_embedding'][0].to(device).permute(1, 2, 0)
H, W, D = aef_emb.shape
aef_np = aef_emb.cpu().numpy()
flat = aef_np.reshape(-1, D)
mean = flat.mean(axis=0)
cov = ((flat - mean).T @ (flat - mean)) / (flat.shape[0] - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

def to_pca_rgb(emb):
    emb_np = emb.cpu().numpy()
    proj = (emb_np.reshape(-1, D) - mean) @ eigvecs[:, :3]
    p2, p98 = np.percentile(proj, [2, 98])
    proj = np.clip((proj - p2) / (p98 - p2 + 1e-8), 0, 1)
    return proj.reshape(H, W, 3)

def analyze_stripes(emb):
    emb_np = emb.cpu().numpy()
    max_corr = 0
    for ch in range(min(16, D)):
        for y in range(H):
            row = emb_np[y, :, ch]
            if row.std() > 1e-6:
                for yy in range(y+1, H):
                    corr = np.corrcoef(row, emb_np[yy, :, ch])[0,1]
                    if not np.isnan(corr):
                        max_corr = max(max_corr, abs(corr))
    return max_corr

exps = [
    ('exp1_spatial_recon', 'outputs/ablation_exp1_spatial_recon/step_000050_seed42.pt'),
    ('exp2_spatial_only', 'outputs/ablation_exp2_spatial_only/step_000050_seed42.pt'),
    ('exp3_global_recon', 'outputs/ablation_exp3_global_recon/step_000050_seed42.pt'),
    ('exp4_spatial_nounif', 'outputs/ablation_exp4_spatial_recon_nounif/step_000050_seed42.pt'),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, (name, ckpt_path) in enumerate(exps):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    has_module = any(k.startswith('module.') for k in state_dict.keys())
    if has_module:
        state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)
        student_emb = out['teacher_embeddings'][0]
    
    stu_rgb = to_pca_rgb(student_emb)
    corr = analyze_stripes(student_emb)
    
    axes[0, i].imshow(stu_rgb)
    axes[0, i].set_title(f'{name}\nmax_y_corr={corr:.3f}')
    axes[0, i].axis('off')
    
    # Show individual channels
    ch0 = student_emb[:, :, 0].cpu().numpy()
    vmin, vmax = ch0.min(), ch0.max()
    ch0_norm = (ch0 - vmin) / (vmax - vmin + 1e-8)
    axes[1, i].imshow(ch0_norm, cmap='viridis')
    axes[1, i].set_title(f'ch0 (stripe check)')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('/workspace/xuannv/aef_reference/ablation_comparison_step50.png', dpi=150)
plt.close()
print('Saved ablation_comparison_step50.png')
