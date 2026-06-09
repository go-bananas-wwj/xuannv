import os, sys
os.environ.setdefault('ASCEND_LAUNCH_BLOCKING', '1')
sys.path.insert(0, '/workspace/xuannv/aef_reference')
import numpy as np, torch, torch_npu
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

# Load trained checkpoint
checkpoint_path = '/workspace/xuannv/aef_reference/outputs/aef_distill_control_exp/step_000200_seed42.pt'
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    has_module = any(k.startswith('module.') for k in state_dict.keys())
    if has_module:
        state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

model.eval()
with torch.no_grad():
    x = model._stack_inputs(source_data)
    first_src = next(iter(model.input_sources.keys()))
    ts = timestamps[first_src]
    feats = model.encoder(x, ts)
    print(f'encoder output shape: {feats.shape}')
    
    # Check stripes in encoder output
    feats_np = feats[0].cpu().numpy()  # T, H, W, C
    T, H, W, C = feats_np.shape
    max_corr = 0
    for t in range(T):
        for ch in range(min(16, C)):
            for y in range(H):
                row = feats_np[t, y, :, ch]
                if row.std() > 1e-6:
                    for yy in range(y+1, H):
                        corr = np.corrcoef(row, feats_np[t, yy, :, ch])[0,1]
                        if not np.isnan(corr):
                            max_corr = max(max_corr, abs(corr))
    print(f'encoder max_y_corr: {max_corr:.4f}')
    
    # summarizer output
    vp = valid_periods
    mu = model.summarizer(feats, ts, vp)
    print(f'summarizer output shape: {mu.shape}')
    
    mu_np = mu[0].cpu().numpy()
    H2, W2, D = mu_np.shape
    max_corr2 = 0
    for ch in range(min(16, D)):
        for y in range(H2):
            row = mu_np[y, :, ch]
            if row.std() > 1e-6:
                for yy in range(y+1, H2):
                    corr = np.corrcoef(row, mu_np[yy, :, ch])[0,1]
                    if not np.isnan(corr):
                        max_corr2 = max(max_corr2, abs(corr))
    print(f'summarizer max_y_corr: {max_corr2:.4f}')
