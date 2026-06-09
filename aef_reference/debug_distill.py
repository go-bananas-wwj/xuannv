import os, sys
os.environ.setdefault('ASCEND_LAUNCH_BLOCKING', '1')
sys.path.insert(0, '/workspace/xuannv/aef_reference')
import numpy as np, torch, torch_npu
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

checkpoint_path = '/workspace/xuannv/aef_reference/outputs/aef_distill_control_exp/step_000200_seed42.pt'
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['model_state_dict']
has_module = any(k.startswith('module.') for k in state_dict.keys())
if has_module:
    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

model.eval()
with torch.no_grad():
    out = model(source_data, timestamps, valid_periods)
    pred = out['teacher_embeddings']  # B, H, W, 64
    target = batch['aef_embedding'].to(device)  # B, 64, H, W
    
    B, H, W, D_pred = pred.shape
    B_t, D_t, H_t, W_t = target.shape
    
    if H_t != H or W_t != W:
        target_2d = F.interpolate(target, size=(H, W), mode='bilinear', align_corners=False)
    else:
        target_2d = target
    
    target_r = rearrange(target_2d, 'b c h w -> b h w c')
    
    pred_n = torch.nn.functional.normalize(pred, p=2, dim=-1)
    tgt_n = torch.nn.functional.normalize(target_r, p=2, dim=-1)
    cosine_sim = (pred_n * tgt_n).sum(dim=-1)
    cosine_loss = (1.0 - cosine_sim)
    
    pred_mag = pred.norm(dim=-1)
    tgt_mag = target_r.norm(dim=-1)
    mag_loss = ((pred_mag - tgt_mag) ** 2) / (tgt_mag ** 2 + 1e-8)
    
    loss = cosine_loss + 0.001 * mag_loss
    
    print(f'pred shape: {pred.shape}, target shape: {target_r.shape}')
    print(f'cosine_sim: mean={cosine_sim.mean():.4f}, min={cosine_sim.min():.4f}, max={cosine_sim.max():.4f}')
    print(f'cosine_loss: mean={cosine_loss.mean():.4f}, min={cosine_loss.min():.4f}, max={cosine_loss.max():.4f}')
    print(f'mag_loss: mean={mag_loss.mean():.4f}')
    print(f'total distill loss: {loss.mean():.4f}')
    print(f'pred norm: mean={pred_mag.mean():.4f}')
    print(f'target norm: mean={tgt_mag.mean():.4f}')
