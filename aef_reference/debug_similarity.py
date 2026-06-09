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
    student_emb = out['teacher_embeddings'][0]  # H, W, 64
    aef_emb = batch['aef_embedding'][0].to(device)  # 64, H, W
    aef_emb = aef_emb.permute(1, 2, 0)  # H, W, 64
    
    # Cosine similarity per spatial location
    stu_norm = torch.nn.functional.normalize(student_emb, p=2, dim=-1)
    aef_norm = torch.nn.functional.normalize(aef_emb, p=2, dim=-1)
    cos_sim = (stu_norm * aef_norm).sum(dim=-1)
    
    print(f'Cosine similarity: mean={cos_sim.mean():.4f}, min={cos_sim.min():.4f}, max={cos_sim.max():.4f}')
    print(f'Percentiles: 10%={cos_sim.quantile(0.1):.4f}, 50%={cos_sim.quantile(0.5):.4f}, 90%={cos_sim.quantile(0.9):.4f}')
    
    # Check if student is close to AEF
    print(f'L2 distance: mean={(student_emb - aef_emb).norm(dim=-1).mean():.4f}')
    print(f'Student norm mean: {student_emb.norm(dim=-1).mean():.4f}')
    print(f'AEF norm mean: {aef_emb.norm(dim=-1).mean():.4f}')
