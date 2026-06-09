import os, sys
os.environ.setdefault('ASCEND_LAUNCH_BLOCKING', '1')
sys.path.insert(0, '/workspace/xuannv/aef_reference')
import torch, torch_npu
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
sample = dataset[0]
batch = collate_fn([sample])
target = batch['aef_embedding']  # B, D, H, W
target_r = target.permute(0, 2, 3, 1)  # B, H, W, D
tgt_mag = target_r.norm(dim=-1)
print(f'AEF target_mag: mean={tgt_mag.mean():.4f}, min={tgt_mag.min():.4f}, max={tgt_mag.max():.4f}')
print(f'Zero-magnitude locations: {(tgt_mag < 0.01).sum()} / {tgt_mag.numel()}')
print(f'Locations with mag < 0.1: {(tgt_mag < 0.1).sum()}')
