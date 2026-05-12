"""测量已有 checkpoint 在不同 uniformity 策略下的实际 loss 值."""
import sys, torch
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config
from src.models.model import AEFModel
from src.data.builder import build_dataloader
from src.training.losses import batch_uniformity_loss_l2, variance_regularizer

def measure(checkpoint_path, label):
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    cfg.data.batch_size = 6
    cfg.data.preload = False
    
    model = AEFModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    
    dataloader = build_dataloader(cfg, training=False, distributed=False)
    batch = next(iter(dataloader))
    batch = {k: v if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        out = model(
            source_frames=batch['source_frames'],
            source_timestamps_ms=batch['source_timestamps_ms'],
            source_frame_mask=batch['source_frame_mask'],
            source_input_mask=batch['source_input_mask'],
            source_type_ids=batch['source_type_ids'],
            valid_start_ms=batch['valid_start_ms'],
            valid_end_ms=batch['valid_end_ms'],
            target_relative_time=batch['target_relative_time'],
            target_metadata=batch['target_metadata'],
        )
    
    emb_vec = out.embedding
    emb_map = out.embedding_map
    pre_vec = out.pre_norm_embedding
    pre_map = out.pre_norm_map
    B, D = emb_vec.shape
    
    print(f"\n{'='*70}")
    print(f"Checkpoint: {label}  (Batch={B}, Dim={D})")
    print('='*70)
    
    targets = {
        'emb_vec (L2 global)': emb_vec,
        'emb_map (L2 spatial)': emb_map,
        'pre_vec (raw global)': pre_vec,
        'pre_map (raw spatial)': pre_map,
    }
    
    print(f"\n{'Target':25s} | Uniformity | Active | Std_mean | VICReg(1.0) | VICReg(0.1)")
    print("-" * 85)
    for name, emb in targets.items():
        u = batch_uniformity_loss_l2(emb).item()
        if emb.dim() == 4:
            std = torch.sqrt(emb.reshape(B, D, -1).var(dim=(0, 2), unbiased=False) + 1e-6)
        else:
            std = torch.sqrt(emb.var(dim=0, unbiased=False) + 1e-6)
        active = (std > 0.05).sum().item()
        
        if emb.dim() == 4:
            flat = emb.permute(0, 2, 3, 1).reshape(-1, D)
        else:
            flat = emb
        v1 = variance_regularizer(flat, min_std=1.0).item()
        v01 = variance_regularizer(flat, min_std=0.1).item()
        
        print(f"{name:25s} | {u:.4f}     | {active:3d}/{D} | {std.mean():.4f}   | {v1:.4f}      | {v01:.4f}")
    
    print(f"\n[策略对比: 相同 weight 下不同目标的 uniformity loss 贡献]")
    for w in [0.05, 0.1, 0.2, 0.3]:
        lv = w * batch_uniformity_loss_l2(emb_vec).item()
        lm = w * batch_uniformity_loss_l2(emb_map).item()
        lp = w * batch_uniformity_loss_l2(pre_vec).item()
        print(f"  w={w:.1f}: emb_vec={lv:.4f}  emb_map={lm:.4f}  pre_vec={lp:.4f}")

if __name__ == '__main__':
    base = '/workspace/outputs/xuannv_v13_monthly'
    for name in ['epoch_best_epoch3.pt', 'epoch_best_epoch4.pt', 'epoch_best_epoch5.pt']:
        measure(f'{base}/{name}', name)
