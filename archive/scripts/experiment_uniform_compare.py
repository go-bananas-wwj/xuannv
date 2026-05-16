"""
快速对比实验: 不同 uniformity 策略对坍缩的影响.
使用 num_blocks=4 降低计算量, 单卡, 3 epoch x 20 steps.
"""
import sys, os, torch, copy, time
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config, Config
from src.models.model import AEFModel
from src.data.builder import build_dataloader
from src.training.optimizer import build_optimizer
from src.training.losses import batch_uniformity_loss_l2, variance_regularizer, reconstruction_loss, consistency_loss_spatial
from src.training.memory_bank import EmbeddingMemoryBank

def quick_train(cfg_override, exp_name):
    """单卡快速训练 3 epoch x 20 steps."""
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    # 覆盖配置
    for k, v in cfg_override.items():
        parts = k.split('.')
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    
    cfg.model.num_blocks = 4
    cfg.data.batch_size = 4
    cfg.data.preload = False
    cfg.training.epochs = 3
    cfg.training.max_steps_per_epoch = 20
    cfg.training.warmup_epochs = 0
    cfg.training.recon_warmup_epochs = 0
    
    device = torch.device('cpu')
    model = AEFModel(cfg).to(device)
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    dataloader = build_dataloader(cfg, training=True, distributed=False)
    optimizer = build_optimizer(model, cfg)
    memory_bank = EmbeddingMemoryBank(K=256, dim=cfg.model.embedding_dim, device=device)
    
    recon_w = cfg.training.reconstruction_weight
    consist_w = cfg.training.consistency_weight
    uniform_w = cfg.training.batch_uniformity_weight
    var_w = getattr(cfg.training, 'variance_weight', 0.3)
    cov_w = getattr(cfg.training, 'covariance_weight', 0.1)
    
    results = []
    for epoch in range(3):
        epoch_losses = {'recon': 0, 'consist': 0, 'unif': 0, 'var': 0, 'cov': 0, 'total': 0}
        n_steps = 0
        for step, batch in enumerate(dataloader):
            if step >= 20:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            kwargs = {
                'source_frames': batch['source_frames'],
                'source_timestamps_ms': batch['source_timestamps_ms'],
                'source_frame_mask': batch['source_frame_mask'],
                'source_input_mask': batch['source_input_mask'],
                'source_type_ids': batch['source_type_ids'],
                'valid_start_ms': batch['valid_start_ms'],
                'valid_end_ms': batch['valid_end_ms'],
                'target_relative_time': batch['target_relative_time'],
                'target_metadata': batch['target_metadata'],
            }
            
            with torch.no_grad():
                t_out = teacher(**kwargs)
            s_out = model(**kwargs)
            
            recon = reconstruction_loss(
                s_out.reconstructions, batch['target_images'], 
                batch['target_mask'], batch.get('target_loss_type'), cfg.data.num_classes
            )
            
            consist = consistency_loss_spatial(t_out.embedding_map.detach(), s_out.embedding_map) if consist_w > 0 else torch.tensor(0.0)
            
            uniform_target = cfg_override.get('uniformity_target', 'vec')
            if uniform_target == 'vec':
                emb = s_out.embedding
            elif uniform_target == 'map':
                emb = s_out.embedding_map
            elif uniform_target == 'pre_vec':
                emb = s_out.pre_norm_embedding
            elif uniform_target == 'pre_map':
                emb = s_out.pre_norm_map
            else:
                emb = s_out.embedding
            
            unif = batch_uniformity_loss_l2(emb.float()) if uniform_w > 0 else torch.tensor(0.0)
            
            pre = s_out.pre_norm_embedding
            var = variance_regularizer(pre.float(), min_std=1.0) if var_w > 0 else torch.tensor(0.0)
            cov = variance_regularizer(pre.float()) if cov_w > 0 else torch.tensor(0.0)
            
            total = recon_w * recon + consist_w * consist + uniform_w * unif + var_w * var + cov_w * cov
            
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            m = 0.996
            for p_t, p_s in zip(teacher.parameters(), model.parameters()):
                p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)
            
            for k in epoch_losses:
                if k == 'recon': epoch_losses[k] += recon.item()
                elif k == 'consist': epoch_losses[k] += consist.item()
                elif k == 'unif': epoch_losses[k] += unif.item()
                elif k == 'var': epoch_losses[k] += var.item()
                elif k == 'cov': epoch_losses[k] += cov.item()
                elif k == 'total': epoch_losses[k] += total.item()
            n_steps += 1
        
        with torch.no_grad():
            diag = {}
            for name, emb in [('vec', s_out.embedding), ('map', s_out.embedding_map), ('pre_vec', s_out.pre_norm_embedding)]:
                if emb is not None:
                    if emb.dim() == 4:
                        std = torch.sqrt(emb.reshape(emb.shape[0], emb.shape[1], -1).var(dim=(0, 2), unbiased=False) + 1e-6)
                    else:
                        std = torch.sqrt(emb.var(dim=0, unbiased=False) + 1e-6)
                    diag[f'active_{name}'] = (std > 0.05).sum().item()
                    diag[f'unif_{name}'] = batch_uniformity_loss_l2(emb).item()
            
            results.append({
                'epoch': epoch + 1,
                **{k: v / n_steps for k, v in epoch_losses.items()},
                **diag
            })
    
    return results

if __name__ == '__main__':
    experiments = [
        ('Baseline (vec, w=0.05)', {'training.batch_uniformity_weight': 0.05, 'uniformity_target': 'vec'}),
        ('Exp-A (map, w=0.05)',    {'training.batch_uniformity_weight': 0.05, 'uniformity_target': 'map'}),
        ('Exp-B (vec, w=0.3)',     {'training.batch_uniformity_weight': 0.30, 'uniformity_target': 'vec'}),
        ('Exp-C (map, w=0.3)',     {'training.batch_uniformity_weight': 0.30, 'uniformity_target': 'map'}),
        ('Exp-D (pre_vec, w=0.3)', {'training.batch_uniformity_weight': 0.30, 'uniformity_target': 'pre_vec'}),
    ]
    
    all_results = {}
    for name, cfg_ov in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        t0 = time.time()
        try:
            res = quick_train(cfg_ov, name)
            all_results[name] = res
            print(f"Done in {time.time()-t0:.1f}s")
            for r in res:
                print(f"  E{r['epoch']}: total={r['total']:.3f} recon={r['recon']:.3f} unif={r['unif']:.3f} | "
                      f"active_vec={r.get('active_vec', 0):3d} active_map={r.get('active_map', 0):3d} "
                      f"u_vec={r.get('unif_vec', 0):.3f} u_map={r.get('unif_map', 0):.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"{'Experiment':25s} | E1_unif | E3_unif | E1_active | E3_active | E1_uvec | E3_uvec")
    print("-" * 85)
    for name, res in all_results.items():
        if len(res) >= 3:
            print(f"{name:25s} | {res[0]['unif']:.3f}   | {res[2]['unif']:.3f}   | "
                  f"{res[0].get('active_vec', 0):3d}       | {res[2].get('active_vec', 0):3d}       | "
                  f"{res[0].get('unif_vec', 0):.3f}   | {res[2].get('unif_vec', 0):.3f}")
