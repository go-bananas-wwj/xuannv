"""分析训练后模型的实际 embedding 统计特性 — 用合成数据避免加载开销."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config
from src.models.model import AEFModel
from src.training.losses import batch_uniformity_loss_l2, variance_regularizer

def analyze(checkpoint_path, label):
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    model = AEFModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    
    # 合成数据，模拟真实 batch
    torch.manual_seed(42)
    B, S, T, C, H, W = 6, 3, 8, 6, 128, 128
    frames = torch.randn(B, S, T, C, H, W, device=device)
    ts = torch.randint(0, 1000000, (B, S, T), device=device)
    frame_mask = torch.ones(B, S, T, dtype=torch.bool, device=device)
    input_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    type_ids = torch.zeros(B, S, dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model(
            source_frames=frames, source_timestamps_ms=ts,
            source_frame_mask=frame_mask, source_input_mask=input_mask,
            source_type_ids=type_ids,
            valid_start_ms=torch.zeros(B, device=device),
            valid_end_ms=torch.ones(B, device=device) * 1000000,
            target_relative_time=torch.zeros(B, 4, device=device),
            target_metadata=torch.zeros(B, 4, device=device),
        )
    
    emb_map = out.embedding_map
    emb_vec = out.embedding
    pre_map = out.pre_norm_map
    pre_vec = out.pre_norm_embedding
    B, D = emb_vec.shape
    
    print(f"\n{'='*60}")
    print(f"{label}  (Batch={B}, Dim={D})")
    print('='*60)
    
    # Uniformity
    u_vec = batch_uniformity_loss_l2(emb_vec)
    u_map = batch_uniformity_loss_l2(emb_map)
    u_pre = batch_uniformity_loss_l2(pre_vec)
    print(f"Uniformity: vec={u_vec.item():.4f}  map={u_map.item():.4f}  pre={u_pre.item():.4f}")
    
    # active_dims
    std_vec = torch.sqrt(emb_vec.var(dim=0, unbiased=False) + 1e-6)
    std_map = torch.sqrt(emb_map.reshape(B, D, -1).var(dim=(0, 2), unbiased=False) + 1e-6)
    std_pre = torch.sqrt(pre_vec.var(dim=0, unbiased=False) + 1e-6)
    a_vec = (std_vec > 0.05).sum().item()
    a_map = (std_map > 0.05).sum().item()
    a_pre = (std_pre > 0.05).sum().item()
    print(f"Active:     vec={a_vec:3d}/{D}  map={a_map:3d}/{D}  pre={a_pre:3d}/{D}")
    print(f"Std_mean:   vec={std_vec.mean():.4f}  map={std_map.mean():.4f}  pre={std_pre.mean():.4f}")
    
    # VICReg
    v_vec_1 = variance_regularizer(emb_vec, min_std=1.0)
    v_vec_01 = variance_regularizer(emb_vec, min_std=0.1)
    print(f"VICReg:     min1.0={v_vec_1.item():.4f}  min0.1={v_vec_01.item():.4f}")
    
    # 空间一致性
    sims = []
    for b in range(min(B, 3)):
        pixels = emb_map[b].permute(1, 2, 0).reshape(-1, D)
        sim = pixels @ pixels.T
        offdiag = sim[~torch.eye(pixels.shape[0], dtype=torch.bool)]
        sims.append(offdiag.mean().item())
    print(f"SpatialSim: {sum(sims)/len(sims):.4f}")

if __name__ == '__main__':
    base = '/workspace/outputs/xuannv_v13_monthly'
    # 先分析随机初始化
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    m = AEFModel(cfg)
    torch.save({'model_state_dict': m.state_dict()}, '/tmp/random_init.pt')
    analyze('/tmp/random_init.pt', 'Random Init')
    
    for name in ['epoch_best_epoch2.pt', 'epoch_best_epoch3.pt', 'epoch_best_epoch4.pt']:
        analyze(f'{base}/{name}', name)
