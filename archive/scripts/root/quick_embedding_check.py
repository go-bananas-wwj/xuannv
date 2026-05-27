"""轻量分析: 只加载权重, 用最小 forward 检查 embedding 特性."""
import sys, torch
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config
from src.models.model import AEFModel
from src.training.losses import batch_uniformity_loss_l2

def check(path, name):
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    cfg.data.batch_size = 2  # 最小 batch
    model = AEFModel(cfg)
    if path != 'random':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    
    B, S, T = 2, 3, 4  # 最小数据
    frames = torch.randn(B, S, T, 6, 128, 128)
    ts = torch.ones(B, S, T, dtype=torch.long) * 20240101
    mask = torch.ones(B, S, T, dtype=torch.bool)
    imask = torch.ones(B, S, dtype=torch.bool)
    tids = torch.zeros(B, S, dtype=torch.long)
    
    with torch.no_grad():
        out = model(frames, ts, mask, imask, tids,
                    torch.zeros(B), torch.ones(B)*1e9,
                    torch.zeros(B, 4), torch.zeros(B, 4))
    
    emb_map = out.embedding_map  # [2, 128, 64, 64]
    emb_vec = out.embedding       # [2, 128]
    
    u_map = batch_uniformity_loss_l2(emb_map).item()
    u_vec = batch_uniformity_loss_l2(emb_vec).item()
    
    std_map = torch.sqrt(emb_map.reshape(2, 128, -1).var(dim=(0, 2), unbiased=False) + 1e-6)
    std_vec = torch.sqrt(emb_vec.var(dim=0, unbiased=False) + 1e-6)
    
    print(f"{name:20s} | u_map={u_map:.3f} u_vec={u_vec:.3f} | active_map={(std_map>0.05).sum().item():3d} active_vec={(std_vec>0.05).sum().item():3d}")

base = '/workspace/outputs/xuannv_v13_monthly'
print(f"{'Checkpoint':20s} | Uniformity (map/vec) | Active (map/vec)")
print("-" * 70)
check('random', 'Random Init')
for name in ['epoch_best_epoch2.pt', 'epoch_best_epoch3.pt', 'epoch_best_epoch4.pt']:
    check(f'{base}/{name}', name)
