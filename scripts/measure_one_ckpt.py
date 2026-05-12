import sys, torch
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config
from src.models.model import AEFModel
from src.training.losses import batch_uniformity_loss_l2, variance_regularizer

cfg = load_config('configs/xuannv_v12_clean.yaml')
model = AEFModel(cfg)
ckpt = torch.load('/workspace/outputs/xuannv_v13_monthly/epoch_best_epoch5.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

torch.manual_seed(42)
B, S, T = 2, 3, 4
frames = torch.randn(B, S, T, 6, 128, 128)
ts = torch.ones(B, S, T, dtype=torch.long) * 20240101
mask = torch.ones(B, S, T, dtype=torch.bool)
imask = torch.ones(B, S, dtype=torch.bool)
tids = torch.zeros(B, S, dtype=torch.long)

with torch.no_grad():
    out = model(frames, ts, mask, imask, tids,
                torch.zeros(B), torch.ones(B)*1e9,
                torch.zeros(B, 4), torch.zeros(B, 4))

emb_vec = out.embedding
emb_map = out.embedding_map
pre_vec = out.pre_norm_embedding
pre_map = out.pre_norm_map
B, D = emb_vec.shape

print(f"Epoch 5 Best Checkpoint  (Batch={B}, Dim={D})")
print("=" * 70)

targets = {
    'emb_vec (L2 global)': emb_vec,
    'emb_map (L2 spatial)': emb_map,
    'pre_vec (raw global)': pre_vec,
    'pre_map (raw spatial)': pre_map,
}

print(f"{'Target':25s} | Uniformity | Active | Std_mean | VICReg(1.0) | VICReg(0.1)")
print("-" * 85)
for name, emb in targets.items():
    u = batch_uniformity_loss_l2(emb).item()
    if emb.dim() == 4:
        std = torch.sqrt(emb.reshape(B, D, -1).var(dim=(0, 2), unbiased=False) + 1e-6)
    else:
        std = torch.sqrt(emb.var(dim=0, unbiased=False) + 1e-6)
    active = (std > 0.05).sum().item()
    flat = emb.permute(0, 2, 3, 1).reshape(-1, D) if emb.dim() == 4 else emb
    v1 = variance_regularizer(flat, min_std=1.0).item()
    v01 = variance_regularizer(flat, min_std=0.1).item()
    print(f"{name:25s} | {u:.4f}     | {active:3d}/{D} | {std.mean():.4f}   | {v1:.4f}      | {v01:.4f}")

print(f"\n[Weight comparison: contribution to total loss]")
for w in [0.05, 0.1, 0.2, 0.3]:
    lv = w * batch_uniformity_loss_l2(emb_vec).item()
    lm = w * batch_uniformity_loss_l2(emb_map).item()
    lp = w * batch_uniformity_loss_l2(pre_vec).item()
    print(f"  w={w:.1f}: emb_vec={lv:.4f}  emb_map={lm:.4f}  pre_vec={lp:.4f}")
