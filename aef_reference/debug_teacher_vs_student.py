"""Compare teacher and student row_mean_corr"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

def rmc(x):
    x = x.detach().cpu().float()
    if x.dim() == 5:
        x = x[0, 0]
        x = x.permute(2, 0, 1)
    elif x.dim() == 4:
        x = x[0]
        x = x.permute(2, 0, 1)
    C, H, W = x.shape
    rm = x.mean(dim=2)
    rm_mean = rm.mean(dim=1, keepdim=True)
    rm_std = rm.std(dim=1, keepdim=True).clamp(min=1e-8)
    rm_norm = (rm - rm_mean) / rm_std
    corrs = []
    for c in range(C):
        for h in range(H - 1):
            corrs.append(rm_norm[c, h].item() * rm_norm[c, h+1].item())
    return np.mean(corrs)

input_sources = {"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4}
decode_sources = {"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4, "dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1}
model = AlphaEarthFoundations(
    model_size="small",
    input_sources=input_sources,
    decode_sources=decode_sources,
    per_source_latent=32,
    enable_text_align=False,
)

checkpoint_path = "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000200_seed42.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["model_state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)

device = "npu:0"
model = model.to(device)
model.eval()

B, T, H, W = 1, 4, 128, 128
source_data = {
    "s1": torch.randn(B, T, H, W, 2, device=device),
    "s2": torch.randn(B, T, H, W, 6, device=device),
    "tianyi_sar": torch.randn(B, T, H, W, 1, device=device),
    "landsat": torch.randn(B, T, H, W, 6, device=device),
    "planet": torch.randn(B, T, H, W, 4, device=device),
}
timestamps = {k: torch.rand(B, T, device=device) for k in source_data.keys()}
valid_periods = [(0.0, 1.0)]

with torch.no_grad():
    x = model._stack_inputs(source_data)
    first_src = next(iter(model.input_sources.keys()))
    ts = timestamps[first_src]
    
    feats_teacher = model.encoder(x, ts)
    print(f"feats_teacher row_mean_corr: {rmc(feats_teacher):.4f}")
    
    student_srcs, student_ts_dict = model._perturb_inputs(source_data, timestamps)
    x_student = model._stack_inputs(student_srcs)
    ts_student = student_ts_dict[first_src]
    feats_student = model.encoder(x_student, ts_student)
    print(f"feats_student row_mean_corr: {rmc(feats_student):.4f}")
    
    vp = torch.tensor(valid_periods, device=device)
    
    mu_t = model.summarizer(feats_teacher, ts, vp)
    mu_s = model.summarizer(feats_student, ts_student, vp)
    
    print(f"mu_t row_mean_corr: {rmc(mu_t):.4f}")
    print(f"mu_s row_mean_corr: {rmc(mu_s):.4f}")
    
    # Check if row centering is actually applied in summarizer
    # Manually replicate summarizer
    B2, T2, H2, W2, C2 = feats_teacher.shape
    feats_2d = feats_teacher.view(B2 * T2, H2, W2, C2).permute(0, 3, 1, 2).contiguous()
    feats_2d = model.summarizer.spatial_smooth(feats_2d)
    feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B2, T2, H2, W2, C2)
    q = model.summarizer.summarizer_q(vp)
    z = model.summarizer.time_pool(feats_smooth, q, mask=None)
    print(f"z (teacher) row_mean_corr: {rmc(z):.4f}")
    
    z_centered = z - z.mean(dim=2, keepdim=True)
    print(f"z_centered row_mean_corr: {rmc(z_centered):.4f}")
    
    mu_manual = model.summarizer.proj_64(z_centered)
    print(f"mu_manual row_mean_corr: {rmc(mu_manual):.4f}")
