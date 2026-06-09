import torch

# Load random init (from step 5 or create fresh)
# Actually, let's compare step 200 with step 5 (very early)
ckpt_5 = torch.load('/workspace/xuannv/aef_reference/outputs/aef_distill_control_exp/step_000005_seed42.pt', map_location='cpu')
ckpt_200 = torch.load('/workspace/xuannv/aef_reference/outputs/aef_distill_control_exp/step_000200_seed42.pt', map_location='cpu')

sd_5 = ckpt_5['model_state_dict']
sd_200 = ckpt_200['model_state_dict']

has_module = any(k.startswith('module.') for k in sd_5.keys())
if has_module:
    sd_5 = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd_5.items()}
    sd_200 = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd_200.items()}

# Compare some key weights
keys = ['summarizer.proj_64.weight', 'encoder.spatial_fusion.0.weight', 'encoder.input_projection.weight']
for k in keys:
    if k in sd_5 and k in sd_200:
        w5 = sd_5[k]
        w200 = sd_200[k]
        diff = (w200 - w5).abs().mean()
        rel_diff = diff / (w5.abs().mean() + 1e-8)
        print(f'{k}: mean_abs_change={diff:.6f}, rel_change={rel_diff:.4f}')
    else:
        print(f'{k}: not found')
