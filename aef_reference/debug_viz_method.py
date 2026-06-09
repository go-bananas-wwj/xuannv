"""Compare different visualization methods"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations

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
    out = model(source_data, timestamps, valid_periods)
    student_emb = out['student_embeddings'][0].cpu().numpy()  # (128, 128, 64)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Method 1: PCA on student only
emb_flat = student_emb.reshape(-1, 64)
mean = emb_flat.mean(axis=0)
centered = emb_flat - mean
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
proj = centered @ eigvecs[:, :3]
for i in range(3):
    c = proj[:, i]
    proj[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
rgb = proj.reshape(128, 128, 3)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Student PCA (own basis)")
axes[0, 0].axis('off')

# Method 2: Same as _embed_to_rgb_shared (student + random aef)
aef_emb = np.random.randn(128, 128, 64) * 0.1
emb = np.concatenate([student_emb.reshape(-1, 64), aef_emb.reshape(-1, 64)], axis=0)
mean = emb.mean(axis=0)
centered = emb - mean
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
proj = centered[:128*128] @ eigvecs[:, :3]
for i in range(3):
    c = proj[:, i]
    proj[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
rgb = proj.reshape(128, 128, 3)
axes[0, 1].imshow(rgb)
axes[0, 1].set_title("Student PCA (shared with random aef)")
axes[0, 1].axis('off')

# Method 3: Per-channel min-max norm to RGB (first 3 channels)
rgb3 = student_emb[:, :, :3]
for i in range(3):
    c = rgb3[:, :, i]
    rgb3[:, :, i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
axes[0, 2].imshow(rgb3)
axes[0, 2].set_title("First 3 channels (min-max)")
axes[0, 2].axis('off')

# Method 4: Show row means for first 5 channels
for i in range(5):
    rm = student_emb[:, :, i].mean(axis=1)
    axes[1, 0].plot(rm, label=f'ch{i}')
axes[1, 0].set_title("Row means (first 5 ch)")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Method 5: Show column means for first 5 channels
for i in range(5):
    cm = student_emb[:, :, i].mean(axis=0)
    axes[1, 1].plot(cm, label=f'ch{i}')
axes[1, 1].set_title("Column means (first 5 ch)")
axes[1, 1].legend()
axes[1, 1].grid(True)

# Method 6: Histogram of all values
axes[1, 2].hist(student_emb.flatten(), bins=100)
axes[1, 2].set_title("Value histogram")

plt.tight_layout()
plt.savefig("/workspace/xuannv/aef_reference/debug_viz_methods.png", dpi=150)
print("Saved debug_viz_methods.png")
