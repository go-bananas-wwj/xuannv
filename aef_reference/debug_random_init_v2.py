"""诊断：随机初始化模型的输出是否天然有水平条带？"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from src.aef.architecture.aef_module import AlphaEarthFoundations

torch.manual_seed(42)

# Build model
model = AlphaEarthFoundations(
    input_sources={"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4},
    decode_sources={"dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1},
    enable_text_align=False,
).npu()

model.eval()

# Random input
B, H, W = 2, 128, 128
x = {
    "s1": torch.randn(B, 4, H, W, 2).npu(),
    "s2": torch.randn(B, 4, H, W, 6).npu(),
    "tianyi_sar": torch.randn(B, 4, H, W, 1).npu(),
    "landsat": torch.randn(B, 4, H, W, 6).npu(),
    "planet": torch.randn(B, 4, H, W, 4).npu(),
}
ts = {k: torch.randn(B, 4).npu() for k in x.keys()}
vp = [(0.0, 1.0) for _ in range(B)]

with torch.no_grad():
    out = model(x, ts, vp)

emb = out["embeddings"].cpu().numpy()  # (B, H, W, 64)

# Check y-correlation
from scipy.stats import pearsonr
b, h, w, d = emb.shape
for bi in range(b):
    print(f"\n=== Sample {bi} ===")
    # y-gradient per channel
    y_grad = np.abs(emb[bi, 1:, :, :] - emb[bi, :-1, :, :]).mean()
    print(f"Mean y-gradient: {y_grad:.4f}")
    
    # y-correlation between adjacent rows
    y_corrs = []
    for c in range(d):
        for yi in range(h - 1):
            c1, c2 = pearsonr(emb[bi, yi, :, c], emb[bi, yi + 1, :, c])
            y_corrs.append(c1)
    print(f"Mean y-corr (per-channel): {np.mean(y_corrs):.4f}")
    
    # Check if ALL channels have same pattern
    flat = emb[bi].reshape(h * w, d)
    corr_matrix = np.corrcoef(flat.T)
    off_diag = corr_matrix[np.triu_indices(d, k=1)]
    print(f"Mean channel-channel corr: {np.mean(off_diag):.4f}")
    print(f"Std channel-channel corr: {np.std(off_diag):.4f}")

# Visualize PCA RGB
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
emb_flat = emb[0].reshape(-1, 64)
pca_rgb = pca.fit_transform(emb_flat)
pca_rgb = (pca_rgb - pca_rgb.min(axis=0)) / (pca_rgb.max(axis=0) - pca_rgb.min(axis=0) + 1e-8)
pca_rgb = pca_rgb.reshape(H, W, 3)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(pca_rgb)
axes[0].set_title('Random Init PCA RGB')
axes[0].axis('off')

# Show mean across channels
mean_img = emb[0].mean(axis=-1)
axes[1].imshow(mean_img, cmap='gray')
axes[1].set_title('Mean across 64 channels')
axes[1].axis('off')

plt.savefig('/workspace/xuannv/aef_reference/debug_random_init_v2.png', dpi=150, bbox_inches='tight')
print("\nSaved to debug_random_init_v2.png")
