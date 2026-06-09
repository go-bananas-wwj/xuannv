"""诊断脚本：只用 S2 输入，检查随机初始化模型的 PCA RGB。"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch_npu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn

# 加载数据集 - 只用 S2
dataset = HaidianAEFDataset(
    data_root="/workspace/xuannv/data_raw/haidian/scenes",
    planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
    stats_dir="/workspace/xuannv/statistics/haidian",
    split="val",
    image_size=128,
    start_date="2025-12-01",
    end_date="2026-04-30",
    source_names=["s2"],
    required_sources=["s2"],
    aef_embedding_root="/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches",
)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
)
batch = next(iter(loader))

# 构建随机初始化模型 - 只用 S2
input_sources = {"s2": 6}
decode_sources = {k: v for k, v in input_sources.items()}
model = AlphaEarthFoundations(
    input_sources=input_sources,
    decode_sources=decode_sources,
    per_source_latent=32,
    enable_text_align=False,
)
model.eval()

device = "npu:0"
model = model.to(device)

with torch.no_grad():
    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"].to(device)
    
    out = model(source_data, timestamps, valid_periods)
    
    student_emb = out["embeddings"].detach().cpu().numpy()
    aef_emb = batch["aef_embedding"].numpy() if batch.get("aef_embedding") is not None else None

def pca_rgb(emb, title, ax):
    B, H, W, D = emb.shape
    flat = emb.reshape(-1, D)
    mean = flat.mean(axis=0)
    centered = flat - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    rgb = (centered @ Vt[:3].T) * S[:3]
    rgb = rgb.reshape(B, H, W, 3)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    ax.imshow(rgb[0])
    ax.set_title(title)
    ax.axis('off')

fig, axes = plt.subplots(1, 2 if aef_emb is not None else 1, figsize=(12, 5))
if aef_emb is not None:
    pca_rgb(aef_emb[:, :, :, :64], "AEF Official (S2 only)", axes[0])
    pca_rgb(student_emb, "Student Random Init (S2 only)", axes[1])
else:
    pca_rgb(student_emb, "Student Random Init (S2 only)", axes)

plt.tight_layout()
plt.savefig("/workspace/xuannv/aef_reference/random_init_s2_only_pca.png", dpi=150)
print("Saved to random_init_s2_only_pca.png")

from scipy.stats import pearsonr
def analyze_y_corr(emb, name):
    H, W, D = emb.shape[1:]
    corrs = []
    for b in range(emb.shape[0]):
        for d in range(D):
            for y in range(H - 1):
                c, _ = pearsonr(emb[b, y, :, d], emb[b, y+1, :, d])
                corrs.append(c)
    print(f"{name}: mean_y_corr={np.mean(corrs):.4f}, max={np.max(corrs):.4f}, min={np.min(corrs):.4f}")

analyze_y_corr(student_emb, "Student Random Init (S2 only)")
student_norm = student_emb / (np.linalg.norm(student_emb, axis=-1, keepdims=True) + 1e-8)
analyze_y_corr(student_norm, "Student Random Init (S2 only, L2 norm)")

if aef_emb is not None:
    analyze_y_corr(aef_emb[:, :, :, :64], "AEF Official (S2 only)")
    aef_norm = aef_emb[:, :, :, :64] / (np.linalg.norm(aef_emb[:, :, :, :64], axis=-1, keepdims=True) + 1e-8)
    analyze_y_corr(aef_norm, "AEF Official (S2 only, L2 norm)")
