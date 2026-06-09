"""Quick visualization of embedding for a single checkpoint"""
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
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn

def load_model(checkpoint_path):
    input_sources = {"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4}
    decode_sources = {"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4, "dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1}
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources=input_sources,
        decode_sources=decode_sources,
        per_source_latent=32,
        enable_text_align=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

device = "npu:0"

# Load sample
dataset = HaidianAEFDataset(
    data_root="/workspace/xuannv/data_raw/haidian/scenes",
    planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
    source_names=["s1", "s2", "tianyi_sar", "landsat", "planet"],
    aef_embedding_root="/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches",
    split="train",
    start_date="2025-12-01",
    end_date="2026-04-30",
)

patch_id = "patch_000036"
sample = None
for s in dataset:
    if s["patch_id"] == patch_id:
        sample = s
        break

batch = collate_fn([sample])
source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
valid_periods = batch["valid_periods"]

def embed_to_rgb(emb):
    """PCA to RGB"""
    H, W, D = emb.shape
    flat = emb.reshape(-1, D)
    mean = flat.mean(axis=0)
    centered = flat - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:3].T
    proj = centered @ basis
    proj = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-8)
    return proj.reshape(H, W, 3)

# Visualize multiple checkpoints
checkpoints = {
    "Baseline (step 200)": "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000200_seed42.pt",
    "ExpP fix-uniform fast-warm (step 200)": "/workspace/xuannv/aef_reference/outputs/aef_distill_expP_fixuniform_seed42/step_000200_seed42.pt",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AEF Official
aef_emb_path = f"/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy"
aef_emb = np.load(aef_emb_path)
aef_emb = np.transpose(aef_emb, (1, 2, 0))
aef_rgb = embed_to_rgb(aef_emb)
axes[0].imshow(aef_rgb)
axes[0].set_title("AEF Official")
axes[0].axis('off')

idx = 1
for name, path in checkpoints.items():
    model = load_model(path).to(device)
    model.eval()
    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)
    student_emb = out["embeddings"][0].detach().cpu().numpy()
    student_rgb = embed_to_rgb(student_emb)
    axes[idx].imshow(student_rgb)
    axes[idx].set_title(name)
    axes[idx].axis('off')
    idx += 1

plt.tight_layout()
plt.savefig("/workspace/xuannv/aef_reference/debug_viz_comparison.png", dpi=150)
print("Saved to debug_viz_comparison.png")
