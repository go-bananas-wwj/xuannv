"""Generate comparison visualization: 200 step vs 1000 step"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch_npu
from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn

def _embed_to_rgb_shared(emb, basis):
    emb_n = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    proj = emb_n @ basis
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    return np.clip(proj, 0, 1)

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

# Load checkpoints
checkpoints = {
    "Step 200": "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000200_seed42.pt",
    "Step 1000 (baseline)": "/workspace/xuannv/aef_reference/outputs/aef_distill_expH_seed42/step_001000_seed42.pt",
}

results = {}
for name, path in checkpoints.items():
    model = load_model(path).to(device)
    model.eval()
    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)
    student_emb = out["embeddings"][0].detach().cpu().numpy()
    results[name] = student_emb

# AEF official
aef_emb_path = f"/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy"
aef_emb = np.load(aef_emb_path)
aef_emb = np.transpose(aef_emb, (1, 2, 0))

# PCA basis from all
D = 64
all_embs = [aef_emb.reshape(-1, D)]
for emb in results.values():
    all_embs.append(emb.reshape(-1, D))
flat = np.concatenate(all_embs, axis=0)
U, S, Vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
basis = Vt[:3].T

# Generate comparison figure
fig = plt.figure(figsize=(20, 5))
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

# AEF Official
ax = fig.add_subplot(gs[0, 0])
ax.imshow(_embed_to_rgb_shared(aef_emb, basis))
ax.set_title("AEF Official (PCA RGB)", fontsize=12, fontweight='bold')
ax.axis('off')

# Step 200
ax = fig.add_subplot(gs[0, 1])
ax.imshow(_embed_to_rgb_shared(results["Step 200"], basis))
ax.set_title("Student @ Step 200", fontsize=12, fontweight='bold')
ax.axis('off')

# Step 1000
ax = fig.add_subplot(gs[0, 2])
ax.imshow(_embed_to_rgb_shared(results["Step 1000 (baseline)"], basis))
ax.set_title("Student @ Step 1000", fontsize=12, fontweight='bold')
ax.axis('off')

# Difference: Step 1000 vs AEF
ax = fig.add_subplot(gs[0, 3])
diff = np.abs(_embed_to_rgb_shared(results["Step 1000 (baseline)"], basis) - _embed_to_rgb_shared(aef_emb, basis)).mean(axis=-1)
im = ax.imshow(diff, cmap='hot')
ax.set_title("|Student1000 - AEF|", fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f"{patch_id}: Striping comparison across training steps", fontsize=14, fontweight='bold', y=0.98)
plt.savefig("/workspace/xuannv/aef_reference/striping_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved striping_comparison.png")
