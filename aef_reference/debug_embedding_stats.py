"""Debug embedding statistics at 2000 steps"""
import sys
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import torch
import numpy as np
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

# Check multiple checkpoints
checkpoints = {
    "Step 200": "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000200_seed42.pt",
    "Step 1000": "/workspace/xuannv/aef_reference/outputs/aef_distill_expI_seed42/step_001000_seed42.pt",
    "Step 2000": "/workspace/xuannv/aef_reference/outputs/aef_distill_expI_seed42/step_002000_seed42.pt",
}

for name, path in checkpoints.items():
    model = load_model(path).to(device)
    model.eval()
    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)
    
    student_emb = out["embeddings"][0].detach().cpu().numpy()  # (H, W, 64)
    
    # Compute statistics
    H, W, D = student_emb.shape
    
    # Row means
    row_means = student_emb.mean(axis=1)  # (H, 64)
    row_mean_std = row_means.std(axis=0).mean()  # Average std across channels
    
    # Within-row std
    within_row_std = student_emb.std(axis=1).mean()  # Average within-row std
    
    # Adjacent row cosine similarity
    row_flat = student_emb.reshape(H, W * D)
    row_norm = row_flat / (np.linalg.norm(row_flat, axis=1, keepdims=True) + 1e-8)
    cos_sims = []
    for h in range(H - 1):
        cos_sims.append(np.dot(row_norm[h], row_norm[h+1]))
    avg_cos_sim = np.mean(cos_sims)
    
    print(f"\n=== {name} ===")
    print(f"  Row mean std (avg across channels): {row_mean_std:.4f}")
    print(f"  Within-row std (avg): {within_row_std:.4f}")
    print(f"  Adjacent row cos_sim: {avg_cos_sim:.4f}")
    print(f"  Shape: {student_emb.shape}")
    
    # Per-channel row mean std
    for c in range(min(5, D)):
        rm = student_emb[:, :, c].mean(axis=1)
        print(f"  Channel {c} row mean range: [{rm.min():.4f}, {rm.max():.4f}], std: {rm.std():.4f}")

# Also check AEF official
aef_emb_path = f"/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy"
aef_emb = np.load(aef_emb_path)
aef_emb = np.transpose(aef_emb, (1, 2, 0))

H, W, D = aef_emb.shape
row_means = aef_emb.mean(axis=1)
row_mean_std = row_means.std(axis=0).mean()
within_row_std = aef_emb.std(axis=1).mean()
row_flat = aef_emb.reshape(H, W * D)
row_norm = row_flat / (np.linalg.norm(row_flat, axis=1, keepdims=True) + 1e-8)
cos_sims = []
for h in range(H - 1):
    cos_sims.append(np.dot(row_norm[h], row_norm[h+1]))
avg_cos_sim = np.mean(cos_sims)

print(f"\n=== AEF Official ===")
print(f"  Row mean std (avg across channels): {row_mean_std:.4f}")
print(f"  Within-row std (avg): {within_row_std:.4f}")
print(f"  Adjacent row cos_sim: {avg_cos_sim:.4f}")
for c in range(min(5, D)):
    rm = aef_emb[:, :, c].mean(axis=1)
    print(f"  Channel {c} row mean range: [{rm.min():.4f}, {rm.max():.4f}], std: {rm.std():.4f}")
