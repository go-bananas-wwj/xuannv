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

checkpoint_path = "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000200_seed42.pt"
print(f"Loading {checkpoint_path}...")
model = load_model(checkpoint_path).to(device)
model.eval()
print("Running forward...")
with torch.no_grad():
    out = model(source_data, timestamps, valid_periods)

student_emb = out["embeddings"][0].detach().cpu().numpy()
H, W, D = student_emb.shape

row_means = student_emb.mean(axis=1)
row_mean_std = row_means.std(axis=0).mean()
within_row_std = student_emb.std(axis=1).mean()

row_flat = student_emb.reshape(H, W * D)
row_norm = row_flat / (np.linalg.norm(row_flat, axis=1, keepdims=True) + 1e-8)
cos_sims = [np.dot(row_norm[h], row_norm[h+1]) for h in range(H - 1)]
avg_cos_sim = np.mean(cos_sims)

print(f"Row mean std: {row_mean_std:.6f}")
print(f"Within-row std: {within_row_std:.6f}")
print(f"Adjacent row cos_sim: {avg_cos_sim:.6f}")
