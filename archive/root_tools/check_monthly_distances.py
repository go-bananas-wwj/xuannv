#!/usr/bin/env python3
"""Compute monthly interval embedding distances for V2 main model."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/epoch_499.pt"

# Single month approximations using mid-month timestamps
SINGLE_MONTHS = {
    "2025-04": (float(1743465600000.0), float(1746057600000.0)),  # Apr 1 - Apr 30
    "2025-05": (float(1746057600000.0), float(1748736000000.0)),  # May 1 - May 31
    "2025-06": (float(1748736000000.0), float(1751328000000.0)),  # Jun 1 - Jun 30
    "2025-07": (float(1751328000000.0), float(1754006400000.0)),  # Jul 1 - Jul 31
    "2025-08": (float(1754006400000.0), float(1756684800000.0)),  # Aug 1 - Aug 31
    "2025-09": (float(1756684800000.0), float(1759267200000.0)),  # Sep 1 - Sep 30
    "2025-10": (float(1759267200000.0), float(1761945600000.0)),  # Oct 1 - Oct 31
}

# Benchmark period windows
PERIOD_WINDOWS = {
    "2025-04~2025-06": (float(1743465600000.0), float(1751328000000.0)),
    "2025-06~2025-08": (float(1748736000000.0), float(1756684800000.0)),
    "2025-08~2025-09": (float(1754006400000.0), float(1759267200000.0)),
    "2025-09~2025-10": (float(1756089600000.0), float(1761945600000.0)),
}

def extract_emb(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
    batch_dev = {k: (v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        output = model(
            source_frames=batch_dev["source_frames"],
            source_timestamps_ms=batch_dev["source_timestamps_ms"],
            source_frame_mask=batch_dev["source_frame_mask"],
            source_input_mask=batch_dev["source_input_mask"],
            source_type_ids=batch_dev["source_type_ids"],
            valid_start_ms=batch_dev["valid_start_ms"],
            valid_end_ms=batch_dev["valid_end_ms"],
            target_relative_time=batch_dev["target_relative_time"],
            target_metadata=batch_dev["target_metadata"],
        )
    emb_map = output.embedding_map
    emb_map = F.normalize(emb_map, p=2, dim=1)
    return emb_map[0].cpu().numpy()

def compute_mean_distance(emb_before, emb_after):
    D, H, W = emb_before.shape
    fb = emb_before.reshape(D, -1)
    fa = emb_after.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    dist_map = ((1.0 - cos_sim) / 2.0).reshape(H, W)
    return float(np.mean(dist_map))

print("="*60)
print("V2 Main Model - Monthly Embedding Distance Analysis")
print("="*60)

cfg = load_config(CONFIG_PATH)
model = AEFModel(cfg).to("cuda:0")
ckpt = torch.load(CKPT_PATH, map_location="cuda:0", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

ds = HarbinPatchDataset(cfg)
ds.training = False
ds._spatial_augmentation = False

n_patches = 15
print(f"\nAnalyzing {n_patches} patches...")

# Pre-extract embeddings for all patches and windows to avoid redundant work
month_names = list(SINGLE_MONTHS.keys())
period_names = list(PERIOD_WINDOWS.keys())

# Extract single-month embeddings
print("\nExtracting single-month embeddings...")
single_month_embs = {}
for i in range(n_patches):
    pid = ds.patches[i]
    single_month_embs[pid] = {}
    for m in month_names:
        try:
            e = extract_emb(model, ds, i, *SINGLE_MONTHS[m])
            single_month_embs[pid][m] = e
        except Exception as e:
            pass
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{n_patches} done")

# Extract period embeddings  
print("\nExtracting period embeddings...")
period_embs = {}
for i in range(n_patches):
    pid = ds.patches[i]
    period_embs[pid] = {}
    for p in period_names:
        try:
            e = extract_emb(model, ds, i, *PERIOD_WINDOWS[p])
            period_embs[pid][p] = e
        except Exception as e:
            pass
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{n_patches} done")

# Compute distances
print("\n--- Single Month Adjacent Pairs ---")
adjacent_pairs = [
    ("Apr-May", "2025-04", "2025-05"),
    ("May-Jun", "2025-05", "2025-06"),
    ("Jun-Jul", "2025-06", "2025-07"),
    ("Jul-Aug", "2025-07", "2025-08"),
    ("Aug-Sep", "2025-08", "2025-09"),
    ("Sep-Oct", "2025-09", "2025-10"),
]
for label, m1, m2 in adjacent_pairs:
    distances = []
    for pid in ds.patches[:n_patches]:
        if m1 in single_month_embs[pid] and m2 in single_month_embs[pid]:
            d = compute_mean_distance(single_month_embs[pid][m1], single_month_embs[pid][m2])
            distances.append(d)
    if distances:
        print(f"{label}: mean_dist={np.mean(distances):.4f}, std={np.std(distances):.4f}, n={len(distances)}")

print("\n--- 2-Month Period Pairs (adjacent, overlapping) ---")
period_pairs = [
    ("Apr-Jun vs Jun-Aug", "2025-04~2025-06", "2025-06~2025-08"),
    ("Jun-Aug vs Aug-Sep", "2025-06~2025-08", "2025-08~2025-09"),
    ("Aug-Sep vs Sep-Oct", "2025-08~2025-09", "2025-09~2025-10"),
]
for label, p1, p2 in period_pairs:
    distances = []
    for pid in ds.patches[:n_patches]:
        if p1 in period_embs[pid] and p2 in period_embs[pid]:
            d = compute_mean_distance(period_embs[pid][p1], period_embs[pid][p2])
            distances.append(d)
    if distances:
        print(f"{label}: mean_dist={np.mean(distances):.4f}, std={np.std(distances):.4f}, n={len(distances)}")

print("\n--- Same Month (Control) ---")
same_pairs = [("Apr-Apr", "2025-04", "2025-04"), ("Jun-Jun", "2025-06", "2025-06")]
for label, m1, m2 in same_pairs:
    distances = []
    for pid in ds.patches[:n_patches]:
        if m1 in single_month_embs[pid] and m2 in single_month_embs[pid]:
            d = compute_mean_distance(single_month_embs[pid][m1], single_month_embs[pid][m2])
            distances.append(d)
    if distances:
        print(f"{label}: mean_dist={np.mean(distances):.4f}, std={np.std(distances):.4f}, n={len(distances)}")

print("\n--- Year-Level Reference ---")
year_before = (float(1672531200000.0), float(1703980800000.0))
year_after = (float(1735689600000.0), float(1767225600000.0))
distances = []
for i in range(n_patches):
    try:
        e1 = extract_emb(model, ds, i, *year_before)
        e2 = extract_emb(model, ds, i, *year_after)
        d = compute_mean_distance(e1, e2)
        distances.append(d)
    except:
        pass
if distances:
    print(f"2023 vs 2025: mean_dist={np.mean(distances):.4f}, std={np.std(distances):.4f}, n={len(distances)}")

print("\n" + "="*60)
print("Done")
print("="*60)
