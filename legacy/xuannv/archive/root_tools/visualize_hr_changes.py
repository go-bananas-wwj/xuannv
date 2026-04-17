#!/usr/bin/env python3
"""Visualize HR-only model outputs: segmentation + adjacent-month change intensity."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

# ── Config ──
CKPT = "/workspace/outputs/aef_qwen_v2_hr_only_small/epoch_59.pt"
CONFIG = "/workspace/xuannv/configs/qwen_v2_hr_only_small.yaml"
OUT_DIR = "/workspace/outputs/aef_qwen_v2_hr_only_small/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# s2_hr timestamps for 2025
MONTHLY_WINDOWS = [
    ("2025-04", 1744171200000.0, 1744171200000.0),
    ("2025-06", 1750564800000.0, 1750564800000.0),
    ("2025-08", 1754280000000.0, 1754280000000.0),
    ("2025-09", 1756699200000.0, 1756699200000.0),
    ("2025-10", 1759291200000.0, 1759291200000.0),
]

ADJACENT_PAIRS = [
    ("Apr-Jun", 1744171200000.0, 1744171200000.0, 1750564800000.0, 1750564800000.0),
    ("Jun-Aug", 1750564800000.0, 1750564800000.0, 1754280000000.0, 1754280000000.0),
    ("Aug-Sep", 1754280000000.0, 1754280000000.0, 1756699200000.0, 1756699200000.0),
    ("Sep-Oct", 1756699200000.0, 1756699200000.0, 1759291200000.0, 1759291200000.0),
]

# Load model
cfg = load_config(CONFIG)
model = AEFModel(cfg).to("cuda:0")
ckpt = torch.load(CKPT, map_location="cuda:0", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ds = HarbinPatchDataset(cfg)
ds.training = False

def extract_outputs(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
    batch_dev = {k: (v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(
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
    emb = F.normalize(out.embedding_map, p=2, dim=1)[0].cpu().numpy()  # [D, H, W]
    recon = out.reconstructions[0].cpu().numpy()  # [T_tgt, C, H, W]
    return emb, recon, batch_dev

# Find target source index for worldcover
target_names = [t["name"] for t in cfg.data.target_sources]
worldcover_idx = target_names.index("worldcover")

# Load grid and annotations
with open("/workspace/index/harbin/grid/harbin_grid.geojson") as f:
    grid_data = json.load(f)
patch_bounds = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

all_changes = []
for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
    gdf = gpd.read_file(f"/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件/{shp_name}")
    if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    for _, row in gdf.iterrows():
        if row.geometry is not None:
            all_changes.append({"geometry": row.geometry, "period": shp_name.replace(".shp", "")})

patch_to_changes = {}
for ch in all_changes:
    geom = ch["geometry"]
    for pid, bounds in patch_bounds.items():
        if geom.intersects(box(*bounds)):
            if pid not in patch_to_changes:
                patch_to_changes[pid] = []
            patch_to_changes[pid].append(ch)

# Pick representative patches: high AUC, low AUC, medium
selected_patches = ["patch_000146", "patch_000235", "patch_000230"]

def embedding_to_rgb(emb):
    """emb: [D, H, W] -> [H, W, 3] via PCA."""
    D, H, W = emb.shape
    flat = emb.reshape(D, -1).T
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat).reshape(H, W, 3)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

def compute_change_map_cosine(e1, e2):
    D, H, W = e1.shape
    a = e1.reshape(D, -1)
    b = e2.reshape(D, -1)
    a = a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-8)
    cs = np.sum(a * b, axis=0)
    return ((1.0 - cs) / 2.0).reshape(H, W)

def rasterize_mask(pid, changes, H=64, W=64):
    bounds = patch_bounds[pid]
    resolution = (bounds[2] - bounds[0]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for ch in changes:
        geom = ch["geometry"]
        for px in range(H):
            for py in range(W):
                wx = bounds[0] + (px + 0.5) * resolution
                wy = bounds[3] - (py + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[px, py] = 1.0
    return mask

for pid in selected_patches:
    if pid not in ds.patches:
        continue
    pidx = ds.patches.index(pid)
    
    # Collect embeddings and segmentations for each month
    embeddings = []
    segmentations = []
    rgb_embs = []
    for name, vs, ve in MONTHLY_WINDOWS:
        emb, recon, batch = extract_outputs(model, ds, pidx, vs, ve)
        embeddings.append((name, emb))
        rgb_embs.append(embedding_to_rgb(emb))
        # WorldCover prediction (argmax over classes)
        wc_logits = recon[worldcover_idx]  # [11, H, W]
        seg = wc_logits.argmax(axis=0)
        segmentations.append(seg)
    
    # Compute adjacent-month change intensity
    change_maps = []
    for name, vs1, ve1, vs2, ve2 in ADJACENT_PAIRS:
        emb1, _, _ = extract_outputs(model, ds, pidx, vs1, ve1)
        emb2, _, _ = extract_outputs(model, ds, pidx, vs2, ve2)
        cd = compute_change_map_cosine(emb1, emb2)
        change_maps.append((name, cd))
    
    # ── Plot 1: 全域分割图 (WorldCover Segmentation) ──
    n_months = len(MONTHLY_WINDOWS)
    fig, axes = plt.subplots(1, n_months, figsize=(4*n_months, 4))
    if n_months == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("tab20", 11)
    for ax, (name, seg) in zip(axes, zip([n for n, _, _ in MONTHLY_WINDOWS], segmentations)):
        ax.imshow(seg, cmap=cmap, vmin=0, vmax=10)
        ax.set_title(f"{pid} - {name}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{pid}_segmentation.png", dpi=150)
    plt.close()
    
    # ── Plot 2: Embedding PCA RGB ──
    fig, axes = plt.subplots(1, n_months, figsize=(4*n_months, 4))
    if n_months == 1:
        axes = [axes]
    for ax, (name, rgb) in zip(axes, zip([n for n, _, _ in MONTHLY_WINDOWS], rgb_embs)):
        ax.imshow(rgb)
        ax.set_title(f"{pid} - {name} embedding RGB")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{pid}_embedding_rgb.png", dpi=150)
    plt.close()
    
    # ── Plot 3: 相邻月份变化强度 (Adjacent Month Change Intensity) ──
    n_pairs = len(change_maps)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4*n_pairs, 4))
    if n_pairs == 1:
        axes = [axes]
    for ax, (name, cd) in zip(axes, change_maps):
        im = ax.imshow(cd, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"{pid} - {name} change intensity")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{pid}_change_intensity.png", dpi=150)
    plt.close()
    
    # ── Plot 4: 变化强度 + 标注叠加 ──
    if pid in patch_to_changes:
        change_mask = rasterize_mask(pid, patch_to_changes[pid])
        fig, axes = plt.subplots(1, n_pairs, figsize=(4*n_pairs, 4))
        if n_pairs == 1:
            axes = [axes]
        for ax, (name, cd) in zip(axes, change_maps):
            ax.imshow(cd, cmap="hot", vmin=0, vmax=1)
            # Overlay contour of annotation mask
            ax.contour(change_mask, levels=[0.5], colors="cyan", linewidths=1.5)
            ax.set_title(f"{pid} - {name} + annotation")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{pid}_change_intensity_annotated.png", dpi=150)
        plt.close()
    
    print(f"Saved visualizations for {pid}")

print(f"\nAll visualizations saved to {OUT_DIR}")
