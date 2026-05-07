#!/usr/bin/env python3
"""
生成每个 SHP 文件的变化检测结果综合大图
- 使用 V2 模型 (aef_qwen_v2)
- 输出季度复合影像版 和 月度单景影像版
- 每个 patch 一行：Before RGB | After RGB | GT Mask | 10-shot Prob | Unsupervised Score | Embedding PCA
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, Point
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

# ── 配置 ─────────────────────────────────────────────────────────
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH   = "/workspace/outputs/aef_qwen_v2/best.pt"
DEVICE      = torch.device("npu:0" if torch.npu.is_available() else "cpu")

SHP_DIR     = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
GRID_PATH   = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
Q_S2_ROOT   = Path("/workspace/raw/harbin/s2")
M_S2_ROOT   = Path("/workspace/raw/harbin_scenes/s2")

OUTPUT_Q    = Path("/workspace/outputs/aef_qwen_v2/shp_maps_quarterly")
OUTPUT_M    = Path("/workspace/outputs/aef_qwen_v2/shp_maps_monthly")

def _date_to_ms(date_str: str) -> float:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return float(dt.timestamp() * 1000)


PERIODS = {
    "2025-04~2025-06": (_date_to_ms("2025-04-01"), _date_to_ms("2025-06-30")),
    "2025-06~2025-08": (_date_to_ms("2025-06-01"), _date_to_ms("2025-08-31")),
    "2025-08~2025-09": (_date_to_ms("2025-08-01"), _date_to_ms("2025-09-30")),
    "2025-09~2025-10": (_date_to_ms("2025-09-01"), _date_to_ms("2025-10-31")),
    "2025-all":        (_date_to_ms("2025-01-01"), _date_to_ms("2025-12-31")),
}

QUARTER_MAP = {
    "2025-04~2025-06": ("2025Q1", "2025Q2"),
    "2025-06~2025-08": ("2025Q2", "2025Q3"),
    "2025-08~2025-09": ("2025Q2", "2025Q3"),
    "2025-09~2025-10": ("2025Q3", "2025Q4"),
    "2025-all":        ("2025Q2", "2025Q3"),
}

SHP_FILES = {
    "june.shp":             {"period": "2025-04~2025-06", "category": "mixed"},
    "aug.shp":              {"period": "2025-06~2025-08", "category": "mixed"},
    "September.shp":        {"period": "2025-08~2025-09", "category": "mixed"},
    "October.shp":          {"period": "2025-09~2025-10", "category": "mixed"},
    "SAR建筑工地.shp":      {"period": "2025-all",        "category": "construction"},
    "SAR房屋拆除.shp":      {"period": "2025-all",        "category": "demolition"},
    "SAR非农非粮.shp":      {"period": "2025-all",        "category": "farmland"},
    "SAR疑似违建.shp":      {"period": "2025-all",        "category": "construction"},
}

SHOT_COUNT = 10
PATCHES_PER_PAGE = 4

# ── 辅助函数 ─────────────────────────────────────────────────────
def _ms_to_date(ms: float) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0)
    return dt.strftime("%Y-%m-%d")


def load_model():
    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset


def extract_embedding(model, dataset, patch_id: str, w_start: float, w_end: float):
    if patch_id not in dataset.patches:
        return None
    pidx = dataset.patches.index(patch_id)
    batch = dataset[pidx]
    batch["valid_start_ms"] = torch.tensor(w_start, dtype=torch.float64)
    batch["valid_end_ms"]   = torch.tensor(w_end,   dtype=torch.float64)
    batch_dev = {k: v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
    emb = out.embedding_map[0].cpu().numpy()  # [D, H, W]
    return emb


def normalize_emb(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=0, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def compute_unsupervised_score(emb_before, emb_after):
    D, H, W = emb_before.shape
    fb = normalize_emb(emb_before).reshape(D, -1)
    fa = normalize_emb(emb_after).reshape(D, -1)
    cos_sim = np.sum(fb * fa, axis=0)
    score = ((1.0 - cos_sim) / 2.0).reshape(H, W)
    return score


def compute_10shot(emb_before, emb_after, gt_mask):
    D, H, W = emb_before.shape
    features = np.zeros((H * W, D * 2), dtype=np.float32)
    for px in range(H):
        for py in range(W):
            idx = px * W + py
            features[idx, :D] = emb_before[:, px, py]
            features[idx, D:] = emb_after[:, px, py]
    labels = gt_mask.flatten()

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None, None

    n_neg = min(len(pos_idx) * 10, len(neg_idx))
    neg_sample = np.random.choice(neg_idx, n_neg, replace=False)
    sampled = np.concatenate([pos_idx, neg_sample])
    np.random.shuffle(sampled)

    X = features[sampled]
    y = labels[sampled]

    rng = np.random.RandomState(42)
    train_samples = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n_s = min(SHOT_COUNT, len(c_idx))
        if n_s > 0:
            train_samples.extend(rng.choice(c_idx, n_s, replace=False).tolist())
    if len(train_samples) < 2 or len(np.unique(y[train_samples])) < 2:
        return None, None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_samples])
    X_all_s   = scaler.transform(features)

    classifiers = [
        ("Linear", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ("kNN-3", KNeighborsClassifier(n_neighbors=min(3, max(1, len(train_samples) // 10)))),
    ]
    best_clf, best_auc = None, -1
    for name, clf in classifiers:
        try:
            clf.fit(X_train_s, y[train_samples])
            train_probs = clf.predict_proba(X_train_s)[:, 1]
            auc = roc_auc_score(y[train_samples], train_probs)
            if auc > best_auc:
                best_auc = auc
                best_clf = clf
        except Exception:
            pass
    if best_clf is None:
        return None, None
    probs = best_clf.predict_proba(X_all_s)[:, 1]
    prob_map = probs.reshape(H, W)
    return prob_map, train_samples


def embedding_pca_rgb(emb_before, emb_after):
    D, H, W = emb_before.shape
    concat = np.concatenate([emb_before, emb_after], axis=0)  # [2D, H, W]
    flat = concat.reshape(concat.shape[0], -1).T  # [H*W, 2D]
    pca = PCA(n_components=3)
    pc = pca.fit_transform(flat)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        ch = pc[:, c]
        lo, hi = np.percentile(ch, (2, 98))
        if hi > lo:
            rgb[:, :, c] = ((ch - lo) / (hi - lo)).reshape(H, W)
        else:
            rgb[:, :, c] = 0.5
    rgb = np.clip(rgb, 0, 1)
    return rgb


# ── RGB 读取 ─────────────────────────────────────────────────────
def _read_s2_rgb(path: Path, target_size: int = 64) -> np.ndarray | None:
    """读取 S2 TIFF 并返回 [H, W, 3] float32 RGB (0-1)."""
    if not path.exists():
        return None
    try:
        with rasterio.open(str(path)) as src:
            data = src.read(out_shape=(src.count, target_size, target_size), resampling=Resampling.bilinear)
        if data.shape[0] < 3:
            return None
        rgb = data[[2, 1, 0]].astype(np.float32)
        valid = rgb[rgb > 0]
        if len(valid) == 0:
            return None
        p2, p98 = np.percentile(valid, [2, 98])
        if p98 > p2:
            rgb = (rgb - p2) / (p98 - p2)
        rgb = np.clip(rgb, 0, 1)
        return rgb.transpose(1, 2, 0)  # [H, W, 3]
    except Exception as e:
        return None


def load_rgb_quarterly(patch_id: str, period: str, target_size: int = 64):
    q_before, q_after = QUARTER_MAP[period]
    b = _read_s2_rgb(Q_S2_ROOT / patch_id / f"{q_before}.tif", target_size)
    a = _read_s2_rgb(Q_S2_ROOT / patch_id / f"{q_after}.tif", target_size)
    return b, a


def load_rgb_monthly(patch_id: str, period: str, target_size: int = 64):
    bs, be = PERIODS[period]
    mid = (bs + be) / 2.0
    b = _load_scene_median(patch_id, bs, mid, target_size)
    a = _load_scene_median(patch_id, mid, be, target_size)
    return b, a


def _load_scene_median(patch_id: str, start_ms: float, end_ms: float, target_size: int):
    s2_dir = M_S2_ROOT / patch_id
    if not s2_dir.exists():
        return None
    files = sorted(s2_dir.glob("*.tif"))
    valid_files = []
    for f in files:
        stem = f.stem
        if len(stem) >= 8 and stem.isdigit():
            ts = float(datetime.strptime(stem[:8], "%Y%m%d").timestamp() * 1000)
            if start_ms <= ts <= end_ms:
                valid_files.append(f)
    # fallback: 取最近的一个
    if not valid_files and files:
        candidates = []
        for f in files:
            stem = f.stem
            if len(stem) >= 8 and stem.isdigit():
                ts = float(datetime.strptime(stem[:8], "%Y%m%d").timestamp() * 1000)
                candidates.append((abs(ts - (start_ms + end_ms) / 2), f))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            valid_files = [candidates[0][1]]
    if not valid_files:
        return None
    frames = []
    for f in valid_files[:5]:
        rgb = _read_s2_rgb(f, target_size)
        if rgb is not None:
            frames.append(rgb)
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0)


# ── SHP / Grid 处理 ──────────────────────────────────────────────
def load_grid():
    gdf = gpd.read_file(str(GRID_PATH))
    return gdf


def find_patches_for_shp(grid, shp_path: Path):
    gdf = gpd.read_file(str(shp_path))
    if gdf.crs is None or gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    joined = gpd.sjoin(grid, gdf, how="inner", predicate="intersects")
    pids = joined["patch_id"].unique().tolist()
    return pids, gdf


def rasterize_shp_for_patch(patch_geom, shp_gdf, size=64):
    minx, miny, maxx, maxy = patch_geom.bounds
    transform = from_bounds(minx, miny, maxx, maxy, size, size)
    geoms = []
    for _, row in shp_gdf.iterrows():
        geom = row.geometry
        if geom is not None and geom.intersects(patch_geom):
            inter = geom.intersection(patch_geom)
            if not inter.is_empty:
                geoms.append((inter, 1))
    if not geoms:
        return np.zeros((size, size), dtype=np.uint8)
    mask = rasterize(geoms, out_shape=(size, size), transform=transform, fill=0, dtype=np.uint8)
    return mask


# ── 可视化 ───────────────────────────────────────────────────────
def draw_page(patches_data, page_idx, shp_name, out_dir, mode_label):
    n = len(patches_data)
    fig, axes = plt.subplots(n, 6, figsize=(24, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    cols = ["Before RGB", "After RGB", "GT Mask", "10-shot Prob", "Unsupervised", "Embedding PCA"]
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=12, fontweight="bold")

    for i, d in enumerate(patches_data):
        # 1. Before RGB
        if d["rgb_before"] is not None:
            axes[i, 0].imshow(d["rgb_before"])
        axes[i, 0].set_ylabel(d["patch_id"], fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # 2. After RGB
        if d["rgb_after"] is not None:
            axes[i, 1].imshow(d["rgb_after"])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # 3. GT Mask
        axes[i, 2].imshow(d["gt_mask"], cmap="Reds", interpolation="nearest")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # 4. 10-shot Prob
        if d["prob_map"] is not None:
            im = axes[i, 3].imshow(d["prob_map"], cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

        # 5. Unsupervised
        im2 = axes[i, 4].imshow(d["score_map"], cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[i, 4], fraction=0.046)
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])

        # 6. PCA
        axes[i, 5].imshow(d["pca_rgb"], interpolation="nearest")
        # overlay GT boundary
        mask_overlay = d["gt_mask"].astype(float)
        mask_overlay[d["gt_mask"] == 0] = np.nan
        axes[i, 5].imshow(mask_overlay, cmap=ListedColormap(["red"]), alpha=0.3, interpolation="nearest")
        axes[i, 5].set_xticks([])
        axes[i, 5].set_yticks([])

    fig.suptitle(f"{shp_name} ({mode_label}) — Page {page_idx + 1}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = out_dir / shp_name.replace(".shp", "") / f"page_{page_idx + 1:02d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── 主流程 ───────────────────────────────────────────────────────
def process_shp(model, dataset, grid, shp_name, info, mode: str):
    out_dir = OUTPUT_Q if mode == "quarterly" else OUTPUT_M
    period = info["period"]
    bs, be = PERIODS[period]
    mid = (bs + be) / 2.0

    pids, shp_gdf = find_patches_for_shp(grid, SHP_DIR / shp_name)
    if not pids:
        print(f"  [!] {shp_name}: no intersecting patches")
        return
    print(f"  {shp_name}: {len(pids)} patches, mode={mode}")

    all_data = []
    for pid in pids:
        patch_row = grid[grid["patch_id"] == pid]
        if len(patch_row) == 0:
            continue
        patch_geom = patch_row.geometry.values[0]

        # embedding
        t0 = time.time()
        emb_before = extract_embedding(model, dataset, pid, bs, mid)
        emb_after  = extract_embedding(model, dataset, pid, mid, be)
        if emb_before is None or emb_after is None:
            print(f"    {pid}: embedding failed")
            continue
        emb_before = normalize_emb(emb_before)
        emb_after  = normalize_emb(emb_after)
        H, W = emb_before.shape[1], emb_before.shape[2]

        # GT mask
        gt_mask = rasterize_shp_for_patch(patch_geom, shp_gdf, size=H)

        # 10-shot
        prob_map, _ = compute_10shot(emb_before, emb_after, gt_mask)

        # unsupervised
        score_map = compute_unsupervised_score(emb_before, emb_after)

        # PCA
        pca_rgb = embedding_pca_rgb(emb_before, emb_after)

        # RGB
        if mode == "quarterly":
            rgb_before, rgb_after = load_rgb_quarterly(pid, period, target_size=H)
        else:
            rgb_before, rgb_after = load_rgb_monthly(pid, period, target_size=H)

        all_data.append({
            "patch_id": pid,
            "rgb_before": rgb_before,
            "rgb_after": rgb_after,
            "gt_mask": gt_mask,
            "prob_map": prob_map,
            "score_map": score_map,
            "pca_rgb": pca_rgb,
        })
        print(f"    {pid}: OK ({time.time()-t0:.1f}s)")

    # 分页输出
    for page_idx in range(0, len(all_data), PATCHES_PER_PAGE):
        page_data = all_data[page_idx:page_idx + PATCHES_PER_PAGE]
        out_path = draw_page(page_data, page_idx // PATCHES_PER_PAGE, shp_name, out_dir, mode)
        print(f"    Saved: {out_path}")


def main():
    print("=" * 60)
    print("Loading V2 model...")
    model, dataset = load_model()
    print(f"Model on {DEVICE}, dataset has {len(dataset.patches)} patches")

    print("Loading grid...")
    grid = load_grid()

    for shp_name, info in SHP_FILES.items():
        print(f"\n--- Processing {shp_name} ---")
        process_shp(model, dataset, grid, shp_name, info, mode="quarterly")
        process_shp(model, dataset, grid, shp_name, info, mode="monthly")

    print("\nAll done.")


if __name__ == "__main__":
    main()
