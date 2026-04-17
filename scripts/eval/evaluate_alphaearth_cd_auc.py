#!/usr/bin/env python3
"""Evaluate AlphaEarth official embedding vs local aef_qwen_v2 on change detection AUC."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import rasterio
from sklearn.metrics import roc_auc_score
from shapely.geometry import box, Point
import geopandas as gpd

sys.path.insert(0, "/workspace/xuannv")
from demo_v2.cache_manager import cache
from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.utils.constants import TIME_WINDOWS

# ── Paths ──
ALPHA_DIR = Path("/workspace/outputs/alphaearth_harbin")
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
OUTPUT_REPORT = Path("/workspace/xuannv/alphaearth_auc_report.md")

# ── Load AlphaEarth GeoTIFFs ──
def load_alphaearth_year(year: int):
    path = ALPHA_DIR / f"alphaearth_harbin_{year}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    ds = rasterio.open(path)
    return ds

# ── Rasterize annotations per patch ──
def rasterize_annotations(patch_id: str, grid_data: dict, all_changes: list, target_shape: tuple):
    """Rasterize change polygons to target_shape for given patch."""
    for feat in grid_data["features"]:
        if feat["properties"]["patch_id"] == patch_id:
            coords = feat["geometry"]["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            break
    else:
        return None

    H, W = target_shape
    resolution_x = (maxx - minx) / W
    resolution_y = (maxy - miny) / H

    mask = np.zeros((H, W), dtype=np.float32)
    patch_box = box(minx, miny, maxx, maxy)

    for ch in all_changes:
        geom = ch["geometry"]
        if geom is None or not geom.intersects(patch_box):
            continue
        # rasterize by point-in-poly sampling
        bounds = geom.bounds
        px_min = max(0, int((bounds[0] - minx) / resolution_x) - 1)
        px_max = min(W, int((bounds[2] - minx) / resolution_x) + 2)
        py_min = max(0, int((maxy - bounds[3]) / resolution_y) - 1)
        py_max = min(H, int((maxy - bounds[1]) / resolution_y) + 2)
        for px in range(px_min, px_max):
            wx = minx + (px + 0.5) * resolution_x
            for py in range(py_min, py_max):
                wy = maxy - (py + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[py, px] = 1.0
    return mask


# ── Extract AlphaEarth per-patch embeddings ──
def extract_alphaearth_patch(ds: rasterio.DatasetReader, patch_id: str, grid_data: dict):
    for feat in grid_data["features"]:
        if feat["properties"]["patch_id"] == patch_id:
            coords = feat["geometry"]["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            break
    else:
        return None

    # Get window in pixel coords
    row_off, col_off = rasterio.transform.rowcol(ds.transform, minx, maxy)
    row_off2, col_off2 = rasterio.transform.rowcol(ds.transform, maxx, miny)
    height = row_off2 - row_off
    width = col_off2 - col_off

    if height <= 0 or width <= 0:
        return None

    window = rasterio.windows.Window(col_off, row_off, width, height)
    data = ds.read(window=window)
    # data shape: [64, H, W]
    return data


# ── Compute cosine distance ──
def cosine_distance_map(emb_before: np.ndarray, emb_after: np.ndarray) -> np.ndarray:
    D, H, W = emb_before.shape
    fb = emb_before.reshape(D, -1)
    fa = emb_after.reshape(D, -1)
    # AlphaEarth embeddings are already unit-length, but normalize just in case
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    score = ((1.0 - np.clip(cos_sim, -1.0, 1.0)) / 2.0).reshape(H, W)
    return score


# ── Local aef_qwen_v2 change score ──
def local_v2_change_score(patch_id: str, before_key: str, after_key: str) -> np.ndarray | None:
    engine = ChangeDetectionEngine("v2")
    before_window = TIME_WINDOWS[before_key]
    after_window = TIME_WINDOWS[after_key]
    emb_before = engine.get_embedding(patch_id, before_window[0], before_window[1], use_precomputed=True)
    emb_after = engine.get_embedding(patch_id, after_window[0], after_window[1], use_precomputed=True)
    if emb_before is None or emb_after is None:
        return None
    # If precomputed same embedding, can't compute real change
    if np.array_equal(emb_before, emb_after):
        return None
    return cosine_distance_map(emb_before, emb_after)


# ── Main evaluation ──
def main():
    print("Loading cache and grid data...")
    cache.load()
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    print("Loading annotations...")
    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        shp_path = ANNOT_DIR / shp_name
        if not shp_path.exists():
            print(f"  Warning: {shp_path} not found, skipping.")
            continue
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                all_changes.append({"geometry": row.geometry, "period": shp_name.replace(".shp", "")})
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  Error loading {shp_name}: {e}")
    print(f"Total annotations: {len(all_changes)}")

    print("Loading AlphaEarth datasets...")
    ds_2023 = load_alphaearth_year(2023)
    ds_2024 = load_alphaearth_year(2024)

    before_key = "2023 全年"
    after_key = "2024 全年"

    results = []
    valid_patches = []

    for pid in cache.patch_ids:
        # Extract AlphaEarth embeddings
        emb_2023 = extract_alphaearth_patch(ds_2023, pid, grid_data)
        emb_2024 = extract_alphaearth_patch(ds_2024, pid, grid_data)
        if emb_2023 is None or emb_2024 is None:
            continue
        if emb_2023.shape != emb_2024.shape:
            continue

        alpha_score = cosine_distance_map(emb_2023, emb_2024)
        mask = rasterize_annotations(pid, grid_data, all_changes, alpha_score.shape)
        if mask is None:
            continue

        # Need both positive and negative samples for AUC
        has_label = mask > 0
        no_label = mask == 0
        if has_label.sum() < 5 or no_label.sum() < 5:
            continue

        alpha_auc = roc_auc_score(mask.flatten(), alpha_score.flatten())

        # Local v2 score
        v2_score = local_v2_change_score(pid, before_key, after_key)
        v2_auc = None
        if v2_score is not None:
            # Resize v2_score to match alpha/mask shape if needed
            if v2_score.shape != alpha_score.shape:
                import torch.nn.functional as F
                v2_tensor = torch.from_numpy(v2_score).unsqueeze(0).unsqueeze(0).float()
                v2_resized = F.interpolate(v2_tensor, size=alpha_score.shape, mode='bilinear', align_corners=False)
                v2_score = v2_resized.squeeze(0).squeeze(0).numpy()
            v2_auc = roc_auc_score(mask.flatten(), v2_score.flatten())

        valid_patches.append(pid)
        results.append({
            "patch_id": pid,
            "alpha_auc": alpha_auc,
            "v2_auc": v2_auc,
            "n_positive": int(has_label.sum()),
            "n_negative": int(no_label.sum()),
        })
        v2_str = f"{v2_auc:.3f}" if v2_auc is not None else "N/A"
        print(f"  {pid}: AlphaEarth AUC={alpha_auc:.3f}, V2 AUC={v2_str}")

    ds_2023.close()
    ds_2024.close()

    # Aggregate
    alpha_aucs = [r["alpha_auc"] for r in results]
    v2_aucs = [r["v2_auc"] for r in results if r["v2_auc"] is not None]

    report_lines = [
        "# AlphaEarth vs AEF_qwen_v2 Change Detection AUC Report",
        "",
        f"**Evaluated patches**: {len(results)}",
        f"**Before window**: {before_key}",
        f"**After window**: {after_key}",
        "",
        "## AlphaEarth Official Embedding (cosine distance)",
        f"- Mean AUC: {np.mean(alpha_aucs):.4f}",
        f"- Std AUC:  {np.std(alpha_aucs):.4f}",
        f"- Median AUC: {np.median(alpha_aucs):.4f}",
        f"- Min AUC: {np.min(alpha_aucs):.4f}",
        f"- Max AUC: {np.max(alpha_aucs):.4f}",
        "",
    ]
    if v2_aucs:
        report_lines += [
            "## Local AEF_qwen_v2 Embedding (cosine distance)",
            f"- Mean AUC: {np.mean(v2_aucs):.4f}",
            f"- Std AUC:  {np.std(v2_aucs):.4f}",
            f"- Median AUC: {np.median(v2_aucs):.4f}",
            f"- Min AUC: {np.min(v2_aucs):.4f}",
            f"- Max AUC: {np.max(v2_aucs):.4f}",
            "",
        ]
    else:
        report_lines += [
            "## Local AEF_qwen_v2 Embedding",
            "- No valid real-time embeddings available for comparison (precomputed mode only).",
            "",
        ]

    report_lines += [
        "## Per-Patch Details",
        "",
        "| Patch ID | AlphaEarth AUC | V2 AUC | Positive Pixels | Negative Pixels |",
        "|----------|----------------|--------|-----------------|-----------------|",
    ]
    for r in results:
        v2_str = f"{r['v2_auc']:.4f}" if r["v2_auc"] is not None else "N/A"
        report_lines.append(
            f"| {r['patch_id']} | {r['alpha_auc']:.4f} | {v2_str} | {r['n_positive']} | {r['n_negative']} |"
        )

    report = "\n".join(report_lines)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
