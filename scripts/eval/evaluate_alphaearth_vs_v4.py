#!/usr/bin/env python3
"""Evaluate AlphaEarth official embedding vs V4 official on change detection AUC."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score
from shapely.geometry import box, Point
import geopandas as gpd

sys.path.insert(0, "/workspace/xuannv")

# ── Paths ──
ALPHA_DIR = Path("/workspace/outputs/alphaearth_harbin")
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
V4_EMB_DIR = Path("/workspace/outputs/aef_qwen_v4_official/monthly_embeddings_2025")
OUTPUT_REPORT = Path("/workspace/outputs/aef_qwen_v4_official/eval/alphaearth_vs_v4_report.md")

# V4 monthly windows for approximating full year
V4_BEFORE_MONTH = "2025-04"
V4_AFTER_MONTH = "2025-10"


def load_alphaearth_year(year: int):
    path = ALPHA_DIR / f"alphaearth_harbin_{year}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return rasterio.open(path)


def rasterize_annotations(patch_id: str, grid_data: dict, all_changes: list, target_shape: tuple):
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

    row_off, col_off = rasterio.transform.rowcol(ds.transform, minx, maxy)
    row_off2, col_off2 = rasterio.transform.rowcol(ds.transform, maxx, miny)
    height = row_off2 - row_off
    width = col_off2 - col_off

    if height <= 0 or width <= 0:
        return None

    window = rasterio.windows.Window(col_off, row_off, width, height)
    data = ds.read(window=window)
    return data


def cosine_distance_map(emb_before: np.ndarray, emb_after: np.ndarray) -> np.ndarray:
    D, H, W = emb_before.shape
    fb = emb_before.reshape(D, -1)
    fa = emb_after.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    score = ((1.0 - np.clip(cos_sim, -1.0, 1.0)) / 2.0).reshape(H, W)
    return score


def v4_change_score(patch_id: str) -> np.ndarray | None:
    bpath = V4_EMB_DIR / f"{patch_id}_{V4_BEFORE_MONTH}.npy"
    apath = V4_EMB_DIR / f"{patch_id}_{V4_AFTER_MONTH}.npy"
    if not bpath.exists() or not apath.exists():
        return None
    emb_b = np.load(bpath)
    emb_a = np.load(apath)
    return cosine_distance_map(emb_b, emb_a)


def main():
    print("=" * 60)
    print("  AlphaEarth vs V4 Official Change Detection AUC")
    print("=" * 60)

    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    print("\nLoading annotations...")
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

    print("\nLoading AlphaEarth datasets...")
    ds_2023 = load_alphaearth_year(2023)
    ds_2024 = load_alphaearth_year(2024)

    # Get all patch IDs from grid
    patch_ids = [feat["properties"]["patch_id"] for feat in grid_data["features"]]

    results = []
    valid_patches = []

    for pid in patch_ids:
        # AlphaEarth
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

        has_label = mask > 0
        no_label = mask == 0
        if has_label.sum() < 5 or no_label.sum() < 5:
            continue

        alpha_auc = roc_auc_score(mask.flatten(), alpha_score.flatten())

        # V4 score
        v4_score = v4_change_score(pid)
        v4_auc = None
        if v4_score is not None:
            if v4_score.shape != alpha_score.shape:
                import torch
                import torch.nn.functional as F
                v4_tensor = torch.from_numpy(v4_score).unsqueeze(0).unsqueeze(0).float()
                v4_resized = F.interpolate(v4_tensor, size=alpha_score.shape, mode='bilinear', align_corners=False)
                v4_score = v4_resized.squeeze(0).squeeze(0).numpy()
            v4_auc = roc_auc_score(mask.flatten(), v4_score.flatten())

        valid_patches.append(pid)
        results.append({
            "patch_id": pid,
            "alpha_auc": alpha_auc,
            "v4_auc": v4_auc,
            "n_positive": int(has_label.sum()),
            "n_negative": int(no_label.sum()),
        })
        v4_str = f"{v4_auc:.3f}" if v4_auc is not None else "N/A"
        print(f"  {pid}: AlphaEarth AUC={alpha_auc:.3f}, V4 AUC={v4_str}")

    ds_2023.close()
    ds_2024.close()

    # Aggregate
    alpha_aucs = [r["alpha_auc"] for r in results]
    v4_aucs = [r["v4_auc"] for r in results if r["v4_auc"] is not None]

    report_lines = [
        "# AlphaEarth vs V4 Official Change Detection AUC Report",
        "",
        f"**Evaluated patches**: {len(results)}",
        f"**Before window**: AlphaEarth=2023 全年, V4=2025-04",
        f"**After window**: AlphaEarth=2024 全年, V4=2025-10",
        "",
        "## AlphaEarth Official Embedding (cosine distance)",
        f"- Mean AUC: {np.mean(alpha_aucs):.4f}",
        f"- Std AUC:  {np.std(alpha_aucs):.4f}",
        f"- Median AUC: {np.median(alpha_aucs):.4f}",
        f"- Min AUC: {np.min(alpha_aucs):.4f}",
        f"- Max AUC: {np.max(alpha_aucs):.4f}",
        "",
    ]
    if v4_aucs:
        report_lines += [
            "## V4 Official Embedding (cosine distance, backbone bare)",
            f"- Mean AUC: {np.mean(v4_aucs):.4f}",
            f"- Std AUC:  {np.std(v4_aucs):.4f}",
            f"- Median AUC: {np.median(v4_aucs):.4f}",
            f"- Min AUC: {np.min(v4_aucs):.4f}",
            f"- Max AUC: {np.max(v4_aucs):.4f}",
            "",
            f"## V4 vs AlphaEarth",
            f"- V4 mean AUC: {np.mean(v4_aucs):.4f} vs AlphaEarth: {np.mean(alpha_aucs):.4f}",
            f"- Delta: {np.mean(v4_aucs) - np.mean(alpha_aucs):+.4f}",
            f"- V4 better patches: {sum(1 for r in results if r['v4_auc'] is not None and r['v4_auc'] > r['alpha_auc'])}/{len(v4_aucs)}",
            "",
        ]
    else:
        report_lines += [
            "## V4 Official Embedding",
            "- No valid V4 embeddings available for comparison.",
            "",
        ]

    report_lines += [
        "## Per-Patch Details",
        "",
        "| Patch ID | AlphaEarth AUC | V4 AUC | Positive Pixels | Negative Pixels |",
        "|----------|----------------|--------|-----------------|-----------------|",
    ]
    for r in results:
        v4_str = f"{r['v4_auc']:.4f}" if r["v4_auc"] is not None else "N/A"
        report_lines.append(
            f"| {r['patch_id']} | {r['alpha_auc']:.4f} | {v4_str} | {r['n_positive']} | {r['n_negative']} |"
        )

    report = "\n".join(report_lines)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved to {OUTPUT_REPORT}")

    # Also save JSON
    json_path = OUTPUT_REPORT.with_suffix(".json")
    import json as _json
    with open(json_path, "w") as f:
        _json.dump({
            "n_patches": len(results),
            "alpha_auc_mean": float(np.mean(alpha_aucs)),
            "alpha_auc_std": float(np.std(alpha_aucs)),
            "alpha_auc_median": float(np.median(alpha_aucs)),
            "v4_auc_mean": float(np.mean(v4_aucs)) if v4_aucs else None,
            "v4_auc_std": float(np.std(v4_aucs)) if v4_aucs else None,
            "v4_auc_median": float(np.median(v4_aucs)) if v4_aucs else None,
            "patch_results": results,
        }, f, indent=2)
    print(f"✅ JSON saved to {json_path}")


if __name__ == "__main__":
    main()
