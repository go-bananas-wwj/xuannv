from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point, box
from sklearn.metrics import roc_auc_score

from . import backbone

warnings.filterwarnings("ignore")

PERIODS = {
    "june": {"before": (2025, 4), "after": (2025, 7)},
    "aug": {"before": (2025, 7), "after": (2025, 10)},
    "September": {"before": (2025, 7), "after": (2025, 10)},
    "October": {"before": (2025, 7), "after": (2025, 10)},
}


def _harbin_cfg(cfg: Any) -> Any:
    cfg.data.manifest_path = "/workspace/xuannv/data_raw/harbin/scenes"
    cfg.data.stats_dir = "/workspace/xuannv/statistics/harbin"
    cfg.data.num_samples = 424
    cfg.data.preload = True
    return cfg


def _load_grid(grid_path: Path) -> dict[str, tuple]:
    with open(grid_path) as f:
        data = json.load(f)
    bounds: dict[str, tuple] = {}
    for feat in data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return bounds


def _load_changes(annot_dir: Path) -> dict[str, list]:
    period_changes: dict[str, list] = {p: [] for p in PERIODS}
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        period = shp_name.replace(".shp", "")
        path = annot_dir / shp_name
        if not path.exists():
            continue
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            warnings.warn(f"无法读取 {path}: {exc}")
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        period_changes[period] = list(gdf.geometry)
    return period_changes


def run_change_detection(
    model_dir: str,
    output_dir: str,
    device: str = "npu:0",
    periods: list[str] | None = None,
    annot_dir: str | None = None,
    grid_path: str | None = None,
    patch_limit: int | None = None,
    patch_ids: list[str] | None = None,
) -> dict[str, Any]:
    if periods is None:
        periods = list(PERIODS.keys())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, cfg = backbone.load_production_model(model_dir, device=device)
    cfg = _harbin_cfg(cfg)

    sys.path.insert(0, str(backbone._project_root()))
    from src.data.dataset import HarbinPatchDataset

    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    patch_ids = patch_ids if patch_ids is not None else [p for p in dataset.patches]
    if patch_limit is not None:
        patch_ids = patch_ids[:patch_limit]

    period_results: dict[str, Any] = {}
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_ch_means: list[float] = []
    all_unch_means: list[float] = []

    evaluate = annot_dir is not None and grid_path is not None
    if evaluate:
        annot_dir = Path(annot_dir)
        grid_path = Path(grid_path)
        patch_bounds = _load_grid(grid_path)
        period_changes = _load_changes(annot_dir)

    for period in periods:
        if period not in PERIODS:
            raise ValueError(f"未知 period: {period}")
        pinfo = PERIODS[period]
        before_y, before_m = pinfo["before"]
        after_y, after_m = pinfo["after"]

        emb_before = backbone.extract_embeddings_for_patches(
            model, dataset, patch_ids, before_y, before_m, device
        )
        emb_after = backbone.extract_embeddings_for_patches(
            model, dataset, patch_ids, after_y, after_m, device
        )
        common_pids = [p for p in patch_ids if p in emb_before and p in emb_after]
        if not common_pids:
            continue

        scores = {
            p: 1.0 - np.sum(emb_before[p] * emb_after[p], axis=0)
            for p in common_pids
        }
        score_arr = np.stack([scores[p] for p in common_pids], axis=0)
        np.savez_compressed(
            out_dir / f"change_score_{period}.npz",
            patch_ids=np.array(common_pids),
            scores=score_arr,
        )

        if not evaluate:
            continue

        changes = period_changes.get(period, [])
        if not changes:
            continue

        annotated_pids = {
            pid
            for geom in changes
            for pid, bounds in patch_bounds.items()
            if box(*bounds).intersects(geom) and pid in common_pids
        }

        p_scores: list[float] = []
        p_labels: list[int] = []
        p_ch: list[float] = []
        p_unch: list[float] = []

        for local_pid in sorted(annotated_pids):
            if local_pid not in common_pids:
                continue
            bounds = patch_bounds[local_pid]
            minx, miny, maxx, maxy = bounds
            eb = emb_before[local_pid]
            ea = emb_after[local_pid]
            H, W = eb.shape[1], eb.shape[2]
            changed_mask = np.zeros((H, W), dtype=bool)

            for geom in changes:
                if not box(minx, miny, maxx, maxy).intersects(geom):
                    continue
                for y in range(H):
                    for x in range(W):
                        px = minx + (x + 0.5) / W * (maxx - minx)
                        py = maxy - (y + 0.5) / H * (maxy - miny)
                        if geom.buffer(1.0).contains(Point(px, py)):
                            changed_mask[y, x] = True

            dist_map = 1.0 - np.sum(eb * ea, axis=0)
            lflat = changed_mask.flatten()
            sflat = dist_map.flatten()
            if lflat.sum() == 0 or lflat.sum() == len(lflat):
                continue

            p_scores.extend(sflat.tolist())
            p_labels.extend(lflat.tolist())
            p_ch.append(float(dist_map[changed_mask].mean()))
            p_unch.append(float(dist_map[~changed_mask].mean()))

            all_scores.extend(sflat.tolist())
            all_labels.extend(lflat.tolist())
            all_ch_means.append(p_ch[-1])
            all_unch_means.append(p_unch[-1])

        if p_labels and 0 < sum(p_labels) < len(p_labels):
            period_results[period] = {
                "auc": float(roc_auc_score(p_labels, p_scores)),
                "changed_mean": float(np.mean(p_ch)),
                "unchanged_mean": float(np.mean(p_unch)),
                "separation": float(np.mean(p_ch) - np.mean(p_unch)),
                "n_samples": len(p_labels),
                "n_positive": int(sum(p_labels)),
            }

    result: dict[str, Any] = {"periods": period_results}

    if evaluate and all_labels:
        result["global"] = {
            "auc": float(roc_auc_score(all_labels, all_scores)),
            "changed_mean": float(np.mean(all_ch_means)),
            "unchanged_mean": float(np.mean(all_unch_means)),
            "separation": float(np.mean(all_ch_means) - np.mean(all_unch_means)),
        }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="双时相变化检测")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output-dir", default="outputs/changedetection")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--periods",
        default="june,aug,September,October",
        help="逗号分隔的 period 列表",
    )
    parser.add_argument(
        "--annot-dir",
        default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件",
    )
    parser.add_argument(
        "--grid",
        default="/workspace/index/harbin/grid/harbin_grid.geojson",
    )
    args = parser.parse_args()

    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    run_change_detection(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        periods=periods,
        annot_dir=args.annot_dir,
        grid_path=args.grid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
