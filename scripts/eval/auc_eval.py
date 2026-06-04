#!/usr/bin/env python
"""变化检测 AUC 评估 — 统一入口。

使用 cosine distance 评估 embedding 的时间敏感性。
时间窗口基于 2025 年哈尔滨变化检测标注：
  - june.shp      : 2025-04 → 2025-06
  - aug.shp       : 2025-06 → 2025-08
  - September.shp : 2025-08 → 2025-09
  - October.shp   : 2025-09 → 2025-10

用法:
    # 使用 normalized embedding（默认）
    python auc_eval.py --config configs/config.yaml --checkpoint epoch_40.pt

    # 使用 pre-norm embedding
    python auc_eval.py --config configs/config.yaml --checkpoint epoch_40.pt \\
        --emb-type pre_norm
"""
from __future__ import annotations

import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

import argparse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
import geopandas as gpd
from shapely.geometry import box
from sklearn.metrics import roc_auc_score

# ── 时间窗口（毫秒，UTC） ────────────────────────────────────────────────────

PERIODS: dict[str, dict] = {
    "june": {
        "before": (1743436800000, 1746028799000),   # 2025-04
        "after":  (1748707200000, 1751299199000),   # 2025-06
    },
    "aug": {
        "before": (1748707200000, 1751299199000),   # 2025-06
        "after":  (1753977600000, 1756655999000),   # 2025-08
    },
    "September": {
        "before": (1753977600000, 1756655999000),   # 2025-08
        "after":  (1756656000000, 1759247999000),   # 2025-09
    },
    "October": {
        "before": (1756656000000, 1759247999000),   # 2025-09
        "after":  (1759248000000, 1761926399000),   # 2025-10
    },
}


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="变化检测 AUC 评估")
    p.add_argument("--config",     default="configs/config.yaml",
                   help="模型配置 YAML 路径")
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint (.pt) 路径")
    p.add_argument("--device",     default="npu:0",
                   help="计算设备，例如 npu:0 / cpu")
    p.add_argument("--emb-type",   default="normalized",
                   choices=["normalized", "pre_norm"],
                   help="normalized: L2 归一化 embedding；pre_norm: 归一化前的 embedding")
    p.add_argument("--annot-dir",
                   default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件",
                   help="标注 shapefile 目录")
    p.add_argument("--grid",
                   default="/workspace/index/harbin/grid/harbin_grid.geojson",
                   help="Patch 格网 GeoJSON 路径")
    p.add_argument("--output",     default="",
                   help="结果输出 JSON 路径（默认存到 checkpoint 同目录）")
    return p.parse_args()


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def load_model(config_path, ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.data.multi_region_dataset import MultiRegionPatchDataset

    cfg = load_config(config_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    cfg.data.preload = True
    if getattr(cfg.data, 'multi_region_manifest', None):
        dataset = MultiRegionPatchDataset(cfg)
    else:
        dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, cfg


# ── 月份推断 ─────────────────────────────────────────────────────────────────

def _month_from_window(start_ms: float, end_ms: float) -> tuple[int, int]:
    from datetime import datetime
    mid = (start_ms + end_ms) / 2
    dt = datetime.fromtimestamp(mid / 1000)
    return dt.year, dt.month


# ── Embedding 提取 ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(model, dataset, patch_month_index, cfg,
                      patch_id, valid_start, valid_end, device, use_pre_norm):
    year, month = _month_from_window(valid_start, valid_end)
    key = (patch_id, year, month)
    if key not in patch_month_index:
        raise ValueError(f"找不到 {patch_id} 在 {year}-{month:02d} 的数据")
    item = dataset[patch_month_index[key]]

    def _to(x):
        return x.unsqueeze(0).to(device)

    use_bf16 = getattr(cfg.training, 'use_bf16', True)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=use_bf16):
        out = model(
            source_frames        = _to(item["source_frames"]),
            source_timestamps_ms = _to(item["source_timestamps_ms"]),
            source_frame_mask    = _to(item["source_frame_mask"]),
            source_input_mask    = _to(item["source_input_mask"]),
            source_type_ids      = _to(item["source_type_ids"]),
            valid_start_ms       = torch.tensor([valid_start], dtype=torch.int64, device=device),
            valid_end_ms         = torch.tensor([valid_end],   dtype=torch.int64, device=device),
            target_relative_time = torch.zeros(1, cfg.data.num_target_sources, device=device),
            target_metadata      = torch.zeros(1, cfg.data.num_target_sources,
                                               cfg.data.metadata_dim, device=device),
            skip_decoder=True,
        )
    emb = out.pre_norm_map if use_pre_norm else out.embedding_map  # [1, D, H, W]
    emb = F.normalize(emb.float(), p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()   # [D, H, W]


# ── 主逻辑 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    use_pre_norm = args.emb_type == "pre_norm"

    print("=" * 60)
    print("  变化检测 AUC 评估")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Emb type:   {args.emb_type}")
    print("=" * 60)

    print("\n加载模型...")
    model, dataset, cfg = load_model(args.config, args.checkpoint, args.device)

    # ── Grid ──────────────────────────────────────────────────────────────
    print("加载 Grid...")
    with open(args.grid) as f:
        grid_data = json.load(f)
    patch_bounds: dict[str, tuple] = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    # ── 标注 ──────────────────────────────────────────────────────────────
    print("加载变化标注...")
    changes_by_period: dict[str, list] = {p: [] for p in PERIODS}
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        period = shp_name.replace(".shp", "")
        try:
            gdf = gpd.read_file(f"{args.annot_dir}/{shp_name}")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                if row.geometry is not None:
                    changes_by_period[period].append(row.geometry)
        except Exception as e:
            print(f"  [警告] 无法加载 {shp_name}: {e}")
    total = sum(len(v) for v in changes_by_period.values())
    print(f"  共 {total} 个变化标注")

    # ── Patch-month 索引 ──────────────────────────────────────────────────
    patch_month_index: dict[tuple, int] = {}
    local_to_full: dict[str, str] = {}  # 用于 grid local_id -> dataset full_id
    for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
        patch_month_index[(pid, year, month)] = idx
        if '_' in pid and not pid.startswith('patch_'):
            local_id = pid.split('_', 1)[1]
            local_to_full[local_id] = pid

    def _resolve_pid(local_pid: str) -> str | None:
        """将 grid 中的 local patch_id 解析为 dataset 中的 full patch_id."""
        if local_pid in local_to_full:
            return local_to_full[local_pid]
        # 如果 dataset 中本身就是 local_id（单区域场景）
        for full_pid in dataset.patches:
            if full_pid == local_pid:
                return full_pid
        return None

    # ── 按 period 评估 ────────────────────────────────────────────────────
    all_scores, all_labels = [], []
    all_ch_means, all_unch_means = [], []
    period_results: dict = {}

    for period, pinfo in PERIODS.items():
        changes = changes_by_period.get(period, [])
        if not changes:
            continue

        annotated_pids = {
            pid
            for geom in changes
            for pid, bounds in patch_bounds.items()
            if box(*bounds).intersects(geom)
        }

        p_scores, p_labels, p_ch, p_unch = [], [], [], []

        for local_pid in sorted(annotated_pids):
            full_pid = _resolve_pid(local_pid)
            if full_pid is None:
                print(f"  [跳过] {local_pid} ({period}): 在 dataset 中找不到对应 patch")
                continue
            try:
                eb = extract_embedding(model, dataset, patch_month_index, cfg,
                                       full_pid, pinfo["before"][0], pinfo["before"][1],
                                       args.device, use_pre_norm)
                ea = extract_embedding(model, dataset, patch_month_index, cfg,
                                       full_pid, pinfo["after"][0], pinfo["after"][1],
                                       args.device, use_pre_norm)
            except Exception as e:
                print(f"  [跳过] {local_pid} -> {full_pid} ({period}): {e}")
                continue

            D, H, W = eb.shape
            changed_mask = np.zeros((H, W), dtype=bool)
            bounds = patch_bounds[pid]
            minx, miny, maxx, maxy = bounds
            for geom in changes:
                if not box(minx, miny, maxx, maxy).intersects(geom):
                    continue
                for y in range(H):
                    for x in range(W):
                        px = minx + (x + 0.5) / W * (maxx - minx)
                        py = maxy - (y + 0.5) / H * (maxy - miny)
                        if geom.contains(box(px, py, px, py)):
                            changed_mask[y, x] = True

            dist_map = 1.0 - np.sum(eb * ea, axis=0)  # cosine distance [H, W]
            lflat = changed_mask.flatten()
            sflat = dist_map.flatten()

            if lflat.sum() == 0 or lflat.sum() == len(lflat):
                continue

            p_scores.extend(sflat.tolist()); p_labels.extend(lflat.tolist())
            all_scores.extend(sflat.tolist()); all_labels.extend(lflat.tolist())

            ch   = float(dist_map[changed_mask].mean()) if changed_mask.any() else 0.0
            unch = float(dist_map[~changed_mask].mean()) if (~changed_mask).any() else 0.0
            p_ch.append(ch); p_unch.append(unch)
            all_ch_means.append(ch); all_unch_means.append(unch)

        if p_labels and 0 < sum(p_labels) < len(p_labels):
            period_results[period] = {
                "auc":            float(roc_auc_score(p_labels, p_scores)),
                "changed_mean":   float(np.mean(p_ch)),
                "unchanged_mean": float(np.mean(p_unch)),
                "separation":     float(np.mean(p_ch) - np.mean(p_unch)),
                "n_samples":      len(p_labels),
                "n_positive":     int(sum(p_labels)),
            }

    # ── 全局 AUC ──────────────────────────────────────────────────────────
    if not all_labels:
        print("\n[ERROR] 没有有效的评估样本")
        sys.exit(1)

    global_auc  = roc_auc_score(all_labels, all_scores)
    global_ch   = float(np.mean(all_ch_means))
    global_unch = float(np.mean(all_unch_means))
    sep         = global_ch - global_unch

    print("\n" + "=" * 60)
    print("  AUC 评估结果")
    print("=" * 60)
    print(f"  全局 AUC:             {global_auc:.4f}")
    print(f"  Changed mean dist:    {global_ch:.4f}")
    print(f"  Unchanged mean dist:  {global_unch:.4f}")
    print(f"  Separation:           {sep:.4f}")
    print("\n  分时期 AUC:")
    for period, res in period_results.items():
        print(f"    {period:12s}: AUC={res['auc']:.4f}  sep={res['separation']:.4f}"
              f"  n={res['n_samples']} pos={res['n_positive']}")
    print("=" * 60)

    result = {
        "config": args.config, "checkpoint": args.checkpoint,
        "device": args.device, "emb_type": args.emb_type,
        "global": {
            "auc": float(global_auc), "changed_mean": global_ch,
            "unchanged_mean": global_unch, "separation": sep,
        },
        "periods": period_results,
    }

    suffix = "_prenorm" if use_pre_norm else ""
    out_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), f"auc_result{suffix}.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
