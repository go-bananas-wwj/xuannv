#!/usr/bin/env python3
"""Validate HR-only model on change detection AUC (2025 windows)."""
import os, sys, json, time, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

# HR-only 验证窗口: 2025 Apr-May vs 2025 Sep-Oct
BEFORE_WINDOW = (1743465600000.0, 1748736000000.0)   # 2025-04-01 ~ 2025-06-01
AFTER_WINDOW = (1754265600000.0, 1759267200000.0)    # 2025-08-05 ~ 2025-11-01

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="/workspace/xuannv/configs/qwen_v2_hr_only_small.yaml")
    parser.add_argument("--gpu", type=str, default="3")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("="*60)
    print("  HR-only 变化检测验证")
    print(f"  CKPT: {args.ckpt}")
    print(f"  CONFIG: {args.config}")
    print("="*60)

    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    def load_model(ckpt_path, config_path):
        cfg = load_config(config_path)
        model = AEFModel(cfg).to("cuda:0")
        ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        dataset = HarbinPatchDataset(cfg)
        dataset.training = False
        dataset._spatial_augmentation = False
        return model, dataset

    print("\n加载模型...")
    model, ds = load_model(args.ckpt, args.config)

    # ──────────────────────────────────────────
    # 加载 Grid 和标注
    # ──────────────────────────────────────────
    print("\n加载 Grid...")
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    print("加载标注数据...")
    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                if row.geometry is not None:
                    all_changes.append({"geometry": row.geometry, "period": shp_name.replace(".shp", "")})
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  {shp_name}: ERROR - {e}")

    print(f"  总计: {len(all_changes)} polygons")

    # ──────────────────────────────────────────
    # 提取 embedding
    # ──────────────────────────────────────────
    def extract_emb(model, dataset, patch_idx, valid_start_ms, valid_end_ms):
        batch = dataset[patch_idx]
        batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
        batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.unsqueeze(0).to("cuda:0")
            else:
                batch_dev[k] = v
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

    def compute_change_map(emb_before, emb_after):
        D, H, W = emb_before.shape
        fb = emb_before.reshape(D, -1)
        fa = emb_after.reshape(D, -1)
        nb = np.linalg.norm(fb, axis=0, keepdims=True)
        na = np.linalg.norm(fa, axis=0, keepdims=True)
        fb = fb / np.maximum(nb, 1e-8)
        fa = fa / np.maximum(na, 1e-8)
        cos_sim = np.sum(fb * fa, axis=0)
        return ((1.0 - cos_sim) / 2.0).reshape(H, W)

    # 找有标注的 patches
    patch_to_changes = {}
    for ch in all_changes:
        geom = ch["geometry"]
        for pid, bounds in patch_bounds.items():
            patch_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
            if geom.intersects(patch_box):
                if pid not in patch_to_changes:
                    patch_to_changes[pid] = []
                patch_to_changes[pid].append(ch)

    # 取 top-20 有最多变化的 patch
    test_patches = sorted(patch_to_changes.items(),
                          key=lambda x: sum(c["geometry"].area for c in x[1]),
                          reverse=True)[:20]

    print(f"\n测试 {len(test_patches)} 个有标注的 patch")

    results = []
    distances = []

    for i, (pid, changes) in enumerate(test_patches):
        if pid not in ds.patches:
            continue
        pidx = ds.patches.index(pid)
        print(f"  [{i+1}/{len(test_patches)}] {pid}... ", end="", flush=True)
        t0 = time.time()

        try:
            eb = extract_emb(model, ds, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1])
            ea = extract_emb(model, ds, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1])
            cd = compute_change_map(eb, ea)

            # 光栅化标注
            bounds = patch_bounds[pid]
            H, W = 64, 64
            resolution = (bounds[2] - bounds[0]) / H
            change_mask = np.zeros((H, W), dtype=np.float32)

            for ch_info in changes:
                geom = ch_info["geometry"]
                for px in range(H):
                    for py in range(W):
                        wx = bounds[0] + (px + 0.5) * resolution
                        wy = bounds[3] - (py + 0.5) * resolution
                        if geom.contains(Point(wx, wy)):
                            change_mask[px, py] = 1.0

            flat_cd = cd.flatten()
            flat_mask = change_mask.flatten()

            if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
                auc = roc_auc_score(flat_mask, flat_cd)
            else:
                auc = None

            elapsed = time.time() - t0
            mean_dist = float(cd.mean())
            print(f"AUC={'%.3f' % auc if auc else 'N/A'}, dist={mean_dist:.4f} ({elapsed:.1f}s)")

            if auc is not None:
                results.append(auc)
            distances.append(mean_dist)

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    # ──────────────────────────────────────────
    # 汇总
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  结果汇总")
    print("="*60)

    if results:
        print(f"\nHR-only: {len(results)} patches, AUC mean={np.mean(results):.3f}, "
              f"median={np.median(results):.3f}, std={np.std(results):.3f}")
        print(f"  AUC > 0.55: {sum(1 for a in results if a > 0.55)}/{len(results)}")
        print(f"  AUC > 0.60: {sum(1 for a in results if a > 0.60)}/{len(results)}")
        print(f"  AUC > 0.70: {sum(1 for a in results if a > 0.70)}/{len(results)}")
    if distances:
        print(f"\nMean embedding distance: {np.mean(distances):.4f}")

    print("\n" + "="*60)
    print("验证完成")
    print("="*60)


if __name__ == "__main__":
    main()
