#!/usr/bin/env python3
"""V8 Clean 变化检测验证 — 像素级多卡并行，计算 per-patch AUC.

用法:
    cd /workspace/xuannv
    python scripts/eval/validate_v8_clean.py \
        --config configs/xuannv_v8_clean.yaml \
        --checkpoint /workspace/outputs/xuannv_backbone_v8_clean/epoch_best_epoch223.pt \
        --devices npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7
"""
import sys, json, time, argparse, multiprocessing as mp
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)


def load_model(cfg_path, ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.inference.engine import extract_embedding_map

    cfg = load_config(cfg_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, extract_embedding_map


def compute_change_map(emb_before, emb_after):
    """像素级 cosine distance change map."""
    D, H, W = emb_before.shape
    fb = emb_before.reshape(D, -1)
    fa = emb_after.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8)
    fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    return ((1.0 - cos_sim) / 2.0).reshape(H, W)


def rasterize_annotations(changes, bounds, H=64, W=64):
    """将变化图斑光栅化为 HxW mask."""
    resolution = (bounds[2] - bounds[0]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


def worker_process(device, cfg_path, ckpt_path, patch_infos, return_dict):
    """子进程：在指定 NPU 上处理一批 patch."""
    torch.npu.set_device(device)
    model, dataset, extract_embedding_map = load_model(cfg_path, ckpt_path, device)

    results = []
    for pid, pidx, bounds, changes in patch_infos:
        try:
            eb = extract_embedding_map(model, dataset, pidx, BEFORE_WINDOW[0], BEFORE_WINDOW[1], device, normalize=True)
            ea = extract_embedding_map(model, dataset, pidx, AFTER_WINDOW[0], AFTER_WINDOW[1], device, normalize=True)
            cd_map = compute_change_map(eb, ea)

            change_mask = rasterize_annotations(changes, bounds)
            flat_cd = cd_map.flatten()
            flat_mask = change_mask.flatten()

            # 需要既有变化像素又有无变化像素才能算AUC
            if flat_mask.sum() > 10 and (1 - flat_mask).sum() > 10:
                auc = roc_auc_score(flat_mask, flat_cd)
                results.append({
                    "patch_id": pid,
                    "auc": auc,
                    "cd_mean": float(cd_map.mean()),
                    "cd_std": float(cd_map.std()),
                    "mask_ratio": float(flat_mask.mean()),
                })
        except Exception as e:
            print(f"  [{device}] {pid} ERROR: {e}")

    return_dict[device] = results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xuannv_v8_clean.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--devices", type=str, default="npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7")
    parser.add_argument("--top-k", type=int, default=0, help="只测变化面积最大的K个patch，0=全部")
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]

    print("=" * 70)
    print("  V8 Clean 像素级变化检测验证 — 多卡并行")
    print("=" * 70)
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Devices:    {devices}")
    print("=" * 70)

    # ── 加载 Grid ──
    print("\n[1/3] 加载 Grid 和标注...")
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    # ── 加载标注 ──
    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                all_changes.append({"geometry": geom, "patch_id": row.get("patch_id", None)})
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

    # 按 patch 聚合标注
    patch_changes = {}
    for change in all_changes:
        if change["patch_id"]:
            pid = change["patch_id"]
            patch_changes.setdefault(pid, []).append(change["geometry"])
        else:
            pt = change["geometry"].centroid
            for pid, bounds in patch_bounds.items():
                if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                    patch_changes.setdefault(pid, []).append(change["geometry"])
                    break

    # 过滤并排序
    from src.config import load_config
    from src.data.dataset import HarbinPatchDataset
    cfg = load_config(args.config)
    cfg.data.preload = False
    ds = HarbinPatchDataset(cfg)

    test_patches = []
    for pid, changes in patch_changes.items():
        if pid not in ds.patches:
            continue
        total_area = sum(g.area for g in changes)
        test_patches.append((pid, ds.patches.index(pid), patch_bounds[pid], changes, total_area))

    # 按变化面积排序
    test_patches.sort(key=lambda x: x[4], reverse=True)

    if args.top_k > 0:
        test_patches = test_patches[:args.top_k]

    print(f"  总标注 patch: {len(test_patches)}")

    # ── 多卡并行提取 ──
    print(f"\n[2/3] 多卡并行提取 embedding ({len(devices)} 卡)...")
    start = time.time()

    manager = mp.Manager()
    return_dict = manager.dict()

    n = len(test_patches)
    chunk_size = (n + len(devices) - 1) // len(devices)
    chunks = [test_patches[i:i + chunk_size] for i in range(0, n, chunk_size)]

    processes = []
    for device, chunk in zip(devices, chunks):
        # 只传递必要信息，去掉area字段
        patch_infos = [(pid, pidx, bounds, changes) for pid, pidx, bounds, changes, _ in chunk]
        p = mp.Process(target=worker_process, args=(device, args.config, args.checkpoint, patch_infos, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start
    print(f"  提取完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ── 汇总 ──
    print("\n[3/3] 计算 AUC...")
    all_results = []
    for device in devices:
        if device in return_dict:
            all_results.extend(return_dict[device])

    if not all_results:
        print("  无有效结果")
        return

    aucs = [r["auc"] for r in all_results]
    print(f"  有效 patch 数: {len(aucs)}")

    print("\n" + "=" * 70)
    print(f"  AUC mean:   {np.mean(aucs):.4f}")
    print(f"  AUC median: {np.median(aucs):.4f}")
    print(f"  AUC std:    {np.std(aucs):.4f}")
    print(f"  AUC min:    {np.min(aucs):.4f}")
    print(f"  AUC max:    {np.max(aucs):.4f}")
    print(f"  AUC > 0.6:  {sum(1 for a in aucs if a > 0.6)}/{len(aucs)}")
    print(f"  AUC > 0.7:  {sum(1 for a in aucs if a > 0.7)}/{len(aucs)}")
    print(f"  AUC > 0.8:  {sum(1 for a in aucs if a > 0.8)}/{len(aucs)}")
    print("=" * 70)

    # Top/Bottom patches
    print("\nTop 5 AUC:")
    for r in sorted(all_results, key=lambda x: x["auc"], reverse=True)[:5]:
        print(f"  {r['patch_id']}: AUC={r['auc']:.4f} cd_mean={r['cd_mean']:.4f} mask_ratio={r['mask_ratio']:.3f}")

    print("\nBottom 5 AUC:")
    for r in sorted(all_results, key=lambda x: x["auc"])[:5]:
        print(f"  {r['patch_id']}: AUC={r['auc']:.4f} cd_mean={r['cd_mean']:.4f} mask_ratio={r['mask_ratio']:.3f}")


if __name__ == "__main__":
    main()
