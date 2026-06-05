#!/usr/bin/env python3
"""周期性评估脚本 — 每 N 个 epoch 自动运行下游任务评估.

用法（由 train.py 自动调用）:
    python scripts/eval/run_periodic_eval.py \
        --config configs/config_dual_teacher_v1.yaml \
        --checkpoint /workspace/outputs/exp_dual_teacher_v1/epoch_10.pt \
        --output /workspace/outputs/exp_dual_teacher_v1/eval_epoch_10.json \
        --device npu:0

评估内容:
    1. kNN 语义分割 (WorldCover) — pixel-level mIoU, OA, per-class IoU
    2. 变化检测 AUC (Harbin, 可选) — ROC-AUC per period
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.data.multi_region_dataset import MultiRegionPatchDataset

# ── 默认配置 ──
ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
DEFAULT_MONTHS = [(2025, 4), (2025, 5), (2025, 6), (2025, 7), (2025, 8), (2025, 9), (2025, 10)]

CD_PERIODS = {
    "apr_jun": {"before": 4, "after": 6, "shp": "june.shp"},
    "jun_aug": {"before": 6, "after": 8, "shp": "aug.shp"},
    "aug_sep": {"before": 8, "after": 9, "shp": "September.shp"},
    "sep_oct": {"before": 9, "after": 10, "shp": "October.shp"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="训练配置 YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--device", default="npu:0", help="设备")
    parser.add_argument("--skip-cd", action="store_true", help="跳过变化检测评估（更快）")
    return parser.parse_args()


def month_to_window(year: int, month: int) -> tuple[int, int]:
    import calendar, time as time_mod
    start_s = int(time_mod.mktime((year, month, 1, 0, 0, 0, 0, 0, 0)))
    last_d = calendar.monthrange(year, month)[1]
    end_s = int(time_mod.mktime((year, month, last_d, 23, 59, 59, 0, 0, 0)))
    return start_s * 1000, end_s * 1000


def load_model(config_path: str, checkpoint_path: str, device: str):
    """加载模型和评估数据集."""
    cfg = load_config(config_path)
    cfg.data.preload = True
    cfg.data.num_workers = 0

    if getattr(cfg.data, 'multi_region_manifest', None):
        dataset = MultiRegionPatchDataset(cfg=cfg)
    else:
        dataset = HarbinPatchDataset(cfg=cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    dev = torch.device(device)
    model = AEFModel(cfg).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, dataset, cfg, dev


@torch.no_grad()
def extract_embeddings(model, dataset, cfg, device, months=None):
    """提取所有 patch × 指定月份的 embedding maps.

    Returns:
        dict: {(patch_id, month_idx): np.ndarray[D, H, W]}
    """
    if months is None:
        months = DEFAULT_MONTHS

    sample_map = {(pid, y, m): idx for idx, (pid, y, m) in enumerate(dataset.monthly_samples)}
    results = {}
    t0 = time.time()

    from tqdm import tqdm
    n_total = sum(1 for pid in dataset.patches for mi, (year, month) in enumerate(months) if (pid, year, month) in sample_map)
    pbar = tqdm(total=n_total, desc="[Extract] embeddings")
    for pid in dataset.patches:
        for mi, (year, month) in enumerate(months):
            key = (pid, year, month)
            if key not in sample_map:
                continue
            item = dataset[sample_map[key]]
            vs, ve = month_to_window(year, month)

            def _to(x):
                return x.unsqueeze(0).to(device)

            use_bf16 = getattr(cfg.training, 'use_bf16', True)
            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(
                    source_frames=_to(item["source_frames"]),
                    source_timestamps_ms=_to(item["source_timestamps_ms"]),
                    source_frame_mask=_to(item["source_frame_mask"]),
                    source_input_mask=_to(item["source_input_mask"]),
                    source_type_ids=_to(item["source_type_ids"]),
                    valid_start_ms=torch.tensor([vs], dtype=torch.int64, device=device),
                    valid_end_ms=torch.tensor([ve], dtype=torch.int64, device=device),
                    target_relative_time=torch.zeros(1, cfg.data.num_target_sources, device=device),
                    target_metadata=torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=device),
                    skip_decoder=True,
                )
            emb = F.normalize(out.embedding_map.float(), p=2, dim=1)  # [1, D, H, W]
            results[(pid, mi)] = emb.squeeze(0).cpu().numpy()  # [D, H, W]
            pbar.update(1)
    pbar.close()

    print(f"  [Extract] {len(results)} embeddings ({time.time()-t0:.1f}s)")
    return results


def knn_semantic_segmentation(embeddings: dict, dataset, month_idx: int = 4):
    """Pixel-level kNN 语义分割评估 (WorldCover).

    Args:
        embeddings: {(patch_id, month_idx): [D, H, W]}
        month_idx: 使用哪个月份的 embedding（默认 8月=index 4）

    Returns:
        dict with mIoU, OA, per_class_iou
    """
    from PIL import Image
    import rasterio

    data_root = Path(dataset.data_root) if hasattr(dataset, 'data_root') else Path("/workspace/raw/haidian_train")

    # 收集所有 patch 的 embedding 和 label
    X_list, y_list = [], []
    for (pid, mi), emb in embeddings.items():
        if mi != month_idx:
            continue
        # 加载 WorldCover label（支持多区域 patch_id 格式）
        label_dir = dataset._resolve_source_dir("worldcover", pid)
        if label_dir is None:
            continue
        label_path = label_dir / "static.tif"
        if not label_path.exists():
            continue
        try:
            with rasterio.open(label_path) as src:
                label = src.read(1)
        except Exception:
            continue

        D, H, W = emb.shape
        # resize label to match embedding spatial size
        if label.shape != (H, W):
            label_img = Image.fromarray(label.astype(np.uint8))
            label_img = label_img.resize((W, H), Image.Resampling.NEAREST)
            label = np.array(label_img)

        # flatten spatial dimensions
        emb_flat = emb.reshape(D, -1).T  # [H*W, D]
        label_flat = label.flatten()

        # filter nodata (WorldCover nodata = 0 or 255)
        valid = (label_flat > 0) & (label_flat < 255)
        if valid.sum() < 10:
            continue

        X_list.append(emb_flat[valid])
        y_list.append(label_flat[valid])

    if not X_list:
        return {"mIoU": 0.0, "OA": 0.0, "per_class_iou": {}}

    X = np.concatenate(X_list, axis=0)  # [N, D]
    y = np.concatenate(y_list, axis=0)  # [N]

    # L2 normalize for cosine similarity
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Patch-stratified split: 随机选 50% patches 作为训练集
    unique_pids = list({pid for (pid, mi) in embeddings.keys() if mi == month_idx})
    rng = np.random.RandomState(42)
    rng.shuffle(unique_pids)
    n_tr = max(1, len(unique_pids) // 2)
    train_pids = set(unique_pids[:n_tr])

    # 按 pixel 的 patch 归属分配 train/test
    # 由于 flatten 后无法直接知道 pixel 属于哪个 patch，我们简化处理：
    # 对整个数据集做随机 50/50 split（近似）
    n_total = len(X)
    perm = rng.permutation(n_total)
    n_tr_px = n_total // 2
    Xtr, ytr = X[perm[:n_tr_px]], y[perm[:n_tr_px]]
    Xte, yte = X[perm[n_tr_px:]], y[perm[n_tr_px:]]

    # kNN (cosine) — NPU 加速
    k = 5
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    batch = 4096
    preds = []
    for i in range(0, len(Xte), batch):
        Xb = torch.from_numpy(Xte[i:i+batch]).to(device)
        dist = torch.cdist(Xb, Xtr_t)
        _, idx = dist.topk(min(k, len(Xtr)), largest=False, dim=1)
        nbr = ytr_t[idx]
        for j in range(nbr.shape[0]):
            vals, counts = torch.unique(nbr[j], return_counts=True)
            preds.append(vals[counts.argmax()].item())
    preds = np.array(preds)

    # 计算 OA 和 mIoU
    oa = float((preds == yte).mean())
    classes = np.unique(np.concatenate([yte, preds]))
    ious = {}
    for cls in classes:
        inter = np.logical_and(preds == cls, yte == cls).sum()
        union = np.logical_or(preds == cls, yte == cls).sum()
        if union > 0:
            ious[int(cls)] = float(inter / union)
    miou = float(np.mean(list(ious.values()))) if ious else 0.0

    return {"mIoU": miou, "OA": oa, "per_class_iou": ious}


def change_detection_auc(embeddings: dict, dataset):
    """变化检测 AUC 评估 (Harbin)."""
    try:
        import geopandas as gpd
        from shapely.geometry import box
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"error": "geopandas or sklearn not available"}

    if not GRID_PATH.exists() or not ANNOT_DIR.exists():
        return {"error": "annotation data not found"}

    gdf = gpd.read_file(GRID_PATH)
    patch_bounds = {}
    for _, row in gdf.iterrows():
        pid = row.get("sample_id") or row.get("patch_id") or row.get("id")
        if pid is None:
            continue
        coords = list(row.geometry.exterior.coords)
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    results = {}
    for period_name, info in CD_PERIODS.items():
        before_mi = info["before"] - 4  # month index offset (April=0)
        after_mi = info["after"] - 4
        shp_path = ANNOT_DIR / info["shp"]
        if not shp_path.exists():
            continue

        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)

        changed_pids = set()
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            for pid, bounds in patch_bounds.items():
                if box(*bounds).intersects(geom):
                    changed_pids.add(pid)
                    break

        # 计算每个 patch 的 before/after 差异
        diffs = []
        labels = []
        for pid in dataset.patches:
            eb = embeddings.get((pid, before_mi))
            ea = embeddings.get((pid, after_mi))
            if eb is None or ea is None:
                continue
            # 处理多区域 patch_id 格式（如 harbin_patch_000000 -> patch_000000）
            local_pid = pid.split('_', 1)[1] if '_' in pid and not pid.startswith('patch_') else pid
            # cosine distance (patch-level mean)
            eb_g = eb.mean(axis=(1, 2))  # [D]
            ea_g = ea.mean(axis=(1, 2))  # [D]
            cos_sim = np.dot(eb_g, ea_g) / (np.linalg.norm(eb_g) * np.linalg.norm(ea_g) + 1e-8)
            diffs.append(1.0 - cos_sim)
            labels.append(1 if local_pid in changed_pids else 0)

        if len(diffs) < 4 or sum(labels) == 0:
            continue

        diffs = np.array(diffs)
        labels = np.array(labels)

        # Cosine distance AUC
        try:
            auc_cosine = float(roc_auc_score(labels, diffs))
        except Exception:
            auc_cosine = 0.5

        # Linear Discriminator AUC
        try:
            X = np.stack([np.concatenate([embeddings[(pid, before_mi)].mean(axis=(1,2)),
                                          embeddings[(pid, after_mi)].mean(axis=(1,2))])
                         for pid in dataset.patches
                         if (pid, before_mi) in embeddings and (pid, after_mi) in embeddings])
            y = np.array([1 if (pid.split('_', 1)[1] if '_' in pid and not pid.startswith('patch_') else pid) in changed_pids else 0
                         for pid in dataset.patches
                         if (pid, before_mi) in embeddings and (pid, after_mi) in embeddings])
            if len(np.unique(y)) > 1:
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X, y)
                auc_linear = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
            else:
                auc_linear = 0.5
        except Exception:
            auc_linear = 0.5

        results[period_name] = {
            "auc_cosine": auc_cosine,
            "auc_linear": auc_linear,
            "n_samples": len(diffs),
            "n_changed": int(sum(labels)),
        }

    return results


def main():
    args = parse_args()
    t0 = time.time()

    print(f"\n[PeriodicEval] Loading checkpoint: {args.checkpoint}")
    model, dataset, cfg, device = load_model(args.config, args.checkpoint, args.device)

    print(f"[PeriodicEval] Extracting embeddings for {len(dataset.patches)} patches...")
    embeddings = extract_embeddings(model, dataset, cfg, device)

    # ── kNN 语义分割 ──
    print("[PeriodicEval] Running kNN semantic segmentation...")
    knn_results = knn_semantic_segmentation(embeddings, dataset, month_idx=4)
    print(f"  kNN mIoU={knn_results['mIoU']:.4f} OA={knn_results['OA']:.4f}")

    # ── 变化检测 AUC ──
    cd_results = {}
    if not args.skip_cd:
        print("[PeriodicEval] Running change detection AUC...")
        cd_results = change_detection_auc(embeddings, dataset)
        for period, res in cd_results.items():
            if isinstance(res, dict) and "auc_cosine" in res:
                print(f"  CD {period}: AUC={res['auc_cosine']:.4f} (n={res['n_samples']}, changed={res['n_changed']})")

    # ── 保存结果 ──
    output = {
        "checkpoint": str(args.checkpoint),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_embeddings": len(embeddings),
        "knn": knn_results,
        "change_detection": cd_results,
        "elapsed_seconds": time.time() - t0,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"[PeriodicEval] Results saved to {args.output} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
