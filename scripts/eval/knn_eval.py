#!/usr/bin/env python
"""KNN 下游评估 — 统一入口。

支持两种后端:
  - pytorch  在指定设备 (cpu / npu:N) 上用 torch.cdist 计算 KNN
  - sklearn   仅 CPU，使用 sklearn.KNeighborsClassifier

用法示例:
    # PyTorch + NPU
    python knn_eval.py --embedding-file patch_embeddings.npz --output-dir eval_results/ \
        --device npu:0 --backend pytorch

    # sklearn CPU
    python knn_eval.py --embedding-file patch_embeddings.npz --output-dir eval_results/ \
        --device cpu --backend sklearn
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_ROOT = Path("/workspace/xuannv/data_raw/harbin_scenes")

TASKS = [
    ("worldcover",   "worldcover",   "static.tif",  10),
    ("jrc_water",    "jrc_water",    "static.tif",   2),
    ("dynamic_world","dynamic_world","static.tif",  12),
]

LABEL_MAPPINGS: dict[str, dict[int, int]] = {
    "worldcover": {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7},
}


# ── 标签加载 ──────────────────────────────────────────────────────────────────

def load_label(patch_id: str, label_dir: str, fname: str, data_roots: list[Path]):
    # 处理多区域 patch_id 格式（如 haidian_patch_000000 -> patch_000000）
    local_id = patch_id.split('_', 1)[1] if '_' in patch_id and not patch_id.startswith('patch_') else patch_id
    for data_root in data_roots:
        candidates = [
            data_root / label_dir / patch_id / fname,
            data_root / label_dir / local_id / fname,
        ]
        for path in candidates:
            if path.exists():
                with rasterio.open(path) as src:
                    return src.read(1), src.nodata
            # 自动查找目录中的第一个 .tif 文件
            patch_dir = path.parent
            if patch_dir.exists():
                tifs = sorted([f for f in patch_dir.iterdir() if f.suffix.lower() == ".tif"])
                if tifs:
                    with rasterio.open(tifs[0]) as src:
                        return src.read(1), src.nodata
    return None, None


def resize_label(label: np.ndarray, h: int, w: int) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=(h, w), mode="nearest").squeeze().numpy().astype(label.dtype)


# ── KNN 后端 ────────────────────────────────────────────────────────────────

def knn_pytorch(X_train, y_train, X_test, k, device):
    """torch.cdist KNN，支持 CPU 和 NPU。"""
    dev = torch.device(device)
    X_tr = torch.from_numpy(X_train).to(dev)
    y_tr = torch.from_numpy(y_train).long().to(dev)
    preds = []
    batch = 512
    for i in range(0, len(X_test), batch):
        q = torch.from_numpy(X_test[i:i + batch]).to(dev)
        dist = torch.cdist(q, X_tr)
        _, idx = dist.topk(min(k, len(X_train)), largest=False, dim=1)
        nbr = y_tr[idx]  # [B, k]
        for j in range(nbr.shape[0]):
            vals, counts = torch.unique(nbr[j], return_counts=True)
            preds.append(vals[counts.argmax()].item())
    return np.array(preds)


def knn_sklearn(X_train, y_train, X_test, k):
    """sklearn KNeighborsClassifier，仅 CPU。"""
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                               algorithm="auto", n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# ── 单任务评估 ───────────────────────────────────────────────────────────────

def evaluate_task(
    task_name, label_dir, label_file, num_classes,
    spatial_maps, patch_ids, device, k, backend, data_roots, seed=42,
):
    """评估单个下游任务。

    Args:
        spatial_maps: np.ndarray [num_patches, D, H, W]
        patch_ids:    list[str]
    """
    _, D, H, W = spatial_maps.shape
    rng = np.random.RandomState(seed)
    n_patches = len(patch_ids)

    mapping = LABEL_MAPPINGS.get(task_name)
    if mapping:
        num_classes = len(mapping)

    all_X, all_y, all_pidx = [], [], []

    for p_idx, pid in enumerate(patch_ids):
        label, nodata = load_label(pid, label_dir, label_file, data_roots)
        if label is None:
            continue
        if label.shape != (H, W):
            label = resize_label(label, H, W)
        # Dynamic World 浮点值四舍五入
        if task_name == "dynamic_world":
            label = np.rint(label).astype(np.int64)
        if mapping:
            mapped = np.full_like(label, -1, dtype=np.int64)
            for k, v in mapping.items():
                mapped[label == k] = v
            label = mapped
            nodata = -1
        mask = (label >= 0) & (label < num_classes)
        if nodata is not None:
            mask &= (label != nodata)
        if mask.sum() == 0:
            continue
        emb = spatial_maps[p_idx]          # [D, H, W]
        all_X.append(emb[:, mask].T)       # [N, D]
        all_y.append(label[mask])
        all_pidx.append(np.full(mask.sum(), p_idx))

    if not all_X:
        print(f"  [{task_name}] 无有效数据")
        return None

    all_X   = np.concatenate(all_X, 0)
    all_y   = np.concatenate(all_y, 0)
    all_pidx = np.concatenate(all_pidx, 0)

    # Patch-stratified 80/20 split
    n_train = max(1, int(n_patches * 0.8))
    train_set = set(rng.choice(n_patches, n_train, replace=False).tolist())
    tr_mask = np.array([p in train_set for p in all_pidx])
    te_mask = ~tr_mask

    X_tr, y_tr = all_X[tr_mask], all_y[tr_mask]
    X_te, y_te = all_X[te_mask], all_y[te_mask]

    if len(X_tr) > 100_000:
        idx = rng.choice(len(X_tr), 100_000, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    if len(X_te) == 0:
        print(f"  [{task_name}] 测试集为空，跳过")
        return None

    # 选后端
    if backend == "pytorch":
        y_pred = knn_pytorch(X_tr, y_tr, X_te, k, device)
    else:
        y_pred = knn_sklearn(X_tr, y_tr, X_te, k)

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred, labels=list(range(num_classes)))

    per_class: dict = {}
    for c in range(num_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        per_class[f"class_{c}"] = {
            "iou": float(tp / (tp + fp + fn + 1e-8)),
            "support": int(cm[c, :].sum()),
        }
    valid_ious = [v["iou"] for v in per_class.values() if v["support"] > 0]
    mean_iou   = float(np.mean(valid_ious)) if valid_ious else 0.0

    print(f"  [{task_name}] Acc={acc:.4f}  mIoU={mean_iou:.4f}")
    return {
        "task": task_name, "k": k, "backend": backend, "device": str(device),
        "accuracy": float(acc), "mean_iou": mean_iou,
        "num_train_pixels": int(len(X_tr)), "num_test_pixels": int(len(X_te)),
        "per_class": per_class, "confusion_matrix": cm.tolist(),
    }


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(description="KNN 下游评估")
    pa.add_argument("--embedding-file", required=True,
                    help=".npz 文件，包含 spatial_maps [N,12,D,H,W] 和 patch_ids")
    pa.add_argument("--output-dir",     required=True)
    pa.add_argument("--device",         default="cpu",
                    help="计算设备，如 cpu / npu:0 / npu:1")
    pa.add_argument("--backend",        default="pytorch",
                    choices=["pytorch", "sklearn"],
                    help="pytorch: torch.cdist KNN；sklearn: KNeighborsClassifier (仅 CPU)")
    pa.add_argument("--k",     type=int, default=5,  help="近邻数")
    pa.add_argument("--month", type=int, default=6,  help="使用第几月的 embedding (1-12)")
    pa.add_argument("--data-root", type=str, default=None,
                    help="数据根目录，默认使用哈尔滨路径")
    args = pa.parse_args()

    # 后端与设备一致性校验
    if args.backend == "sklearn" and args.device != "cpu":
        print("[警告] sklearn 后端仅支持 CPU，已强制 --device cpu")
        args.device = "cpu"

    if "npu" in args.device:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("[警告] 未找到 torch_npu，回退到 CPU")
            args.device = "cpu"

    if args.data_root:
        data_roots = [Path(p.strip()) for p in args.data_root.split(',')]
    else:
        data_roots = [
            DEFAULT_DATA_ROOT,
            Path("/workspace/xuannv/data_raw/haidian_train"),
        ]
    print(f"[KNN] 数据根目录: {data_roots}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[KNN] 加载 embedding: {args.embedding_file}")
    data = np.load(args.embedding_file)
    spatial_maps = data["spatial_maps"]      # [N, 12, D, H, W]
    patch_ids    = list(data["patch_ids"])   # [N]

    month_idx = args.month - 1
    spatial_maps = spatial_maps[:, month_idx]   # [N, D, H, W]
    print(f"      使用第 {args.month} 月，形状: {spatial_maps.shape}")
    print(f"      后端: {args.backend}  设备: {args.device}  k={args.k}")

    all_reports: dict = {}
    for task_name, label_dir, label_file, num_classes in TASKS:
        print(f"[KNN] 评估 {task_name}...")
        report = evaluate_task(
            task_name, label_dir, label_file, num_classes,
            spatial_maps, patch_ids, args.device, args.k, args.backend,
            data_roots,
        )
        if report:
            all_reports[task_name] = report
            (output_dir / f"knn_{task_name}.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False)
            )

    summary = {k: {"accuracy": v["accuracy"], "mean_iou": v["mean_iou"]}
               for k, v in all_reports.items()}
    (output_dir / "knn_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n[KNN] 完成!")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
