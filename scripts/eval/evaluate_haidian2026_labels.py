#!/usr/bin/env python3
"""海淀区 2026 年标注下游评估。

标注来源: /workspace/xuannv/haidian_label/labeljson/*.json (LabelMe 格式)
任务: 基于现有 embedding (exp_multires_v1_0612 epoch 40 best) 评估
      施工工地 / 建筑用地 / 疑似违建 / 农用地变化 / 建筑消失 / 施工道路
      的像素级分类/检测效果。

用法:
    python scripts/eval/evaluate_haidian2026_labels.py \
        --embedding-file outputs/exp_multires_v1_0612/eval_best40/patch_embeddings.npz \
        --label-dir /workspace/xuannv/haidian_label/labeljson \
        --output-dir outputs/exp_multires_v1_0612/haidian2026_eval \
        --device npu:0
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

warnings.filterwarnings("ignore")


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--embedding-file", required=True)
    pa.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    pa.add_argument("--output-dir", required=True)
    pa.add_argument("--device", default="npu:0")
    pa.add_argument("--month", default="2025-10")
    pa.add_argument("--k", type=int, default=5)
    pa.add_argument("--seed", type=int, default=42)
    return pa.parse_args()


# 标注名称标准化（原始 JSON 中存在拼音简写/错字）
LABEL_NORMALIZE = {
    "gongdi": "gongdi",
    "jiazhudongdi": "jianzhudongdi",
    "jianzhudongdi": "jianzhudongdi",
    "weijian": "weijian",
    "nongyongdi": "nongyongdi",
    "chachu": "chaichu",
    "chaichu": "chaichu",
    "daolubianhuo": "daolubianhua",
    "daolubianhua": "daolubianhua",
}

CLASS_NAMES = [
    "gongdi",        # 施工工地
    "jianzhudongdi", # 建筑用地
    "weijian",       # 疑似违建
    "nongyongdi",    # 农用地变化
    "chaichu",       # 建筑消失
    "daolubianhua",  # 施工道路
]

CLASS_NAMES_CN = {
    "gongdi": "施工工地",
    "jianzhudongdi": "建筑用地",
    "weijian": "疑似违建",
    "nongyongdi": "农用地变化",
    "chaichu": "建筑消失",
    "daolubianhua": "施工道路",
}


def load_label_json(json_path: Path, image_size: tuple[int, int] = (427, 427)) -> dict[str, np.ndarray]:
    """将 LabelMe JSON 中的 polygon 标注栅格化为每个类别的二值掩膜."""
    with open(json_path) as f:
        data = json.load(f)

    h, w = image_size
    masks: dict[str, np.ndarray] = {name: np.zeros((h, w), dtype=np.uint8) for name in CLASS_NAMES}

    for shape in data.get("shapes", []):
        raw_label = shape.get("label", "").strip().lower()
        norm_label = LABEL_NORMALIZE.get(raw_label)
        if norm_label is None or norm_label not in masks:
            continue
        pts = [(int(p[0]), int(p[1])) for p in shape["points"]]
        if len(pts) < 3:
            continue
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
        masks[norm_label] |= np.array(img, dtype=np.uint8)

    return masks


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize to square embedding size."""
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def build_dataset(npz_path: str, label_dir: Path, month: str):
    """返回 {patch_id: (embedding [D,H,W], label [C,H,W])} 字典."""
    data = np.load(npz_path)
    spatial_maps = data["spatial_maps"].astype(np.float32)  # [N, M, D, H, W]
    month_labels = list(data["month_labels"])
    patch_ids = [str(p) for p in data["patch_ids"]]

    if month not in month_labels:
        raise ValueError(f"月份 {month} 不在 embedding 中: {month_labels}")
    mi = month_labels.index(month)
    spatial_maps = spatial_maps[:, mi]  # [N, D, H, W]
    _, D, H, W = spatial_maps.shape

    results = {}
    for i, pid in enumerate(patch_ids):
        json_path = label_dir / f"{pid}_20260430_rgb_uint8.json"
        if not json_path.exists():
            continue
        masks = load_label_json(json_path, image_size=(427, 427))
        label = np.stack([resize_mask(masks[name], H) for name in CLASS_NAMES], axis=0)  # [C, H, W]
        emb = spatial_maps[i]  # [D, H, W]
        results[pid] = (emb, label)

    return results, (D, H, W)


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def knn_eval(X_train, y_train, X_test, k: int, device: torch.device):
    """Pixel-level kNN (cosine) on NPU."""
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    Xte = torch.from_numpy(X_test).to(device)
    Xtr = F.normalize(Xtr, p=2, dim=1)
    Xte = F.normalize(Xte, p=2, dim=1)

    batch = 2048
    preds = []
    for i in range(0, len(Xte), batch):
        sim = Xte[i:i+batch] @ Xtr.T
        _, idx = sim.topk(min(k, len(Xtr)), largest=True, dim=1)
        nbr = ytr[idx]
        for j in range(nbr.shape[0]):
            vals, counts = torch.unique(nbr[j], return_counts=True)
            preds.append(vals[counts.argmax()].item())
    return np.array(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    oa = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    miou = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = {}
    for c in range(num_classes):
        yt = (y_true == c).astype(np.int64)
        yp = (y_pred == c).astype(np.int64)
        inter = int((yt & yp).sum())
        union = int((yt | yp).sum())
        iou = inter / (union + 1e-8)
        if c < len(CLASS_NAMES):
            name = CLASS_NAMES[c]
            name_cn = CLASS_NAMES_CN[name]
        else:
            name = "background"
            name_cn = "背景"
        per_class[c] = {
            "name": name,
            "name_cn": name_cn,
            "iou": float(iou),
            "support": int(yt.sum()),
        }
    return {"oa": oa, "bacc": bacc, "mf1": mf1, "miou": miou, "per_class": per_class}


def pixel_multiclass_eval(results: dict, D: int, H: int, W: int, k: int, device: torch.device, seed: int):
    """6 类像素级分类：kNN + 线性探测 + MLP."""
    rng = np.random.RandomState(seed)
    pids = list(results.keys())
    rng.shuffle(pids)
    n_train = int(len(pids) * 0.8)
    train_pids = set(pids[:n_train])
    test_pids = set(pids[n_train:])

    # 像素级特征/标签
    X_train, y_train = [], []
    X_test, y_test = [], []
    for pid, (emb, label) in results.items():
        emb_flat = emb.reshape(D, -1).T  # [H*W, D]
        # 多类标签：背景=6（未标注），前景类 0-5
        label_flat = label.reshape(len(CLASS_NAMES), -1).T  # [H*W, C]
        cls = np.full(H * W, len(CLASS_NAMES), dtype=np.int64)  # background class
        mask_any = label_flat.sum(axis=1) > 0
        cls[mask_any] = np.argmax(label_flat[mask_any], axis=1)

        if pid in train_pids:
            X_train.append(emb_flat)
            y_train.append(cls)
        else:
            X_test.append(emb_flat)
            y_test.append(cls)

    X_train = np.concatenate(X_train, 0)
    y_train = np.concatenate(y_train, 0)
    X_test = np.concatenate(X_test, 0)
    y_test = np.concatenate(y_test, 0)

    print(f"[多分类] train patches={len(train_pids)} test patches={len(test_pids)}")
    print(f"         train pixels={len(X_train)} test pixels={len(X_test)}")
    print(f"         类别分布: {dict(Counter(int(y) for y in y_train).most_common())}")

    reports = {}

    # kNN
    y_pred = knn_eval(X_train, y_train, X_test, k, device)
    reports["knn"] = compute_metrics(y_test, y_pred, len(CLASS_NAMES) + 1)
    print(f"  kNN  : OA={reports['knn']['oa']:.4f} BAcc={reports['knn']['bacc']:.4f} mF1={reports['knn']['mf1']:.4f} mIoU={reports['knn']['miou']:.4f}")

    # 线性探测
    clf = LogisticRegression(max_iter=500, n_jobs=4, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    reports["linear"] = compute_metrics(y_test, y_pred, len(CLASS_NAMES) + 1)
    print(f"  Linear: OA={reports['linear']['oa']:.4f} BAcc={reports['linear']['bacc']:.4f} mF1={reports['linear']['mf1']:.4f} mIoU={reports['linear']['miou']:.4f}")

    # MLP
    mlp = TinyMLP(D, len(CLASS_NAMES) + 1, hidden=128).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).to(device)
    batch = 4096
    for epoch in range(50):
        mlp.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch):
            b = perm[i:i+batch]
            logits = mlp(Xt[b])
            loss = F.cross_entropy(logits, yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()

    mlp.eval()
    preds = []
    with torch.no_grad():
        Xe = torch.from_numpy(X_test).float().to(device)
        for i in range(0, len(Xe), batch):
            preds.append(mlp(Xe[i:i+batch]).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    reports["mlp"] = compute_metrics(y_test, y_pred, len(CLASS_NAMES) + 1)
    print(f"  MLP  : OA={reports['mlp']['oa']:.4f} BAcc={reports['mlp']['bacc']:.4f} mF1={reports['mlp']['mf1']:.4f} mIoU={reports['mlp']['miou']:.4f}")

    return reports


def binary_construction_eval(results: dict, D: int, H: int, W: int, k: int, device: torch.device, seed: int):
    """建设相关目标二分类：gongdi + jianzhudongdi + weijian + chaichu + daolubianhua 合并为 foreground."""
    construction_indices = [CLASS_NAMES.index(n) for n in ["gongdi", "jianzhudongdi", "weijian", "chaichu", "daolubianhua"]]
    rng = np.random.RandomState(seed)
    pids = list(results.keys())
    rng.shuffle(pids)
    n_train = int(len(pids) * 0.8)
    train_pids = set(pids[:n_train])

    X_train, y_train = [], []
    X_test, y_test = [], []
    for pid, (emb, label) in results.items():
        emb_flat = emb.reshape(D, -1).T
        fg = label[construction_indices].sum(axis=0) > 0  # [H, W]
        cls = fg.astype(np.int64).flatten()
        if pid in train_pids:
            X_train.append(emb_flat)
            y_train.append(cls)
        else:
            X_test.append(emb_flat)
            y_test.append(cls)

    X_train = np.concatenate(X_train, 0)
    y_train = np.concatenate(y_train, 0)
    X_test = np.concatenate(X_test, 0)
    y_test = np.concatenate(y_test, 0)

    pos_ratio = y_train.mean()
    print(f"[建设二分类] train={len(y_train)} test={len(y_test)} pos_ratio={pos_ratio:.4f}")

    reports = {}
    y_pred = knn_eval(X_train, y_train, X_test, k, device)
    reports["knn"] = compute_metrics(y_test, y_pred, 2)
    print(f"  kNN  : OA={reports['knn']['oa']:.4f} BAcc={reports['knn']['bacc']:.4f} mF1={reports['knn']['mf1']:.4f} mIoU={reports['knn']['miou']:.4f}")

    clf = LogisticRegression(max_iter=500, n_jobs=4, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    reports["linear"] = compute_metrics(y_test, y_pred, 2)
    print(f"  Linear: OA={reports['linear']['oa']:.4f} BAcc={reports['linear']['bacc']:.4f} mF1={reports['linear']['mf1']:.4f} mIoU={reports['linear']['miou']:.4f}")

    mlp = TinyMLP(D, 2, hidden=128).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).to(device)
    batch = 4096
    for epoch in range(50):
        mlp.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch):
            b = perm[i:i+batch]
            loss = F.cross_entropy(mlp(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()

    mlp.eval()
    preds = []
    with torch.no_grad():
        Xe = torch.from_numpy(X_test).float().to(device)
        for i in range(0, len(Xe), batch):
            preds.append(mlp(Xe[i:i+batch]).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    reports["mlp"] = compute_metrics(y_test, y_pred, 2)
    print(f"  MLP  : OA={reports['mlp']['oa']:.4f} BAcc={reports['mlp']['bacc']:.4f} mF1={reports['mlp']['mf1']:.4f} mIoU={reports['mlp']['miou']:.4f}")

    return reports


def main():
    args = parse_args()
    if "npu" in args.device:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            pass
    device = torch.device(args.device)

    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[加载] embedding: {args.embedding_file}")
    print(f"[加载] labels: {label_dir}")
    results, (D, H, W) = build_dataset(args.embedding_file, label_dir, args.month)
    print(f"[信息] 有效标注 patch 数: {len(results)}, embedding: D={D}, H={H}, W={W}")

    print("\n" + "=" * 60)
    print("任务一：6 类像素级分类（含背景）")
    print("=" * 60)
    multi_reports = pixel_multiclass_eval(results, D, H, W, args.k, device, args.seed)

    print("\n" + "=" * 60)
    print("任务二：建设相关目标二分类")
    print("=" * 60)
    binary_reports = binary_construction_eval(results, D, H, W, args.k, device, args.seed)

    summary = {
        "month": args.month,
        "n_patches": len(results),
        "embedding_shape": [D, H, W],
        "multiclass": multi_reports,
        "binary_construction": binary_reports,
    }
    out_json = output_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 结果写入 {out_json}")


if __name__ == "__main__":
    main()
