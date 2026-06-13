#!/usr/bin/env python3
"""可视化海淀区 2026 标注与 MLP 预测结果（支持双时相 embedding）。

用法:
    python scripts/eval/visualize_haidian2026_predictions.py \
        --embedding-file outputs/exp_multires_v1_0612/embeddings_202512/patch_embeddings.npz \
        --month 2025-12 \
        --second-embedding-file outputs/exp_multires_v1_0612/embeddings_202604/patch_embeddings.npz \
        --second-month 2026-04 \
        --label-dir /workspace/xuannv/haidian_label/labeljson \
        --output-dir outputs/exp_multires_v1_0612/haidian2026_eval_bitemporal/visualizations \
        --device npu:0 \
        --n-examples 4
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

# import helpers from main eval script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.evaluate_haidian2026_labels import (
    build_dataset,
    CLASS_NAMES,
    TinyMLP,
)


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--embedding-file", required=True)
    pa.add_argument("--month", default="2025-10")
    pa.add_argument("--second-embedding-file", default="")
    pa.add_argument("--second-month", default="")
    pa.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    pa.add_argument("--output-dir", required=True)
    pa.add_argument("--device", default="npu:0")
    pa.add_argument("--n-examples", type=int, default=4)
    pa.add_argument("--seed", type=int, default=42)
    return pa.parse_args()


def load_s2_rgb_for_month(patch_id: str, year_month: str) -> np.ndarray | None:
    """加载指定月份的第一帧 S2 RGB。year_month 格式如 2025-12。"""
    s2_dir = Path(f"/workspace/xuannv/data_raw/haidian/scenes/{patch_id}/s2")
    if not s2_dir.exists():
        return None
    tifs = sorted(s2_dir.glob(f"{year_month.replace('-', '')}*.tif"))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        s2 = src.read()
    if s2.shape[0] < 3:
        return None
    rgb = np.stack([s2[2], s2[1], s2[0]], axis=-1)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb


def rasterize_label(json_path: Path, image_size: tuple[int, int] = (427, 427)) -> np.ndarray:
    """将 polygon 标注转为多类掩膜（0-5），无标注为背景（6）。"""
    from scripts.eval.evaluate_haidian2026_labels import load_label_json
    masks = load_label_json(json_path, image_size)
    h, w = image_size
    label = np.full((h, w), len(CLASS_NAMES), dtype=np.int64)
    for i, name in enumerate(CLASS_NAMES):
        label[masks[name] > 0] = i
    return label


def visualize_patch(pid: str, emb: np.ndarray, label_json: Path, model: nn.Module,
                    device: torch.device, output_path: Path, month1: str, month2: str | None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    D, H, W = emb.shape
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(emb.reshape(D, -1).T).float().to(device)
        pred = model(X).argmax(dim=1).cpu().numpy().reshape(H, W)

    label_427 = rasterize_label(label_json)
    label_64 = np.array(Image.fromarray((label_427 * 40).astype(np.uint8)).resize((W, H), Image.Resampling.NEAREST))
    label_64 = np.rint(label_64 / 40).astype(np.int64)
    label_64 = np.clip(label_64, 0, len(CLASS_NAMES))

    def resize_rgb(rgb):
        if rgb is None:
            return np.zeros((H, W, 3))
        return np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)) / 255.0

    rgb1 = resize_rgb(load_s2_rgb_for_month(pid, month1))
    rgb2 = resize_rgb(load_s2_rgb_for_month(pid, month2)) if month2 else np.zeros((H, W, 3))

    colors = [
        "#006400",  # gongdi
        "#fa0000",  # jianzhudongdi
        "#ff00ff",  # weijian
        "#f096ff",  # nongyongdi
        "#ffff4c",  # chaichu
        "#0064c8",  # daolubianhua
        "#e0e0e0",  # background
    ]
    names = ["施工工地", "建筑用地", "疑似违建", "农用地变化", "建筑消失", "施工道路", "背景"]

    ncols = 4 if month2 else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes[0].imshow(rgb1)
    axes[0].set_title(f"{pid} S2 {month1}")
    axes[0].axis("off")

    if month2:
        axes[1].imshow(rgb2)
        axes[1].set_title(f"{pid} S2 {month2}")
        axes[1].axis("off")

    ax_label = axes[2] if month2 else axes[1]
    ax_label.imshow(label_64, cmap=plt.cm.colors.ListedColormap(colors), vmin=0, vmax=len(colors)-1)
    ax_label.set_title("2026 标注")
    ax_label.axis("off")

    ax_pred = axes[3] if month2 else axes[2]
    ax_pred.imshow(pred, cmap=plt.cm.colors.ListedColormap(colors), vmin=0, vmax=len(colors)-1)
    ax_pred.set_title("MLP 预测")
    ax_pred.axis("off")

    patches = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(len(colors))]
    ax_pred.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] {output_path}")


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

    results, (D, H, W) = build_dataset(
        args.embedding_file, label_dir, args.month,
        second_npz_path=args.second_embedding_file or None,
        second_month=args.second_month or None,
    )
    print(f"[信息] 有效 patch: {len(results)}, embedding: {D}x{H}x{W}")

    # 80/20 split same as eval script
    rng = np.random.RandomState(args.seed)
    pids = list(results.keys())
    rng.shuffle(pids)
    n_train = int(len(pids) * 0.8)
    train_pids = set(pids[:n_train])

    # Train MLP on multiclass task
    X_train, y_train = [], []
    for pid, (emb, label) in results.items():
        if pid not in train_pids:
            continue
        emb_flat = emb.reshape(D, -1).T
        label_flat = label.reshape(len(CLASS_NAMES), -1).T
        cls = np.full(H * W, len(CLASS_NAMES), dtype=np.int64)
        mask_any = label_flat.sum(axis=1) > 0
        cls[mask_any] = np.argmax(label_flat[mask_any], axis=1)
        X_train.append(emb_flat)
        y_train.append(cls)
    X_train = np.concatenate(X_train, 0)
    y_train = np.concatenate(y_train, 0)

    model = TinyMLP(D, len(CLASS_NAMES) + 1, hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).to(device)
    batch = 4096
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch):
            b = perm[i:i+batch]
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Visualize test patches with foreground pixels
    test_pids = [p for p in pids[n_train:] if results[p][1].sum() > 0]
    test_pids = test_pids[:args.n_examples]
    for pid in test_pids:
        emb, _ = results[pid]
        label_json = label_dir / f"{pid}_20260430_rgb_uint8.json"
        out_path = output_dir / f"{pid}_mlp_pred.png"
        visualize_patch(pid, emb, label_json, model, device, out_path, args.month, args.second_month or None)


if __name__ == "__main__":
    main()
