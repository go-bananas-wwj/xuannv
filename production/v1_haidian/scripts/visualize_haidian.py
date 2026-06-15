#!/usr/bin/env python3
"""为海淀 6 任务生成可视化对比图.

每行 5 个图: 变化前 RGB、变化后 RGB、变化前 embedding RGB、变化后 embedding RGB、预测概率图.
下方 2 个图: 真实 label、模型预测 label.

用法:
    cd production/v1_haidian
    PYTHONPATH=. python scripts/visualize_haidian.py \
        --model-dir model \
        --output-dir visualizations
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# 使用系统自带中文字体，避免标题显示为方块
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 二值标签 colormap：背景浅灰，正样本红色，便于观察稀疏变化
_LABEL_CMAP = ListedColormap(["#e8e8e8", "#d62728"])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xuannv_v1 import backbone


def parse_args():
    parser = argparse.ArgumentParser(description="海淀 6 任务可视化")
    parser.add_argument("--model-dir", default="model", help="生产模型目录")
    parser.add_argument("--output-dir", default="visualizations", help="可视化输出目录")
    parser.add_argument("--device", default="npu:0", help="推理设备")
    parser.add_argument("--pred-dir", default="outputs/haidian", help="haidian_tasks 预测输出目录")
    parser.add_argument("--scene-dir", default="/workspace/xuannv/data_raw/beijing/planetscene", help="Planet RGB 目录")
    return parser.parse_args()


def _rgb_tiff_to_array(tiff_path: Path) -> np.ndarray:
    """读取 4 波段 Planet RGB tiff，取前 3 波段并做 percentile stretch."""
    with rasterio.open(tiff_path) as src:
        rgb = src.read([1, 2, 3])  # [3, H, W]
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    for c in range(3):
        band = rgb[..., c]
        low, high = np.percentile(band, [2, 98])
        band = np.clip((band - low) / (high - low + 1e-6), 0, 1)
        rgb[..., c] = band
    return rgb


def _embed_to_rgb(emb: np.ndarray, pca: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """将 [D, H, W] embedding 映射到 [H, W, 3] RGB.

    如果传入 pca，则使用同一个 pca 进行 transform；否则新建并返回.
    """
    D, H, W = emb.shape
    flat = emb.reshape(D, -1).T  # [H*W, D]
    if pca is None:
        pca = PCA(n_components=3)
        proj = pca.fit_transform(flat)
    else:
        proj = pca.transform(flat)
    for i in range(3):
        low, high = np.percentile(proj[:, i], [2, 98])
        proj[:, i] = np.clip((proj[:, i] - low) / (high - low + 1e-6), 0, 1)
    rgb = proj.reshape(H, W, 3)
    return rgb, pca


def _draw_row5(fig, axes, before_rgb, after_rgb, before_emb_rgb, after_emb_rgb, prob):
    """绘制上方 5 个子图."""
    titles = ["变化前 RGB", "变化后 RGB", "变化前 embedding", "变化后 embedding", "预测概率"]
    images = [before_rgb, after_rgb, before_emb_rgb, after_emb_rgb, prob]
    cmaps = [None, None, None, None, "hot"]
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if title == "预测概率":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_row2(axes, true_label, pred_label):
    """绘制下方 2 个子图."""
    titles = ["真实 label", "预测 label"]
    for ax, label, title in zip(axes, [true_label, pred_label], titles):
        im = ax.imshow(label, cmap=_LABEL_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        # 标出正样本比例，避免稀疏目标被忽略
        pos_ratio = label.mean() * 100
        ax.text(
            0.02,
            0.98,
            f"正样本: {label.sum()} ({pos_ratio:.2f}%)",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        fig = ax.figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])


def visualize_patch(
    patch_id: str,
    task_name: str,
    prob: np.ndarray,
    true_label: np.ndarray,
    before_emb: np.ndarray,
    after_emb: np.ndarray,
    scene_dir: Path,
    output_dir: Path,
):
    before_tiff = scene_dir / patch_id / "20251209.tif"
    after_tiff = scene_dir / patch_id / "20260430.tif"
    if not before_tiff.exists() or not after_tiff.exists():
        print(f"[skip] {patch_id} 缺少 RGB 时相")
        return

    before_rgb = _rgb_tiff_to_array(before_tiff)
    after_rgb = _rgb_tiff_to_array(after_tiff)

    # 用同一个 PCA 投影前后两期 embedding，保证色彩一致
    flat_before = before_emb.reshape(before_emb.shape[0], -1).T
    flat_after = after_emb.reshape(after_emb.shape[0], -1).T
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([flat_before, flat_after], axis=0))
    before_emb_rgb, _ = _embed_to_rgb(before_emb, pca)
    after_emb_rgb, _ = _embed_to_rgb(after_emb, pca)

    pred_label = (prob > 0.5).astype(np.uint8)

    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    _draw_row5(fig, top_axes, before_rgb, after_rgb, before_emb_rgb, after_emb_rgb, prob)

    ax_true = fig.add_subplot(gs[1, :2])
    ax_pred = fig.add_subplot(gs[1, 3:])
    _draw_row2([ax_true, ax_pred], true_label, pred_label)

    task_cn = {
        "gongdi": "施工工地",
        "jianzhudongdi": "建筑用地",
        "weijian": "疑似违建",
        "nongyongdi": "农用地变化",
        "chaichu": "建筑消失",
        "daolubianhua": "施工道路",
    }
    fig.suptitle(f"{task_cn.get(task_name, task_name)} ({task_name}) - {patch_id}", fontsize=14)

    out_dir = output_dir / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> int:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(args.scene_dir)

    model, dataset, cfg = backbone.load_production_model(args.model_dir, device=args.device)

    task_names = ["gongdi", "jianzhudongdi", "weijian", "nongyongdi", "chaichu", "daolubianhua"]
    for task in task_names:
        pred_npz = pred_dir / task / "pred.npz"
        if not pred_npz.exists():
            print(f"[skip] 缺少预测文件: {pred_npz}")
            continue
        data = np.load(pred_npz)
        patch_ids = [str(p) for p in data["patch_ids"]]
        prob_maps = data["prob_map"]  # [N, H, W]
        label_maps = data["label_map"]  # [N, H, W]

        print(f"\n[Task] {task} - {len(patch_ids)} patches")
        for idx, pid in enumerate(patch_ids):
            emb_before = backbone.extract_embedding_for_month(model, dataset, pid, 2025, 12, args.device)
            emb_after = backbone.extract_embedding_for_month(model, dataset, pid, 2026, 4, args.device)
            visualize_patch(
                patch_id=pid,
                task_name=task,
                prob=prob_maps[idx],
                true_label=label_maps[idx],
                before_emb=emb_before,
                after_emb=emb_after,
                scene_dir=scene_dir,
                output_dir=output_dir,
            )

    print(f"\n可视化完成，输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
