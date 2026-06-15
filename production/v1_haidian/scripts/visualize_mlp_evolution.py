#!/usr/bin/env python3
"""施工工地监测：原 MLP / 后处理 MLP / 改进 MLP 三方案对比可视化.

输出：
- visualizations/mlp_evolution/{task}/{patch}.png
- visualizations/mlp_evolution/{task}/summary_metrics.json
- visualizations/mlp_evolution/{task}/summary_bar.png
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

_LABEL_CMAP = ListedColormap(["#e8e8e8", "#d62728"])
_PRED_CMAP = LinearSegmentedColormap.from_list(
    "pred", ["#ADD8E6", "#8B0000"]
)


def parse_args():
    parser = argparse.ArgumentParser(description="MLP 三方案进化对比可视化")
    parser.add_argument("--pred-dir", default="outputs/merged_construction_ablation")
    parser.add_argument("--output-dir", default="visualizations/mlp_evolution")
    parser.add_argument("--scene-dir", default="/workspace/xuannv/data_raw/beijing/planetscene")
    parser.add_argument("--cache", default="outputs/head_ablation/.cache/embeddings.npz")
    parser.add_argument("--task", default="shigongjiandu")
    return parser.parse_args()


def _load_embeddings(cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = np.load(cache_path, allow_pickle=False)
    pids = [str(p) for p in data["patch_ids"]]
    emb_dec_arr = data["emb_dec"]
    emb_apr_arr = data["emb_apr"]
    emb_dec = {pid: emb_dec_arr[i] for i, pid in enumerate(pids)}
    emb_apr = {pid: emb_apr_arr[i] for i, pid in enumerate(pids)}
    return emb_dec, emb_apr


def _rgb_tiff_to_array(tiff_path: Path) -> np.ndarray:
    with rasterio.open(tiff_path) as src:
        rgb = src.read([1, 2, 3])
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    for c in range(3):
        band = rgb[..., c]
        low, high = np.percentile(band, [2, 98])
        band = np.clip((band - low) / (high - low + 1e-6), 0, 1)
        rgb[..., c] = band
    return rgb


def _embed_to_rgb(emb: np.ndarray, pca: PCA | None = None) -> tuple[np.ndarray, PCA]:
    D, H, W = emb.shape
    flat = emb.reshape(D, -1).T
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


def _patch_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def _global_metrics(prob: np.ndarray, true: np.ndarray, pred: np.ndarray | None = None) -> dict[str, float]:
    true_flat = true.flatten()
    prob_flat = prob.flatten()
    if pred is None:
        pred_flat = (prob_flat >= 0.5).astype(np.uint8)
    else:
        pred_flat = pred.flatten()
    metrics: dict[str, float] = {
        "f1": float(f1_score(true_flat, pred_flat, zero_division=0)),
        "iou": float(jaccard_score(true_flat, pred_flat, zero_division=0)),
    }
    if len(np.unique(true_flat)) == 2:
        metrics["auc"] = float(roc_auc_score(true_flat, prob_flat))
    return metrics


def visualize_patch(
    patch_id: str,
    task: str,
    true_label: np.ndarray,
    mlp_prob: np.ndarray,
    mlp_pred: np.ndarray,
    post_mask: np.ndarray,
    v2_prob: np.ndarray,
    v2_pred: np.ndarray,
    before_emb: np.ndarray,
    after_emb: np.ndarray,
    scene_dir: Path,
    output_dir: Path,
) -> None:
    before_tiff = scene_dir / patch_id / "20251209.tif"
    after_tiff = scene_dir / patch_id / "20260430.tif"
    if not before_tiff.exists() or not after_tiff.exists():
        return

    before_rgb = _rgb_tiff_to_array(before_tiff)
    after_rgb = _rgb_tiff_to_array(after_tiff)

    flat_before = before_emb.reshape(before_emb.shape[0], -1).T
    flat_after = after_emb.reshape(after_emb.shape[0], -1).T
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([flat_before, flat_after], axis=0))
    before_emb_rgb, _ = _embed_to_rgb(before_emb, pca)
    after_emb_rgb, _ = _embed_to_rgb(after_emb, pca)

    mlp_m = _patch_metrics(mlp_pred, true_label)
    post_m = _patch_metrics(post_mask, true_label)
    v2_m = _patch_metrics(v2_pred, true_label)

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 7, hspace=0.35, wspace=0.35)

    top_imgs = [before_rgb, after_rgb, before_emb_rgb, after_emb_rgb, true_label]
    top_titles = ["变化前 RGB", "变化后 RGB", "变化前 embedding", "变化后 embedding", "真实 label"]
    top_cmaps = [None, None, None, None, _LABEL_CMAP]
    for i, (img, title, cmap) in enumerate(zip(top_imgs, top_titles, top_cmaps)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap is _LABEL_CMAP else None,
                  interpolation="nearest" if cmap is _LABEL_CMAP else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if title == "真实 label":
            fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])

    # 下方：三种方案概率 + 预测
    def _draw(ax_prob, ax_pred, prob, pred, title, metrics):
        im = ax_prob.imshow(prob, cmap="hot", vmin=0, vmax=1)
        ax_prob.set_title(f"{title} 概率", fontsize=10)
        ax_prob.axis("off")
        ax_prob.figure.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)

        im2 = ax_pred.imshow(pred, cmap=_PRED_CMAP, vmin=0, vmax=1)
        ax_pred.set_title(f"{title} 预测", fontsize=10)
        ax_pred.axis("off")
        ax_pred.text(
            0.02, 0.02,
            f"F1={metrics['f1']:.2f}\nIoU={metrics['iou']:.2f}\nP={metrics['precision']:.2f}\nR={metrics['recall']:.2f}",
            transform=ax_pred.transAxes, fontsize=7, verticalalignment="bottom",
            color="black", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    _draw(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), mlp_prob, mlp_pred, "原 MLP", mlp_m)
    _draw(fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]), mlp_prob, post_mask, "后处理 MLP", post_m)
    _draw(fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[1, 5]), v2_prob, v2_pred, "改进 MLP", v2_m)

    # 第 6 列留空，整体标题
    fig.suptitle(
        f"施工工地监测 ({task}) - {patch_id} | 原 MLP F1={mlp_m['f1']:.2f} | "
        f"后处理 F1={post_m['f1']:.2f} | 改进 MLP F1={v2_m['f1']:.2f}",
        fontsize=14,
    )

    out_dir = output_dir / task
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
    task = args.task

    mlp_npz = pred_dir / "mlp_torch" / task / "pred.npz"
    post_npz = pred_dir / "mlp_torch_post" / task / "pred.npz"
    v2_npz = pred_dir / "mlp_torch_v2" / task / "pred.npz"
    for p in (mlp_npz, post_npz, v2_npz):
        if not p.exists():
            print(f"[error] 缺少预测文件: {p}")
            return 1

    emb_dec, emb_apr = _load_embeddings(Path(args.cache))

    mlp_data = np.load(mlp_npz)
    post_data = np.load(post_npz)
    v2_data = np.load(v2_npz)

    pids = [str(p) for p in mlp_data["patch_ids"]]
    assert list(post_data["patch_ids"]) == pids and list(v2_data["patch_ids"]) == pids

    # 只可视化含真实正样本的 patch
    positive_indices = [i for i, pid in enumerate(pids) if mlp_data["label_map"][i].sum() > 0]
    print(f"[visualize] {task}: {len(pids)} patches, {len(positive_indices)} 含正样本")

    summary = {
        "mlp_torch": _global_metrics(mlp_data["prob_map"], mlp_data["label_map"]),
        "mlp_torch_post": _global_metrics(
            mlp_data["prob_map"], mlp_data["label_map"], pred=post_data["post_label_map"]
        ),
        "mlp_torch_v2": _global_metrics(v2_data["prob_map"], v2_data["label_map"]),
    }

    for idx in positive_indices:
        pid = pids[idx]
        if pid not in emb_dec or pid not in emb_apr:
            print(f"[skip] {pid} 缺少 embedding")
            continue
        visualize_patch(
            patch_id=pid,
            task=task,
            true_label=mlp_data["label_map"][idx],
            mlp_prob=mlp_data["prob_map"][idx],
            mlp_pred=(mlp_data["prob_map"][idx] >= 0.5).astype(np.uint8),
            post_mask=post_data["post_label_map"][idx],
            v2_prob=v2_data["prob_map"][idx],
            v2_pred=(v2_data["prob_map"][idx] >= 0.5).astype(np.uint8),
            before_emb=emb_dec[pid],
            after_emb=emb_apr[pid],
            scene_dir=Path(args.scene_dir),
            output_dir=output_dir,
        )

    # 保存全局指标
    task_out = output_dir / task
    task_out.mkdir(parents=True, exist_ok=True)
    (task_out / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"[summary] {json.dumps(summary, ensure_ascii=False, indent=2)}")

    # 绘制全局指标对比柱状图
    methods = ["原 MLP", "后处理 MLP", "改进 MLP"]
    keys = ["mlp_torch", "mlp_torch_post", "mlp_torch_v2"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["f1", "iou", "auc"]):
        vals = [summary[k].get(metric, 0.0) for k in keys]
        bars = ax.bar(methods, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} 对比")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(task_out / "summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {task_out / 'summary_bar.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
