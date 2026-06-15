#!/usr/bin/env python3
"""将不同下游头（linear / mlp_torch / unet）的预测结果在同一张图里对比可视化.

每行:
- 上方: 变化前 RGB、变化后 RGB、变化前 embedding RGB、变化后 embedding RGB、真实 label
- 下方: linear 概率 / pred, mlp_torch 概率 / pred, unet 概率 / pred

用法:
    cd production/v1_haidian
    python scripts/visualize_head_comparison.py
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

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

_LABEL_CMAP = ListedColormap(["#e8e8e8", "#d62728"])
_PRED_CMAP = LinearSegmentedColormap.from_list(
    "pred", ["#ADD8E6", "#8B0000"]
)

TASK_CN = {
    "gongdi": "施工工地",
    "jianzhudongdi": "建筑用地",
    "weijian": "疑似违建",
    "nongyongdi": "农用地变化",
    "chaichu": "建筑消失",
    "daolubianhua": "施工道路",
    "shigongjiandu": "施工工地监测",
}

HEAD_CN = {
    "linear": "Linear",
    "mlp_torch": "MLP",
    "unet": "U-Net",
}


def parse_args():
    parser = argparse.ArgumentParser(description="多 head 预测结果对比可视化")
    parser.add_argument(
        "--pred-dir",
        default="outputs/head_ablation",
        help="各 head 预测结果父目录",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/comparison",
        help="对比图输出目录",
    )
    parser.add_argument(
        "--scene-dir",
        default="/workspace/xuannv/data_raw/beijing/planetscene",
        help="Planet RGB 目录",
    )
    parser.add_argument(
        "--cache",
        default="outputs/head_ablation/.cache/embeddings.npz",
        help="embedding 缓存",
    )
    parser.add_argument(
        "--heads",
        default="linear,mlp_torch,unet",
        help="逗号分隔的 head 列表",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="只可视化指定任务（例如 shigongjiandu），不指定则处理所有任务",
    )
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


def _load_metrics(pred_dir: Path, head: str) -> dict[str, dict]:
    path = pred_dir / head / "metrics_all.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_head_pred(ax_prob, ax_pred, prob: np.ndarray, true_label: np.ndarray, head: str, task: str):
    im = ax_prob.imshow(prob, cmap="hot", vmin=0, vmax=1)
    ax_prob.set_title(f"{HEAD_CN.get(head, head)} 概率", fontsize=10)
    ax_prob.axis("off")
    ax_prob.figure.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)

    # 预测 label 用预测分数连续显示：浅蓝 -> 深红
    im2 = ax_pred.imshow(prob, cmap=_PRED_CMAP, vmin=0, vmax=1)
    ax_pred.set_title(f"{HEAD_CN.get(head, head)} 预测 label", fontsize=10)
    ax_pred.axis("off")
    ax_pred.figure.colorbar(im2, ax=ax_pred, fraction=0.046, pad=0.04)

    # 计算该 patch 的 F1/IoU（仍用 0.5 阈值）
    pred = (prob > 0.5).astype(np.uint8)
    tp = ((pred == 1) & (true_label == 1)).sum()
    fp = ((pred == 1) & (true_label == 0)).sum()
    fn = ((pred == 0) & (true_label == 1)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    ax_pred.text(
        0.02,
        0.02,
        f"F1={f1:.2f}\nIoU={iou:.2f}",
        transform=ax_pred.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )


def visualize_patch_comparison(
    patch_id: str,
    task: str,
    head_probs: dict[str, np.ndarray],
    true_label: np.ndarray,
    before_emb: np.ndarray,
    after_emb: np.ndarray,
    scene_dir: Path,
    output_dir: Path,
    head_auc: dict[str, float | None],
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

    heads = list(head_probs.keys())
    n_heads = len(heads)

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 5 + n_heads, hspace=0.35, wspace=0.35)

    # 上方 5 个公共图
    top_imgs = [before_rgb, after_rgb, before_emb_rgb, after_emb_rgb, true_label]
    top_titles = ["变化前 RGB", "变化后 RGB", "变化前 embedding", "变化后 embedding", "真实 label"]
    top_cmaps = [None, None, None, None, _LABEL_CMAP]
    for i, (img, title, cmap) in enumerate(zip(top_imgs, top_titles, top_cmaps)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap is _LABEL_CMAP else None, interpolation="nearest" if cmap is _LABEL_CMAP else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if title == "真实 label":
            fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])

    # 下方每个 head 两个子图：概率 + 预测
    for col, head in enumerate(heads):
        ax_prob = fig.add_subplot(gs[1, col * 2])
        ax_pred = fig.add_subplot(gs[1, col * 2 + 1])
        _draw_head_pred(ax_prob, ax_pred, head_probs[head], true_label, head, task)

    # 标题加入各 head 在该任务上的整体 AUC
    auc_parts = [
        f"{HEAD_CN.get(h, h)} AUC={head_auc.get(h):.3f}" if head_auc.get(h) is not None else f"{HEAD_CN.get(h, h)} AUC=-"
        for h in heads
    ]
    fig.suptitle(
        f"{TASK_CN.get(task, task)} ({task}) - {patch_id} | " + " | ".join(auc_parts),
        fontsize=13,
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
    scene_dir = Path(args.scene_dir)
    cache_path = Path(args.cache)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]

    if not cache_path.exists():
        print(f"[error] embedding 缓存不存在: {cache_path}，请先运行 head 对比实验")
        return 1

    emb_dec, emb_apr = _load_embeddings(cache_path)

    task_list = [args.task] if args.task else list(TASK_CN.keys())
    if args.task and args.task not in TASK_CN:
        raise ValueError(f"未知任务: {args.task}，可选: {list(TASK_CN.keys())}")

    for task in task_list:
        # 加载每个 head 的预测和任务级 AUC
        head_data: dict[str, dict] = {}
        head_auc: dict[str, float | None] = {}
        common_pids: set[str] | None = None
        for head in heads:
            npz_path = pred_dir / head / task / "pred.npz"
            if not npz_path.exists():
                print(f"[skip] 缺少 {head}/{task}/pred.npz")
                continue
            data = np.load(npz_path)
            pids = [str(p) for p in data["patch_ids"]]
            probs = data["prob_map"]
            labels = data["label_map"]
            head_data[head] = {"pids": pids, "probs": probs, "labels": labels}
            if common_pids is None:
                common_pids = set(pids)
            else:
                common_pids &= set(pids)

            metrics = _load_metrics(pred_dir, head)
            head_auc[head] = metrics.get(task, {}).get("auc")

        if not head_data or not common_pids:
            continue

        # 只可视化包含真实标注正样本的 patch
        ref_head = next(iter(head_data))
        ref = head_data[ref_head]
        positive_pids = {
            pid
            for pid in common_pids
            if ref["labels"][ref["pids"].index(pid)].sum() > 0
        }
        if not positive_pids:
            print(f"\n[Task] {task} - 无包含正样本的公共 patch，跳过")
            continue
        common_pids = positive_pids

        print(f"\n[Task] {task} - 含正样本的公共 patch: {len(common_pids)}")
        for pid in sorted(common_pids):
            head_probs = {}
            true_label = None
            for head, d in head_data.items():
                idx = d["pids"].index(pid)
                head_probs[head] = d["probs"][idx]
                true_label = d["labels"][idx]
            if pid not in emb_dec or pid not in emb_apr:
                print(f"[skip] {pid} 缺少 embedding")
                continue
            visualize_patch_comparison(
                patch_id=pid,
                task=task,
                head_probs=head_probs,
                true_label=true_label,
                before_emb=emb_dec[pid],
                after_emb=emb_apr[pid],
                scene_dir=scene_dir,
                output_dir=output_dir,
                head_auc=head_auc,
            )

    print(f"\n对比可视化完成，输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
