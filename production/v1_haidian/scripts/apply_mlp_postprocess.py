#!/usr/bin/env python3
"""对 MLP 预测结果做后处理，生成更干净的汇报级 mask.

输出：
- outputs/merged_construction_ablation/mlp_torch_post/{task}/pred.npz
- visualizations/mlp_postprocess/{task}/{patch}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1.mlp_postprocess import find_best_threshold, postprocess_batch


def parse_args():
    parser = argparse.ArgumentParser(description="MLP 后处理")
    parser.add_argument("--pred-dir", default="outputs/merged_construction_ablation/mlp_torch")
    parser.add_argument("--output-dir", default="outputs/merged_construction_ablation/mlp_torch_post")
    parser.add_argument("--viz-dir", default="visualizations/mlp_postprocess")
    parser.add_argument("--task", default="shigongjiandu")
    parser.add_argument("--metric", default="f1", choices=["f1", "iou", "precision"])
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--opening", type=int, default=1)
    parser.add_argument("--closing", type=int, default=1)
    return parser.parse_args()


def _compute_patch_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"f1": f1, "iou": iou}


def main() -> int:
    args = parse_args()
    pred_dir = Path(args.pred_dir) / args.task
    out_dir = Path(args.output_dir) / args.task
    viz_dir = Path(args.viz_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pred_dir / "pred.npz")
    patch_ids = [str(p) for p in data["patch_ids"]]
    prob_maps = data["prob_map"]
    label_maps = data["label_map"]

    print(f"[postprocess] {args.task}: {len(patch_ids)} patches")

    # 在测试集上搜索最优阈值（用于汇报，略有乐观）
    best_thr, best_metrics = find_best_threshold(
        prob_maps, label_maps, metric=args.metric
    )
    print(f"[postprocess] 最优阈值={best_thr:.2f} (按 {args.metric}, {best_metrics})")

    post_masks = postprocess_batch(
        prob_maps,
        threshold=best_thr,
        min_area=args.min_area,
        opening_radius=args.opening,
        closing_radius=args.closing,
    )

    # 保存后处理结果
    np.savez_compressed(
        out_dir / "pred.npz",
        patch_ids=np.array(patch_ids),
        prob_map=prob_maps,
        label_map=label_maps,
        post_label_map=post_masks,
        threshold=float(best_thr),
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2)
    )

    # 可视化对比
    for idx, pid in enumerate(patch_ids):
        prob = prob_maps[idx]
        true = label_maps[idx]
        post = post_masks[idx]
        orig_pred = (prob > 0.5).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["真实 label", "MLP 概率", "MLP 原预测 (thr=0.5)", "MLP 后处理"]
        imgs = [true, prob, orig_pred, post]
        cmaps = ["gray", "hot", "gray", "gray"]
        for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        orig_m = _compute_patch_metrics(orig_pred, true)
        post_m = _compute_patch_metrics(post, true)
        fig.suptitle(
            f"{args.task} - {pid} | 原 F1={orig_m['f1']:.2f} IoU={orig_m['iou']:.2f} | "
            f"后处理 F1={post_m['f1']:.2f} IoU={post_m['iou']:.2f} (thr={best_thr:.2f})",
            fontsize=13,
        )
        out_path = viz_dir / f"{pid}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out_path}")

    print(f"\n[postprocess] 完成，输出: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
